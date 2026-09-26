from __future__ import annotations

import torch

from nmd.hira import HIRACore
from nmd.regime_transfer_authority import (
    BASES_PER_DOMAIN,
    K_VALUES,
    PARAPHRASE_VIEWS,
    generate_w16_domain,
)
from nmd.regime_transfer_cache import compile_w16_cache
from nmd.regime_transfer_eval import (
    EXTRA_STATE_TOKEN_DILUTION,
    OUTCOME_STABLE,
    classify_domain,
    cross_domain_outcome,
    directional_scores,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w16_authority_shape_and_rendering_identity():
    rows = generate_w16_domain("CG")
    assert len(rows) == BASES_PER_DOMAIN * len(K_VALUES) * len(PARAPHRASE_VIEWS)
    by_base = {}
    for row in rows:
        by_base.setdefault(row.base_id, []).append(row)
    assert len(by_base) == BASES_PER_DOMAIN
    for group in by_base.values():
        assert len(group) == 9
        assert len({row.bare_state_text for row in group}) == 1
        assert len({row.decorated_state_text for row in group}) == 1
        assert all("Reported impact is " in row.decorated_state_text for row in group)
        by_k = {}
        for row in group:
            by_k.setdefault(row.diagnosis_k, []).append(row)
        assert set(by_k) == {4, 8, 16}
        for k, views in by_k.items():
            assert {row.view_id for row in views} == {"D0", "D1", "D2"}
            assert len({row.option_ids for row in views}) == 1
            assert len({row.gold_index for row in views}) == 1
            assert len(next(iter({row.option_ids for row in views}))) == k


def test_w16_cache_r2_reuses_decorated_prefix_without_extra_encode():
    rows = generate_w16_domain("CG")
    base_id = rows[0].base_id
    rows = [row for row in rows if row.base_id == base_id]

    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=96,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    cache = compile_w16_cache(model, rows)
    assert cache["metadata"]["base_count"] == 1
    assert cache["metadata"]["view_count"] == 9
    assert cache["metadata"]["encoded_state_texts_per_base"] == 2.0
    assert cache["metadata"]["r2_additional_encoder_calls"] == 0
    assert cache["metadata"]["prefix_token_identity_rate"] == 1.0

    renderings = cache["bases"][0]["state_renderings"]
    assert torch.equal(
        renderings["R0"]["content_token_ids"],
        renderings["R2"]["content_token_ids"],
    )
    assert torch.equal(
        renderings["R1"]["content_token_ids"][
            : renderings["R0"]["content_token_ids"].numel()
        ],
        renderings["R0"]["content_token_ids"],
    )
    assert renderings["R1"]["content_tokens"].shape[0] > renderings["R0"]["content_tokens"].shape[0]


def test_w16_directional_score_contract():
    # Two options, two option tokens each. Identity projection makes the
    # directionality easy to verify.
    state = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    options = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    scores = directional_scores(state, options, mask, torch.eye(2))
    assert set(scores) == {"D2S", "S2D", "SYM"}
    assert torch.allclose(scores["SYM"], 0.5 * (scores["D2S"] + scores["S2D"]))
    assert scores["SYM"][0] > scores["SYM"][1]


def _classification_fixture():
    def row(top1):
        return {"top1": top1, "top5": 1.0, "mrr": top1, "mean_margin": top1}

    metrics = {}
    for rendering in ("R0", "R1", "R2"):
        metrics[rendering] = {}
        for operator in ("D2S", "S2D", "SYM"):
            metrics[rendering][operator] = {
                "4": row(.9),
                "8": row(.85),
                "16": row(.8),
            }

    # Full decoration hurts symmetric mainly through S2D.
    metrics["R1"]["SYM"]["4"] = row(.70)
    metrics["R1"]["SYM"]["16"] = row(.60)
    metrics["R2"]["SYM"]["4"] = row(.88)
    metrics["R2"]["SYM"]["16"] = row(.77)

    metrics["R1"]["D2S"]["4"] = row(.87)
    metrics["R1"]["D2S"]["16"] = row(.77)
    metrics["R1"]["S2D"]["4"] = row(.72)
    metrics["R1"]["S2D"]["16"] = row(.62)

    reference = {
        "R0": {
            "4": row(.95),
            "8": row(.93),
            "16": row(.90),
        },
        "R1": {
            "4": row(.90),
            "8": row(.88),
            "16": row(.84),
        },
    }
    return metrics, reference


def test_w16_extra_token_dilution_classifier_contract():
    metrics, reference = _classification_fixture()
    result = classify_domain(metrics, reference)
    assert result["classification"] == EXTRA_STATE_TOKEN_DILUTION
    assert result["adequate"] is True


def test_w16_cross_domain_stability_contract():
    per_domain = {
        domain: {"classification": EXTRA_STATE_TOKEN_DILUTION}
        for domain in ("CG", "CH", "CI")
    }
    per_domain["CJ"] = {"classification": "REGIME_TRANSFER_UNRESOLVED"}
    result = cross_domain_outcome(per_domain)
    assert result["outcome"] == OUTCOME_STABLE
    assert result["stable_classification"] == EXTRA_STATE_TOKEN_DILUTION
