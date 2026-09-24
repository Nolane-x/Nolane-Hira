import copy

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.high_k_localization import (
    _rank_metrics,
    aggregate_w6f_records,
    compile_w6f_cache,
    diagnose_cached_view,
    localization_classification,
    uniform_salience_logits,
)
from nmd.high_k_localization_authority import generate_w6f_domain
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def _small_runtime():
    torch.manual_seed(4001)
    encoder = TrainableSemanticEncoder(
        vocab_size=8192,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    hira = HIRACore(d_model=256, dropout=0.0)
    model = NolaneHira(encoder, hira)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    return model, hira, scorer


def test_w6f_cache_reuses_production_compiler_contract():
    model, _, _ = _small_runtime()
    views = generate_w6f_domain("N")[:4]
    cache = compile_w6f_cache(model, views)

    assert cache["metadata"]["schema_version"] == "r8-w6f-high-k-cache-v1"
    assert cache["metadata"]["view_count"] == 4
    assert cache["metadata"]["base_count"] == 1
    assert cache["metadata"]["state_encode_calls_per_case"] == 1.0

    for case, view in zip(cache["cases"], views):
        assert case["base_id"] == view.base_id
        assert case["domain_id"] == "N"
        assert tuple(case["option_distances"]) == view.option_distances
        assert len(case["decisions"]) == 1
        assert case["decisions"][0]["question_id"] == "diagnosis"


def test_uniform_salience_ablation_is_permutation_equivariant_and_nonmutating():
    torch.manual_seed(4002)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    before = {
        key: value.detach().clone()
        for key, value in scorer.state_dict().items()
    }

    batch, k, t, s, q, d = 1, 8, 5, 7, 3, 256
    state_tokens = torch.randn(batch, s, d)
    question_tokens = torch.randn(batch, q, d)
    option_tokens = torch.randn(batch, k, t, d)
    option_ids = torch.randint(1, 500, (batch, k, t), dtype=torch.long)
    state_mask = torch.ones(batch, s, dtype=torch.bool)
    question_mask = torch.ones(batch, q, dtype=torch.bool)
    option_mask = torch.ones(batch, k, t, dtype=torch.bool)

    logits = uniform_salience_logits(
        scorer,
        state_tokens=state_tokens,
        state_mask=state_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_tokens=option_tokens,
        option_token_ids=option_ids,
        option_mask=option_mask,
    )

    perm = torch.tensor([4, 0, 7, 1, 5, 2, 6, 3])
    permuted = uniform_salience_logits(
        scorer,
        state_tokens=state_tokens,
        state_mask=state_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_tokens=option_tokens[:, perm],
        option_token_ids=option_ids[:, perm],
        option_mask=option_mask[:, perm],
    )

    assert logits.shape == (1, 8)
    assert torch.isfinite(logits).all()
    assert torch.allclose(permuted, logits[:, perm], atol=1e-6, rtol=1e-6)
    for key, value in scorer.state_dict().items():
        assert torch.equal(value, before[key])


def test_diagnose_cached_view_exposes_coarse_final_and_relation_anatomy():
    model, hira, scorer = _small_runtime()
    view = generate_w6f_domain("N")[0]
    cache = compile_w6f_cache(model, [view])
    result = diagnose_cached_view(hira, scorer, cache["cases"][0])

    assert result["diagnosis_k"] == 8
    for name in ("native_coarse", "uniform_coarse", "final"):
        metrics = result[name]
        assert 1 <= metrics["rank"] <= 8
        assert isinstance(metrics["top1"], bool)
        assert isinstance(metrics["top5"], bool)
        assert 0 < metrics["reciprocal_rank"] <= 1.0
        assert torch.isfinite(torch.tensor(metrics["gold_logit"]))
        assert torch.isfinite(torch.tensor(metrics["margin"]))

    relation = result["relation"]
    assert isinstance(relation["rescued"], bool)
    assert isinstance(relation["damaged"], bool)
    assert isinstance(relation["gold_rank_change"], int)
    assert torch.isfinite(torch.tensor(relation["gold_delta"]))
    assert result["winner_distance"] in {0, 1, 2, 3, 4}


def _k_fixture(
    *,
    native_top1,
    native_top5,
    uniform_top1,
    uniform_top5,
    final_top1,
    rescue=0.0,
    damage=0.0,
):
    return {
        "native_coarse_top1": native_top1,
        "native_coarse_top5": native_top5,
        "uniform_coarse_top1": uniform_top1,
        "uniform_coarse_top5": uniform_top5,
        "final_top1": final_top1,
        "relation_rescue_rate": rescue,
        "relation_damage_rate": damage,
    }


def test_localization_rules_are_frozen_and_nonoverclaiming():
    k8 = _k_fixture(
        native_top1=0.80,
        native_top5=0.95,
        uniform_top1=0.79,
        uniform_top5=0.96,
        final_top1=0.80,
    )
    salience_k64 = _k_fixture(
        native_top1=0.35,
        native_top5=0.68,
        uniform_top1=0.45,
        uniform_top5=0.79,
        final_top1=0.35,
    )
    out = localization_classification(k8, salience_k64)
    assert out["classification"] == "CANDIDATE_RELATIVE_SALIENCE_IMPLICATED"

    relation_k64 = _k_fixture(
        native_top1=0.60,
        native_top5=0.85,
        uniform_top1=0.61,
        uniform_top5=0.86,
        final_top1=0.50,
        rescue=0.02,
        damage=0.10,
    )
    out = localization_classification(k8, relation_k64)
    assert out["classification"] == "RELATION_RERANKING_IMPLICATED"

    coarse_k64 = _k_fixture(
        native_top1=0.38,
        native_top5=0.70,
        uniform_top1=0.40,
        uniform_top5=0.72,
        final_top1=0.39,
        rescue=0.03,
        damage=0.02,
    )
    out = localization_classification(k8, coarse_k64)
    assert out["classification"] == "COARSE_BINDING_LIMIT"

    mixed_k64 = _k_fixture(
        native_top1=0.50,
        native_top5=0.80,
        uniform_top1=0.60,
        uniform_top5=0.88,
        final_top1=0.42,
        rescue=0.01,
        damage=0.09,
    )
    out = localization_classification(k8, mixed_k64)
    assert out["classification"] == "MIXED_HIGH_K_FAILURE"


def _record(base_id, k, native_rank, final_rank, uniform_rank):
    return {
        "base_id": base_id,
        "case_id": f"{base_id}-k{k}",
        "domain_id": "N",
        "diagnosis_k": k,
        "winner_mismatch_fields": ("field-a",) if final_rank > 1 else (),
        "winner_distance": 1 if final_rank > 1 else 0,
        "native_coarse": {
            "rank": native_rank,
            "top1": native_rank == 1,
            "top5": native_rank <= 5,
            "reciprocal_rank": 1.0 / native_rank,
            "gold_logit": 1.0 - 0.01 * k,
            "margin": 0.4 - 0.01 * k,
        },
        "uniform_coarse": {
            "rank": uniform_rank,
            "top1": uniform_rank == 1,
            "top5": uniform_rank <= 5,
            "reciprocal_rank": 1.0 / uniform_rank,
            "gold_logit": 1.1 - 0.005 * k,
            "margin": 0.5 - 0.005 * k,
        },
        "final": {
            "rank": final_rank,
            "top1": final_rank == 1,
            "top5": final_rank <= 5,
            "reciprocal_rank": 1.0 / final_rank,
            "gold_logit": 0.9 - 0.01 * k,
            "margin": 0.3 - 0.01 * k,
        },
        "relation": {
            "gold_delta": -0.1,
            "coarse_best_negative_delta": 0.05,
            "final_best_negative_delta": 0.07,
            "rescued": native_rank > 1 and final_rank == 1,
            "damaged": native_rank == 1 and final_rank > 1,
            "gold_rank_change": final_rank - native_rank,
        },
    }


def test_aggregate_reports_cardinality_trajectory_and_error_anatomy():
    rows = []
    for base in ("a", "b"):
        for k in (8, 16, 32, 64):
            native = 1 if k < 64 else 2
            final = 1 if k < 32 else 2
            uniform = 1
            rows.append(_record(base, k, native, final, uniform))

    out = aggregate_w6f_records(rows)
    assert out["view_count"] == 8
    trajectory = out["base_trajectory"]
    assert trajectory["complete_base_count"] == 2
    assert trajectory["mean_native_coarse_rank_drift_k8_to_k64"] == 1.0
    assert trajectory["mean_uniform_coarse_rank_drift_k8_to_k64"] == 0.0
    assert trajectory["uniform_minus_native_rank_drift_k8_to_k64"] == -1.0
    assert trajectory["mean_final_rank_drift_k8_to_k64"] == 1.0
    assert trajectory["first_native_top1_loss_k"] == {"64": 2}
    assert trajectory["first_uniform_top1_loss_k"] == {"never": 2}
    assert trajectory["first_final_top1_loss_k"] == {"32": 2}
    assert out["winning_wrong_field_counts"]["field-a"] == 4
    assert out["winning_wrong_distance_counts"][1] == 4


def test_rank_metrics_match_deterministic_argmax_under_exact_ties():
    logits = torch.tensor([[1.0, 1.0, 0.5]])
    first = _rank_metrics(logits[0], 0)
    tied_later = _rank_metrics(logits[0], 1)
    assert first["predicted_index"] == 0
    assert first["rank"] == 1
    assert first["top1"] is True
    assert tied_later["predicted_index"] == 0
    assert tied_later["rank"] == 2
    assert tied_later["top1"] is False
    assert tied_later["margin"] == 0.0
