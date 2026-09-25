from __future__ import annotations

import torch

from nmd.anchor_preserving_residual import AnchorPreservingResidualMixer
from nmd.anchor_preserving_residual_authority import generate_w15_dev
from nmd.anchor_preserving_residual_cache import (
    compile_w15_cache,
    validate_w15_cache,
)
from nmd.anchor_preserving_residual_training import (
    CANDIDATES,
    bound_causal_gates,
    candidate_logits,
    decision_components,
    dev_selection_key,
    evaluate_w15,
)
from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def _one_case_cache():
    torch.manual_seed(7)
    encoder = TrainableSemanticEncoder(
        vocab_size=2048,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    model.eval()
    case = generate_w15_dev()[0]
    cache = compile_w15_cache(
        model,
        [case],
        expected_split="dev-cd",
    )
    return cache


def test_w15_cache_is_state_once_and_preserves_multiview_identity():
    cache = _one_case_cache()
    validate_w15_cache(cache, expected_split="dev-cd")
    assert cache["metadata"]["case_count"] == 1
    assert cache["metadata"]["decision_count"] == 5
    assert cache["metadata"]["schema_view_count"] == 15
    assert cache["metadata"]["state_encode_calls"] == 1
    assert cache["metadata"]["state_encode_calls_per_case"] == 1.0

    row = cache["cases"][0]
    assert len(row["decisions"]) == 5
    for decision in row["decisions"]:
        views = decision["views"]
        assert set(views) == {"D0", "D1", "D2"}
        ids = views["D0"]["option_ids"]
        assert views["D1"]["option_ids"] == ids
        assert views["D2"]["option_ids"] == ids
        assert views["D0"]["option_texts"] != views["D1"]["option_texts"]
        assert views["D0"]["option_texts"] != views["D2"]["option_texts"]


def test_w15_components_and_all_candidate_logits_have_valid_shapes():
    cache = _one_case_cache()
    case = cache["cases"][0]
    torch.manual_seed(11)
    hira = HIRACore(d_model=256, dropout=0.0)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    hira.eval()
    scorer.eval()

    for decision in case["decisions"]:
        components = decision_components(hira, scorer, case, decision)
        k = len(decision["views"]["D0"]["option_ids"])
        assert components["anchor_logits"].shape == (k,)
        assert components["d0_anchor_logits"].shape == (k,)
        assert components["competitive_logits"].shape == (k,)
        assert components["relation_delta"].shape == (k,)
        assert components["production_logits"].shape == (k,)

        for candidate in CANDIDATES:
            mixer = None
            if "residual" in candidate:
                mixer = AnchorPreservingResidualMixer(
                    bounded=not candidate.startswith("unbounded")
                )
            logits, residual = candidate_logits(
                candidate,
                components,
                decision,
                mixer,
            )
            assert logits.shape == (k,)
            assert bool(torch.isfinite(logits).all())
            if mixer is not None:
                assert residual is not None


def test_w15_one_case_evaluation_reports_anchor_transitions_and_integrity():
    cache = _one_case_cache()
    case = cache["cases"]
    torch.manual_seed(13)
    hira = HIRACore(d_model=256, dropout=0.0)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    mixer = AnchorPreservingResidualMixer(bounded=True)

    metrics = evaluate_w15(
        "bounded-residual-primary",
        hira,
        scorer,
        case,
        mixer=mixer,
    )
    assert metrics["case_count"] == 1
    assert metrics["decision_count"] == 5
    assert metrics["source_state_encodes_per_case"] == 1.0
    assert metrics["probability_mass_max_error"] <= 1e-12
    k = str(case[0]["diagnosis_k"])
    assert metrics["diagnosis_per_k"][k]["n"] == 1
    assert metrics["anchor_diagnosis_per_k"][k]["n"] == 1
    assert metrics["production_diagnosis_per_k"][k]["n"] == 1
    assert (
        metrics["anchor_transitions_per_k"][k]["anchor_correct_count"]
        + metrics["anchor_transitions_per_k"][k]["anchor_wrong_count"]
        == 1
    )
    assert metrics["residual_diagnostics"]["max_bound_violation"] <= 1e-6


def test_w15_dev_selection_key_prefers_k16_then_retention():
    base = {
        "diagnosis_per_k": {
            "4": {"top1": 0.8},
            "8": {"top1": 0.7},
            "16": {"top1": 0.6},
        },
        "anchor_transitions_per_k": {
            "16": {"anchor_correct_retention": 0.9},
        },
        "accuracy": 0.75,
        "score_mae": 0.2,
        "soft_ece": 0.1,
    }
    better = {
        **base,
        "diagnosis_per_k": {
            **base["diagnosis_per_k"],
            "16": {"top1": 0.61},
        },
    }
    assert dev_selection_key(better, 8) < dev_selection_key(base, 1)


def test_w15_bound_causal_gate_semantics_are_frozen():
    def metrics(k16, retention, overall=0.9, non_diag=0.9):
        return {
            "diagnosis_per_k": {
                "4": {"top1": 0.9},
                "8": {"top1": 0.85},
                "16": {"top1": k16},
            },
            "anchor_transitions_per_k": {
                "16": {"anchor_correct_retention": retention},
            },
            "accuracy": overall,
            "non_diagnosis_accuracy": non_diag,
        }

    bounded = metrics(0.75, 0.95)
    unbounded = metrics(0.69, 0.90)
    gates = bound_causal_gates(bounded, unbounded)
    assert gates == {
        "retention_or_k16": True,
        "overall_nonregression": True,
        "non_diag_nonregression": True,
    }
