import copy

import torch

from nmd.hira import HIRACore
from nmd.mainline import HiraV0Mainline, HiraV0Manifest
from nmd.mainline_m2_authority import generate_m2_high_k_semantic
from nmd.mainline_m2_eval import (
    M2_DEV_GATES,
    M2_SEALED_GATES,
    evaluate_m2_semantic_cases,
    m2_dev_qualification,
    m2_sealed_qualification,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.w33_coevidence_semantic import CoEvidenceSemanticScorer


def _model():
    torch.manual_seed(21801)
    encoder = TrainableSemanticEncoder(
        vocab_size=4096,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    scorer = CoEvidenceSemanticScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(torch.randn(128, 256) * 0.02, freeze=True)
    scorer.freeze_candidate()
    runtime = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
        coevidence_symmetric_semantic_scorer=scorer,
    )
    return HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m2_mechanics_available(),
    )


def _metrics(k, top1, top5, mrr):
    return {
        "k": k,
        "case_count": 16,
        "top1": top1,
        "top5": top5,
        "mrr": mrr,
        "mean_gold_rank": 1.5,
        "mean_gold_probability": 0.5,
        "mean_confidence": 0.7,
        "probability_mass_max_error": 1e-7,
        "permutation_max_error": 1e-9,
        "selected_option_invariant_rate": 1.0,
        "full_k_rate": 1.0,
        "state_once_rate": 1.0,
        "relation_delta_max": 0.0,
        "finite_rate": 1.0,
        "mean_case_ms": 1.0,
    }


def test_m2_semantic_dev_gates_are_frozen():
    assert M2_DEV_GATES == {
        64: {"top1": 0.70, "top5": 0.90, "mrr": 0.75},
        128: {"top1": 0.60, "top5": 0.85, "mrr": 0.65},
    }
    assert M2_SEALED_GATES == {
        255: {"top1": 0.50, "top5": 0.80, "mrr": 0.58},
    }


def test_m2_dev_qualification_requires_semantics_and_mechanics():
    evaluation = {
        "per_k": {
            "64": _metrics(64, 0.80, 0.95, 0.82),
            "128": _metrics(128, 0.70, 0.90, 0.72),
        }
    }
    passed = m2_dev_qualification(evaluation)
    assert passed["pass"] is True
    assert passed["per_k"]["64"]["pass"] is True
    assert passed["per_k"]["128"]["pass"] is True

    broken = copy.deepcopy(evaluation)
    broken["per_k"]["128"]["top1"] = 0.20
    failed = m2_dev_qualification(broken)
    assert failed["pass"] is False
    assert failed["per_k"]["128"]["semantic"]["top1"] is False


def test_m2_sealed_gate_requires_k255_semantic_quality():
    evaluation = {
        "per_k": {
            "255": _metrics(255, 0.60, 0.90, 0.65),
        }
    }
    passed = m2_sealed_qualification(evaluation)
    assert passed["pass"] is True

    broken = copy.deepcopy(evaluation)
    broken["per_k"]["255"]["permutation_max_error"] = 1e-2
    failed = m2_sealed_qualification(broken)
    assert failed["pass"] is False
    assert failed["mechanics"]["permutation"] is False


def test_actual_evaluator_preserves_mechanics_on_fresh_small_k_fixture():
    model = _model()
    rows = [
        row
        for row in generate_m2_high_k_semantic("train")
        if row.k == 4
    ][:2]

    result = evaluate_m2_semantic_cases(model, rows)
    metrics = result["per_k"]["4"]

    assert result["case_count"] == 2
    assert result["state_encode_count"] == 2
    assert result["state_encodes_per_case"] == 1.0
    assert result["candidate_pruning_used"] is False
    assert result["relation_refinement_used"] is False
    assert result["adaptive_budget_used"] is False

    assert metrics["k"] == 4
    assert metrics["case_count"] == 2
    assert metrics["probability_mass_max_error"] <= 1e-6
    assert metrics["permutation_max_error"] <= 2e-6
    assert metrics["selected_option_invariant_rate"] == 1.0
    assert metrics["full_k_rate"] == 1.0
    assert metrics["state_once_rate"] == 1.0
    assert metrics["relation_delta_max"] == 0.0
    assert metrics["finite_rate"] == 1.0
