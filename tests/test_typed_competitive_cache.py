import math

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.typed_competitive_authority import generate_w6b_authority
from nmd.typed_competitive_cache import (
    HIRA_PARAMETER_COUNT,
    JOINT_PARAMETER_COUNT,
    SCORER_PARAMETER_COUNT,
    absolute_production_gates,
    compile_w6b_cache,
    configure_candidate_trainability,
    _ece15,
    confirm_verdict,
    dev_selection_key,
    evaluate_w6b_cases,
    mechanism_gates,
    validate_w6b_cache,
)


def make_small_cache(case_count: int = 2):
    torch.manual_seed(9001)
    encoder = TrainableSemanticEncoder(
        vocab_size=4096,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    cases = generate_w6b_authority("train")[:case_count]
    return compile_w6b_cache(
        model,
        cases,
        expected_split="train",
    ), model


def test_w6b_cache_compiles_state_once_and_preserves_token_artifacts():
    cache, model = make_small_cache(2)
    validate_w6b_cache(cache, expected_split="train")
    assert cache["metadata"]["case_count"] == 2
    assert cache["metadata"]["decision_count"] == 10
    assert cache["metadata"]["state_encode_calls"] == 2
    assert cache["metadata"]["state_encode_calls_per_case"] == 1.0
    assert cache["metadata"]["confirm_exposed"] is False
    assert model.state_encode_calls == 2

    case = cache["cases"][0]
    assert case["state_content_tokens"].shape[-1] == 256
    assert len(case["decisions"]) == 5
    diagnosis = case["decisions"][0]
    assert diagnosis["option_token_ids"].dtype == torch.long
    assert diagnosis["option_content_mask"].dtype == torch.bool
    assert diagnosis["option_tokens"].shape[:2] == (
        diagnosis["option_token_ids"].shape
    )


def test_cached_legacy_and_competitive_paths_are_finite_and_normalized():
    cache, _ = make_small_cache(1)
    torch.manual_seed(9002)
    legacy = HIRACore(d_model=256, dropout=0.0)
    competitive_hira = HIRACore(d_model=256, dropout=0.0)
    competitive_hira.load_state_dict(legacy.state_dict(), strict=True)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)

    legacy_metrics = evaluate_w6b_cases(
        legacy,
        None,
        cache["cases"],
        competitive=False,
    )
    competitive_metrics = evaluate_w6b_cases(
        competitive_hira,
        scorer,
        cache["cases"],
        competitive=True,
    )
    for metrics in (legacy_metrics, competitive_metrics):
        assert math.isfinite(float(metrics["accuracy"]))
        assert math.isfinite(float(metrics["hard_brier"]))
        assert math.isfinite(float(metrics["ece"]))
        assert math.isfinite(float(metrics["raw_ece"]))
        assert math.isfinite(float(metrics["soft_ece"]))
        assert math.isfinite(float(metrics["score_mae"]))
        assert metrics["probability_mass_max_error"] <= 1e-6
        assert metrics["source_state_encodes_per_case"] == 1.0
        assert metrics["decision_count"] == 5


def test_candidate_trainability_counts_are_frozen():
    legacy = HIRACore(d_model=256, dropout=0.0)
    competitive_hira = HIRACore(d_model=256, dropout=0.0)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)

    competitive, params = configure_candidate_trainability(
        "legacy-w3-joint",
        legacy,
        None,
    )
    assert competitive is False
    assert sum(p.numel() for p in params) == HIRA_PARAMETER_COUNT

    competitive, params = configure_candidate_trainability(
        "competitive-w5i-joint",
        competitive_hira,
        scorer,
    )
    assert competitive is True
    assert sum(p.numel() for p in params) == JOINT_PARAMETER_COUNT

    competitive_hira = HIRACore(d_model=256, dropout=0.0)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    competitive, params = configure_candidate_trainability(
        "competitive-w5i-scorer-only",
        competitive_hira,
        scorer,
    )
    assert competitive is True
    assert sum(p.numel() for p in params) == SCORER_PARAMETER_COUNT
    assert not any(p.requires_grad for p in competitive_hira.parameters())
    assert all(p.requires_grad for p in scorer.parameters())


def metric_fixture(
    *,
    accuracy: float,
    choice: float,
    noul: float,
    score: float,
    k64: float,
    brier: float,
    ece: float,
    score_mae: float,
):
    return {
        "accuracy": accuracy,
        "primitive_accuracy": {
            "choice": choice,
            "noul": noul,
            "score": score,
        },
        "diagnosis_per_k": {
            "8": {"accuracy": k64},
            "16": {"accuracy": k64},
            "32": {"accuracy": k64},
            "64": {"accuracy": k64},
        },
        "hard_brier": brier,
        "soft_accuracy": max(0.0, accuracy - 0.1),
        "ece": ece,
        "raw_ece": ece,
        "soft_ece": ece,
        "score_mae": score_mae,
        "probability_mass_max_error": 1e-7,
        "source_state_encodes_per_case": 1.0,
    }


def test_w6b_rescue_requires_absolute_and_mechanism_gates():
    legacy = metric_fixture(
        accuracy=0.66,
        choice=0.66,
        noul=0.72,
        score=0.62,
        k64=0.56,
        brier=0.45,
        ece=0.12,
        score_mae=0.50,
    )
    selected = metric_fixture(
        accuracy=0.72,
        choice=0.72,
        noul=0.78,
        score=0.68,
        k64=0.64,
        brier=0.41,
        ece=0.10,
        score_mae=0.47,
    )
    verdict, absolute, mechanism = confirm_verdict(selected, legacy)
    assert verdict == "PRODUCTION_COMPETITIVE_RESCUE"
    assert all(absolute.values())
    assert all(mechanism.values())


def test_legacy_already_strong_requires_legacy_absolute_competence():
    legacy = metric_fixture(
        accuracy=0.68,
        choice=0.68,
        noul=0.74,
        score=0.64,
        k64=0.58,
        brier=0.44,
        ece=0.11,
        score_mae=0.49,
    )
    selected = metric_fixture(
        accuracy=0.69,
        choice=0.69,
        noul=0.75,
        score=0.65,
        k64=0.59,
        brier=0.43,
        ece=0.11,
        score_mae=0.48,
    )
    verdict, _, mechanism = confirm_verdict(selected, legacy)
    assert verdict == "PRODUCTION_LEGACY_ALREADY_STRONG"
    assert not all(mechanism.values())


def test_noncompetent_legacy_cannot_be_mislabeled_already_strong():
    legacy = metric_fixture(
        accuracy=0.58,
        choice=0.58,
        noul=0.65,
        score=0.55,
        k64=0.45,
        brier=0.60,
        ece=0.20,
        score_mae=0.70,
    )
    selected = metric_fixture(
        accuracy=0.62,
        choice=0.62,
        noul=0.68,
        score=0.58,
        k64=0.50,
        brier=0.55,
        ece=0.18,
        score_mae=0.64,
    )
    verdict, _, _ = confirm_verdict(selected, legacy)
    assert verdict == "PRODUCTION_COMPETITIVE_PARTIAL"


def test_dev_selection_key_uses_frozen_order():
    base = metric_fixture(
        accuracy=0.70,
        choice=0.70,
        noul=0.75,
        score=0.65,
        k64=0.60,
        brier=0.45,
        ece=0.10,
        score_mae=0.50,
    )
    better_accuracy = {**base, "accuracy": 0.71}
    assert dev_selection_key(better_accuracy, 6) < dev_selection_key(base, 1)

    equal_acc_better_brier = {**base, "hard_brier": 0.44}
    assert dev_selection_key(
        equal_acc_better_brier,
        6,
    ) < dev_selection_key(base, 1)


def test_absolute_and_mechanism_gate_helpers_are_explicit():
    selected = metric_fixture(
        accuracy=0.70,
        choice=0.70,
        noul=0.75,
        score=0.65,
        k64=0.60,
        brier=0.45,
        ece=0.10,
        score_mae=0.50,
    )
    legacy = metric_fixture(
        accuracy=0.65,
        choice=0.65,
        noul=0.70,
        score=0.60,
        k64=0.54,
        brier=0.46,
        ece=0.11,
        score_mae=0.51,
    )
    absolute = absolute_production_gates(selected)
    mechanism = mechanism_gates(selected, legacy)
    assert set(absolute) == {
        "overall_accuracy",
        "choice_accuracy",
        "noul_accuracy",
        "score_accuracy",
        "diagnosis_k64_accuracy",
        "hard_brier",
        "soft_ece",
        "score_mae",
        "probability_mass",
        "state_once",
    }
    assert set(mechanism) == {
        "overall_accuracy_gain",
        "diagnosis_k64_gain",
        "hard_brier_nonregression",
        "soft_ece_nonregression",
        "score_mae_nonregression",
    }


def test_soft_ece_respects_declared_reliability_targets():
    confidence = [0.90, 0.75, 0.60]
    hard_correctness = [1.0, 1.0, 1.0]
    soft_targets = [0.90, 0.75, 0.60]
    assert _ece15(confidence, soft_targets) < 1e-12
    assert _ece15(confidence, hard_correctness) > 0.15


def test_selection_and_gates_use_soft_ece_not_raw_hard_ece():
    base = metric_fixture(
        accuracy=0.70,
        choice=0.70,
        noul=0.75,
        score=0.65,
        k64=0.60,
        brier=0.45,
        ece=0.10,
        score_mae=0.50,
    )
    reliability_calibrated = {
        **base,
        "ece": 0.40,
        "raw_ece": 0.40,
        "soft_ece": 0.08,
    }
    hard_calibrated_only = {
        **base,
        "ece": 0.01,
        "raw_ece": 0.01,
        "soft_ece": 0.18,
    }
    assert dev_selection_key(
        reliability_calibrated, 6
    ) < dev_selection_key(hard_calibrated_only, 6)
    gates = absolute_production_gates(reliability_calibrated)
    assert gates["soft_ece"] is True
