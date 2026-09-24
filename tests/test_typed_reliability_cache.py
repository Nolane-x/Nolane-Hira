import math

import torch

from nmd.calibration import TypedReliabilityCalibrator
from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.typed_reliability_authority import generate_w6c_authority
from nmd.typed_reliability_cache import (
    CANDIDATES,
    absolute_production_gates,
    calibration_case_loss,
    compile_w6c_logit_cache,
    confirm_verdict,
    dev_selection_key,
    evaluate_w6c_cases,
    mechanism_gates,
    train_w6c_candidate,
    validate_w6c_logit_cache,
)


def make_cache(split="train", count=2):
    torch.manual_seed(10101)
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
        coarse_scorer=CompetitiveCoarseScorer(
            d_model=256,
            d_rel=128,
        ),
    )
    cases = generate_w6c_authority(split)[:count]
    cache = compile_w6c_logit_cache(
        model,
        cases,
        expected_split=split,
    )
    return cache, model


def test_logit_cache_is_state_once_and_contains_no_trainable_model():
    cache, model = make_cache("train", 2)
    validate_w6c_logit_cache(
        cache,
        expected_split="train",
    )
    assert cache["metadata"]["case_count"] == 2
    assert cache["metadata"]["decision_count"] == 10
    assert cache["metadata"]["state_encode_calls"] == 2
    assert cache["metadata"]["state_encode_calls_per_case"] == 1.0
    assert model.state_encode_calls == 2
    for case in cache["cases"]:
        for decision in case["decisions"]:
            assert decision["raw_logits"].requires_grad is False


def test_control_and_identity_calibrator_are_exactly_equivalent():
    cache, _ = make_cache("train", 2)
    control = evaluate_w6c_cases(None, cache)
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    identity = evaluate_w6c_cases(calibrator, cache)
    for key in (
        "accuracy",
        "soft_accuracy",
        "hard_brier",
        "soft_brier",
        "raw_ece",
        "soft_ece",
        "score_mae",
    ):
        assert control[key] == identity[key]
    assert control["primitive_accuracy"] == identity["primitive_accuracy"]
    assert control["diagnosis_per_k"] == identity["diagnosis_per_k"]


def test_calibration_loss_backprop_reaches_only_calibrator():
    cache, _ = make_cache("train", 1)
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    loss = calibration_case_loss(
        calibrator,
        cache["cases"][0],
    )
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert calibrator.log_temperature.grad is not None
    assert calibrator.noul_true_bias is not None
    assert calibrator.noul_true_bias.grad is not None


def test_candidate_parameter_contract_is_zero_three_four():
    expected = {
        "frozen-production-control": 0,
        "primitive-temperature": 3,
        "primitive-temperature-noul-bias": 4,
    }
    for candidate in CANDIDATES:
        if candidate == "frozen-production-control":
            count = 0
        else:
            calibrator = TypedReliabilityCalibrator(candidate)
            count = calibrator.trainable_parameter_count
        assert count == expected[candidate]


def metric_fixture(
    *,
    accuracy=0.80,
    choice=0.80,
    noul=0.72,
    score=0.85,
    k64=0.60,
    hard_brier=0.30,
    soft_brier=0.12,
    soft_ece=0.10,
    score_mae=0.25,
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
        "soft_accuracy": 0.55,
        "hard_brier": hard_brier,
        "soft_brier": soft_brier,
        "raw_ece": 0.20,
        "soft_ece": soft_ece,
        "score_mae": score_mae,
        "probability_mass_max_error": 1e-7,
        "source_state_encodes_per_case": 1.0,
    }


def test_dev_selection_key_prioritizes_soft_ece_then_noul():
    base = metric_fixture(soft_ece=0.10, noul=0.72)
    lower_ece = metric_fixture(soft_ece=0.09, noul=0.70)
    assert dev_selection_key(
        lower_ece, 8
    ) < dev_selection_key(base, 1)

    equal_ece_higher_noul = metric_fixture(
        soft_ece=0.10,
        noul=0.73,
    )
    assert dev_selection_key(
        equal_ece_higher_noul, 8
    ) < dev_selection_key(base, 1)


def test_reliability_rescue_requires_absolute_and_mechanism_gates():
    control = metric_fixture(
        accuracy=0.80,
        noul=0.68,
        soft_ece=0.18,
    )
    selected = metric_fixture(
        accuracy=0.80,
        noul=0.72,
        soft_ece=0.12,
    )
    verdict, absolute, mechanism = confirm_verdict(
        selected,
        control,
    )
    assert verdict == "RELIABILITY_CALIBRATION_RESCUE"
    assert all(absolute.values())
    assert all(mechanism.values())


def test_control_already_rescues_is_not_misattributed():
    control = metric_fixture(
        accuracy=0.80,
        noul=0.72,
        soft_ece=0.12,
    )
    selected = metric_fixture(
        accuracy=0.80,
        noul=0.72,
        soft_ece=0.11,
    )
    verdict, _, mechanism = confirm_verdict(
        selected,
        control,
    )
    assert verdict == "RELIABILITY_CONTROL_ALREADY_RESCUES"
    assert not all(mechanism.values())


def test_partial_signal_is_frozen_before_confirm():
    control = metric_fixture(
        accuracy=0.80,
        noul=0.68,
        soft_ece=0.18,
    )
    selected = metric_fixture(
        accuracy=0.79,
        noul=0.69,
        soft_ece=0.155,
    )
    verdict, _, _ = confirm_verdict(
        selected,
        control,
    )
    assert verdict == "RELIABILITY_CALIBRATION_PARTIAL"


def test_gate_key_sets_are_explicit():
    selected = metric_fixture()
    control = metric_fixture(
        noul=0.68,
        soft_ece=0.18,
    )
    assert set(absolute_production_gates(selected)) == {
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
    assert set(mechanism_gates(selected, control)) == {
        "soft_ece_improvement",
        "noul_gain_or_control_pass",
        "overall_nonregression",
        "diagnosis_k64_nonregression",
        "hard_brier_nonregression",
        "score_mae_nonregression",
    }


def test_temperature_only_preserves_all_hard_predictions_on_cached_production_logits():
    cache, _ = make_cache("train", 4)
    control = evaluate_w6c_cases(None, cache)
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature"
    )
    with torch.no_grad():
        calibrator.log_temperature.copy_(
            torch.tensor([1.0, -0.7, 0.5])
        )
    calibrated = evaluate_w6c_cases(
        calibrator,
        cache,
    )
    assert calibrated["accuracy"] == control["accuracy"]
    assert (
        calibrated["primitive_accuracy"]
        == control["primitive_accuracy"]
    )
    assert (
        calibrated["diagnosis_per_k"]
        == control["diagnosis_per_k"]
    )


def test_noul_bias_cannot_change_choice_score_or_diagnosis_predictions():
    cache, _ = make_cache("train", 8)
    control = evaluate_w6c_cases(None, cache)
    calibrator = TypedReliabilityCalibrator(
        "primitive-temperature-noul-bias"
    )
    with torch.no_grad():
        calibrator.noul_true_bias.fill_(5.0)
    calibrated = evaluate_w6c_cases(
        calibrator,
        cache,
    )
    assert (
        calibrated["primitive_accuracy"]["choice"]
        == control["primitive_accuracy"]["choice"]
    )
    assert (
        calibrated["primitive_accuracy"]["score"]
        == control["primitive_accuracy"]["score"]
    )
    assert (
        calibrated["diagnosis_per_k"]
        == control["diagnosis_per_k"]
    )
