import copy
from unittest.mock import patch

import torch

from nmd.mainline import W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
from nmd.mainline_m1_cache import M1_CACHE_SCHEMA, OOD_FEATURE_NAMES
from nmd.mainline_m1_r2_training import (
    RISK_CANDIDATES,
    RISK_FEATURE_INDICES,
    TinySelectiveRiskHead,
    evaluate_risk_head,
    m1_r2_dev_qualification,
    safe_calibration_tournament,
    train_selective_risk_tournament,
)
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256


def _metadata(partition, count):
    return {
        "schema_version": M1_CACHE_SCHEMA,
        "partition": partition,
        "case_count": count,
        "state_encode_count": count,
        "state_encodes_per_case": 1.0,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "transfer_core_maturity": "provisional",
        "production_ready": False,
        "decision_core_frozen": True,
        "calibration_applied": False,
        "ood_feature_names": list(OOD_FEATURE_NAMES),
        "semantic_ood_feature_count": 4,
        "confidence_ood_feature_count": 3,
    }


def _case(case_id, correct, feature_signal):
    logits = torch.tensor(
        [2.5, 0.0] if correct else [0.0, 2.5],
        dtype=torch.float32,
    )
    probabilities = torch.softmax(logits, dim=-1)
    return {
        "case_id": case_id,
        "domain_id": "r2-unit",
        "partition": "unit",
        "primitive": "noul",
        "primitive_id": 2,
        "is_ood": False,
        "ood_kind": None,
        "confidence_band": "unit",
        "gold_index": 0,
        "gold_probabilities": torch.tensor([0.9, 0.1]),
        "logits": logits,
        "probabilities": probabilities,
        "ood_features": torch.tensor(
            [
                feature_signal,
                feature_signal * 0.9,
                feature_signal * 0.8,
                0.05,
                float(probabilities.max()),
                0.2,
                float(probabilities.max() - probabilities.min()),
                0.1,
            ],
            dtype=torch.float32,
        ),
        "option_values": (0.0, 1.0),
        "option_ids": ("no", "yes"),
    }


def _cache(partition, cases):
    rows = copy.deepcopy(cases)
    for row in rows:
        row["partition"] = partition
    return {
        "metadata": _metadata(partition, len(rows)),
        "cases": rows,
    }


def test_safe_calibration_rejects_lower_ece_candidate_that_breaks_accuracy():
    control = {
        "accuracy": 0.70,
        "soft_ece": 0.13,
        "soft_nll": 1.0,
        "soft_brier": 0.25,
        "probability_mass_max_error": 1e-7,
    }
    good = {
        "candidate": "primitive-temperature",
        "parameter_count": 3,
        "selected_epoch": 5,
        "selected_dev": {
            "accuracy": 0.70,
            "soft_ece": 0.10,
            "soft_nll": 0.95,
            "soft_brier": 0.23,
            "probability_mass_max_error": 1e-7,
        },
        "state_dict": {"x": torch.tensor([1.0])},
        "history": [],
    }
    bad = {
        "candidate": "primitive-temperature-noul-bias",
        "parameter_count": 4,
        "selected_epoch": 1,
        "selected_dev": {
            "accuracy": 0.60,
            "soft_ece": 0.05,
            "soft_nll": 0.96,
            "soft_brier": 0.24,
            "probability_mass_max_error": 1e-7,
        },
        "state_dict": {"x": torch.tensor([2.0])},
        "history": [],
    }
    control_row = {
        "candidate": "control",
        "parameter_count": 0,
        "selected_epoch": 0,
        "selected_dev": control,
        "state_dict": None,
        "history": [],
    }
    fake = {
        "control_dev": control,
        "candidates": [control_row, good, bad],
    }

    with patch(
        "nmd.mainline_m1_r2_training.train_calibration_tournament",
        return_value=fake,
    ):
        out = safe_calibration_tournament(
            {"cases": [1]},
            {"cases": [1]},
        )

    assert out["selected_candidate"] == "primitive-temperature"
    summaries = {row["candidate"]: row for row in out["candidate_summaries"]}
    assert summaries["primitive-temperature"]["eligible"] is True
    assert summaries["primitive-temperature-noul-bias"]["eligible"] is False
    assert summaries["primitive-temperature-noul-bias"]["accuracy_guardrail"] is False


def _risk_caches():
    train = []
    dev = []
    for i in range(80):
        correct = i % 4 != 0
        signal = 0.9 if correct else -0.8
        train.append(_case(f"train-{i}", correct, signal))
    for i in range(40):
        correct = i % 4 != 0
        signal = 0.85 if correct else -0.75
        dev.append(_case(f"dev-{i}", correct, signal))
    return _cache("train", train), _cache("dev", dev)


def test_risk_candidates_are_tiny_and_have_semantic_features():
    assert RISK_CANDIDATES == (
        "semantic-risk-linear",
        "semantic-confidence-risk-linear",
    )
    assert len(RISK_FEATURE_INDICES["semantic-risk-linear"]) == 5
    assert len(RISK_FEATURE_INDICES["semantic-confidence-risk-linear"]) == 8
    assert set(range(4)).issubset(RISK_FEATURE_INDICES["semantic-risk-linear"])
    assert set(range(4)).issubset(RISK_FEATURE_INDICES["semantic-confidence-risk-linear"])

    assert TinySelectiveRiskHead(5).trainable_parameter_count == 6
    assert TinySelectiveRiskHead(8).trainable_parameter_count == 9


def test_selective_risk_tournament_can_separate_correctness():
    train, dev = _risk_caches()
    result = train_selective_risk_tournament(train, dev, None)

    assert result["selected_candidate"] in RISK_CANDIDATES
    assert int(result["selected_parameter_count"]) in {6, 9}
    assert bool(result["selected_dev"]["meets_target"]) is True
    assert float(result["selected_dev"]["coverage"]) >= 0.25
    assert float(result["selected_dev"]["selective_accuracy"]) >= 0.90
    assert float(result["selected_dev"]["selective_risk"]) <= 0.10


def test_risk_head_fixed_threshold_evaluation_is_bounded():
    train, dev = _risk_caches()
    result = train_selective_risk_tournament(train, dev, None)
    head = TinySelectiveRiskHead(len(result["selected_feature_indices"]))
    head.load_state_dict(result["selected_state_dict"], strict=True)
    head.eval()

    metrics = evaluate_risk_head(
        head,
        result["selected_feature_mean"],
        result["selected_feature_std"],
        dev,
        None,
        feature_indices=tuple(result["selected_feature_indices"]),
        threshold=float(result["selected_threshold"]),
    )
    assert 0.0 <= float(metrics["coverage"]) <= 1.0
    assert 0.0 <= float(metrics["selective_accuracy"]) <= 1.0
    assert 0.0 <= float(metrics["correctness_auroc"]) <= 1.0


def test_r2_dev_qualification_requires_safe_calibration_and_selective_risk():
    calibration = {
        "control_dev": {
            "accuracy": 0.70,
            "soft_ece": 0.16,
        },
        "selected_dev": {
            "accuracy": 0.70,
            "soft_ece": 0.10,
            "probability_mass_max_error": 1e-7,
        },
    }
    risk = {
        "selected_dev": {
            "meets_target": True,
            "selective_accuracy": 0.94,
            "coverage": 0.35,
            "selective_risk": 0.06,
        }
    }
    passed = m1_r2_dev_qualification(calibration, risk)
    assert passed["pass"] is True
    assert all(passed["calibration"].values())
    assert all(passed["risk"].values())

    broken = copy.deepcopy(risk)
    broken["selected_dev"]["selective_accuracy"] = 0.85
    broken["selected_dev"]["selective_risk"] = 0.15
    broken["selected_dev"]["meets_target"] = False
    failed = m1_r2_dev_qualification(calibration, broken)
    assert failed["pass"] is False
    assert failed["risk"]["accuracy"] is False
