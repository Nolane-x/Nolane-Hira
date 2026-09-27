import copy

import torch

from nmd.mainline import W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
from nmd.mainline_m1_cache import M1_CACHE_SCHEMA, OOD_FEATURE_NAMES
from nmd.mainline_m1_training import (
    CALIBRATION_CANDIDATES,
    CALIBRATION_EPOCHS,
    OOD_CANDIDATES,
    OOD_FEATURE_INDICES,
    SELECTIVE_MIN_COVERAGE,
    SELECTIVE_TARGET_ACCURACY,
    evaluate_calibration_cache,
    evaluate_final_reliability_policy,
    m1_dev_qualification,
    m1_sealed_qualification,
    select_selective_threshold,
    train_calibration_tournament,
    train_ood_tournament,
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


def _id_case(case_id, primitive, logits, gold_index, gold_mass, features):
    probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    k = len(logits)
    other = (1.0 - gold_mass) / (k - 1)
    gold = torch.tensor(
        [gold_mass if i == gold_index else other for i in range(k)],
        dtype=torch.float32,
    )
    values = (
        (0.0, 1.0)
        if primitive == "noul"
        else ((0.0, 1.0, 2.0) if primitive == "score" else (None,) * k)
    )
    return {
        "case_id": case_id,
        "domain_id": "unit",
        "partition": "unit",
        "primitive": primitive,
        "primitive_id": {"choice": 0, "score": 1, "noul": 2}[primitive],
        "is_ood": False,
        "ood_kind": None,
        "confidence_band": "unit",
        "gold_index": gold_index,
        "gold_probabilities": gold,
        "logits": torch.tensor(logits, dtype=torch.float32),
        "probabilities": probabilities,
        "ood_features": torch.tensor(features, dtype=torch.float32),
        "option_values": values,
        "option_ids": tuple(f"o-{i}" for i in range(k)),
    }


def _ood_case(case_id, primitive, logits, features):
    probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    k = len(logits)
    values = (
        (0.0, 1.0)
        if primitive == "noul"
        else ((0.0, 1.0, 2.0) if primitive == "score" else (None,) * k)
    )
    return {
        "case_id": case_id,
        "domain_id": "unit-ood",
        "partition": "unit",
        "primitive": primitive,
        "primitive_id": {"choice": 0, "score": 1, "noul": 2}[primitive],
        "is_ood": True,
        "ood_kind": "unit",
        "confidence_band": None,
        "gold_index": None,
        "gold_probabilities": None,
        "logits": torch.tensor(logits, dtype=torch.float32),
        "probabilities": probabilities,
        "ood_features": torch.tensor(features, dtype=torch.float32),
        "option_values": values,
        "option_ids": tuple(f"o-{i}" for i in range(k)),
    }


def _cache(partition, cases):
    cases = copy.deepcopy(cases)
    for case in cases:
        case["partition"] = partition
    return {"metadata": _metadata(partition, len(cases)), "cases": cases}


def _calibration_caches():
    train = []
    dev = []
    primitives = ("choice", "score", "noul")
    for i in range(18):
        primitive = primitives[i % 3]
        k = 2 if primitive == "noul" else 3
        gold = i % k
        logits = [-1.0] * k
        logits[gold] = 2.8
        features = [0.8, 0.75, 0.65, 0.08, 0.92, 0.2, 0.75, 0.1]
        train.append(
            _id_case(
                f"train-{i}",
                primitive,
                logits,
                gold,
                0.72,
                features,
            )
        )
    for i in range(12):
        primitive = primitives[i % 3]
        k = 2 if primitive == "noul" else 3
        gold = (i + 1) % k
        logits = [-0.8] * k
        logits[gold] = 2.4
        features = [0.78, 0.72, 0.62, 0.1, 0.9, 0.25, 0.7, 0.1]
        dev.append(
            _id_case(
                f"dev-{i}",
                primitive,
                logits,
                gold,
                0.70,
                features,
            )
        )
    return _cache("cal_train", train), _cache("cal_dev", dev)


def _ood_caches():
    id_train = []
    ood_train = []
    id_dev = []
    ood_dev = []
    for i in range(20):
        primitive = ("choice", "score", "noul")[i % 3]
        k = 2 if primitive == "noul" else 3
        logits = [2.0] + [0.0] * (k - 1)
        id_features = [
            0.85 + 0.005 * (i % 3),
            0.80,
            0.68,
            0.06,
            0.82,
            0.30,
            0.55,
            0.10,
        ]
        ood_features = [
            -0.25,
            -0.18,
            -0.30,
            0.04,
            0.76,
            0.38,
            0.42,
            0.10,
        ]
        id_train.append(
            _id_case(f"id-train-{i}", primitive, logits, 0, 0.9, id_features)
        )
        ood_train.append(
            _ood_case(f"ood-train-{i}", primitive, logits, ood_features)
        )

    for i in range(12):
        primitive = ("choice", "score", "noul")[i % 3]
        k = 2 if primitive == "noul" else 3
        logits = [1.8] + [0.0] * (k - 1)
        id_features = [0.82, 0.77, 0.65, 0.07, 0.80, 0.32, 0.52, 0.10]
        ood_features = [-0.20, -0.15, -0.28, 0.05, 0.79, 0.34, 0.50, 0.10]
        id_dev.append(
            _id_case(f"id-dev-{i}", primitive, logits, 0, 0.9, id_features)
        )
        ood_dev.append(
            _ood_case(f"ood-dev-{i}", primitive, logits, ood_features)
        )

    return (
        _cache("cal_train", id_train),
        _cache("ood_train", ood_train),
        _cache("cal_dev", id_dev),
        _cache("ood_dev", ood_dev),
    )


def test_m1_training_hyperparameter_surface_is_frozen():
    assert CALIBRATION_EPOCHS == 8
    assert CALIBRATION_CANDIDATES == (
        "control",
        "primitive-temperature",
        "primitive-temperature-noul-bias",
    )
    assert OOD_CANDIDATES == (
        "semantic-linear",
        "semantic-confidence-linear",
    )
    assert SELECTIVE_TARGET_ACCURACY == 0.90
    assert SELECTIVE_MIN_COVERAGE == 0.25

    for mode in OOD_CANDIDATES:
        indices = OOD_FEATURE_INDICES[mode]
        assert set(range(4)).issubset(set(indices))


def test_calibration_tournament_is_tiny_and_dev_selected():
    train, dev = _calibration_caches()
    before = [case["logits"].clone() for case in train["cases"]]
    result = train_calibration_tournament(train, dev)

    assert result["selected_candidate"] in CALIBRATION_CANDIDATES
    assert int(result["selected_parameter_count"]) in {0, 3, 4}
    assert int(result["selected_epoch"]) in range(0, 9)
    assert result["train_case_count"] == len(train["cases"])
    assert result["dev_case_count"] == len(dev["cases"])
    assert result["selected_dev"]["probability_mass_max_error"] <= 1e-6

    for frozen, original in zip(before, train["cases"]):
        assert torch.equal(frozen, original["logits"])


def test_calibration_metrics_are_finite():
    _, dev = _calibration_caches()
    metrics = evaluate_calibration_cache(dev, None)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["soft_nll"] >= 0.0
    assert metrics["soft_brier"] >= 0.0
    assert 0.0 <= metrics["soft_ece"] <= 1.0


def test_ood_tournament_uses_semantic_features_and_tiny_heads():
    id_train, ood_train, id_dev, ood_dev = _ood_caches()
    result = train_ood_tournament(
        id_train,
        ood_train,
        id_dev,
        ood_dev,
    )
    assert result["selected_candidate"] in OOD_CANDIDATES
    assert int(result["selected_parameter_count"]) in {6, 9}
    assert 0.0 <= float(result["selected_threshold"]) <= 1.0
    assert float(result["selected_dev"]["auroc"]) >= 0.95

    selected_names = tuple(result["selected_feature_names"])
    for required in OOD_FEATURE_NAMES[:4]:
        assert required in selected_names


def test_selective_threshold_is_dev_only_and_can_meet_target_on_easy_data():
    _, dev = _calibration_caches()
    result = select_selective_threshold(dev, None)
    assert 0.0 <= float(result["threshold"]) <= 0.99
    assert 0.0 <= float(result["coverage"]) <= 1.0
    assert 0.0 <= float(result["selective_accuracy"]) <= 1.0
    assert bool(result["meets_target"]) is True


def test_dev_qualification_requires_all_three_authorities():
    train, dev = _calibration_caches()
    calibration = train_calibration_tournament(train, dev)
    calibrator = None
    if calibration["selected_candidate"] != "control":
        from nmd.calibration import TypedReliabilityCalibrator
        calibrator = TypedReliabilityCalibrator(calibration["selected_candidate"])
        calibrator.load_state_dict(calibration["selected_state_dict"], strict=True)
        calibrator.eval()

    selective = select_selective_threshold(dev, calibrator)
    id_train, ood_train, id_dev, ood_dev = _ood_caches()
    ood = train_ood_tournament(id_train, ood_train, id_dev, ood_dev)

    result = m1_dev_qualification(calibration, selective, ood)
    assert set(result) == {"pass", "calibration", "selective", "ood"}
    assert all(result["calibration"].values())
    assert all(result["selective"].values())
    assert all(result["ood"].values())
    assert result["pass"] is True

    broken_ood = copy.deepcopy(ood)
    broken_ood["selected_dev"]["auroc"] = 0.1
    failed = m1_dev_qualification(calibration, selective, broken_ood)
    assert failed["ood"]["auroc"] is False
    assert failed["pass"] is False


def test_final_reliability_policy_combines_ood_and_selective_thresholds():
    id_train, ood_train, id_dev, ood_dev = _ood_caches()
    selection = train_ood_tournament(
        id_train,
        ood_train,
        id_dev,
        ood_dev,
    )
    from nmd.mainline_m1_training import TinyOODHead

    head = TinyOODHead(len(selection["selected_feature_indices"]))
    head.load_state_dict(selection["selected_state_dict"], strict=True)
    head.eval()

    result = evaluate_final_reliability_policy(
        id_dev,
        ood_dev,
        None,
        head,
        selection["selected_feature_mean"],
        selection["selected_feature_std"],
        feature_indices=tuple(selection["selected_feature_indices"]),
        ood_threshold=float(selection["selected_threshold"]),
        selective_threshold=0.0,
    )
    assert 0.0 <= result["id_coverage"] <= 1.0
    assert 0.0 <= result["id_accepted_accuracy"] <= 1.0
    assert 0.0 <= result["ood_final_accept_rate"] <= 1.0


def test_sealed_qualification_requires_every_component():
    selected_calibration = {
        "soft_ece": 0.10,
        "probability_mass_max_error": 1e-7,
        "accuracy": 0.95,
    }
    control_calibration = {
        "soft_ece": 0.14,
        "accuracy": 0.95,
    }
    ood_metrics = {
        "auroc": 0.90,
        "balanced_accuracy": 0.85,
        "ood_recall": 0.90,
        "ood_false_accept_rate": 0.10,
        "id_accept_rate": 0.80,
    }
    final_policy = {
        "id_coverage": 0.50,
        "id_accepted_accuracy": 0.95,
        "id_selective_risk": 0.05,
        "ood_final_accept_rate": 0.05,
    }
    passed = m1_sealed_qualification(
        selected_calibration,
        control_calibration,
        ood_metrics,
        final_policy,
    )
    assert passed["pass"] is True
    assert all(passed["calibration"].values())
    assert all(passed["ood"].values())
    assert all(passed["selective"].values())
    assert all(passed["final_ood"].values())

    broken = dict(final_policy)
    broken["ood_final_accept_rate"] = 0.50
    failed = m1_sealed_qualification(
        selected_calibration,
        control_calibration,
        ood_metrics,
        broken,
    )
    assert failed["final_ood"]["ood_final_accept"] is False
    assert failed["pass"] is False
