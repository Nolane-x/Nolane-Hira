from __future__ import annotations

from copy import deepcopy
import random
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .calibration import TypedReliabilityCalibrator
from .mainline_m1_cache import OOD_FEATURE_NAMES, validate_m1_frozen_cache
from .mainline_m1_training import (
    DEV_CALIBRATION_ECE_IMPROVEMENT,
    DEV_CALIBRATION_ECE_MAX,
    DEV_CALIBRATION_MAX_ACCURACY_REGRESSION,
    SELECTIVE_MIN_COVERAGE,
    SELECTIVE_TARGET_ACCURACY,
    calibrated_case_logits,
    train_calibration_tournament,
)
from .mainline_reliability import confidence_diagnostics

RISK_EPOCHS = 80
RISK_LR = 0.02
RISK_WEIGHT_DECAY = 0.001
RISK_SEED = 12031

RISK_CANDIDATES = (
    "semantic-risk-linear",
    "semantic-confidence-risk-linear",
)

RISK_FEATURE_NAMES = (
    "state_question_cosine",
    "state_option_max_cosine",
    "state_option_mean_cosine",
    "state_option_std_cosine",
    "max_probability",
    "normalized_entropy",
    "top_margin",
    "normalized_log_k",
)
RISK_FEATURE_INDICES = {
    "semantic-risk-linear": (0, 1, 2, 3, 7),
    "semantic-confidence-risk-linear": tuple(range(8)),
}


def safe_calibration_tournament(
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
) -> dict[str, object]:
    """Select calibration only among candidates preserving hard accuracy."""
    tournament = train_calibration_tournament(train_cache, dev_cache)
    control = tournament["control_dev"]
    eligible = []
    summaries = []

    for candidate in tournament["candidates"]:
        metrics = candidate["selected_dev"]
        accuracy_ok = (
            float(metrics["accuracy"])
            >= float(control["accuracy"]) - DEV_CALIBRATION_MAX_ACCURACY_REGRESSION
        )
        probability_ok = float(metrics["probability_mass_max_error"]) <= 1e-6
        is_eligible = bool(accuracy_ok and probability_ok)
        summaries.append(
            {
                "candidate": candidate["candidate"],
                "parameter_count": candidate["parameter_count"],
                "selected_epoch": candidate["selected_epoch"],
                "selected_dev": metrics,
                "accuracy_guardrail": accuracy_ok,
                "probability_integrity": probability_ok,
                "eligible": is_eligible,
            }
        )
        if is_eligible:
            eligible.append(candidate)

    if not eligible:
        raise RuntimeError("M1-R2 calibration guardrail rejected every candidate")

    selected = max(
        eligible,
        key=lambda row: (
            -float(row["selected_dev"]["soft_ece"]),
            -float(row["selected_dev"]["soft_nll"]),
            -float(row["selected_dev"]["soft_brier"]),
            -int(row["parameter_count"]),
            -int(row["selected_epoch"]),
        ),
    )
    return {
        "schema_version": "hira-v0-m1-r2-safe-calibration-selection-v1",
        "selected_candidate": selected["candidate"],
        "selected_parameter_count": selected["parameter_count"],
        "selected_epoch": selected["selected_epoch"],
        "selected_dev": selected["selected_dev"],
        "selected_state_dict": selected["state_dict"],
        "control_dev": control,
        "candidate_summaries": summaries,
        "train_case_count": len(train_cache["cases"]),
        "dev_case_count": len(dev_cache["cases"]),
    }


def build_selected_calibrator(
    selection: Mapping[str, object],
) -> TypedReliabilityCalibrator | None:
    candidate = str(selection["selected_candidate"])
    if candidate == "control":
        return None
    calibrator = TypedReliabilityCalibrator(candidate)
    state = selection["selected_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("M1-R2 selected calibrator state is missing")
    calibrator.load_state_dict(state, strict=True)
    if calibrator.trainable_parameter_count != int(selection["selected_parameter_count"]):
        raise RuntimeError("M1-R2 calibrator parameter count changed")
    calibrator.eval()
    return calibrator


def _case_risk_features(
    case: Mapping[str, object],
    calibrator: TypedReliabilityCalibrator | None,
) -> Tensor:
    base = case["ood_features"].detach().float()
    if base.shape != (len(OOD_FEATURE_NAMES),):
        raise ValueError("M1-R2 cached feature vector changed")

    logits = calibrated_case_logits(case, calibrator)
    probabilities = torch.softmax(logits, dim=-1)
    diagnostics = confidence_diagnostics(probabilities)

    features = torch.tensor(
        [
            float(base[0]),
            float(base[1]),
            float(base[2]),
            float(base[3]),
            diagnostics.max_probability,
            diagnostics.normalized_entropy,
            diagnostics.top_margin,
            float(base[7]),
        ],
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("M1-R2 risk features contain non-finite values")
    return features


def _case_correct(
    case: Mapping[str, object],
    calibrator: TypedReliabilityCalibrator | None,
) -> float:
    logits = calibrated_case_logits(case, calibrator)
    selected = int(logits.argmax())
    return float(selected == int(case["gold_index"]))


class TinySelectiveRiskHead(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        if feature_count < 5:
            raise ValueError("M1-R2 risk head requires semantic geometry")
        self.linear = nn.Linear(feature_count, 1)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.linear.in_features:
            raise ValueError("M1-R2 risk feature matrix shape changed")
        return self.linear(features).squeeze(-1)


def _risk_dataset(
    cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
    feature_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    validate_m1_frozen_cache(cache)
    rows = [case for case in cache["cases"] if not case["is_ood"]]
    if not rows:
        raise ValueError("M1-R2 risk dataset requires ID rows")
    features = torch.stack(
        [_case_risk_features(case, calibrator)[list(feature_indices)] for case in rows]
    )
    labels = torch.tensor(
        [_case_correct(case, calibrator) for case in rows],
        dtype=torch.float32,
    )
    if labels.min() == labels.max():
        raise ValueError("M1-R2 risk TRAIN requires correct and incorrect decisions")
    return features, labels


def _binary_auroc(scores: Tensor, labels: Tensor) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return 0.5
    greater = (positive[:, None] > negative[None, :]).float().mean()
    equal = (positive[:, None] == negative[None, :]).float().mean()
    return float(greater + 0.5 * equal)


def _select_risk_threshold(
    scores: Tensor,
    labels: Tensor,
) -> dict[str, float | bool]:
    reports = []
    for index in range(5, 96):
        threshold = index / 100.0
        accepted_mask = scores >= threshold
        accepted_count = int(accepted_mask.sum())
        coverage = accepted_count / max(1, labels.numel())
        accuracy = (
            float(labels[accepted_mask].mean())
            if accepted_count
            else 0.0
        )
        reports.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "selective_accuracy": accuracy,
                "selective_risk": 1.0 - accuracy if accepted_count else 1.0,
                "accepted_count": float(accepted_count),
                "case_count": float(labels.numel()),
                "meets_target": (
                    coverage >= SELECTIVE_MIN_COVERAGE
                    and accuracy >= SELECTIVE_TARGET_ACCURACY
                ),
            }
        )

    passing = [row for row in reports if row["meets_target"]]
    if passing:
        return max(
            passing,
            key=lambda row: (
                float(row["coverage"]),
                float(row["selective_accuracy"]),
                -float(row["threshold"]),
            ),
        )
    return max(
        reports,
        key=lambda row: (
            float(row["selective_accuracy"]),
            float(row["coverage"]),
            float(row["threshold"]),
        ),
    )


@torch.inference_mode()
def evaluate_risk_head(
    head: TinySelectiveRiskHead,
    feature_mean: Tensor,
    feature_std: Tensor,
    cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
    *,
    feature_indices: tuple[int, ...],
    threshold: float | None = None,
) -> dict[str, float | bool]:
    features, labels = _risk_dataset(cache, calibrator, feature_indices)
    normalized = (features - feature_mean) / feature_std
    scores = torch.sigmoid(head(normalized))
    if threshold is None:
        metrics = _select_risk_threshold(scores, labels)
    else:
        accepted = scores >= float(threshold)
        accepted_count = int(accepted.sum())
        coverage = accepted_count / max(1, labels.numel())
        accuracy = float(labels[accepted].mean()) if accepted_count else 0.0
        metrics = {
            "threshold": float(threshold),
            "coverage": coverage,
            "selective_accuracy": accuracy,
            "selective_risk": 1.0 - accuracy if accepted_count else 1.0,
            "accepted_count": float(accepted_count),
            "case_count": float(labels.numel()),
            "meets_target": (
                coverage >= SELECTIVE_MIN_COVERAGE
                and accuracy >= SELECTIVE_TARGET_ACCURACY
            ),
        }
    metrics["correctness_auroc"] = _binary_auroc(scores, labels)
    return metrics


def _risk_selection_key(
    metrics: Mapping[str, object],
    *,
    mode: str,
    parameter_count: int,
    epoch: int,
) -> tuple[float, ...]:
    simple_bonus = 1.0 if mode == "semantic-risk-linear" else 0.0
    return (
        float(bool(metrics["meets_target"])),
        float(metrics["coverage"]),
        float(metrics["selective_accuracy"]),
        float(metrics["correctness_auroc"]),
        simple_bonus,
        -float(parameter_count),
        -float(epoch),
    )


def _train_one_risk_candidate(
    mode: str,
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
) -> dict[str, object]:
    if mode not in RISK_CANDIDATES:
        raise ValueError(f"unsupported M1-R2 risk mode: {mode}")
    feature_indices = RISK_FEATURE_INDICES[mode]
    train_x, train_y = _risk_dataset(train_cache, calibrator, feature_indices)
    feature_mean = train_x.mean(0)
    feature_std = train_x.std(0, unbiased=False).clamp_min(1e-4)
    normalized = (train_x - feature_mean) / feature_std

    torch.manual_seed(RISK_SEED)
    random.seed(RISK_SEED)
    head = TinySelectiveRiskHead(len(feature_indices))
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=RISK_LR,
        weight_decay=RISK_WEIGHT_DECAY,
    )

    best_key = None
    best_state = None
    best_epoch = None
    best_metrics = None
    history = []

    for epoch in range(1, RISK_EPOCHS + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(normalized)
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        loss.backward()
        optimizer.step()

        head.eval()
        metrics = evaluate_risk_head(
            head,
            feature_mean,
            feature_std,
            dev_cache,
            calibrator,
            feature_indices=feature_indices,
        )
        key = _risk_selection_key(
            metrics,
            mode=mode,
            parameter_count=head.trainable_parameter_count,
            epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "train_bce": float(loss.detach()),
                "dev": deepcopy(metrics),
                "selection_key": list(key),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
            best_epoch = epoch
            best_metrics = deepcopy(metrics)

    if best_state is None or best_metrics is None or best_epoch is None:
        raise RuntimeError("M1-R2 risk tournament produced no checkpoint")
    return {
        "candidate": mode,
        "feature_indices": feature_indices,
        "feature_names": tuple(RISK_FEATURE_NAMES[i] for i in feature_indices),
        "parameter_count": head.trainable_parameter_count,
        "selected_epoch": best_epoch,
        "selected_threshold": best_metrics["threshold"],
        "selected_dev": best_metrics,
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "state_dict": best_state,
        "history": history,
    }


def train_selective_risk_tournament(
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
) -> dict[str, object]:
    candidates = [
        _train_one_risk_candidate(mode, train_cache, dev_cache, calibrator)
        for mode in RISK_CANDIDATES
    ]
    selected = max(
        candidates,
        key=lambda row: _risk_selection_key(
            row["selected_dev"],
            mode=str(row["candidate"]),
            parameter_count=int(row["parameter_count"]),
            epoch=int(row["selected_epoch"]),
        ),
    )
    return {
        "schema_version": "hira-v0-m1-r2-risk-selection-v1",
        "selected_candidate": selected["candidate"],
        "selected_parameter_count": selected["parameter_count"],
        "selected_epoch": selected["selected_epoch"],
        "selected_threshold": selected["selected_threshold"],
        "selected_dev": selected["selected_dev"],
        "selected_feature_indices": selected["feature_indices"],
        "selected_feature_names": selected["feature_names"],
        "selected_feature_mean": selected["feature_mean"],
        "selected_feature_std": selected["feature_std"],
        "selected_state_dict": selected["state_dict"],
        "candidates": candidates,
    }


@torch.inference_mode()
def risk_scores_for_cache(
    head: TinySelectiveRiskHead,
    feature_mean: Tensor,
    feature_std: Tensor,
    cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
    *,
    feature_indices: tuple[int, ...],
) -> Tensor:
    validate_m1_frozen_cache(cache)
    rows = [case for case in cache["cases"] if not case["is_ood"]]
    if len(rows) != len(cache["cases"]):
        # OOD rows have no correctness target, but the feature transform itself
        # is still defined. Callers needing OOD final-policy scores should use
        # risk_scores_for_any_cache below.
        raise ValueError("M1-R2 correctness risk score requires ID-only cache")
    features = torch.stack(
        [_case_risk_features(case, calibrator)[list(feature_indices)] for case in rows]
    )
    normalized = (features - feature_mean) / feature_std
    return torch.sigmoid(head(normalized))


@torch.inference_mode()
def risk_scores_for_any_cache(
    head: TinySelectiveRiskHead,
    feature_mean: Tensor,
    feature_std: Tensor,
    cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
    *,
    feature_indices: tuple[int, ...],
) -> Tensor:
    validate_m1_frozen_cache(cache)
    rows = cache["cases"]
    if not rows:
        raise ValueError("M1-R2 risk score cache is empty")
    features = torch.stack(
        [_case_risk_features(case, calibrator)[list(feature_indices)] for case in rows]
    )
    normalized = (features - feature_mean) / feature_std
    return torch.sigmoid(head(normalized))


@torch.inference_mode()
def evaluate_r2_final_policy(
    id_cache: dict[str, object],
    ood_cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
    risk_head: TinySelectiveRiskHead,
    risk_feature_mean: Tensor,
    risk_feature_std: Tensor,
    ood_head,
    ood_feature_mean: Tensor,
    ood_feature_std: Tensor,
    *,
    risk_feature_indices: tuple[int, ...],
    risk_threshold: float,
    ood_feature_indices: tuple[int, ...],
    ood_threshold: float,
) -> dict[str, float]:
    from .mainline_m1_training import ood_scores_for_cache

    validate_m1_frozen_cache(id_cache)
    validate_m1_frozen_cache(ood_cache)
    if not 0.0 <= float(risk_threshold) <= 1.0:
        raise ValueError("M1-R2 risk threshold must be in [0,1]")
    if not 0.0 <= float(ood_threshold) <= 1.0:
        raise ValueError("M1-R2 OOD threshold must be in [0,1]")

    id_ood_scores = ood_scores_for_cache(
        ood_head,
        ood_feature_mean,
        ood_feature_std,
        id_cache,
        feature_indices=ood_feature_indices,
    )
    ood_ood_scores = ood_scores_for_cache(
        ood_head,
        ood_feature_mean,
        ood_feature_std,
        ood_cache,
        feature_indices=ood_feature_indices,
    )
    id_risk_scores = risk_scores_for_any_cache(
        risk_head,
        risk_feature_mean,
        risk_feature_std,
        id_cache,
        calibrator,
        feature_indices=risk_feature_indices,
    )
    ood_risk_scores = risk_scores_for_any_cache(
        risk_head,
        risk_feature_mean,
        risk_feature_std,
        ood_cache,
        calibrator,
        feature_indices=risk_feature_indices,
    )

    id_accepted: list[int] = []
    for index, case in enumerate(id_cache["cases"]):
        if case["is_ood"]:
            continue
        accept = (
            float(id_ood_scores[index]) < float(ood_threshold)
            and float(id_risk_scores[index]) >= float(risk_threshold)
        )
        if accept:
            logits = calibrated_case_logits(case, calibrator)
            selected = int(logits.argmax())
            id_accepted.append(selected == int(case["gold_index"]))

    ood_accept_count = 0
    for index, case in enumerate(ood_cache["cases"]):
        if not case["is_ood"]:
            continue
        accept = (
            float(ood_ood_scores[index]) < float(ood_threshold)
            and float(ood_risk_scores[index]) >= float(risk_threshold)
        )
        ood_accept_count += int(accept)

    id_total = sum(not case["is_ood"] for case in id_cache["cases"])
    ood_total = sum(bool(case["is_ood"]) for case in ood_cache["cases"])
    id_coverage = len(id_accepted) / max(1, id_total)
    id_accuracy = (
        sum(id_accepted) / len(id_accepted)
        if id_accepted
        else 0.0
    )
    return {
        "id_case_count": float(id_total),
        "id_accepted_count": float(len(id_accepted)),
        "id_coverage": id_coverage,
        "id_accepted_accuracy": id_accuracy,
        "id_selective_risk": 1.0 - id_accuracy if id_accepted else 1.0,
        "ood_case_count": float(ood_total),
        "ood_final_accept_count": float(ood_accept_count),
        "ood_final_accept_rate": ood_accept_count / max(1, ood_total),
    }


def m1_r2_sealed_qualification(
    calibration_gates: Mapping[str, object],
    risk_metrics: Mapping[str, object],
    base_sealed_gates: Mapping[str, object],
) -> dict[str, object]:
    """Require both standalone correctness-risk quality and final M1 policy."""
    risk_gate = {
        "meets_target": bool(risk_metrics["meets_target"]),
        "accuracy": (
            float(risk_metrics["selective_accuracy"]) >= SELECTIVE_TARGET_ACCURACY
        ),
        "coverage": float(risk_metrics["coverage"]) >= SELECTIVE_MIN_COVERAGE,
        "selective_risk": float(risk_metrics["selective_risk"]) <= 0.10,
    }
    passed = (
        bool(base_sealed_gates["pass"])
        and all(risk_gate.values())
        and all(bool(value) for value in calibration_gates.values())
    )
    return {
        "pass": passed,
        "calibration": dict(calibration_gates),
        "standalone_risk": risk_gate,
        "base_m1": dict(base_sealed_gates),
    }


def m1_r2_dev_qualification(
    calibration: Mapping[str, object],
    risk: Mapping[str, object],
) -> dict[str, object]:
    selected_cal = calibration["selected_dev"]
    control_cal = calibration["control_dev"]
    selected_risk = risk["selected_dev"]

    calibration_gate = {
        "accuracy_non_regression": (
            float(selected_cal["accuracy"])
            >= float(control_cal["accuracy"]) - DEV_CALIBRATION_MAX_ACCURACY_REGRESSION
        ),
        "ece_absolute": float(selected_cal["soft_ece"]) <= DEV_CALIBRATION_ECE_MAX,
        "ece_mechanism": (
            float(control_cal["soft_ece"]) <= 0.15
            or (
                float(control_cal["soft_ece"]) - float(selected_cal["soft_ece"])
                >= DEV_CALIBRATION_ECE_IMPROVEMENT
            )
        ),
        "probability_integrity": (
            float(selected_cal["probability_mass_max_error"]) <= 1e-6
        ),
    }
    risk_gate = {
        "meets_target": bool(selected_risk["meets_target"]),
        "accuracy": (
            float(selected_risk["selective_accuracy"]) >= SELECTIVE_TARGET_ACCURACY
        ),
        "coverage": float(selected_risk["coverage"]) >= SELECTIVE_MIN_COVERAGE,
        "selective_risk": float(selected_risk["selective_risk"]) <= 0.10,
    }
    return {
        "pass": all(calibration_gate.values()) and all(risk_gate.values()),
        "calibration": calibration_gate,
        "risk": risk_gate,
    }


__all__ = [
    "RISK_CANDIDATES",
    "RISK_EPOCHS",
    "RISK_FEATURE_INDICES",
    "RISK_FEATURE_NAMES",
    "RISK_LR",
    "RISK_SEED",
    "RISK_WEIGHT_DECAY",
    "TinySelectiveRiskHead",
    "build_selected_calibrator",
    "evaluate_risk_head",
    "evaluate_r2_final_policy",
    "risk_scores_for_any_cache",
    "risk_scores_for_cache",
    "m1_r2_dev_qualification",
    "m1_r2_sealed_qualification",
    "safe_calibration_tournament",
    "train_selective_risk_tournament",
]
