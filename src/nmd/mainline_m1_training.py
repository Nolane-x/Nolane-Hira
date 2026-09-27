from __future__ import annotations

from copy import deepcopy
import random
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .calibration import TypedReliabilityCalibrator
from .mainline_m1_cache import (
    OOD_FEATURE_NAMES,
    validate_m1_frozen_cache,
)

CALIBRATION_EPOCHS = 8
CALIBRATION_LR = 0.01
CALIBRATION_SEED = 11017
OOD_EPOCHS = 60
OOD_LR = 0.02
OOD_WEIGHT_DECAY = 0.001
OOD_SEED = 11029
SELECTIVE_TARGET_ACCURACY = 0.90
SELECTIVE_MIN_COVERAGE = 0.25

CALIBRATION_CANDIDATES = (
    "control",
    "primitive-temperature",
    "primitive-temperature-noul-bias",
)
OOD_CANDIDATES = (
    "semantic-linear",
    "semantic-confidence-linear",
)

OOD_FEATURE_INDICES = {
    # Four semantic geometry features plus K; never confidence-only.
    "semantic-linear": (0, 1, 2, 3, 7),
    "semantic-confidence-linear": tuple(range(len(OOD_FEATURE_NAMES))),
}


def _noul_true_mask(case: Mapping[str, object], device) -> Tensor | None:
    if case["primitive"] != "noul":
        return None
    values = tuple(case["option_values"])
    if sorted(float(v) for v in values if v is not None) != [0.0, 1.0]:
        raise ValueError("M1 noul calibration requires values 0 and 1")
    return torch.tensor(
        [[float(v) == 1.0 for v in values]],
        dtype=torch.bool,
        device=device,
    )


def calibrated_case_logits(
    case: Mapping[str, object],
    calibrator: TypedReliabilityCalibrator | None,
) -> Tensor:
    logits = case["logits"].detach().float()
    if calibrator is None:
        return logits
    qtype = torch.tensor(
        [int(case["primitive_id"])],
        dtype=torch.long,
        device=logits.device,
    )
    return calibrator(
        logits.unsqueeze(0),
        qtype,
        noul_true_mask=_noul_true_mask(case, logits.device),
    )[0]


def _soft_ece(
    confidences: list[float],
    selected_target_mass: list[float],
    *,
    bins: int = 10,
) -> float:
    if len(confidences) != len(selected_target_mass) or not confidences:
        raise ValueError("M1 ECE requires aligned non-empty values")
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            i
            for i, value in enumerate(confidences)
            if (
                low <= value < high
                or (index == bins - 1 and value == 1.0)
            )
        ]
        if not members:
            continue
        mean_conf = sum(confidences[i] for i in members) / len(members)
        mean_target = (
            sum(selected_target_mass[i] for i in members) / len(members)
        )
        error += (len(members) / total) * abs(mean_conf - mean_target)
    return error


@torch.inference_mode()
def evaluate_calibration_cache(
    cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
) -> dict[str, float]:
    validate_m1_frozen_cache(cache)
    rows = [case for case in cache["cases"] if not case["is_ood"]]
    if not rows:
        raise ValueError("M1 calibration evaluation requires ID rows")

    nll = 0.0
    brier = 0.0
    correct = 0
    confidences: list[float] = []
    target_mass: list[float] = []
    probability_error = 0.0

    if calibrator is not None:
        calibrator.eval()

    for case in rows:
        logits = calibrated_case_logits(case, calibrator)
        probabilities = torch.softmax(logits, dim=-1)
        gold = case["gold_probabilities"].to(probabilities)
        selected = int(probabilities.argmax())
        gold_index = int(case["gold_index"])

        nll += float(-(gold * torch.log_softmax(logits, dim=-1)).sum())
        brier += float(((probabilities - gold) ** 2).sum())
        correct += int(selected == gold_index)
        confidences.append(float(probabilities[selected]))
        target_mass.append(float(gold[selected]))
        probability_error = max(
            probability_error,
            abs(float(probabilities.sum()) - 1.0),
        )

    count = len(rows)
    return {
        "case_count": float(count),
        "accuracy": correct / count,
        "soft_nll": nll / count,
        "soft_brier": brier / count,
        "soft_ece": _soft_ece(confidences, target_mass),
        "probability_mass_max_error": probability_error,
    }


def _calibration_selection_key(
    metrics: Mapping[str, float],
    *,
    epoch: int,
    parameter_count: int,
) -> tuple[float, ...]:
    return (
        -float(metrics["soft_ece"]),
        -float(metrics["soft_nll"]),
        -float(metrics["soft_brier"]),
        float(metrics["accuracy"]),
        -float(parameter_count),
        -float(epoch),
    )


def _train_one_calibrator(
    mode: str,
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
) -> dict[str, object]:
    if mode not in {
        "primitive-temperature",
        "primitive-temperature-noul-bias",
    }:
        raise ValueError(f"unsupported M1 calibration mode: {mode}")

    validate_m1_frozen_cache(train_cache)
    validate_m1_frozen_cache(dev_cache)
    train_rows = [case for case in train_cache["cases"] if not case["is_ood"]]
    if not train_rows:
        raise ValueError("M1 calibration TRAIN is empty")

    torch.manual_seed(CALIBRATION_SEED)
    random.seed(CALIBRATION_SEED)
    calibrator = TypedReliabilityCalibrator(mode)
    optimizer = torch.optim.Adam(
        calibrator.parameters(),
        lr=CALIBRATION_LR,
        weight_decay=0.0,
    )
    history = []
    best_key = None
    best_state = None
    best_epoch = None
    best_metrics = None

    for epoch in range(1, CALIBRATION_EPOCHS + 1):
        calibrator.train()
        indices = list(range(len(train_rows)))
        random.Random(CALIBRATION_SEED + epoch).shuffle(indices)
        total = 0.0

        for row_index in indices:
            case = train_rows[row_index]
            optimizer.zero_grad(set_to_none=True)
            logits = calibrated_case_logits(case, calibrator)
            gold = case["gold_probabilities"].to(logits)
            probabilities = torch.softmax(logits, dim=-1)
            loss = (
                -(gold * torch.log_softmax(logits, dim=-1)).sum()
                + ((probabilities - gold) ** 2).sum()
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach())

        metrics = evaluate_calibration_cache(dev_cache, calibrator)
        key = _calibration_selection_key(
            metrics,
            epoch=epoch,
            parameter_count=calibrator.trainable_parameter_count,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(train_rows),
                "dev": metrics,
                "selection_key": list(key),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in calibrator.state_dict().items()
            }
            best_epoch = epoch
            best_metrics = deepcopy(metrics)

    if best_state is None or best_metrics is None or best_epoch is None:
        raise RuntimeError("M1 calibrator selection produced no checkpoint")
    return {
        "candidate": mode,
        "parameter_count": calibrator.trainable_parameter_count,
        "selected_epoch": best_epoch,
        "selected_dev": best_metrics,
        "state_dict": best_state,
        "history": history,
    }


def train_calibration_tournament(
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
) -> dict[str, object]:
    control = evaluate_calibration_cache(dev_cache, None)
    candidates: list[dict[str, object]] = [
        {
            "candidate": "control",
            "parameter_count": 0,
            "selected_epoch": 0,
            "selected_dev": control,
            "state_dict": None,
            "history": [],
        }
    ]
    for mode in CALIBRATION_CANDIDATES[1:]:
        candidates.append(_train_one_calibrator(mode, train_cache, dev_cache))

    selected = max(
        candidates,
        key=lambda row: _calibration_selection_key(
            row["selected_dev"],
            epoch=int(row["selected_epoch"]),
            parameter_count=int(row["parameter_count"]),
        ),
    )
    return {
        "schema_version": "hira-v0-m1-calibration-selection-v1",
        "selected_candidate": selected["candidate"],
        "selected_parameter_count": selected["parameter_count"],
        "selected_epoch": selected["selected_epoch"],
        "selected_dev": selected["selected_dev"],
        "selected_state_dict": selected["state_dict"],
        "control_dev": control,
        "candidates": candidates,
        "train_case_count": len(train_cache["cases"]),
        "dev_case_count": len(dev_cache["cases"]),
    }


class TinyOODHead(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        if feature_count < 4:
            raise ValueError("M1 OOD head requires semantic features")
        self.linear = nn.Linear(feature_count, 1)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.linear.in_features:
            raise ValueError("M1 OOD feature matrix shape changed")
        return self.linear(features).squeeze(-1)


def _ood_dataset(
    id_cache: dict[str, object],
    ood_cache: dict[str, object],
    feature_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    validate_m1_frozen_cache(id_cache)
    validate_m1_frozen_cache(ood_cache)
    rows = []
    labels = []
    for case in id_cache["cases"]:
        if case["is_ood"]:
            continue
        rows.append(case["ood_features"][list(feature_indices)].float())
        labels.append(0.0)
    for case in ood_cache["cases"]:
        if not case["is_ood"]:
            continue
        rows.append(case["ood_features"][list(feature_indices)].float())
        labels.append(1.0)
    if not rows or 0.0 not in labels or 1.0 not in labels:
        raise ValueError("M1 OOD dataset requires ID and OOD examples")
    return torch.stack(rows), torch.tensor(labels, dtype=torch.float32)


def _auroc(scores: Tensor, labels: Tensor) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("M1 AUROC requires both classes")
    greater = (positive[:, None] > negative[None, :]).float().mean()
    equal = (positive[:, None] == negative[None, :]).float().mean()
    return float(greater + 0.5 * equal)


def _threshold_metrics(
    scores: Tensor,
    labels: Tensor,
    threshold: float,
) -> dict[str, float]:
    predicted = scores >= threshold
    truth = labels.bool()
    tp = int((predicted & truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    fp = int((predicted & ~truth).sum())
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "threshold": float(threshold),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "ood_recall": tpr,
        "id_accept_rate": tnr,
        "ood_false_accept_rate": fn / max(1, tp + fn),
        "id_false_abstain_rate": fp / max(1, tn + fp),
    }


def _select_ood_threshold(scores: Tensor, labels: Tensor) -> dict[str, float]:
    candidates = [i / 100.0 for i in range(5, 96)]
    rows = [_threshold_metrics(scores, labels, threshold) for threshold in candidates]
    return max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            -row["ood_false_accept_rate"],
            row["id_accept_rate"],
            row["threshold"],
        ),
    )


@torch.inference_mode()
def evaluate_ood_head(
    head: TinyOODHead,
    feature_mean: Tensor,
    feature_std: Tensor,
    id_cache: dict[str, object],
    ood_cache: dict[str, object],
    *,
    feature_indices: tuple[int, ...],
    threshold: float | None = None,
) -> dict[str, float]:
    features, labels = _ood_dataset(id_cache, ood_cache, feature_indices)
    normalized = (features - feature_mean) / feature_std
    scores = torch.sigmoid(head(normalized))
    selected = (
        _select_ood_threshold(scores, labels)
        if threshold is None
        else _threshold_metrics(scores, labels, threshold)
    )
    selected["auroc"] = _auroc(scores, labels)
    selected["id_count"] = float((labels == 0).sum())
    selected["ood_count"] = float((labels == 1).sum())
    return selected


def _ood_selection_key(
    metrics: Mapping[str, float],
    *,
    parameter_count: int,
    mode: str,
    epoch: int,
) -> tuple[float, ...]:
    semantic_only_bonus = 1.0 if mode == "semantic-linear" else 0.0
    return (
        float(metrics["auroc"]),
        float(metrics["balanced_accuracy"]),
        -float(metrics["ood_false_accept_rate"]),
        float(metrics["id_accept_rate"]),
        semantic_only_bonus,
        -float(parameter_count),
        -float(epoch),
    )


def _train_one_ood_candidate(
    mode: str,
    id_train: dict[str, object],
    ood_train: dict[str, object],
    id_dev: dict[str, object],
    ood_dev: dict[str, object],
) -> dict[str, object]:
    if mode not in OOD_CANDIDATES:
        raise ValueError(f"unsupported M1 OOD mode: {mode}")
    feature_indices = OOD_FEATURE_INDICES[mode]
    train_x, train_y = _ood_dataset(id_train, ood_train, feature_indices)
    feature_mean = train_x.mean(0)
    feature_std = train_x.std(0, unbiased=False).clamp_min(1e-4)
    train_x = (train_x - feature_mean) / feature_std

    torch.manual_seed(OOD_SEED)
    head = TinyOODHead(len(feature_indices))
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=OOD_LR,
        weight_decay=OOD_WEIGHT_DECAY,
    )

    best_key = None
    best_state = None
    best_epoch = None
    best_metrics = None
    history = []

    for epoch in range(1, OOD_EPOCHS + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        loss.backward()
        optimizer.step()

        head.eval()
        metrics = evaluate_ood_head(
            head,
            feature_mean,
            feature_std,
            id_dev,
            ood_dev,
            feature_indices=feature_indices,
        )
        key = _ood_selection_key(
            metrics,
            parameter_count=head.trainable_parameter_count,
            mode=mode,
            epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "train_bce": float(loss.detach()),
                "dev": metrics,
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
        raise RuntimeError("M1 OOD selection produced no checkpoint")
    return {
        "candidate": mode,
        "feature_indices": feature_indices,
        "feature_names": tuple(OOD_FEATURE_NAMES[i] for i in feature_indices),
        "parameter_count": head.trainable_parameter_count,
        "selected_epoch": best_epoch,
        "selected_dev": best_metrics,
        "selected_threshold": best_metrics["threshold"],
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "state_dict": best_state,
        "history": history,
    }


def train_ood_tournament(
    id_train: dict[str, object],
    ood_train: dict[str, object],
    id_dev: dict[str, object],
    ood_dev: dict[str, object],
) -> dict[str, object]:
    candidates = [
        _train_one_ood_candidate(
            mode,
            id_train,
            ood_train,
            id_dev,
            ood_dev,
        )
        for mode in OOD_CANDIDATES
    ]
    selected = max(
        candidates,
        key=lambda row: _ood_selection_key(
            row["selected_dev"],
            parameter_count=int(row["parameter_count"]),
            mode=str(row["candidate"]),
            epoch=int(row["selected_epoch"]),
        ),
    )
    return {
        "schema_version": "hira-v0-m1-ood-selection-v1",
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
def select_selective_threshold(
    dev_cache: dict[str, object],
    calibrator: TypedReliabilityCalibrator | None,
) -> dict[str, float | bool]:
    validate_m1_frozen_cache(dev_cache)
    rows = [case for case in dev_cache["cases"] if not case["is_ood"]]
    values = []
    for case in rows:
        logits = calibrated_case_logits(case, calibrator)
        probabilities = torch.softmax(logits, dim=-1)
        selected = int(probabilities.argmax())
        values.append(
            (
                float(probabilities[selected]),
                int(selected == int(case["gold_index"])),
            )
        )

    candidates = [i / 100.0 for i in range(0, 100)]
    reports = []
    for threshold in candidates:
        accepted = [correct for confidence, correct in values if confidence >= threshold]
        coverage = len(accepted) / len(values)
        accuracy = (
            sum(accepted) / len(accepted)
            if accepted
            else 0.0
        )
        reports.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "selective_accuracy": accuracy,
                "selective_risk": 1.0 - accuracy if accepted else 1.0,
                "meets_target": (
                    coverage >= SELECTIVE_MIN_COVERAGE
                    and accuracy >= SELECTIVE_TARGET_ACCURACY
                ),
            }
        )

    passing = [row for row in reports if row["meets_target"]]
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                row["coverage"],
                row["selective_accuracy"],
                -row["threshold"],
            ),
        )
    else:
        selected = max(
            reports,
            key=lambda row: (
                row["selective_accuracy"],
                row["coverage"],
                row["threshold"],
            ),
        )
    return selected


__all__ = [
    "CALIBRATION_CANDIDATES",
    "CALIBRATION_EPOCHS",
    "CALIBRATION_LR",
    "CALIBRATION_SEED",
    "OOD_CANDIDATES",
    "OOD_EPOCHS",
    "OOD_FEATURE_INDICES",
    "OOD_LR",
    "OOD_SEED",
    "OOD_WEIGHT_DECAY",
    "SELECTIVE_MIN_COVERAGE",
    "SELECTIVE_TARGET_ACCURACY",
    "TinyOODHead",
    "calibrated_case_logits",
    "evaluate_calibration_cache",
    "evaluate_ood_head",
    "select_selective_threshold",
    "train_calibration_tournament",
    "train_ood_tournament",
]
