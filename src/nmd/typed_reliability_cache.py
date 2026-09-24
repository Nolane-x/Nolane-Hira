from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .calibration import TypedReliabilityCalibrator
from .runtime import NolaneHira, PRIMITIVE_TO_ID
from .typed_reliability_authority import (
    CALIBRATION_SEED,
    ReliabilityAuthorityCase,
)


CANDIDATES = (
    "frozen-production-control",
    "primitive-temperature",
    "primitive-temperature-noul-bias",
)
TRAINABLE_CANDIDATES = (
    "primitive-temperature",
    "primitive-temperature-noul-bias",
)

W6B_HIRA_SHA256 = (
    "925f74094ac4ae583c015ea2a0be32ec692d885ec64dcf9cf3d94b64be0ccf42"
)
W6B_SCORER_SHA256 = (
    "50abb2e8136599bcaf5c41d61036e3c335a7589c6244536cc0292dcea15b1ef0"
)

EPOCHS = 8
LR = 0.01
WEIGHT_DECAY = 0.0


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_case_ids(cases: Sequence[ReliabilityAuthorityCase]) -> str:
    payload = "\n".join(
        sorted(case.typed.case_id for case in cases)
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _noul_true_mask(decision) -> Tensor:
    values = [option.value for option in decision.options]
    if (
        any(value is None for value in values)
        or sorted(float(value) for value in values) != [0.0, 1.0]
    ):
        raise ValueError(
            "W6c noul decision requires semantic option values 0 and 1"
        )
    return torch.tensor(
        [float(value) == 1.0 for value in values],
        dtype=torch.bool,
    )


@torch.inference_mode()
def compile_w6c_logit_cache(
    model: NolaneHira,
    cases: Sequence[ReliabilityAuthorityCase],
    *,
    expected_split: str,
) -> dict:
    if model.reliability_calibrator is not None:
        raise ValueError(
            "W6c logit cache must be compiled before reliability calibration"
        )
    if model.coarse_scorer is None:
        raise ValueError(
            "W6c logit cache requires frozen competitive production scorer"
        )

    model.eval()
    before = model.state_encode_calls
    rows: list[dict] = []

    for authority_case in cases:
        if authority_case.split != expected_split:
            raise ValueError("W6c authority split mismatch")
        typed = authority_case.typed
        memory = model.compile_state(
            typed.state_text,
            segment_tokens=32,
        )

        decisions: list[dict] = []
        for decision in typed.decisions:
            schema, _ = model.compile_schema(
                primitive=decision.primitive,
                question_text=decision.question_text,
                options=decision.options,
                use_cache=False,
                include_token_artifacts=True,
            )
            out = model.forward_compiled(
                memory,
                schema,
                forced_budget=255,
                adaptive_budget=False,
                coarse_mode="competitive",
            )
            if not torch.isfinite(out.logits).all():
                raise RuntimeError(
                    "W6c frozen production path produced non-finite logits"
                )

            raw_values = [option.value for option in decision.options]
            if decision.primitive == "score":
                if any(value is None for value in raw_values):
                    raise RuntimeError(
                        "W6c score decision requires numeric support"
                    )
                score_support = torch.tensor(
                    raw_values,
                    dtype=torch.float32,
                )
            else:
                score_support = torch.empty(
                    0,
                    dtype=torch.float32,
                )

            true_mask = (
                _noul_true_mask(decision)
                if decision.primitive == "noul"
                else torch.zeros(
                    len(decision.options),
                    dtype=torch.bool,
                )
            )
            decisions.append(
                {
                    "question_id": decision.question_id,
                    "primitive": decision.primitive,
                    "qtype": PRIMITIVE_TO_ID[decision.primitive],
                    "raw_logits": out.logits.detach().cpu().float(),
                    "gold_index": int(decision.gold_index),
                    "gold_probabilities": torch.tensor(
                        decision.gold_probabilities,
                        dtype=torch.float32,
                    ),
                    "noul_true_mask": true_mask,
                    "gold_score": (
                        None
                        if decision.gold_score is None
                        else float(decision.gold_score)
                    ),
                    "score_support": score_support,
                }
            )

        rows.append(
            {
                "case_id": typed.case_id,
                "split": authority_case.split,
                "template_id": authority_case.template_id,
                "diagnosis_k": int(authority_case.diagnosis_k),
                "severity": int(authority_case.severity),
                "confidence": authority_case.confidence,
                "decisions": decisions,
            }
        )

    state_calls = model.state_encode_calls - before
    if state_calls != len(cases):
        raise RuntimeError(
            "W6c cache violated one-state-encode-per-case contract"
        )

    cache = {
        "metadata": {
            "schema_version": "r8-w6c-production-logit-cache-v1",
            "split": expected_split,
            "case_count": len(rows),
            "decision_count": sum(
                len(row["decisions"])
                for row in rows
            ),
            "case_id_sha256": hash_case_ids(cases),
            "state_encode_calls": state_calls,
            "state_encode_calls_per_case": (
                state_calls / max(1, len(rows))
            ),
            "confirm_exposed": expected_split == "confirm",
            "cached_production_logits": True,
        },
        "cases": rows,
    }
    validate_w6c_logit_cache(
        cache,
        expected_split=expected_split,
    )
    return cache


def validate_w6c_logit_cache(
    cache: dict,
    *,
    expected_split: str | None = None,
) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W6c cache must be a dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("W6c cache requires metadata and cases")
    if (
        metadata.get("schema_version")
        != "r8-w6c-production-logit-cache-v1"
    ):
        raise ValueError("unexpected W6c cache schema")
    split = metadata.get("split")
    if expected_split is not None and split != expected_split:
        raise ValueError("W6c cache split mismatch")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W6c cache case count mismatch")
    if float(
        metadata.get("state_encode_calls_per_case", -1.0)
    ) != 1.0:
        raise ValueError(
            "W6c cache must encode source state exactly once per case"
        )

    seen: set[str] = set()
    decision_count = 0
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("invalid or duplicate W6c case ID")
        seen.add(case_id)
        if case.get("split") != split:
            raise ValueError("W6c case/cache split mismatch")
        if int(case.get("diagnosis_k", -1)) not in {8, 16, 32, 64}:
            raise ValueError("invalid W6c diagnosis K")
        decisions = case.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 5:
            raise ValueError("W6c case must contain five decisions")
        for decision in decisions:
            decision_count += 1
            primitive = decision.get("primitive")
            if primitive not in {"choice", "score", "noul"}:
                raise ValueError("invalid W6c primitive")
            logits = decision.get("raw_logits")
            gold = decision.get("gold_probabilities")
            if (
                not isinstance(logits, Tensor)
                or logits.ndim != 1
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("invalid W6c cached logits")
            if (
                not isinstance(gold, Tensor)
                or gold.shape != logits.shape
                or not torch.isfinite(gold).all()
                or abs(float(gold.sum()) - 1.0) > 1e-6
            ):
                raise ValueError("invalid W6c soft target")
            gold_index = int(decision.get("gold_index", -1))
            if not 0 <= gold_index < logits.numel():
                raise ValueError("invalid W6c gold index")
            if primitive == "noul":
                mask = decision.get("noul_true_mask")
                if (
                    not isinstance(mask, Tensor)
                    or mask.dtype != torch.bool
                    or mask.shape != logits.shape
                    or int(mask.sum()) != 1
                ):
                    raise ValueError(
                        "W6c noul semantic true mask invalid"
                    )
    if int(metadata.get("decision_count", -1)) != decision_count:
        raise ValueError("W6c decision count mismatch")


def save_w6c_logit_cache(cache: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w6c_logit_cache(
    path: str | Path,
    *,
    expected_split: str | None = None,
) -> dict:
    cache = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    validate_w6c_logit_cache(
        cache,
        expected_split=expected_split,
    )
    return cache


def _ece15(
    confidence: list[float],
    targets: list[float],
) -> float:
    if not confidence:
        return float("nan")
    total = len(confidence)
    result = 0.0
    for index in range(15):
        lo = index / 15
        hi = (index + 1) / 15
        selected = [
            row
            for row, value in enumerate(confidence)
            if value > lo and value <= hi
        ]
        if not selected:
            continue
        mean_conf = sum(
            confidence[row]
            for row in selected
        ) / len(selected)
        mean_target = sum(
            targets[row]
            for row in selected
        ) / len(selected)
        result += (
            len(selected)
            / total
            * abs(mean_conf - mean_target)
        )
    return result


def _probabilities(
    decision: dict,
    calibrator: TypedReliabilityCalibrator | None,
) -> Tensor:
    logits = decision["raw_logits"].float().unsqueeze(0)
    if calibrator is not None:
        qtype = torch.tensor(
            [int(decision["qtype"])],
            dtype=torch.long,
        )
        mask = None
        if decision["primitive"] == "noul":
            mask = decision[
                "noul_true_mask"
            ].bool().unsqueeze(0)
        logits = calibrator(
            logits,
            qtype,
            noul_true_mask=mask,
        )
    return torch.softmax(logits, dim=-1)[0]


def evaluate_w6c_cases(
    calibrator: TypedReliabilityCalibrator | None,
    cache: dict,
) -> dict[str, object]:
    validate_w6c_logit_cache(cache)
    if calibrator is not None:
        calibrator.eval()

    cases = cache["cases"]
    case_count = 0
    decision_count = 0
    correct_total = 0
    primitive_counts = {"choice": 0, "score": 0, "noul": 0}
    primitive_correct = {"choice": 0, "score": 0, "noul": 0}
    diagnosis = {
        8: {"n": 0, "correct": 0},
        16: {"n": 0, "correct": 0},
        32: {"n": 0, "correct": 0},
        64: {"n": 0, "correct": 0},
    }
    soft_accuracy_sum = 0.0
    hard_brier_sum = 0.0
    soft_brier_sum = 0.0
    confidence_values: list[float] = []
    correctness_values: list[float] = []
    soft_target_values: list[float] = []
    score_errors: list[float] = []
    max_mass_error = 0.0

    with torch.inference_mode():
        for case in cases:
            case_count += 1
            for decision in case["decisions"]:
                p = _probabilities(decision, calibrator).to(torch.float64)
                gold = decision[
                    "gold_probabilities"
                ].to(torch.float64)
                gold_index = int(decision["gold_index"])
                predicted = int(p.argmax())
                is_correct = float(predicted == gold_index)

                decision_count += 1
                correct_total += int(is_correct)
                primitive = decision["primitive"]
                primitive_counts[primitive] += 1
                primitive_correct[primitive] += int(is_correct)

                if decision["question_id"] == "diagnosis":
                    slot = diagnosis[int(case["diagnosis_k"])]
                    slot["n"] += 1
                    slot["correct"] += int(is_correct)

                soft_accuracy_sum += float((p * gold).sum())
                onehot = torch.zeros_like(p)
                onehot[gold_index] = 1.0
                hard_brier_sum += float(
                    ((p - onehot) ** 2).sum()
                )
                soft_brier_sum += float(
                    ((p - gold) ** 2).sum()
                )
                confidence_values.append(float(p.max()))
                correctness_values.append(is_correct)
                soft_target_values.append(
                    float(gold[predicted])
                )
                max_mass_error = max(
                    max_mass_error,
                    abs(float(p.sum()) - 1.0),
                )

                if primitive == "score":
                    support = decision[
                        "score_support"
                    ].to(dtype=p.dtype)
                    predicted_score = float(
                        (p * support).sum()
                    )
                    score_errors.append(
                        abs(
                            predicted_score
                            - float(decision["gold_score"])
                        )
                    )

    primitive_accuracy = {
        primitive: primitive_correct[primitive]
        / primitive_counts[primitive]
        for primitive in primitive_counts
    }
    diagnosis_per_k = {
        str(k): {
            "n": slot["n"],
            "accuracy": slot["correct"] / slot["n"],
        }
        for k, slot in diagnosis.items()
        if slot["n"]
    }
    return {
        "case_count": case_count,
        "decision_count": decision_count,
        "accuracy": correct_total / decision_count,
        "primitive_counts": primitive_counts,
        "primitive_accuracy": primitive_accuracy,
        "diagnosis_per_k": diagnosis_per_k,
        "soft_accuracy": soft_accuracy_sum / decision_count,
        "hard_brier": hard_brier_sum / decision_count,
        "soft_brier": soft_brier_sum / decision_count,
        "raw_ece": _ece15(
            confidence_values,
            correctness_values,
        ),
        "soft_ece": _ece15(
            confidence_values,
            soft_target_values,
        ),
        "score_mae": (
            sum(score_errors) / len(score_errors)
        ),
        "score_count": len(score_errors),
        "probability_mass_max_error": max_mass_error,
        "source_state_encodes_per_case": float(
            cache["metadata"]["state_encode_calls_per_case"]
        ),
        "cached_production_logits": True,
    }


def calibration_case_loss(
    calibrator: TypedReliabilityCalibrator,
    case: dict,
) -> Tensor:
    losses: list[Tensor] = []
    for decision in case["decisions"]:
        p = _probabilities(
            decision,
            calibrator,
        ).clamp_min(1e-8)
        gold = decision[
            "gold_probabilities"
        ].to(dtype=p.dtype)
        positive = gold > 0
        teacher_kl = (
            gold[positive]
            * (
                gold[positive].log()
                - p[positive].log()
            )
        ).sum()
        soft_brier = ((p - gold) ** 2).sum()
        losses.append(teacher_kl + soft_brier)
    return torch.stack(losses).mean()


def dev_selection_key(
    metrics: dict[str, object],
    epoch: int,
) -> tuple:
    return (
        float(metrics["soft_ece"]),
        -float(metrics["primitive_accuracy"]["noul"]),
        float(metrics["soft_brier"]),
        float(metrics["hard_brier"]),
        -float(metrics["accuracy"]),
        float(metrics["score_mae"]),
        int(epoch),
    )


def candidate_to_mode(candidate: str) -> str | None:
    if candidate == "frozen-production-control":
        return None
    if candidate == "primitive-temperature":
        return "primitive-temperature"
    if candidate == "primitive-temperature-noul-bias":
        return "primitive-temperature-noul-bias"
    raise ValueError(f"unknown W6c candidate: {candidate}")


def train_w6c_candidate(
    candidate: str,
    train_cache: dict,
    dev_cache: dict,
) -> tuple[
    list[dict[str, object]],
    dict[str, Tensor] | None,
    dict[str, object],
]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown W6c candidate: {candidate}")
    validate_w6c_logit_cache(
        train_cache,
        expected_split="train",
    )
    validate_w6c_logit_cache(
        dev_cache,
        expected_split="dev",
    )

    mode = candidate_to_mode(candidate)
    if mode is None:
        metrics = evaluate_w6c_cases(
            None,
            dev_cache,
        )
        metrics["epoch"] = 0
        return [metrics], None, dict(metrics)

    torch.manual_seed(CALIBRATION_SEED)
    calibrator = TypedReliabilityCalibrator(mode)
    expected = (
        3
        if candidate == "primitive-temperature"
        else 4
    )
    if calibrator.trainable_parameter_count != expected:
        raise RuntimeError(
            "unexpected W6c calibrator parameter count"
        )
    optimizer = torch.optim.Adam(
        calibrator.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_metrics = None

    train_cases = train_cache["cases"]
    for epoch in range(1, EPOCHS + 1):
        calibrator.train()
        order = list(range(len(train_cases)))
        random.Random(
            CALIBRATION_SEED + epoch
        ).shuffle(order)
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = calibration_case_loss(
                calibrator,
                train_cases[index],
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())

        metrics = evaluate_w6c_cases(
            calibrator,
            dev_cache,
        )
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = (
            loss_sum / len(train_cases)
        )
        history.append(metrics)
        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor
                in calibrator.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError(
            "W6c calibrator produced no checkpoint"
        )
    calibrator.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics


def absolute_production_gates(
    metrics: dict[str, object],
) -> dict[str, bool]:
    primitive = metrics["primitive_accuracy"]
    return {
        "overall_accuracy": float(metrics["accuracy"]) >= 0.65,
        "choice_accuracy": float(primitive["choice"]) >= 0.65,
        "noul_accuracy": float(primitive["noul"]) >= 0.70,
        "score_accuracy": float(primitive["score"]) >= 0.60,
        "diagnosis_k64_accuracy": (
            float(
                metrics["diagnosis_per_k"]["64"]["accuracy"]
            )
            >= 0.55
        ),
        "hard_brier": float(metrics["hard_brier"]) <= 0.50,
        "soft_ece": float(metrics["soft_ece"]) <= 0.15,
        "score_mae": float(metrics["score_mae"]) <= 0.55,
        "probability_mass": (
            float(
                metrics["probability_mass_max_error"]
            )
            <= 1e-6
        ),
        "state_once": (
            float(
                metrics["source_state_encodes_per_case"]
            )
            == 1.0
        ),
    }


def mechanism_gates(
    selected: dict[str, object],
    control: dict[str, object],
) -> dict[str, bool]:
    control_noul = float(
        control["primitive_accuracy"]["noul"]
    )
    return {
        "soft_ece_improvement": (
            float(control["soft_ece"])
            - float(selected["soft_ece"])
            >= 0.04
        ),
        "noul_gain_or_control_pass": (
            float(
                selected["primitive_accuracy"]["noul"]
            )
            - control_noul
            >= 0.01
            or control_noul >= 0.70
        ),
        "overall_nonregression": (
            float(selected["accuracy"])
            >= float(control["accuracy"]) - 0.01
        ),
        "diagnosis_k64_nonregression": (
            float(
                selected["diagnosis_per_k"]["64"]["accuracy"]
            )
            >= float(
                control["diagnosis_per_k"]["64"]["accuracy"]
            )
        ),
        "hard_brier_nonregression": (
            float(selected["hard_brier"])
            <= float(control["hard_brier"]) + 0.02
        ),
        "score_mae_nonregression": (
            float(selected["score_mae"])
            <= float(control["score_mae"]) + 0.05
        ),
    }


def confirm_verdict(
    selected: dict[str, object],
    control: dict[str, object],
) -> tuple[str, dict[str, bool], dict[str, bool]]:
    absolute = absolute_production_gates(selected)
    control_absolute = absolute_production_gates(control)
    mechanism = mechanism_gates(selected, control)

    if all(absolute.values()) and all(mechanism.values()):
        return (
            "RELIABILITY_CALIBRATION_RESCUE",
            absolute,
            mechanism,
        )
    if all(control_absolute.values()) and not all(mechanism.values()):
        return (
            "RELIABILITY_CONTROL_ALREADY_RESCUES",
            absolute,
            mechanism,
        )

    ece_improvement = (
        float(control["soft_ece"])
        - float(selected["soft_ece"])
    )
    noul_gain = (
        float(selected["primitive_accuracy"]["noul"])
        - float(control["primitive_accuracy"]["noul"])
    )
    overall_delta = (
        float(selected["accuracy"])
        - float(control["accuracy"])
    )
    if (
        (ece_improvement >= 0.02 or noul_gain >= 0.01)
        and overall_delta >= -0.02
    ):
        return (
            "RELIABILITY_CALIBRATION_PARTIAL",
            absolute,
            mechanism,
        )
    return (
        "RELIABILITY_CALIBRATION_FAIL",
        absolute,
        mechanism,
    )
