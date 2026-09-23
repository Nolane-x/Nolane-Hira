from __future__ import annotations

import math
from typing import Iterable

import torch

from .runtime import CoarseMode, NolaneHira
from .typed_decisions import (
    TypedDecisionCase,
    execute_typed_case,
)


def _ece(confidence: list[float], correct: list[float], bins: int = 15) -> float:
    if not confidence:
        return float("nan")
    total = len(confidence)
    result = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        selected = [
            i
            for i, value in enumerate(confidence)
            if value > lo and value <= hi
        ]
        if not selected:
            continue
        mean_confidence = sum(confidence[i] for i in selected) / len(selected)
        mean_correct = sum(correct[i] for i in selected) / len(selected)
        result += (
            len(selected)
            / total
            * abs(mean_confidence - mean_correct)
        )
    return result


@torch.inference_mode()
def evaluate_typed_cases(
    model: NolaneHira,
    cases: Iterable[TypedDecisionCase],
    *,
    forced_budget: int | None = None,
    adaptive_budget: bool = False,
    coarse_mode: CoarseMode = "legacy",
) -> dict[str, object]:
    model.eval()

    case_count = 0
    decision_count = 0
    state_encode_calls = 0
    correct_total = 0
    primitive_counts = {"choice": 0, "score": 0, "noul": 0}
    primitive_correct = {"choice": 0, "score": 0, "noul": 0}

    soft_accuracy_sum = 0.0
    hard_brier_sum = 0.0
    soft_brier_sum = 0.0
    nll_sum = 0.0
    kl_sum = 0.0
    confidence: list[float] = []
    correctness: list[float] = []
    score_abs_errors: list[float] = []
    probability_mass_max_error = 0.0

    for case in cases:
        execution = execute_typed_case(
            model,
            case,
            forced_budget=forced_budget,
            adaptive_budget=adaptive_budget,
            use_schema_cache=True,
            coarse_mode=coarse_mode,
        )
        case_count += 1
        state_encode_calls += execution.receipt.state_encode_calls

        for decision, output in zip(case.decisions, execution.outputs):
            p = output.probabilities.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            gold = torch.tensor(
                decision.gold_probabilities,
                dtype=torch.float64,
            )
            if p.shape != gold.shape:
                raise RuntimeError(
                    "prediction/gold probability shape mismatch"
                )
            mass_error = abs(float(p.sum()) - 1.0)
            probability_mass_max_error = max(
                probability_mass_max_error,
                mass_error,
            )

            predicted = int(p.argmax().item())
            is_correct = float(predicted == decision.gold_index)
            correct_total += int(is_correct)
            decision_count += 1
            primitive_counts[decision.primitive] += 1
            primitive_correct[decision.primitive] += int(is_correct)

            soft_accuracy_sum += float((p * gold).sum())
            onehot = torch.zeros_like(p)
            onehot[decision.gold_index] = 1.0
            hard_brier_sum += float(((p - onehot) ** 2).sum())
            soft_brier_sum += float(((p - gold) ** 2).sum())
            nll_sum += -math.log(
                max(float(p[decision.gold_index]), 1e-12)
            )
            ratio = gold / p.clamp_min(1e-12)
            kl_sum += float(
                (
                    gold
                    * torch.log(ratio.clamp(min=1e-12, max=1e4))
                ).sum()
            )
            confidence.append(float(p.max()))
            correctness.append(is_correct)

            if decision.primitive == "score":
                if decision.gold_score is None:
                    raise RuntimeError(
                        "score decision is missing gold_score"
                    )
                score_abs_errors.append(
                    abs(float(output.value) - decision.gold_score)
                )

    if decision_count == 0 or case_count == 0:
        raise ValueError("typed evaluator requires at least one case")

    primitive_accuracy = {
        primitive: (
            primitive_correct[primitive] / primitive_counts[primitive]
            if primitive_counts[primitive]
            else None
        )
        for primitive in primitive_counts
    }
    score_mae = (
        sum(score_abs_errors) / len(score_abs_errors)
        if score_abs_errors
        else None
    )

    return {
        "case_count": case_count,
        "decision_count": decision_count,
        "primitive_counts": primitive_counts,
        "accuracy": correct_total / decision_count,
        "primitive_accuracy": primitive_accuracy,
        "soft_accuracy": soft_accuracy_sum / decision_count,
        "hard_brier": hard_brier_sum / decision_count,
        "soft_brier": soft_brier_sum / decision_count,
        "nll": nll_sum / decision_count,
        "kl_gold_to_prediction": kl_sum / decision_count,
        "ece": _ece(confidence, correctness, bins=15),
        "score_mae": score_mae,
        "score_count": len(score_abs_errors),
        "state_encode_calls": state_encode_calls,
        "state_encode_calls_per_case": state_encode_calls / case_count,
        "decisions_per_state_encode": decision_count / state_encode_calls,
        "probability_mass_max_error": probability_mass_max_error,
    }
