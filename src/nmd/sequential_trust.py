from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SafeStepCandidate:
    eta: float
    mean_teacher_kl: float
    teacher_argmax_agreement_count: int
    trust_n: int
    finite: bool
    non_relation_bit_identical: bool
    first_order_structural_predicted_benefit: float
    applied_update_l2: float

    @property
    def trust_safe(self) -> bool:
        return (
            self.finite
            and self.non_relation_bit_identical
            and self.teacher_argmax_agreement_count == self.trust_n
            and self.first_order_structural_predicted_benefit > 0.0
        )


def choose_largest_safe_step(
    candidates: Iterable[SafeStepCandidate],
    *,
    mean_kl_ceiling: float,
) -> SafeStepCandidate | None:
    if mean_kl_ceiling < 0:
        raise ValueError("mean_kl_ceiling must be non-negative")

    safe = [
        row
        for row in candidates
        if row.trust_safe and row.mean_teacher_kl <= mean_kl_ceiling
    ]
    if not safe:
        return None

    return min(
        safe,
        key=lambda row: (
            -float(row.eta),
            float(row.mean_teacher_kl),
            float(row.applied_update_l2),
        ),
    )


def exact_zero_drop(
    *,
    candidate_correct: int,
    baseline_correct: int,
) -> bool:
    if candidate_correct < 0 or baseline_correct < 0:
        raise ValueError("correct counts must be non-negative")
    return candidate_correct >= baseline_correct
