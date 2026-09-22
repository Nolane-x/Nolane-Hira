from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class FunctionalTrustMetrics:
    mean_teacher_kl: float
    teacher_argmax_agreement_count: int
    n: int
    teacher_argmax_agreement_fraction: float
    finite: bool


def functional_trust_metrics(
    student_logits: Tensor,
    teacher_probabilities: Tensor,
) -> FunctionalTrustMetrics:
    if student_logits.ndim != 2:
        raise ValueError("student_logits must be [N,C]")
    teacher = teacher_probabilities.detach().to(
        device=student_logits.device,
        dtype=student_logits.dtype,
    )
    if teacher.shape != student_logits.shape:
        raise ValueError("teacher probabilities shape mismatch")
    if student_logits.shape[0] <= 0:
        raise ValueError("functional trust batch must be non-empty")

    finite = bool(
        torch.isfinite(student_logits).all()
        and torch.isfinite(teacher).all()
    )
    if not finite:
        return FunctionalTrustMetrics(
            mean_teacher_kl=float("inf"),
            teacher_argmax_agreement_count=0,
            n=int(student_logits.shape[0]),
            teacher_argmax_agreement_fraction=0.0,
            finite=False,
        )

    if (teacher < 0).any():
        raise ValueError("teacher probabilities must be non-negative")
    row_sums = teacher.sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("teacher probabilities must sum to one")

    log_student = F.log_softmax(student_logits, dim=-1)
    per_example_kl = F.kl_div(
        log_student,
        teacher,
        reduction="none",
    ).sum(dim=-1)
    mean_kl = float(per_example_kl.mean().detach().cpu())

    student_argmax = student_logits.argmax(dim=-1)
    teacher_argmax = teacher.argmax(dim=-1)
    agreement_count = int(
        (student_argmax == teacher_argmax).sum().detach().cpu()
    )
    n = int(student_logits.shape[0])

    return FunctionalTrustMetrics(
        mean_teacher_kl=mean_kl,
        teacher_argmax_agreement_count=agreement_count,
        n=n,
        teacher_argmax_agreement_fraction=agreement_count / n,
        finite=True,
    )


def trust_safe(
    metrics: FunctionalTrustMetrics,
    *,
    mean_kl_ceiling: float,
    require_full_argmax_agreement: bool = True,
) -> bool:
    if mean_kl_ceiling < 0:
        raise ValueError("mean_kl_ceiling must be non-negative")
    if not metrics.finite:
        return False
    if metrics.mean_teacher_kl > float(mean_kl_ceiling):
        return False
    if require_full_argmax_agreement:
        return metrics.teacher_argmax_agreement_count == metrics.n
    return True
