from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def teacher_kl_loss(student_logits: Tensor, teacher_probabilities: Tensor) -> Tensor:
    if student_logits.ndim != 2:
        raise ValueError("student_logits must be [N,C]")
    teacher = teacher_probabilities.detach().to(
        device=student_logits.device,
        dtype=student_logits.dtype,
    )
    if teacher.shape != student_logits.shape:
        raise ValueError("teacher probabilities shape mismatch")
    if not torch.isfinite(teacher).all():
        raise ValueError("teacher probabilities must be finite")
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
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher,
        reduction="batchmean",
    )


def weighted_batch_gradient_row(
    flat_batch_mean_gradient: Tensor,
    *,
    batch_n: int,
    total_n: int,
) -> Tensor:
    if batch_n <= 0 or total_n <= 0 or batch_n > total_n:
        raise ValueError("invalid batch/total size")
    row = flat_batch_mean_gradient.detach().cpu().to(torch.float64)
    if row.ndim != 1 or not torch.isfinite(row).all():
        raise ValueError("gradient row must be finite 1-D")
    return row * math.sqrt(batch_n / total_n)
