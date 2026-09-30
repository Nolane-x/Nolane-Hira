from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .v1_norm_balanced_gradient import NormBalancedGradientDiagnostics


def _validate(primary: Sequence[Tensor], relation: Sequence[Tensor]) -> None:
    if len(primary) != len(relation) or not primary:
        raise ValueError("S22 gradient lists must be nonempty and equal length")
    for i, (p, r) in enumerate(zip(primary, relation)):
        if p.shape != r.shape or p.device != r.device or p.dtype != r.dtype:
            raise ValueError(f"S22 gradient mismatch at {i}")
        if not bool(torch.isfinite(p).all() and torch.isfinite(r).all()):
            raise ValueError(f"S22 non-finite gradient at {i}")


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    total = a[0].new_zeros(())
    for x, y in zip(a, b):
        total = total + torch.sum(x * y)
    return total


def _sq(a: Sequence[Tensor]) -> Tensor:
    return _dot(a, a)


def primary_priority_norm_balanced_gradient_update(
    primary: Sequence[Tensor],
    relation: Sequence[Tensor],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], NormBalancedGradientDiagnostics]:
    """S22: norm balance while protecting the primary direction on conflict.

    This is the exact S17/S21 rule with one controlled change: when the
    normalized primary and relation directions conflict, relation is projected
    away from primary instead of primary being projected away from relation.
    """
    if epsilon <= 0:
        raise ValueError("S22 epsilon must be positive")
    _validate(primary, relation)

    gp = [x.detach().clone() for x in primary]
    gr = [x.detach().clone() for x in relation]
    np_t = torch.sqrt(_sq(gp))
    nr_t = torch.sqrt(_sq(gr))
    np = float(np_t.item())
    nr = float(nr_t.item())

    if np == 0.0 and nr == 0.0:
        combined = [torch.zeros_like(x) for x in gp]
        return combined, NormBalancedGradientDiagnostics(
            np, nr, 0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, "both_zero"
        )
    if nr == 0.0:
        return gp, NormBalancedGradientDiagnostics(
            np, nr, 0.0, 0.0, False, 0.0, np, 1.0, np, "relation_zero"
        )
    if np == 0.0:
        return gr, NormBalancedGradientDiagnostics(
            np, nr, 0.0, 0.0, False, 0.0, nr, 1.0, nr, "primary_zero"
        )

    up = [x / (np_t + epsilon) for x in gp]
    ur = [x / (nr_t + epsilon) for x in gr]
    pre = _dot(up, ur)
    conflict = bool(pre.item() < 0.0)

    coeff = pre.new_zeros(())
    projected_relation = ur
    if conflict:
        up_sq = _sq(up)
        coeff = pre / (up_sq + epsilon)
        projected_relation = [
            r - coeff * p for p, r in zip(up, ur)
        ]

    post = _dot(up, projected_relation)
    direction = [p + r for p, r in zip(up, projected_relation)]
    direction_norm_t = torch.sqrt(_sq(direction))
    scale_t = 0.5 * (np_t + nr_t)

    if float(direction_norm_t.item()) == 0.0:
        combined = [torch.zeros_like(x) for x in direction]
    else:
        combined = [
            scale_t * x / (direction_norm_t + epsilon)
            for x in direction
        ]

    combined_norm_t = torch.sqrt(_sq(combined))
    return combined, NormBalancedGradientDiagnostics(
        primary_norm=np,
        relation_norm=nr,
        normalized_pre_dot=float(pre.item()),
        normalized_post_dot=float(post.item()),
        conflict=conflict,
        projection_coefficient=float(coeff.item()),
        reference_scale=float(scale_t.item()),
        direction_norm=float(direction_norm_t.item()),
        combined_norm=float(combined_norm_t.item()),
        special_case="balanced",
    )


__all__ = ["primary_priority_norm_balanced_gradient_update"]
