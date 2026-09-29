from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class NormBalancedGradientDiagnostics:
    primary_norm: float
    relation_norm: float
    normalized_pre_dot: float
    normalized_post_dot: float
    conflict: bool
    projection_coefficient: float
    reference_scale: float
    direction_norm: float
    combined_norm: float
    special_case: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "primary_norm": self.primary_norm,
            "relation_norm": self.relation_norm,
            "normalized_pre_dot": self.normalized_pre_dot,
            "normalized_post_dot": self.normalized_post_dot,
            "conflict": self.conflict,
            "projection_coefficient": self.projection_coefficient,
            "reference_scale": self.reference_scale,
            "direction_norm": self.direction_norm,
            "combined_norm": self.combined_norm,
            "special_case": self.special_case,
        }


def _validate(primary: Sequence[Tensor], relation: Sequence[Tensor]) -> None:
    if len(primary) != len(relation) or not primary:
        raise ValueError("S17 gradient lists must be nonempty and equal length")
    for i, (p, r) in enumerate(zip(primary, relation)):
        if p.shape != r.shape or p.device != r.device or p.dtype != r.dtype:
            raise ValueError(f"S17 gradient mismatch at {i}")
        if not bool(torch.isfinite(p).all() and torch.isfinite(r).all()):
            raise ValueError(f"S17 non-finite gradient at {i}")


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    total = a[0].new_zeros(())
    for x, y in zip(a, b):
        total = total + torch.sum(x * y)
    return total


def _sq(a: Sequence[Tensor]) -> Tensor:
    return _dot(a, a)


def norm_balanced_gradient_update(
    primary: Sequence[Tensor],
    relation: Sequence[Tensor],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], NormBalancedGradientDiagnostics]:
    if epsilon <= 0:
        raise ValueError("S17 epsilon must be positive")
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
    projected = up
    if conflict:
        ur_sq = _sq(ur)
        coeff = pre / (ur_sq + epsilon)
        projected = [p - coeff * r for p, r in zip(up, ur)]

    post = _dot(projected, ur)
    direction = [p + r for p, r in zip(projected, ur)]
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


def apply_gradient_update(
    parameters: Iterable[torch.nn.Parameter],
    gradients: Sequence[Tensor],
) -> None:
    params = list(parameters)
    if len(params) != len(gradients):
        raise ValueError("S17 parameter/gradient length mismatch")
    for i, (p, g) in enumerate(zip(params, gradients)):
        if p.shape != g.shape or not p.requires_grad:
            raise ValueError(f"S17 invalid target parameter at {i}")
        p.grad = g.detach().clone()


__all__ = [
    "NormBalancedGradientDiagnostics",
    "apply_gradient_update",
    "norm_balanced_gradient_update",
]
