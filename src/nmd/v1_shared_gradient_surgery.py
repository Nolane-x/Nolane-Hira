from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GradientSurgeryDiagnostics:
    pre_dot: float
    post_primary_relation_dot: float
    primary_norm: float
    relation_norm: float
    projected_primary_norm: float
    combined_norm: float
    conflict: bool
    projection_coefficient: float

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "pre_dot": self.pre_dot,
            "post_primary_relation_dot": self.post_primary_relation_dot,
            "primary_norm": self.primary_norm,
            "relation_norm": self.relation_norm,
            "projected_primary_norm": self.projected_primary_norm,
            "combined_norm": self.combined_norm,
            "conflict": self.conflict,
            "projection_coefficient": self.projection_coefficient,
        }


def _validate_pair(primary: Sequence[Tensor], relation: Sequence[Tensor]) -> None:
    if len(primary) != len(relation):
        raise ValueError("S16 gradient lists must have identical length")
    if not primary:
        raise ValueError("S16 gradient lists cannot be empty")
    for index, (gp, gr) in enumerate(zip(primary, relation)):
        if gp.shape != gr.shape:
            raise ValueError(f"S16 gradient shape mismatch at {index}")
        if gp.device != gr.device:
            raise ValueError(f"S16 gradient device mismatch at {index}")
        if gp.dtype != gr.dtype:
            raise ValueError(f"S16 gradient dtype mismatch at {index}")
        if not bool(torch.isfinite(gp).all() and torch.isfinite(gr).all()):
            raise ValueError(f"S16 non-finite gradient at {index}")


def _sum_dot(left: Sequence[Tensor], right: Sequence[Tensor]) -> Tensor:
    total = left[0].new_zeros(())
    for a, b in zip(left, right):
        total = total + torch.sum(a * b)
    return total


def _sum_sq(values: Sequence[Tensor]) -> Tensor:
    total = values[0].new_zeros(())
    for value in values:
        total = total + torch.sum(value * value)
    return total


def relation_priority_gradient_surgery(
    primary: Sequence[Tensor],
    relation: Sequence[Tensor],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], list[Tensor], GradientSurgeryDiagnostics]:
    """Resolve shared-surface conflict without learned parameters.

    If the global primary/relation gradient dot product is negative, remove
    exactly the component of the primary gradient that opposes the relation
    gradient. Otherwise leave the primary gradient unchanged.

    Returns:
      projected_primary, combined_update, diagnostics

    The combined update is projected_primary + relation.
    """
    if epsilon <= 0:
        raise ValueError("S16 gradient-surgery epsilon must be positive")
    _validate_pair(primary, relation)

    gp = [x.detach().clone() for x in primary]
    gr = [x.detach().clone() for x in relation]

    pre_dot_t = _sum_dot(gp, gr)
    primary_sq = _sum_sq(gp)
    relation_sq = _sum_sq(gr)
    conflict = bool(pre_dot_t.item() < 0.0)

    coefficient_t = pre_dot_t.new_zeros(())
    projected = gp
    if conflict and float(relation_sq.item()) > 0.0:
        coefficient_t = pre_dot_t / (relation_sq + epsilon)
        projected = [
            p - coefficient_t * r
            for p, r in zip(gp, gr)
        ]

    post_dot_t = _sum_dot(projected, gr)
    combined = [p + r for p, r in zip(projected, gr)]
    combined_sq = _sum_sq(combined)

    diagnostics = GradientSurgeryDiagnostics(
        pre_dot=float(pre_dot_t.item()),
        post_primary_relation_dot=float(post_dot_t.item()),
        primary_norm=float(torch.sqrt(primary_sq).item()),
        relation_norm=float(torch.sqrt(relation_sq).item()),
        projected_primary_norm=float(torch.sqrt(_sum_sq(projected)).item()),
        combined_norm=float(torch.sqrt(combined_sq).item()),
        conflict=conflict,
        projection_coefficient=float(coefficient_t.item()),
    )
    return projected, combined, diagnostics


def apply_gradient_update(
    parameters: Iterable[torch.nn.Parameter],
    gradients: Sequence[Tensor],
) -> None:
    params = list(parameters)
    if len(params) != len(gradients):
        raise ValueError("S16 parameter/gradient length mismatch")
    for index, (parameter, gradient) in enumerate(zip(params, gradients)):
        if parameter.shape != gradient.shape:
            raise ValueError(f"S16 parameter/gradient shape mismatch at {index}")
        if not parameter.requires_grad:
            raise ValueError(f"S16 target parameter {index} is frozen")
        parameter.grad = gradient.detach().clone()


__all__ = [
    "GradientSurgeryDiagnostics",
    "apply_gradient_update",
    "relation_priority_gradient_surgery",
]
