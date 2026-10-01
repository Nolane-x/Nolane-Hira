from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .v1_neutral_bisector_gradient import (
    neutral_bisector_norm_balanced_gradient_update,
)
from .v1_norm_balanced_gradient import NormBalancedGradientDiagnostics


@dataclass(frozen=True)
class S25GradientOwnershipDiagnostics:
    shared: NormBalancedGradientDiagnostics
    primary_to_relation_private_max_abs: float
    relation_to_primary_private_max_abs: float
    primary_private_l1: float
    relation_private_l1: float
    primary_shared_l1: float
    relation_shared_l1: float


def _materialize(
    grads: Sequence[Tensor | None],
    params: Sequence[nn.Parameter],
) -> list[Tensor]:
    return [
        torch.zeros_like(p) if g is None else g.detach().clone()
        for g, p in zip(grads, params)
    ]


def _l1(grads: Sequence[Tensor]) -> float:
    return float(sum(g.abs().sum() for g in grads).detach().cpu())


def _max_abs_optional(grads: Sequence[Tensor | None]) -> float:
    values = [
        float(g.detach().abs().max().cpu())
        for g in grads
        if g is not None and g.numel()
    ]
    return max(values, default=0.0)


def apply_s25_decoupled_gradient_update(
    *,
    primary_block: Tensor,
    relation_block: Tensor,
    shared: Sequence[nn.Parameter],
    primary_private: Sequence[nn.Parameter],
    relation_private: Sequence[nn.Parameter],
    epsilon: float = 1e-12,
) -> S25GradientOwnershipDiagnostics:
    if not shared or not primary_private or not relation_private:
        raise ValueError("S25 gradient ownership requires all parameter groups")
    if len({id(p) for p in (*shared, *primary_private, *relation_private)}) != (
        len(shared) + len(primary_private) + len(relation_private)
    ):
        raise ValueError("S25 parameter groups must be disjoint")

    targets = [*shared, *primary_private, *relation_private]
    p_all = torch.autograd.grad(
        primary_block,
        targets,
        retain_graph=True,
        allow_unused=True,
    )
    r_all = torch.autograd.grad(
        relation_block,
        targets,
        retain_graph=False,
        allow_unused=True,
    )

    ns = len(shared)
    np = len(primary_private)
    p_shared_o = p_all[:ns]
    p_primary_o = p_all[ns : ns + np]
    p_relation_cross = p_all[ns + np :]
    r_shared_o = r_all[:ns]
    r_primary_cross = r_all[ns : ns + np]
    r_relation_o = r_all[ns + np :]

    p_to_r = _max_abs_optional(p_relation_cross)
    r_to_p = _max_abs_optional(r_primary_cross)
    if p_to_r != 0.0:
        raise RuntimeError("S25 primary block reached relation-private surface")
    if r_to_p != 0.0:
        raise RuntimeError("S25 relation block reached primary-private surface")

    p_shared = _materialize(p_shared_o, shared)
    r_shared = _materialize(r_shared_o, shared)
    p_private = _materialize(p_primary_o, primary_private)
    r_private = _materialize(r_relation_o, relation_private)

    combined, shared_diag = neutral_bisector_norm_balanced_gradient_update(
        p_shared,
        r_shared,
        epsilon=epsilon,
    )

    for param, g in zip(shared, combined):
        param.grad = g
    for param, g in zip(primary_private, p_private):
        param.grad = g
    for param, g in zip(relation_private, r_private):
        param.grad = g

    return S25GradientOwnershipDiagnostics(
        shared=shared_diag,
        primary_to_relation_private_max_abs=p_to_r,
        relation_to_primary_private_max_abs=r_to_p,
        primary_private_l1=_l1(p_private),
        relation_private_l1=_l1(r_private),
        primary_shared_l1=_l1(p_shared),
        relation_shared_l1=_l1(r_shared),
    )


__all__ = [
    "S25GradientOwnershipDiagnostics",
    "apply_s25_decoupled_gradient_update",
]
