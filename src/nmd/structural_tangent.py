from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .subspace_projection import RetentionSubspace, project_onto_retention_subspace


@dataclass(frozen=True)
class StructuralNullTangent:
    raw_descent: Tensor
    projected_damage: Tensor
    null_descent: Tensor
    retained_energy_fraction: float
    leakage_energy_fraction: float


def functional_null_structural_tangent(
    structural_gradient: Tensor,
    subspace: RetentionSubspace,
    *,
    rank: int,
) -> StructuralNullTangent:
    g = structural_gradient.detach().cpu().to(torch.float64)
    if g.ndim != 1 or g.shape[0] != subspace.rows.shape[1]:
        raise ValueError("structural_gradient dimension mismatch")
    if not torch.isfinite(g).all():
        raise ValueError("structural_gradient must be finite")
    if float(g.norm()) <= 0.0:
        raise ValueError("structural_gradient must be nonzero")

    raw = -g
    projected = project_onto_retention_subspace(raw, subspace, rank=rank)
    null = raw - projected
    null_norm_sq = null.square().sum()
    raw_norm_sq = raw.square().sum()
    if not torch.isfinite(null).all() or float(null_norm_sq) <= 0.0:
        raise ValueError("functional-null tangent must be finite and nonzero")

    # Re-project the residual. This is a direct numerical check that the
    # returned direction lies in the measured damage-subspace nullspace.
    leakage = project_onto_retention_subspace(null, subspace, rank=rank)
    leakage_fraction = float(leakage.square().sum() / null_norm_sq)

    return StructuralNullTangent(
        raw_descent=raw,
        projected_damage=projected,
        null_descent=null,
        retained_energy_fraction=float(null_norm_sq / raw_norm_sq),
        leakage_energy_fraction=leakage_fraction,
    )


def normalize_to_reference_l2(vector: Tensor, reference: Tensor) -> Tensor:
    x = vector.detach().cpu().to(torch.float64)
    ref = reference.detach().cpu().to(torch.float64)
    if x.ndim != 1 or ref.ndim != 1 or x.shape != ref.shape:
        raise ValueError("vector/reference must be equal 1-D tensors")
    if not torch.isfinite(x).all() or not torch.isfinite(ref).all():
        raise ValueError("vector/reference must be finite")
    x_norm = x.norm()
    ref_norm = ref.norm()
    if float(x_norm) <= 0.0:
        raise ValueError("vector must be nonzero")
    if float(ref_norm) <= 0.0:
        raise ValueError("reference must be nonzero")
    return x * (ref_norm / x_norm)
