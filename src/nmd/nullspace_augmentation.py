from __future__ import annotations

import math

import torch
from torch import Tensor

from .subspace_projection import (
    RetentionSubspace,
    project_onto_retention_subspace,
)


def nullspace_component(
    vector: Tensor,
    subspace: RetentionSubspace,
    *,
    rank: int,
) -> tuple[Tensor, Tensor]:
    """Split a vector into rank-k damage-subspace and orthogonal components."""
    x = vector.detach().cpu().to(torch.float64)
    if x.ndim != 1 or x.shape[0] != subspace.rows.shape[1]:
        raise ValueError("vector dimension mismatch")
    if not torch.isfinite(x).all():
        raise ValueError("vector must be finite")
    projected = project_onto_retention_subspace(
        x,
        subspace,
        rank=rank,
    )
    residual = x - projected
    return residual, projected


def subspace_coordinates(
    vector: Tensor,
    subspace: RetentionSubspace,
    *,
    rank: int,
) -> Tensor:
    """Return coordinates Q^T x in the implicit orthonormal rank-k basis."""
    x = vector.detach().cpu().to(torch.float64)
    if x.ndim != 1 or x.shape[0] != subspace.rows.shape[1]:
        raise ValueError("vector dimension mismatch")
    if not torch.isfinite(x).all():
        raise ValueError("vector must be finite")
    rank = int(rank)
    if rank <= 0 or rank > subspace.rank:
        raise ValueError("rank outside retention subspace")
    u = subspace.eigenvectors[:, :rank]
    lam = subspace.eigenvalues[:rank]
    gx = subspace.rows @ x
    return (u.T @ gx) / lam.sqrt()


def normalized_subspace_leakage(
    vector: Tensor,
    subspace: RetentionSubspace,
    *,
    rank: int,
) -> float:
    """Max absolute orthonormal-basis coordinate divided by vector L2."""
    x = vector.detach().cpu().to(torch.float64)
    norm = float(x.norm())
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must have finite positive L2 norm")
    coords = subspace_coordinates(x, subspace, rank=rank)
    return float(coords.abs().max()) / norm


def normalize_like(direction: Tensor, reference: Tensor) -> Tensor:
    """Scale direction to the exact L2 norm of reference."""
    d = direction.detach().cpu().to(torch.float64)
    r = reference.detach().cpu().to(torch.float64)
    if d.ndim != 1 or r.ndim != 1 or d.shape != r.shape:
        raise ValueError("direction/reference must be equal 1-D tensors")
    if not torch.isfinite(d).all() or not torch.isfinite(r).all():
        raise ValueError("direction/reference must be finite")
    d_norm = float(d.norm())
    r_norm = float(r.norm())
    if d_norm <= 0.0 or r_norm <= 0.0:
        raise ValueError("direction/reference must have positive L2 norm")
    return d * (r_norm / d_norm)


def combine_updates(
    protected_update: Tensor,
    structural_null_direction: Tensor,
    *,
    alpha: float,
    gamma: float,
) -> Tensor:
    p = protected_update.detach().cpu().to(torch.float64)
    n = structural_null_direction.detach().cpu().to(torch.float64)
    if p.ndim != 1 or n.ndim != 1 or p.shape != n.shape:
        raise ValueError("updates must be equal 1-D tensors")
    if not torch.isfinite(p).all() or not torch.isfinite(n).all():
        raise ValueError("updates must be finite")
    alpha = float(alpha)
    gamma = float(gamma)
    if not math.isfinite(alpha) or not math.isfinite(gamma):
        raise ValueError("alpha/gamma must be finite")
    if alpha < 0.0 or gamma < 0.0:
        raise ValueError("alpha/gamma must be non-negative")
    return alpha * p + gamma * n


def first_order_structural_benefit(
    structural_gradient: Tensor,
    update: Tensor,
) -> float:
    g = structural_gradient.detach().cpu().to(torch.float64)
    u = update.detach().cpu().to(torch.float64)
    if g.ndim != 1 or u.ndim != 1 or g.shape != u.shape:
        raise ValueError("gradient/update must be equal 1-D tensors")
    if not torch.isfinite(g).all() or not torch.isfinite(u).all():
        raise ValueError("gradient/update must be finite")
    return float(-torch.dot(g, u))
