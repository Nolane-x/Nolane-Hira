from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .block_interpolation import atomic_group_for_key
from .delta_surgery import RelationSpan, relation_layout


@dataclass(frozen=True)
class RetentionSubspace:
    rows: Tensor
    eigenvalues: Tensor
    eigenvectors: Tensor
    cumulative_coverage: Tensor
    orthonormal_error: float

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.numel())


def build_retention_subspace(
    weighted_gradient_rows: Tensor,
    *,
    relative_eigenvalue_floor: float = 1e-12,
) -> RetentionSubspace:
    rows = weighted_gradient_rows.detach().cpu().to(torch.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError("weighted_gradient_rows must be non-empty [B,D]")
    if not torch.isfinite(rows).all():
        raise ValueError("weighted_gradient_rows must be finite")
    gram = rows @ rows.T
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0.0)
    evecs = evecs[:, order]

    max_eval = float(evals[0]) if evals.numel() else 0.0
    floor = max_eval * float(relative_eigenvalue_floor)
    keep = evals > floor
    evals = evals[keep]
    evecs = evecs[:, keep]
    if evals.numel() == 0:
        raise ValueError("retention gradient subspace has zero numerical rank")

    total = evals.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        raise ValueError("retention gradient energy must be positive")
    coverage = evals.cumsum(0) / total

    # Q = G^T U Lambda^-1/2. Check Q^T Q without materializing Q.
    scaled = evecs / evals.sqrt().unsqueeze(0)
    qtq = scaled.T @ gram @ scaled
    eye = torch.eye(qtq.shape[0], dtype=qtq.dtype)
    orth_error = float((qtq - eye).abs().max())

    return RetentionSubspace(
        rows=rows,
        eigenvalues=evals,
        eigenvectors=evecs,
        cumulative_coverage=coverage,
        orthonormal_error=orth_error,
    )


def rank_for_coverage(subspace: RetentionSubspace, target: float) -> tuple[int, float]:
    target = float(target)
    if not 0.0 < target <= 1.0:
        raise ValueError("coverage target must be in (0,1]")
    hits = torch.nonzero(
        subspace.cumulative_coverage >= target,
        as_tuple=False,
    ).squeeze(1)
    rank = subspace.rank if hits.numel() == 0 else int(hits[0]) + 1
    realized = float(subspace.cumulative_coverage[rank - 1])
    return rank, realized


def project_onto_retention_subspace(
    vector: Tensor,
    subspace: RetentionSubspace,
    *,
    rank: int,
) -> Tensor:
    x = vector.detach().cpu().to(torch.float64)
    if x.ndim != 1 or x.shape[0] != subspace.rows.shape[1]:
        raise ValueError("vector dimension mismatch")
    rank = int(rank)
    if rank <= 0 or rank > subspace.rank:
        raise ValueError("rank outside retention subspace")
    u = subspace.eigenvectors[:, :rank]
    lam = subspace.eigenvalues[:rank]
    gx = subspace.rows @ x
    coeff = (u.T @ gx) / lam
    return subspace.rows.T @ (u @ coeff)


def rotate_delta_away_from_retention(
    delta: Tensor,
    subspace: RetentionSubspace,
    *,
    rank: int,
    removal_strength: float,
) -> tuple[Tensor, Tensor]:
    removal_strength = float(removal_strength)
    if not 0.0 <= removal_strength <= 1.0:
        raise ValueError("removal_strength must be in [0,1]")
    d = delta.detach().cpu().to(torch.float64)
    projected = project_onto_retention_subspace(d, subspace, rank=rank)
    rotated = d - removal_strength * projected
    return rotated, projected


def apply_relation_update(
    base: Mapping[str, Tensor],
    relation_update: Tensor,
    *,
    alpha: float,
    layout: Sequence[RelationSpan] | None = None,
) -> dict[str, Tensor]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    spans = tuple(layout) if layout is not None else relation_layout(base)
    update = relation_update.detach().cpu().to(torch.float64)
    if update.ndim != 1 or update.shape[0] != spans[-1].end:
        raise ValueError("relation_update shape mismatch")

    relation_keys = {span.key for span in spans}
    out: dict[str, Tensor] = {
        key: value.detach().cpu().clone() for key, value in base.items()
    }
    for span in spans:
        original = base[span.key].detach().cpu()
        local = update[span.start:span.end].reshape(span.shape)
        out[span.key] = (
            original.to(torch.float64) + alpha * local
        ).to(original.dtype)

    for key in base:
        if atomic_group_for_key(key) == "relation" and key not in relation_keys:
            raise ValueError(f"relation key not covered by layout: {key}")
    return out


def non_relation_bit_identical(
    base: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    *,
    layout: Sequence[RelationSpan] | None = None,
) -> bool:
    spans = tuple(layout) if layout is not None else relation_layout(base)
    relation_keys = {span.key for span in spans}
    if set(base) != set(candidate):
        return False
    return all(
        torch.equal(base[key].detach().cpu(), candidate[key].detach().cpu())
        for key in base
        if key not in relation_keys
    )


def projection_metrics(
    original_delta: Tensor,
    rotated_delta: Tensor,
    structural_gradient: Tensor,
) -> dict[str, float | int]:
    d = original_delta.detach().cpu().to(torch.float64)
    r = rotated_delta.detach().cpu().to(torch.float64)
    g = structural_gradient.detach().cpu().to(torch.float64)
    if not (d.ndim == r.ndim == g.ndim == 1 and d.shape == r.shape == g.shape):
        raise ValueError("projection metric vectors must be equal 1-D tensors")
    d_norm = d.norm()
    r_norm = r.norm()
    if float(d_norm) <= 0.0:
        raise ValueError("original delta must be nonzero")
    removed = d - r
    removed_energy_fraction = float(removed.square().sum() / d.square().sum())
    cosine = 0.0
    if float(r_norm) > 0.0:
        cosine = float(torch.dot(d, r) / (d_norm * r_norm))
    return {
        "original_delta_l2": float(d_norm),
        "retained_delta_l2": float(r_norm),
        "removed_delta_l2": float(removed.norm()),
        "removed_delta_energy_fraction": removed_energy_fraction,
        "cosine_with_original_delta": cosine,
        "first_order_structural_predicted_benefit": float(-torch.dot(g, r)),
        "changed_relation_coordinates": int((r.abs() > 1e-12).sum()),
    }
