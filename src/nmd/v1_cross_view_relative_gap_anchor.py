from __future__ import annotations

import torch
from torch import Tensor

from nmd.v1_cross_view_relational_anchor import cross_view_similarity_matrix


def relative_gap_tensor(geometry: Tensor) -> Tensor:
    """Return row-wise same-option-vs-option gaps with zero diagonal.

    For cross-view similarity geometry G[B,K,K]:
        R[B,i,j] = G[B,i,i] - G[B,i,j]

    Diagonal entries are identically zero and are excluded by the anchor loss.
    """
    if geometry.ndim != 3:
        raise ValueError("S43 geometry must have shape [B,K,K]")
    if geometry.shape[-1] != geometry.shape[-2]:
        raise ValueError("S43 geometry must be square over options")
    if geometry.shape[-1] < 2:
        raise ValueError("S43 relative-gap geometry requires K>=2")
    if not bool(torch.isfinite(geometry).all()):
        raise ValueError("S43 non-finite geometry input")

    diagonal = torch.diagonal(geometry, dim1=-2, dim2=-1).unsqueeze(-1)
    return diagonal - geometry


def relative_gap_geometry_loss(
    treatment_geometry: Tensor,
    reference_geometry: Tensor,
) -> Tensor:
    """MSE over all off-diagonal same-option-vs-wrong-option gaps.

    Reference geometry is always detached. No gold labels, max operator,
    weighting coefficient, or numeric margin target enters this loss.
    """
    if treatment_geometry.shape != reference_geometry.shape:
        raise ValueError("S43 treatment/reference geometry shape mismatch")
    if treatment_geometry.ndim != 3:
        raise ValueError("S43 geometry must have shape [B,K,K]")
    if treatment_geometry.shape[-1] != treatment_geometry.shape[-2]:
        raise ValueError("S43 geometry must be square over options")
    k = treatment_geometry.shape[-1]
    if k < 2:
        raise ValueError("S43 relative-gap geometry requires K>=2")
    if not bool(
        torch.isfinite(treatment_geometry).all()
        and torch.isfinite(reference_geometry).all()
    ):
        raise ValueError("S43 non-finite geometry input")

    treatment_gap = relative_gap_tensor(treatment_geometry)
    reference_gap = relative_gap_tensor(reference_geometry.detach())

    offdiag = ~torch.eye(k, dtype=torch.bool, device=treatment_geometry.device)
    delta = treatment_gap[..., offdiag] - reference_gap[..., offdiag]
    return torch.mean(delta**2)


def relative_gap_signature_geometry_anchor(
    treatment_canonical: Tensor,
    treatment_paraphrase: Tensor,
    reference_canonical: Tensor,
    reference_paraphrase: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    """Label-free cross-view relative-gap signature geometry anchor."""
    if treatment_canonical.shape != reference_canonical.shape:
        raise ValueError("S43 canonical treatment/reference signature shape mismatch")
    if treatment_paraphrase.shape != reference_paraphrase.shape:
        raise ValueError("S43 paraphrase treatment/reference signature shape mismatch")

    treatment_geometry = cross_view_similarity_matrix(
        treatment_canonical,
        treatment_paraphrase,
        epsilon=epsilon,
    )
    reference_geometry = cross_view_similarity_matrix(
        reference_canonical.detach(),
        reference_paraphrase.detach(),
        epsilon=epsilon,
    )
    return relative_gap_geometry_loss(treatment_geometry, reference_geometry)


__all__ = [
    "relative_gap_tensor",
    "relative_gap_geometry_loss",
    "relative_gap_signature_geometry_anchor",
]
