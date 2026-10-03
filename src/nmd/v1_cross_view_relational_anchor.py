from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def cross_view_similarity_matrix(
    canonical_signatures: Tensor,
    paraphrase_signatures: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    """Return full BxKxK cross-view cosine geometry."""
    if epsilon <= 0:
        raise ValueError("S42 relational anchor epsilon must be positive")
    if canonical_signatures.shape != paraphrase_signatures.shape:
        raise ValueError("S42 canonical/paraphrase signature shape mismatch")
    if canonical_signatures.ndim != 3:
        raise ValueError("S42 signatures must have shape [B,K,D]")
    if canonical_signatures.shape[1] < 1 or canonical_signatures.shape[2] < 1:
        raise ValueError("S42 signatures must have nonempty option/native dimensions")
    if not bool(
        torch.isfinite(canonical_signatures).all()
        and torch.isfinite(paraphrase_signatures).all()
    ):
        raise ValueError("S42 non-finite signature input")

    canonical = F.normalize(canonical_signatures, dim=-1, eps=epsilon)
    paraphrase = F.normalize(paraphrase_signatures, dim=-1, eps=epsilon)
    return canonical @ paraphrase.transpose(-1, -2)


def relational_geometry_matrix_loss(
    treatment_geometry: Tensor,
    reference_geometry: Tensor,
) -> Tensor:
    """Unweighted full-matrix MSE; reference is always detached."""
    if treatment_geometry.shape != reference_geometry.shape:
        raise ValueError("S42 treatment/reference geometry shape mismatch")
    if treatment_geometry.ndim != 3:
        raise ValueError("S42 geometry must have shape [B,K,K]")
    if treatment_geometry.shape[-1] != treatment_geometry.shape[-2]:
        raise ValueError("S42 geometry must be square over options")
    if not bool(
        torch.isfinite(treatment_geometry).all()
        and torch.isfinite(reference_geometry).all()
    ):
        raise ValueError("S42 non-finite geometry input")
    reference = reference_geometry.detach()
    return torch.mean((treatment_geometry - reference) ** 2)


def relational_signature_geometry_anchor(
    treatment_canonical: Tensor,
    treatment_paraphrase: Tensor,
    reference_canonical: Tensor,
    reference_paraphrase: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    """Full cross-view KxK relational geometry anchor."""
    if treatment_canonical.shape != reference_canonical.shape:
        raise ValueError("S42 canonical treatment/reference signature shape mismatch")
    if treatment_paraphrase.shape != reference_paraphrase.shape:
        raise ValueError("S42 paraphrase treatment/reference signature shape mismatch")
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
    return relational_geometry_matrix_loss(treatment_geometry, reference_geometry)


__all__ = [
    "cross_view_similarity_matrix",
    "relational_geometry_matrix_loss",
    "relational_signature_geometry_anchor",
]
