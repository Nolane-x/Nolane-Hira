from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .v1_evidence_fusion import standardize_full_k_evidence


@dataclass(frozen=True)
class StandardizedTriadicConsistencyDiagnostics:
    canonical_rms: float
    paraphrase_rms: float
    canonical_flat_rate: float
    paraphrase_flat_rate: float
    mean_squared_distance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "canonical_rms": self.canonical_rms,
            "paraphrase_rms": self.paraphrase_rms,
            "canonical_flat_rate": self.canonical_flat_rate,
            "paraphrase_flat_rate": self.paraphrase_flat_rate,
            "mean_squared_distance": self.mean_squared_distance,
        }


def standardized_triadic_consistency_loss(
    canonical_logits: Tensor,
    paraphrase_logits: Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, StandardizedTriadicConsistencyDiagnostics]:
    """Scale-free S20 consistency on the same evidence geometry used by fusion."""

    if canonical_logits.shape != paraphrase_logits.shape:
        raise ValueError("S20 triadic views must share the exact [B,K] shape")
    if canonical_logits.ndim != 2 or canonical_logits.shape[-1] < 2:
        raise ValueError("S20 triadic evidence must be [B,K>=2]")
    if epsilon <= 0:
        raise ValueError("S20 standardization epsilon must be positive")

    canonical, canonical_rms = standardize_full_k_evidence(
        canonical_logits,
        epsilon=epsilon,
    )
    paraphrase, paraphrase_rms = standardize_full_k_evidence(
        paraphrase_logits,
        epsilon=epsilon,
    )

    loss = (canonical - paraphrase).pow(2).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("S20 standardized consistency became non-finite")

    diagnostics = StandardizedTriadicConsistencyDiagnostics(
        canonical_rms=float(canonical_rms.mean().detach().cpu()),
        paraphrase_rms=float(paraphrase_rms.mean().detach().cpu()),
        canonical_flat_rate=float((canonical_rms <= epsilon).to(loss.dtype).mean().detach().cpu()),
        paraphrase_flat_rate=float((paraphrase_rms <= epsilon).to(loss.dtype).mean().detach().cpu()),
        mean_squared_distance=float(loss.detach().cpu()),
    )
    return loss, canonical, paraphrase, diagnostics


__all__ = [
    "StandardizedTriadicConsistencyDiagnostics",
    "standardized_triadic_consistency_loss",
]
