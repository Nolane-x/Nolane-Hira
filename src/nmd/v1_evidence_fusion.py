from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EvidenceFusionDiagnostics:
    triadic_rms: Tensor
    relation_rms: Tensor
    standardized_cosine: Tensor
    expert_top1_agreement: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "triadic_rms": float(self.triadic_rms.detach().cpu()),
            "relation_rms": float(self.relation_rms.detach().cpu()),
            "standardized_cosine": float(
                self.standardized_cosine.detach().cpu()
            ),
            "expert_top1_agreement": float(
                self.expert_top1_agreement.detach().cpu()
            ),
        }


def standardize_full_k_evidence(
    logits: Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError("S14 evidence logits must be [B,K>=2]")
    if epsilon <= 0:
        raise ValueError("S14 fusion epsilon must be positive")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S14 evidence logits must be finite")

    centered = logits - logits.mean(dim=-1, keepdim=True)
    rms = centered.pow(2).mean(dim=-1, keepdim=True).sqrt()
    standardized = torch.where(
        rms > epsilon,
        centered / rms.clamp_min(epsilon),
        torch.zeros_like(centered),
    )
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("S14 standardized evidence became non-finite")
    return standardized, rms.squeeze(-1)


class SymmetricFullKEvidenceFusion(nn.Module):
    """Zero-parameter S14 fusion of triadic and canonical-relation evidence.

    Each expert is centered and RMS-normalized independently across the same
    full-K logical option set. The two standardized vectors are then averaged
    with exact equal weight. There is no learned gate, fitted coefficient,
    expert-specific temperature, or option-specific parameter.
    """

    def __init__(self, *, epsilon: float = 1e-6):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("S14 fusion epsilon must be positive")
        self.epsilon = float(epsilon)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        triadic_logits: Tensor,
        relation_logits: Tensor,
    ) -> tuple[Tensor, EvidenceFusionDiagnostics]:
        if triadic_logits.shape != relation_logits.shape:
            raise ValueError("S14 experts must share the exact [B,K] shape")

        triadic, triadic_rms = standardize_full_k_evidence(
            triadic_logits,
            epsilon=self.epsilon,
        )
        relation, relation_rms = standardize_full_k_evidence(
            relation_logits,
            epsilon=self.epsilon,
        )
        fused = 0.5 * (triadic + relation)
        if not bool(torch.isfinite(fused).all()):
            raise ValueError("S14 fused evidence became non-finite")

        cosine = F.cosine_similarity(triadic, relation, dim=-1)
        # If one expert is exactly flat its standardized vector is zero and
        # cosine_similarity returns zero; this is the intended neutral value.
        agreement = (
            triadic_logits.argmax(dim=-1)
            == relation_logits.argmax(dim=-1)
        ).to(fused.dtype)

        diagnostics = EvidenceFusionDiagnostics(
            triadic_rms=triadic_rms.mean(),
            relation_rms=relation_rms.mean(),
            standardized_cosine=cosine.mean(),
            expert_top1_agreement=agreement.mean(),
        )
        return fused, diagnostics


def fused_gold_vs_max_wrong_margin(
    logits: Tensor,
    gold: Tensor,
) -> Tensor:
    if logits.ndim != 2 or gold.shape != logits.shape[:1]:
        raise ValueError("S14 fused margin shape mismatch")
    if gold.dtype != torch.long:
        raise ValueError("S14 fused gold must be torch.long")
    if gold.numel() and (
        int(gold.min().item()) < 0 or int(gold.max().item()) >= logits.shape[-1]
    ):
        raise ValueError("S14 fused gold index out of range")
    chosen = logits.gather(-1, gold[:, None]).squeeze(-1)
    index = torch.arange(logits.shape[-1], device=logits.device)
    strongest_wrong = logits.masked_fill(
        index[None, :].eq(gold[:, None]),
        float("-inf"),
    ).max(-1).values
    return chosen - strongest_wrong


__all__ = [
    "EvidenceFusionDiagnostics",
    "SymmetricFullKEvidenceFusion",
    "fused_gold_vs_max_wrong_margin",
    "standardize_full_k_evidence",
]
