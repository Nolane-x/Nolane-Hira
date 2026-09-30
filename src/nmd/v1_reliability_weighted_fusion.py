from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_evidence_fusion import standardize_full_k_evidence


@dataclass(frozen=True)
class ReliabilityWeightedFusionDiagnostics:
    primary_rms: Tensor
    relation_rms: Tensor
    primary_reliability: Tensor
    relation_reliability: Tensor
    primary_weight: Tensor
    relation_weight: Tensor
    standardized_cosine: Tensor
    expert_top1_agreement: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "primary_rms": float(self.primary_rms.detach().cpu()),
            "relation_rms": float(self.relation_rms.detach().cpu()),
            "primary_reliability": float(self.primary_reliability.detach().cpu()),
            "relation_reliability": float(self.relation_reliability.detach().cpu()),
            "primary_weight": float(self.primary_weight.detach().cpu()),
            "relation_weight": float(self.relation_weight.detach().cpu()),
            "standardized_cosine": float(self.standardized_cosine.detach().cpu()),
            "expert_top1_agreement": float(self.expert_top1_agreement.detach().cpu()),
        }


def standardized_top2_gap(z: Tensor) -> Tensor:
    if z.ndim != 2 or z.shape[-1] < 2:
        raise ValueError("S24 standardized evidence must be [B,K>=2]")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("S24 standardized evidence must be finite")
    top2 = torch.topk(z, k=2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    if not bool(torch.isfinite(gap).all()) or bool((gap < 0).any()):
        raise ValueError("S24 reliability gap became invalid")
    return gap


class ReliabilityWeightedFullKEvidenceFusion(nn.Module):
    """S24 deterministic reliability-weighted full-K expert fusion.

    Both experts are center/RMS standardized exactly as S14. Reliability is
    the standardized top1-top2 gap plus frozen epsilon. The two reliabilities
    are normalized to symmetric nonnegative weights. The relation expert is
    detached from the fused-primary objective exactly as in S15-S23.
    """

    def __init__(self, *, epsilon: float = 1e-6):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("S24 fusion epsilon must be positive")
        self.epsilon = float(epsilon)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        primary_logits: Tensor,
        relation_logits: Tensor,
    ) -> tuple[Tensor, ReliabilityWeightedFusionDiagnostics]:
        if primary_logits.shape != relation_logits.shape:
            raise ValueError("S24 experts must share the exact [B,K] shape")
        relation_logits = relation_logits.detach()

        primary, primary_rms = standardize_full_k_evidence(
            primary_logits,
            epsilon=self.epsilon,
        )
        relation, relation_rms = standardize_full_k_evidence(
            relation_logits,
            epsilon=self.epsilon,
        )

        primary_gap = standardized_top2_gap(primary)
        relation_gap = standardized_top2_gap(relation)
        primary_rel = primary_gap + self.epsilon
        relation_rel = relation_gap + self.epsilon
        denom = primary_rel + relation_rel
        primary_w = primary_rel / denom
        relation_w = relation_rel / denom

        fused = (
            primary_w[:, None] * primary
            + relation_w[:, None] * relation
        )
        if not bool(
            torch.isfinite(fused).all()
            and torch.isfinite(primary_w).all()
            and torch.isfinite(relation_w).all()
        ):
            raise ValueError("S24 reliability fusion became non-finite")

        cosine = F.cosine_similarity(primary, relation, dim=-1)
        agreement = (
            primary_logits.argmax(dim=-1)
            == relation_logits.argmax(dim=-1)
        ).to(fused.dtype)

        diagnostics = ReliabilityWeightedFusionDiagnostics(
            primary_rms=primary_rms.mean(),
            relation_rms=relation_rms.mean(),
            primary_reliability=primary_rel.mean(),
            relation_reliability=relation_rel.mean(),
            primary_weight=primary_w.mean(),
            relation_weight=relation_w.mean(),
            standardized_cosine=cosine.mean(),
            expert_top1_agreement=agreement.mean(),
        )
        return fused, diagnostics


__all__ = [
    "ReliabilityWeightedFusionDiagnostics",
    "ReliabilityWeightedFullKEvidenceFusion",
    "standardized_top2_gap",
]
