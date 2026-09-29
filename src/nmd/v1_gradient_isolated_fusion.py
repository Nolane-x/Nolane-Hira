from __future__ import annotations

from torch import Tensor, nn

from .v1_evidence_fusion import (
    EvidenceFusionDiagnostics,
    SymmetricFullKEvidenceFusion,
)


class GradientIsolatedFullKEvidenceFusion(nn.Module):
    """Numerically identical S14 fusion with protected relation gradients.

    Forward evidence is unchanged: the same centered/RMS-normalized 0.5/0.5
    full-K fusion is used. During autograd, however, the relation expert is
    detached from the fused-primary objective. Relation semantics remain
    trainable through their dedicated relation CE/canonicalization losses.

    This is gradient-route isolation, not parameter isolation: both experts
    still share the inherited A13-LoRA/projection physical surface.
    """

    def __init__(self, *, epsilon: float = 1e-6):
        super().__init__()
        self.fusion = SymmetricFullKEvidenceFusion(epsilon=epsilon)

    @property
    def epsilon(self) -> float:
        return self.fusion.epsilon

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        triadic_logits: Tensor,
        relation_logits: Tensor,
    ) -> tuple[Tensor, EvidenceFusionDiagnostics]:
        return self.fusion(triadic_logits, relation_logits.detach())


__all__ = ["GradientIsolatedFullKEvidenceFusion"]
