from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .v1_evidence_fusion import standardize_full_k_evidence


@dataclass(frozen=True)
class ThreeExpertConsensusDiagnostics:
    primary_native_top1_agreement: Tensor
    primary_corrected_top1_agreement: Tensor
    native_corrected_top1_agreement: Tensor
    all_three_top1_agreement: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "primary_native_top1_agreement": float(
                self.primary_native_top1_agreement.detach().cpu()
            ),
            "primary_corrected_top1_agreement": float(
                self.primary_corrected_top1_agreement.detach().cpu()
            ),
            "native_corrected_top1_agreement": float(
                self.native_corrected_top1_agreement.detach().cpu()
            ),
            "all_three_top1_agreement": float(
                self.all_three_top1_agreement.detach().cpu()
            ),
        }


class RobustThreeExpertMedianFusion(nn.Module):
    """S46 zero-parameter robust consensus over three full-K experts.

    Each expert is independently centered/RMS-normalized with the existing S14
    primitive. The final full-K evidence is the coordinate-wise median across:
    primary triadic, native relation, and corrected private relation evidence.

    This module is a decision shell. It adds no learned parameter, scalar gate,
    temperature, routing network, or extra encoder path.
    """

    def __init__(self, *, epsilon: float = 1e-6):
        super().__init__()
        if epsilon != 1e-6:
            raise ValueError("S46 fusion epsilon is frozen at 1e-6")
        self.epsilon = float(epsilon)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        primary_logits: Tensor,
        native_relation_logits: Tensor,
        corrected_relation_logits: Tensor,
    ) -> tuple[Tensor, ThreeExpertConsensusDiagnostics]:
        if (
            primary_logits.ndim != 2
            or native_relation_logits.ndim != 2
            or corrected_relation_logits.ndim != 2
        ):
            raise ValueError("S46 experts must be [B,K]")
        if not (
            primary_logits.shape
            == native_relation_logits.shape
            == corrected_relation_logits.shape
        ):
            raise ValueError("S46 experts must share exact [B,K] shape")
        if primary_logits.shape[-1] < 2:
            raise ValueError("S46 requires K >= 2")

        primary, _ = standardize_full_k_evidence(
            primary_logits,
            epsilon=self.epsilon,
        )
        native, _ = standardize_full_k_evidence(
            native_relation_logits,
            epsilon=self.epsilon,
        )
        corrected, _ = standardize_full_k_evidence(
            corrected_relation_logits,
            epsilon=self.epsilon,
        )

        stacked = torch.stack((primary, native, corrected), dim=1)
        fused = torch.median(stacked, dim=1).values
        if not bool(torch.isfinite(fused).all()):
            raise ValueError("S46 fused evidence became non-finite")

        p_top = primary_logits.argmax(dim=-1)
        n_top = native_relation_logits.argmax(dim=-1)
        c_top = corrected_relation_logits.argmax(dim=-1)

        diagnostics = ThreeExpertConsensusDiagnostics(
            primary_native_top1_agreement=(p_top == n_top).to(fused.dtype).mean(),
            primary_corrected_top1_agreement=(p_top == c_top).to(fused.dtype).mean(),
            native_corrected_top1_agreement=(n_top == c_top).to(fused.dtype).mean(),
            all_three_top1_agreement=((p_top == n_top) & (n_top == c_top))
            .to(fused.dtype)
            .mean(),
        )
        return fused, diagnostics


__all__ = [
    "RobustThreeExpertMedianFusion",
    "ThreeExpertConsensusDiagnostics",
]
