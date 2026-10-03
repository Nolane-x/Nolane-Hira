from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_native_relation_geometry import NativeA13RelationCanonicalizer
from .v1_relation_canonicalization import CanonicalRelationDiagnostics


class NativeGradientIsolatedBilinearCorrectnessReadout(nn.Module):
    """S39 full bilinear correction with a hard detached correction path."""

    def __init__(
        self,
        *,
        role_temperature: float = 0.10,
        pair_temperature: float = 0.10,
        contrastive_temperature: float = 0.10,
        native_dimension: int = 256,
        query_norm_epsilon: float = 1e-12,
        residual_scale: float = 1.0,
        train_readout: bool = True,
    ):
        super().__init__()
        if native_dimension != 256:
            raise ValueError("S39 native dimension is frozen at 256")
        if query_norm_epsilon != 1e-12:
            raise ValueError("S39 query normalization epsilon is frozen at 1e-12")
        if residual_scale != 1.0:
            raise ValueError("S39 residual scale is frozen at 1.0")
        self.native_dimension = int(native_dimension)
        self.query_norm_epsilon = float(query_norm_epsilon)
        self.residual_scale = float(residual_scale)
        self.base = NativeA13RelationCanonicalizer(
            role_temperature=role_temperature,
            pair_temperature=pair_temperature,
            contrastive_temperature=contrastive_temperature,
            native_dimension=native_dimension,
        )
        self.bilinear_weight = nn.Parameter(
            torch.zeros(native_dimension, native_dimension, dtype=torch.float32),
            requires_grad=bool(train_readout),
        )

    @property
    def added_parameter_count(self) -> int:
        return int(self.bilinear_weight.numel())

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def readout_state_dict(self) -> dict[str, Tensor]:
        return {"bilinear.weight": self.bilinear_weight.detach().cpu().clone()}

    def load_readout_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        if set(state_dict) != {"bilinear.weight"}:
            raise ValueError("S39 bilinear checkpoint keys changed")
        value = state_dict["bilinear.weight"]
        if tuple(value.shape) != (self.native_dimension, self.native_dimension):
            raise ValueError("S39 bilinear weight shape changed")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("S39 bilinear tensor non-finite")
        with torch.no_grad():
            self.bilinear_weight.copy_(
                value.to(
                    device=self.bilinear_weight.device,
                    dtype=self.bilinear_weight.dtype,
                )
            )
        self.bilinear_weight.requires_grad_(not freeze)

    def query_summary(
        self,
        *,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        if question_tokens.ndim != 3:
            raise ValueError("S39 question_tokens must be [B,Q,D]")
        if question_tokens.shape[-1] != self.native_dimension:
            raise ValueError("S39 question native dimension changed")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("S39 question_mask mismatch")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S39 requires question content")

        mask = question_mask.to(question_tokens.dtype)
        count = mask.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (question_tokens * mask[..., None]).sum(dim=1) / count
        summary = F.normalize(mean, dim=-1, eps=self.query_norm_epsilon)
        if not bool(torch.isfinite(summary).all()):
            raise ValueError("S39 query summary produced non-finite values")
        return summary

    def detached_readout_residual(
        self,
        *,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        if signatures.ndim != 3 or signatures.shape[-1] != self.native_dimension:
            raise ValueError("S39 signatures must be [B,K,256]")
        detached_signatures = signatures.detach()
        detached_question = question_tokens.detach()
        query = self.query_summary(
            question_tokens=detached_question,
            question_mask=question_mask,
        ).detach()
        if query.shape[0] != detached_signatures.shape[0]:
            raise ValueError("S39 signature/query batch mismatch")
        weight = self.bilinear_weight.to(dtype=detached_signatures.dtype)
        residual = torch.einsum(
            "bkd,de,be->bk",
            detached_signatures,
            weight,
            query,
        )
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("S39 bilinear residual produced non-finite values")
        return residual

    def correction_logits(
        self,
        *,
        native_logits: Tensor,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        residual = self.detached_readout_residual(
            signatures=signatures,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        if native_logits.shape != residual.shape:
            raise ValueError("S39 native-logit/residual shape mismatch")
        logits = native_logits.detach() + self.residual_scale * residual
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S39 correction logits produced non-finite values")
        return logits

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> tuple[Tensor, Tensor, CanonicalRelationDiagnostics]:
        native_logits, signatures, diagnostics = self.base(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )
        residual = self.detached_readout_residual(
            signatures=signatures,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        logits = native_logits + self.residual_scale * residual
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S39 evaluation logits produced non-finite values")
        return logits, signatures, diagnostics


__all__ = ["NativeGradientIsolatedBilinearCorrectnessReadout"]
