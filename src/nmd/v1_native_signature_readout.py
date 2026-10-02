from __future__ import annotations

import torch
from torch import Tensor, nn

from .v1_native_relation_geometry import NativeA13RelationCanonicalizer
from .v1_relation_canonicalization import CanonicalRelationDiagnostics


class NativeSignatureLinearCorrectnessReadout(nn.Module):
    """S36 native relation logits plus one shared zero-init correctness vector."""

    def __init__(
        self,
        *,
        role_temperature: float = 0.10,
        pair_temperature: float = 0.10,
        contrastive_temperature: float = 0.10,
        native_dimension: int = 256,
        residual_scale: float = 1.0,
        train_readout: bool = True,
    ):
        super().__init__()
        if native_dimension != 256:
            raise ValueError("S36 native dimension is frozen at 256")
        if residual_scale != 1.0:
            raise ValueError("S36 residual scale is frozen at 1.0")
        self.native_dimension = int(native_dimension)
        self.residual_scale = float(residual_scale)
        self.base = NativeA13RelationCanonicalizer(
            role_temperature=role_temperature,
            pair_temperature=pair_temperature,
            contrastive_temperature=contrastive_temperature,
            native_dimension=native_dimension,
        )
        self.readout_weight = nn.Parameter(
            torch.zeros(native_dimension, dtype=torch.float32),
            requires_grad=bool(train_readout),
        )

    @property
    def added_parameter_count(self) -> int:
        return int(self.readout_weight.numel())

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def readout_state_dict(self) -> dict[str, Tensor]:
        return {
            "readout.weight": self.readout_weight.detach().cpu().clone(),
        }

    def load_readout_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        if set(state_dict) != {"readout.weight"}:
            raise ValueError("S36 readout checkpoint keys changed")
        value = state_dict["readout.weight"]
        if tuple(value.shape) != (self.native_dimension,):
            raise ValueError("S36 readout shape changed")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("S36 readout tensor non-finite")
        with torch.no_grad():
            self.readout_weight.copy_(
                value.to(
                    device=self.readout_weight.device,
                    dtype=self.readout_weight.dtype,
                )
            )
        self.readout_weight.requires_grad_(not freeze)

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
        base_logits, signatures, diagnostics = self.base(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )
        if signatures.shape[-1] != self.native_dimension:
            raise RuntimeError("S36 native signature width changed")
        residual = torch.einsum(
            "bkd,d->bk",
            signatures,
            self.readout_weight.to(dtype=signatures.dtype),
        )
        logits = base_logits + self.residual_scale * residual
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S36 readout produced non-finite logits")
        return logits, signatures, diagnostics


__all__ = ["NativeSignatureLinearCorrectnessReadout"]
