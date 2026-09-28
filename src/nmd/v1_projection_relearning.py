from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_triadic_semantic import ParameterFreeTriadicScorer


class TrainableProjectionTriadicScorer(ParameterFreeTriadicScorer):
    """S5: relearn the shared A13->relation projection directly.

    The triadic decision operator remains parameter-free. The only learned
    surface is the shared bias-free 256->128 projection inherited from
    SymmetricSemanticScorer.
    """

    @property
    def projection_parameter_count(self) -> int:
        return self.projection.weight.numel()

    @property
    def projection_trainable_parameter_count(self) -> int:
        return (
            self.projection.weight.numel()
            if self.projection.weight.requires_grad
            else 0
        )

    def load_projection_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        if set(state_dict) != {"projection.weight"}:
            raise ValueError("Hira v1 S5 projection checkpoint keys changed")
        weight = state_dict["projection.weight"]
        self.load_projection_weight(weight, freeze=freeze)

    def option_view_alignment_loss(
        self,
        *,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
        temperature: float = 0.10,
    ) -> Tensor:
        """Symmetric InfoNCE between two semantic views of each option.

        Shapes:
            option_view_tokens: [B,K,V,T,D]
            option_view_token_mask: [B,K,V,T]
            option_view_mask: [B,K,V]

        S5 authority requires at least two active views per option.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if option_view_tokens.ndim != 5:
            raise ValueError("option_view_tokens must be [B,K,V,T,D]")
        if option_view_tokens.shape[-1] != self.d_model:
            raise ValueError("option_view_tokens d_model mismatch")
        if option_view_token_mask.shape != option_view_tokens.shape[:4]:
            raise ValueError("option_view_token_mask mismatch")
        if option_view_token_mask.dtype != torch.bool:
            raise ValueError("option_view_token_mask must be bool")
        if option_view_mask.shape != option_view_tokens.shape[:3]:
            raise ValueError("option_view_mask mismatch")
        if option_view_mask.dtype != torch.bool:
            raise ValueError("option_view_mask must be bool")
        if option_view_tokens.shape[2] < 2:
            raise ValueError("S5 alignment requires at least two views")
        if bool((option_view_mask[..., :2].sum(-1) < 2).any()):
            raise ValueError("S5 alignment requires two active views per option")

        projected = self._project(option_view_tokens)
        weights = option_view_token_mask.to(projected.dtype)[..., None]
        pooled = (projected * weights).sum(-2) / weights.sum(-2).clamp_min(1)
        pooled = F.normalize(pooled, dim=-1)

        left = pooled[:, :, 0]
        right = pooled[:, :, 1]
        logits_lr = torch.einsum("bkd,bjd->bkj", left, right) / temperature
        logits_rl = torch.einsum("bkd,bjd->bkj", right, left) / temperature

        batch, k, _ = logits_lr.shape
        labels = torch.arange(k, device=logits_lr.device).expand(batch, k)
        loss_lr = F.cross_entropy(
            logits_lr.reshape(batch * k, k),
            labels.reshape(batch * k),
        )
        loss_rl = F.cross_entropy(
            logits_rl.reshape(batch * k, k),
            labels.reshape(batch * k),
        )
        loss = 0.5 * (loss_lr + loss_rl)
        if not bool(torch.isfinite(loss)):
            raise ValueError("S5 alignment loss became non-finite")
        return loss


__all__ = ["TrainableProjectionTriadicScorer"]
