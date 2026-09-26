from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SymmetricSemanticScorer(nn.Module):
    """Production port of the W28 rescued symmetric token semantic operator.

    The scorer intentionally has exactly one trainable component: a shared
    bias-free projection. W29 loads an already frozen W28 projection and does
    not train this module.
    """

    def __init__(self, d_model: int = 256, d_rel: int = 128):
        super().__init__()
        self.d_model = int(d_model)
        self.d_rel = int(d_rel)
        self.projection = nn.Linear(self.d_model, self.d_rel, bias=False)

    def load_projection_weight(self, weight: Tensor, *, freeze: bool = True) -> None:
        if not isinstance(weight, Tensor) or tuple(weight.shape) != (
            self.d_rel,
            self.d_model,
        ):
            raise ValueError(
                f"projection weight must be [{self.d_rel},{self.d_model}]"
            )
        if not torch.isfinite(weight).all():
            raise ValueError("projection weight must be finite")
        with torch.no_grad():
            self.projection.weight.copy_(
                weight.to(
                    device=self.projection.weight.device,
                    dtype=self.projection.weight.dtype,
                )
            )
        self.projection.weight.requires_grad_(not freeze)

    def freeze_projection(self) -> None:
        self.projection.weight.requires_grad_(False)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _project(self, x: Tensor) -> Tensor:
        return F.normalize(self.projection(x), dim=-1)

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> Tensor:
        """Return full-K semantic logits.

        Shapes:
            state_tokens: [B,S,D]
            state_mask: [B,S]
            option_view_tokens: [B,K,V,T,D]
            option_view_token_mask: [B,K,V,T]
            option_view_mask: [B,K,V]

        One valid view exactly matches the W28 q0 symmetric operator.
        Multiple views are scored independently and averaged arithmetically.
        """

        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.d_model:
            raise ValueError("state_tokens must be [B,S,D]")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("state_mask mismatch")
        if option_view_tokens.ndim != 5 or option_view_tokens.shape[-1] != self.d_model:
            raise ValueError("option_view_tokens must be [B,K,V,T,D]")
        if (
            option_view_token_mask.shape != option_view_tokens.shape[:4]
            or option_view_token_mask.dtype != torch.bool
        ):
            raise ValueError("option_view_token_mask mismatch")
        if (
            option_view_mask.shape != option_view_tokens.shape[:3]
            or option_view_mask.dtype != torch.bool
        ):
            raise ValueError("option_view_mask mismatch")
        if state_tokens.shape[0] != option_view_tokens.shape[0]:
            raise ValueError("batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("symmetric semantic scorer requires >=2 options")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("every state requires at least one content token")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("every option requires at least one semantic view")

        active_view_tokens = option_view_token_mask.sum(-1)
        if bool(((active_view_tokens < 1) & option_view_mask).any()):
            raise ValueError("every active semantic view requires content tokens")
        if bool((option_view_token_mask & ~option_view_mask[..., None]).any()):
            raise ValueError("inactive views cannot expose token positions")

        state = self._project(state_tokens)
        options = self._project(option_view_tokens)

        # [B,K,V,T,S]
        similarity = torch.einsum(
            "bkvtd,bsd->bkvts",
            options,
            state,
        )

        # W28 option->state direction:
        # for each option token take max over valid state tokens, then mean
        # over valid option tokens.
        option_to_state = similarity.masked_fill(
            ~state_mask[:, None, None, None, :],
            -1e4,
        ).max(dim=-1).values
        option_to_state = option_to_state.masked_fill(
            ~option_view_token_mask,
            0.0,
        )
        option_counts = option_view_token_mask.sum(-1).clamp_min(1).to(
            option_to_state.dtype
        )
        option_mean = option_to_state.sum(-1) / option_counts

        # W28 state->option direction:
        # for each state token take max over valid tokens of this one view,
        # then mean over valid state tokens.
        state_to_option = similarity.masked_fill(
            ~option_view_token_mask[..., None],
            -1e4,
        ).max(dim=-2).values
        state_to_option = state_to_option.masked_fill(
            ~state_mask[:, None, None, :],
            0.0,
        )
        state_counts = state_mask.sum(-1).clamp_min(1).to(state_to_option.dtype)
        state_mean = state_to_option.sum(-1) / state_counts[:, None, None]

        view_scores = 0.5 * (option_mean + state_mean)
        view_scores = view_scores.masked_fill(~option_view_mask, 0.0)
        view_counts = option_view_mask.sum(-1).clamp_min(1).to(view_scores.dtype)
        logits = view_scores.sum(-1) / view_counts

        if not torch.isfinite(logits).all():
            raise ValueError("symmetric semantic scorer produced non-finite logits")
        return logits


def count_symmetric_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "SymmetricSemanticScorer",
    "count_symmetric_parameters",
]
