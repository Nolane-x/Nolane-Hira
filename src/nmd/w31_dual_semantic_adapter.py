from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .semantic_transfer_bridge import LowRankSemanticTransferBridge
from .symmetric_semantic import SymmetricSemanticScorer


class DualAdapterSymmetricSemanticScorer(SymmetricSemanticScorer):
    """W31 asymmetric state/schema residual geometry.

    The rescued T0 projection is shared and remains frozen. State tokens and
    schema-option tokens then pass through independent identity-initialized
    low-rank residual adapters before the exact W28 symmetric operator.
    """

    def __init__(self, d_model: int = 256, d_rel: int = 128, rank: int = 8):
        super().__init__(d_model=d_model, d_rel=d_rel)
        self.rank = int(rank)
        self.state_adapter = LowRankSemanticTransferBridge(
            d_rel=d_rel,
            rank=rank,
        )
        self.schema_adapter = LowRankSemanticTransferBridge(
            d_rel=d_rel,
            rank=rank,
        )

    @property
    def adapter_parameter_count(self) -> int:
        return (
            self.state_adapter.parameter_count
            + self.schema_adapter.parameter_count
        )

    @property
    def adapter_trainable_parameter_count(self) -> int:
        return (
            self.state_adapter.trainable_parameter_count
            + self.schema_adapter.trainable_parameter_count
        )

    def freeze_adapters(self) -> None:
        self.state_adapter.freeze()
        self.schema_adapter.freeze()

    def load_adapter_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected = {
            "state_adapter.down.weight",
            "state_adapter.up.weight",
            "schema_adapter.down.weight",
            "schema_adapter.up.weight",
        }
        if set(state_dict) != expected:
            raise ValueError("W31 adapter checkpoint keys changed")

        expected_shapes = {
            "state_adapter.down.weight": (self.rank, self.d_rel),
            "state_adapter.up.weight": (self.d_rel, self.rank),
            "schema_adapter.down.weight": (self.rank, self.d_rel),
            "schema_adapter.up.weight": (self.d_rel, self.rank),
        }
        for key, shape in expected_shapes.items():
            value = state_dict[key]
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"W31 adapter shape mismatch: {key}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"W31 adapter contains non-finite values: {key}")

        own = self.state_dict()
        with torch.no_grad():
            for key in expected:
                own[key].copy_(
                    state_dict[key].to(
                        device=own[key].device,
                        dtype=own[key].dtype,
                    )
                )
        if freeze:
            self.freeze_adapters()

    def _project_state(self, x: Tensor) -> Tensor:
        z = self.projection(x)
        z = self.state_adapter(z)
        return F.normalize(z, dim=-1)

    def _project_schema(self, x: Tensor) -> Tensor:
        z = self.projection(x)
        z = self.schema_adapter(z)
        return F.normalize(z, dim=-1)

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> Tensor:
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
            raise ValueError("dual semantic scorer requires >=2 options")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("every state requires at least one content token")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("every option requires at least one semantic view")

        active_view_tokens = option_view_token_mask.sum(-1)
        if bool(((active_view_tokens < 1) & option_view_mask).any()):
            raise ValueError("every active semantic view requires content tokens")
        if bool((option_view_token_mask & ~option_view_mask[..., None]).any()):
            raise ValueError("inactive views cannot expose token positions")

        state = self._project_state(state_tokens)
        options = self._project_schema(option_view_tokens)

        similarity = torch.einsum(
            "bkvtd,bsd->bkvts",
            options,
            state,
        )

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

        state_to_option = similarity.masked_fill(
            ~option_view_token_mask[..., None],
            -1e4,
        ).max(dim=-2).values
        state_to_option = state_to_option.masked_fill(
            ~state_mask[:, None, None, :],
            0.0,
        )
        state_counts = state_mask.sum(-1).clamp_min(1).to(
            state_to_option.dtype
        )
        state_mean = state_to_option.sum(-1) / state_counts[:, None, None]

        view_scores = 0.5 * (option_mean + state_mean)
        view_scores = view_scores.masked_fill(~option_view_mask, 0.0)
        view_counts = option_view_mask.sum(-1).clamp_min(1).to(
            view_scores.dtype
        )
        logits = view_scores.sum(-1) / view_counts

        if not bool(torch.isfinite(logits).all()):
            raise ValueError("dual semantic scorer produced non-finite logits")
        return logits


__all__ = ["DualAdapterSymmetricSemanticScorer"]
