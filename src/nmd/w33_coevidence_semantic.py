from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn

from .w32_interaction_semantic_adapter import InteractionSemanticScorer


class CoEvidenceSemanticScorer(InteractionSemanticScorer):
    """W33 shared low-rank co-evidence compositional semantic scorer.

    W33 preserves W32's from-scratch residual + interaction geometry and adds
    a second shared low-rank composition space. For every option semantic view,
    the composition term uses the second strongest distinct state-token match.
    This makes the added signal require support from more than one state token,
    while remaining factor/domain/primitive agnostic.
    """

    def __init__(self, d_model: int = 256, d_rel: int = 128, rank: int = 8):
        super().__init__(d_model=d_model, d_rel=d_rel, rank=rank)
        self.composition_state = nn.Linear(self.d_rel, self.rank, bias=False)
        self.composition_schema = nn.Linear(self.d_rel, self.rank, bias=False)
        # Exact identity boundary: schema composition keeps its seeded basis,
        # while a zero state map makes the complete co-evidence term zero.
        nn.init.zeros_(self.composition_state.weight)

    @property
    def composition_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (self.composition_state, self.composition_schema)
            for p in module.parameters()
        )

    @property
    def candidate_parameter_count(self) -> int:
        return super().candidate_parameter_count + self.composition_parameter_count

    @property
    def candidate_trainable_parameter_count(self) -> int:
        modules = (
            self.state_adapter,
            self.schema_adapter,
            self.interaction_state,
            self.interaction_schema,
            self.composition_state,
            self.composition_schema,
        )
        return sum(
            p.numel()
            for module in modules
            for p in module.parameters()
            if p.requires_grad
        )

    def freeze_candidate(self) -> None:
        super().freeze_candidate()
        for module in (self.composition_state, self.composition_schema):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def load_candidate_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected_shapes = {
            "state_adapter.down.weight": (self.rank, self.d_rel),
            "state_adapter.up.weight": (self.d_rel, self.rank),
            "schema_adapter.down.weight": (self.rank, self.d_rel),
            "schema_adapter.up.weight": (self.d_rel, self.rank),
            "interaction_state.weight": (self.rank, self.d_rel),
            "interaction_schema.weight": (self.rank, self.d_rel),
            "composition_state.weight": (self.rank, self.d_rel),
            "composition_schema.weight": (self.rank, self.d_rel),
        }
        if set(state_dict) != set(expected_shapes):
            raise ValueError("W33 candidate checkpoint keys changed")

        own = self.state_dict()
        with torch.no_grad():
            for key, shape in expected_shapes.items():
                value = state_dict[key]
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(f"W33 candidate shape mismatch: {key}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"W33 candidate contains non-finite values: {key}")
                own[key].copy_(
                    value.to(device=own[key].device, dtype=own[key].dtype)
                )
        if freeze:
            self.freeze_candidate()

    def _composition_similarity(
        self,
        options: Tensor,
        state: Tensor,
    ) -> Tensor:
        state_low = self.composition_state(state)
        option_low = self.composition_schema(options)
        return torch.einsum(
            "bkvtr,bsr->bkvts",
            option_low,
            state_low,
        ) / sqrt(float(self.rank))

    @staticmethod
    def _second_distinct_state_support(
        composition: Tensor,
        *,
        state_mask: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> Tensor:
        """Return second-strongest valid state-token support per option view."""
        # Max over valid schema tokens -> support for each distinct state token.
        state_support = composition.masked_fill(
            ~option_view_token_mask[..., None],
            -1e4,
        ).max(dim=-2).values
        state_support = state_support.masked_fill(
            ~state_mask[:, None, None, :],
            -1e4,
        )

        valid_state_count = state_mask.sum(-1)
        k = min(2, composition.shape[-1])
        top = state_support.topk(k=k, dim=-1).values
        if k < 2:
            second = torch.zeros_like(top[..., 0])
        else:
            second = top[..., 1]
            enough = valid_state_count[:, None, None] >= 2
            second = torch.where(enough, second, torch.zeros_like(second))

        second = second.masked_fill(~option_view_mask, 0.0)
        return second

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
            raise ValueError("co-evidence semantic scorer requires >=2 options")
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
        similarity = similarity + self._interaction_similarity(options, state)

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

        composition = self._composition_similarity(options, state)
        coevidence = self._second_distinct_state_support(
            composition,
            state_mask=state_mask,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        view_scores = 0.5 * (option_mean + state_mean) + coevidence
        view_scores = view_scores.masked_fill(~option_view_mask, 0.0)
        view_counts = option_view_mask.sum(-1).clamp_min(1).to(
            view_scores.dtype
        )
        logits = view_scores.sum(-1) / view_counts

        if not bool(torch.isfinite(logits).all()):
            raise ValueError("W33 co-evidence scorer produced non-finite logits")
        return logits


__all__ = ["CoEvidenceSemanticScorer"]
