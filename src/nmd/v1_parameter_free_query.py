from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .w33_coevidence_semantic import CoEvidenceSemanticScorer


class ParameterFreeQueryCoEvidenceScorer(CoEvidenceSemanticScorer):
    """Zero-new-parameter query-conditioned W34 baseline.

    This baseline exists to separate the value of *using the question at all*
    from the value of learning an additional query-binding adapter.

    It reuses only the frozen W28/W34 parameters. Question content tokens are
    projected through the exact frozen W28 projection, pooled, and used as a
    bounded multiplicative relevance gate over state↔option evidence.
    """

    @property
    def added_parameter_count(self) -> int:
        return 0

    def _question_context(
        self,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        question = F.normalize(self.projection(question_tokens), dim=-1)
        weights = question_mask.to(question.dtype)[..., None]
        pooled = (question * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1)

    @staticmethod
    def _fixed_binding_multiplier(
        *,
        state: Tensor,
        options: Tensor,
        question_context: Tensor,
    ) -> Tensor:
        state_relevance = torch.einsum(
            "bsd,bd->bs",
            state,
            question_context,
        )
        option_relevance = torch.einsum(
            "bkvtd,bd->bkvt",
            options,
            question_context,
        )
        # Both relevance terms are cosine-like values in [-1,1]. Their sum is
        # therefore in [-2,2], and 0.25 keeps the residual multiplier in
        # [0.5,1.5] without trainable scale parameters.
        return 1.0 + 0.25 * (
            state_relevance[:, None, None, None, :]
            + option_relevance[..., None]
        )

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
    ) -> Tensor:
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.d_model:
            raise ValueError("state_tokens must be [B,S,D]")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("state_mask mismatch")
        if question_tokens.ndim != 3 or question_tokens.shape[-1] != self.d_model:
            raise ValueError("question_tokens must be [B,Q,D]")
        if (
            question_mask.shape != question_tokens.shape[:2]
            or question_mask.dtype != torch.bool
        ):
            raise ValueError("question_mask mismatch")
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
        if (
            state_tokens.shape[0] != question_tokens.shape[0]
            or state_tokens.shape[0] != option_view_tokens.shape[0]
        ):
            raise ValueError("batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("parameter-free query scorer requires >=2 options")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("every state requires at least one content token")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("every question requires at least one content token")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("every option requires at least one semantic view")

        active_view_tokens = option_view_token_mask.sum(-1)
        if bool(((active_view_tokens < 1) & option_view_mask).any()):
            raise ValueError("every active semantic view requires content tokens")
        if bool((option_view_token_mask & ~option_view_mask[..., None]).any()):
            raise ValueError("inactive views cannot expose token positions")

        state = self._project_state(state_tokens)
        options = self._project_schema(option_view_tokens)
        question_context = self._question_context(
            question_tokens,
            question_mask,
        )
        multiplier = self._fixed_binding_multiplier(
            state=state,
            options=options,
            question_context=question_context,
        )

        similarity = torch.einsum(
            "bkvtd,bsd->bkvts",
            options,
            state,
        )
        similarity = similarity + self._interaction_similarity(options, state)
        similarity = similarity * multiplier

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

        composition = self._composition_similarity(options, state) * multiplier
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
            raise ValueError(
                "parameter-free query scorer produced non-finite logits"
            )
        return logits


__all__ = ["ParameterFreeQueryCoEvidenceScorer"]
