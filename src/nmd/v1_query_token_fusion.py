from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .w33_coevidence_semantic import CoEvidenceSemanticScorer


class QuestionAsEvidenceScorer(CoEvidenceSemanticScorer):
    """S2-A0 zero-parameter baseline: append question tokens to state evidence."""

    @property
    def added_parameter_count(self) -> int:
        return 0

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
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("question_mask mismatch")
        if state_tokens.shape[0] != question_tokens.shape[0]:
            raise ValueError("batch mismatch")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("every question requires at least one content token")

        evidence = torch.cat([state_tokens, question_tokens], dim=1)
        evidence_mask = torch.cat([state_mask, question_mask], dim=1)
        return super().forward(
            state_tokens=evidence,
            state_mask=evidence_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )


class QueryTokenResidualFusionScorer(CoEvidenceSemanticScorer):
    """S2 learned query-token cross-attention residual fusion.

    All original state tokens are preserved. Frozen W34 state relation vectors
    query the compiled question tokens through a shared rank-16 attention path.
    The resulting question context is projected back into relation space and
    fused residually before exact W34 interaction/composition scoring.
    """

    fusion_rank = 16

    def __init__(self, d_model: int = 256, d_rel: int = 128, rank: int = 8):
        super().__init__(d_model=d_model, d_rel=d_rel, rank=rank)
        self.state_query = nn.Linear(self.d_rel, self.fusion_rank, bias=False)
        self.question_key = nn.Linear(self.d_rel, self.fusion_rank, bias=False)
        self.question_value = nn.Linear(self.d_rel, self.fusion_rank, bias=False)
        self.fusion_up = nn.Linear(self.fusion_rank, self.d_rel, bias=False)

        # Exact W34 functional starting point before optimization.
        nn.init.zeros_(self.fusion_up.weight)

    @property
    def fusion_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (
                self.state_query,
                self.question_key,
                self.question_value,
                self.fusion_up,
            )
            for p in module.parameters()
        )

    @property
    def fusion_trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (
                self.state_query,
                self.question_key,
                self.question_value,
                self.fusion_up,
            )
            for p in module.parameters()
            if p.requires_grad
        )

    @property
    def candidate_parameter_count(self) -> int:
        return super().candidate_parameter_count + self.fusion_parameter_count

    @property
    def candidate_trainable_parameter_count(self) -> int:
        return (
            super().candidate_trainable_parameter_count
            + self.fusion_trainable_parameter_count
        )

    def load_w34_base_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze_base: bool = True,
    ) -> None:
        CoEvidenceSemanticScorer.load_candidate_state_dict(
            self,
            state_dict,
            freeze=False,
        )
        if freeze_base:
            CoEvidenceSemanticScorer.freeze_candidate(self)

    def freeze_fusion(self) -> None:
        for module in (
            self.state_query,
            self.question_key,
            self.question_value,
            self.fusion_up,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def freeze_candidate(self) -> None:
        CoEvidenceSemanticScorer.freeze_candidate(self)
        self.freeze_fusion()

    def load_fusion_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected = {
            "state_query.weight": (self.fusion_rank, self.d_rel),
            "question_key.weight": (self.fusion_rank, self.d_rel),
            "question_value.weight": (self.fusion_rank, self.d_rel),
            "fusion_up.weight": (self.d_rel, self.fusion_rank),
        }
        if set(state_dict) != set(expected):
            raise ValueError("Hira v1 S2 fusion checkpoint keys changed")
        own = self.state_dict()
        with torch.no_grad():
            for key, shape in expected.items():
                value = state_dict[key]
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(f"Hira v1 S2 fusion shape mismatch: {key}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Hira v1 S2 fusion non-finite tensor: {key}")
                own[key].copy_(value.to(device=own[key].device, dtype=own[key].dtype))
        if freeze:
            self.freeze_fusion()

    def _fuse_state(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        state = self._project_state(state_tokens)
        question = F.normalize(self.projection(question_tokens), dim=-1)

        state_q = self.state_query(state)
        question_k = self.question_key(question)
        question_v = self.question_value(question)

        attention_logits = torch.einsum(
            "bsr,bqr->bsq",
            state_q,
            question_k,
        ) / sqrt(float(self.fusion_rank))
        attention_logits = attention_logits.masked_fill(
            ~question_mask[:, None, :],
            -1e4,
        )
        attention = torch.softmax(attention_logits, dim=-1)
        context = torch.einsum("bsq,bqr->bsr", attention, question_v)
        delta = self.fusion_up(context)

        fused = F.normalize(state + delta, dim=-1)
        fused = fused.masked_fill(~state_mask[..., None], 0.0)
        return fused, attention

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
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("question_mask mismatch")
        if option_view_tokens.ndim != 5 or option_view_tokens.shape[-1] != self.d_model:
            raise ValueError("option_view_tokens must be [B,K,V,T,D]")
        if option_view_token_mask.shape != option_view_tokens.shape[:4]:
            raise ValueError("option_view_token_mask mismatch")
        if option_view_mask.shape != option_view_tokens.shape[:3]:
            raise ValueError("option_view_mask mismatch")
        if option_view_token_mask.dtype != torch.bool or option_view_mask.dtype != torch.bool:
            raise ValueError("option masks must be bool")
        if state_tokens.shape[0] != question_tokens.shape[0] or state_tokens.shape[0] != option_view_tokens.shape[0]:
            raise ValueError("batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S2 fusion scorer requires >=2 options")
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

        state, _ = self._fuse_state(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
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
        option_counts = option_view_token_mask.sum(-1).clamp_min(1).to(option_to_state.dtype)
        option_mean = option_to_state.sum(-1) / option_counts

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

        composition = self._composition_similarity(options, state)
        coevidence = self._second_distinct_state_support(
            composition,
            state_mask=state_mask,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        view_scores = 0.5 * (option_mean + state_mean) + coevidence
        view_scores = view_scores.masked_fill(~option_view_mask, 0.0)
        view_counts = option_view_mask.sum(-1).clamp_min(1).to(view_scores.dtype)
        logits = view_scores.sum(-1) / view_counts

        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Hira v1 S2 fusion scorer produced non-finite logits")
        return logits


__all__ = [
    "QuestionAsEvidenceScorer",
    "QueryTokenResidualFusionScorer",
]
