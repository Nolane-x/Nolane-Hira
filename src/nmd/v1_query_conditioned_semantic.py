from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .w33_coevidence_semantic import CoEvidenceSemanticScorer


class QueryConditionedCoEvidenceScorer(CoEvidenceSemanticScorer):
    """Hira v1 query-conditioned co-evidence semantic scorer.

    Hira v0's W34 path compares state tokens with option-view tokens but does
    not consume the compiled question tokens. V1 preserves W34's compact
    state/schema/interactions/composition geometry and adds one shared
    question-binding residual path.

    The query branch is exact-identity initialized:
    - query_basis keeps a random basis;
    - query_state and query_schema initialize to zero;
    - therefore the multiplicative binding term is exactly 1.0 initially;
    - with identical inherited weights, initial logits equal W34 exactly.

    No dataset-, factor-, language-, or primitive-specific head is introduced.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_rel: int = 128,
        rank: int = 8,
        query_rank: int = 8,
    ):
        super().__init__(d_model=d_model, d_rel=d_rel, rank=rank)
        self.query_rank = int(query_rank)
        if self.query_rank < 1 or self.query_rank > self.d_rel:
            raise ValueError("query_rank must satisfy 1 <= query_rank <= d_rel")

        self.query_basis = nn.Linear(self.d_rel, self.query_rank, bias=False)
        self.query_state = nn.Linear(self.d_rel, self.query_rank, bias=False)
        self.query_schema = nn.Linear(self.d_rel, self.query_rank, bias=False)

        # Exact W34 identity boundary at initialization while retaining a
        # useful random query basis for the first optimizer step.
        nn.init.zeros_(self.query_state.weight)
        nn.init.zeros_(self.query_schema.weight)

    @property
    def query_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (
                self.query_basis,
                self.query_state,
                self.query_schema,
            )
            for p in module.parameters()
        )

    @property
    def query_trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (
                self.query_basis,
                self.query_state,
                self.query_schema,
            )
            for p in module.parameters()
            if p.requires_grad
        )

    @property
    def candidate_parameter_count(self) -> int:
        return super().candidate_parameter_count + self.query_parameter_count

    @property
    def candidate_trainable_parameter_count(self) -> int:
        return (
            super().candidate_trainable_parameter_count
            + self.query_trainable_parameter_count
        )

    def freeze_query_binding(self) -> None:
        for module in (
            self.query_basis,
            self.query_state,
            self.query_schema,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def freeze_candidate(self) -> None:
        super().freeze_candidate()
        self.freeze_query_binding()

    def load_query_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected_shapes = {
            "query_basis.weight": (self.query_rank, self.d_rel),
            "query_state.weight": (self.query_rank, self.d_rel),
            "query_schema.weight": (self.query_rank, self.d_rel),
        }
        if set(state_dict) != set(expected_shapes):
            raise ValueError("Hira v1 query checkpoint keys changed")

        own = self.state_dict()
        with torch.no_grad():
            for key, shape in expected_shapes.items():
                value = state_dict[key]
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(
                        f"Hira v1 query checkpoint shape mismatch: {key}"
                    )
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(
                        f"Hira v1 query checkpoint contains non-finite values: {key}"
                    )
                own[key].copy_(
                    value.to(device=own[key].device, dtype=own[key].dtype)
                )
        if freeze:
            self.freeze_query_binding()

    def load_candidate_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        base_keys = {
            "state_adapter.down.weight",
            "state_adapter.up.weight",
            "schema_adapter.down.weight",
            "schema_adapter.up.weight",
            "interaction_state.weight",
            "interaction_schema.weight",
            "composition_state.weight",
            "composition_schema.weight",
        }
        query_shapes = {
            "query_basis.weight": (self.query_rank, self.d_rel),
            "query_state.weight": (self.query_rank, self.d_rel),
            "query_schema.weight": (self.query_rank, self.d_rel),
        }
        if set(state_dict) != base_keys | set(query_shapes):
            raise ValueError("Hira v1 QCCE candidate checkpoint keys changed")

        super().load_candidate_state_dict(
            {key: state_dict[key] for key in base_keys},
            freeze=False,
        )
        own = self.state_dict()
        with torch.no_grad():
            for key, shape in query_shapes.items():
                value = state_dict[key]
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(
                        f"Hira v1 QCCE candidate shape mismatch: {key}"
                    )
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(
                        f"Hira v1 QCCE candidate contains non-finite values: {key}"
                    )
                own[key].copy_(
                    value.to(device=own[key].device, dtype=own[key].dtype)
                )
        if freeze:
            self.freeze_candidate()

    def load_w34_base_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze_base: bool = True,
    ) -> None:
        """Load W34 tensors while preserving the v1 query trainable surface."""
        CoEvidenceSemanticScorer.load_candidate_state_dict(
            self,
            state_dict,
            freeze=False,
        )
        if freeze_base:
            CoEvidenceSemanticScorer.freeze_candidate(self)

    def _project_question(self, question_tokens: Tensor) -> Tensor:
        return F.normalize(self.projection(question_tokens), dim=-1)

    def _question_context(
        self,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        question = self._project_question(question_tokens)
        low = self.query_basis(question)
        weights = question_mask.to(low.dtype)[..., None]
        pooled = (low * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1)

    def _query_binding_multiplier(
        self,
        *,
        state: Tensor,
        options: Tensor,
        question_context: Tensor,
    ) -> Tensor:
        state_low = self.query_state(state)
        option_low = self.query_schema(options)

        state_relevance = torch.einsum(
            "bsr,br->bs",
            state_low,
            question_context,
        )
        option_relevance = torch.einsum(
            "bkvtr,br->bkvt",
            option_low,
            question_context,
        )
        binding = (
            state_relevance[:, None, None, None, :]
            + option_relevance[..., None]
        ) / sqrt(float(self.query_rank))
        # Residual multiplicative gate. At zero query-state/schema maps this is
        # exactly one, preserving W34 logits bit-for-bit.
        return 1.0 + torch.tanh(binding)

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
            state_tokens.shape[0] != option_view_tokens.shape[0]
            or question_tokens.shape[0] != state_tokens.shape[0]
        ):
            raise ValueError("batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("query-conditioned scorer requires >=2 options")
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

        similarity = torch.einsum(
            "bkvtd,bsd->bkvts",
            options,
            state,
        )
        similarity = similarity + self._interaction_similarity(options, state)

        binding_multiplier = self._query_binding_multiplier(
            state=state,
            options=options,
            question_context=question_context,
        )
        similarity = similarity * binding_multiplier

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

        composition = (
            self._composition_similarity(options, state) * binding_multiplier
        )
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
                "Hira v1 query-conditioned scorer produced non-finite logits"
            )
        return logits


__all__ = ["QueryConditionedCoEvidenceScorer"]
