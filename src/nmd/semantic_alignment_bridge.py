from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .competitive import (
    MIN_COVERAGE_WEIGHT,
    SALIENCE_THRESHOLD,
    CompetitiveCoarseScorer,
    candidate_relative_idf,
)


DEFAULT_D_REL = 128
DEFAULT_RANK = 16
SHARED_BRIDGE_PARAMETER_COUNT = 4096
ASYMMETRIC_BRIDGE_PARAMETER_COUNT = 8192


class LowRankSemanticBridge(nn.Module):
    """Identity-initialized residual map in the competitive relation space."""

    def __init__(
        self,
        d_rel: int = DEFAULT_D_REL,
        rank: int = DEFAULT_RANK,
    ):
        super().__init__()
        self.d_rel = int(d_rel)
        self.rank = int(rank)
        if self.d_rel < 1 or self.rank < 1:
            raise ValueError("semantic bridge dimensions must be positive")
        self.down = nn.Linear(self.d_rel, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.d_rel, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, projected: Tensor) -> Tensor:
        if projected.shape[-1] != self.d_rel:
            raise ValueError("semantic bridge relation dimension mismatch")
        residual = self.up(F.gelu(self.down(projected)))
        return F.normalize(projected + residual, dim=-1)


@dataclass(frozen=True)
class BridgeParameterCounts:
    base: int
    bridge: int
    total: int


class SemanticAlignmentBridgeScorer(nn.Module):
    """Competitive scorer with optional shared or asymmetric semantic bridges.

    The frozen production projection remains the base semantic projection.
    W9 bridge paths learn only low-rank corrections after that projection.

    mode="shared":
        the same correction is used for state/question and option tokens.
    mode="asymmetric":
        context and schema tokens receive independent corrections.

    Because every up projection is zero-initialized, both modes start as a
    numerical identity relative to the provided base scorer.
    """

    MODES = ("shared", "asymmetric")

    def __init__(
        self,
        base: CompetitiveCoarseScorer,
        *,
        mode: str,
        rank: int = DEFAULT_RANK,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown semantic bridge mode: {mode}")
        self.base = base
        self.mode = str(mode)
        self.d_model = int(base.d_model)
        self.d_rel = int(base.d_rel)
        self.rank = int(rank)

        if self.mode == "shared":
            self.shared_bridge = LowRankSemanticBridge(
                d_rel=self.d_rel,
                rank=self.rank,
            )
            self.context_bridge = None
            self.schema_bridge = None
        else:
            self.shared_bridge = None
            self.context_bridge = LowRankSemanticBridge(
                d_rel=self.d_rel,
                rank=self.rank,
            )
            self.schema_bridge = LowRankSemanticBridge(
                d_rel=self.d_rel,
                rank=self.rank,
            )

    def scale(self) -> Tensor:
        return self.base.scale()

    def _base_project(self, x: Tensor) -> Tensor:
        return F.normalize(self.base.projection(x), dim=-1)

    def project_context(self, x: Tensor) -> Tensor:
        projected = self._base_project(x)
        if self.mode == "shared":
            assert self.shared_bridge is not None
            return self.shared_bridge(projected)
        assert self.context_bridge is not None
        return self.context_bridge(projected)

    def project_schema(self, x: Tensor) -> Tensor:
        projected = self._base_project(x)
        if self.mode == "shared":
            assert self.shared_bridge is not None
            return self.shared_bridge(projected)
        assert self.schema_bridge is not None
        return self.schema_bridge(projected)

    def bridge_parameters(self) -> list[nn.Parameter]:
        if self.mode == "shared":
            assert self.shared_bridge is not None
            return list(self.shared_bridge.parameters())
        assert self.context_bridge is not None
        assert self.schema_bridge is not None
        return [
            *self.context_bridge.parameters(),
            *self.schema_bridge.parameters(),
        ]

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.bridge_parameters():
            parameter.requires_grad_(True)

    def parameter_counts(self) -> BridgeParameterCounts:
        base = sum(p.numel() for p in self.base.parameters())
        bridge = sum(p.numel() for p in self.bridge_parameters())
        return BridgeParameterCounts(
            base=base,
            bridge=bridge,
            total=base + bridge,
        )

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_tokens: Tensor,
        option_token_ids: Tensor,
        option_mask: Tensor,
    ) -> Tensor:
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.d_model:
            raise ValueError("state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3 or question_tokens.shape[-1] != self.d_model:
            raise ValueError("question_tokens must be [B,Q,D]")
        if option_tokens.ndim != 4 or option_tokens.shape[-1] != self.d_model:
            raise ValueError("option_tokens must be [B,K,T,D]")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("state_mask mismatch")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("question_mask mismatch")
        if option_mask.shape != option_tokens.shape[:3] or option_mask.dtype != torch.bool:
            raise ValueError("option_mask mismatch")
        if option_token_ids.shape != option_mask.shape:
            raise ValueError("option_token_ids mismatch")
        if option_token_ids.dtype != torch.long:
            raise ValueError("option_token_ids must be torch.long")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_tokens.shape[0]
        ):
            raise ValueError("semantic bridge batch mismatch")
        if option_tokens.shape[1] < 2:
            raise ValueError("semantic bridge scorer requires at least two options")
        if (state_mask.sum(-1) < 1).any():
            raise ValueError("state requires at least one content token")
        if (question_mask.sum(-1) < 1).any():
            raise ValueError("question requires at least one content token")
        if (option_mask.sum(-1) < 1).any():
            raise ValueError("every option requires content tokens")

        context = torch.cat([state_tokens, question_tokens], dim=1)
        context_mask = torch.cat([state_mask, question_mask], dim=1)
        projected_context = self.project_context(context)
        projected_options = self.project_schema(option_tokens)

        similarity = torch.einsum(
            "bktd,bcd->bktc",
            projected_options,
            projected_context,
        )
        similarity = similarity.masked_fill(
            ~context_mask[:, None, None, :],
            -1e4,
        )

        salience = candidate_relative_idf(option_token_ids, option_mask)
        salience = salience.to(
            device=similarity.device,
            dtype=similarity.dtype,
        )
        weights = salience * option_mask.to(salience.dtype)
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-8)

        common = (
            similarity * weights[..., None]
        ).sum(dim=2) / denom
        adjusted = similarity - common[:, :, None, :]
        adjusted = adjusted.masked_fill(
            ~context_mask[:, None, None, :],
            -1e4,
        )
        coverage = adjusted.max(dim=-1).values

        salient_mask = option_mask & (salience >= SALIENCE_THRESHOLD)
        empty = salient_mask.sum(-1) == 0
        if empty.any():
            salient_mask = salient_mask.clone()
            salient_mask[empty] = option_mask[empty]

        weighted_mean = (
            (coverage * weights).sum(-1)
            / weights.sum(-1).clamp_min(1e-8)
        )
        min_coverage = coverage.masked_fill(
            ~salient_mask,
            1e4,
        ).min(dim=-1).values
        raw = weighted_mean + MIN_COVERAGE_WEIGHT * min_coverage
        logits = raw * self.scale().to(raw.device, raw.dtype)
        if not torch.isfinite(logits).all():
            raise ValueError("semantic bridge scorer produced non-finite logits")
        return logits


def semantic_alignment_scores(
    scorer: CompetitiveCoarseScorer | SemanticAlignmentBridgeScorer,
    *,
    state_tokens: Tensor,
    state_mask: Tensor,
    definition_tokens: Tensor,
    definition_mask: Tensor,
) -> Tensor:
    """Candidate-independent symmetric token alignment scores.

    Shapes:
        state_tokens: [B,S,D]
        state_mask: [B,S]
        definition_tokens: [B,K,T,D]
        definition_mask: [B,K,T]

    The score intentionally excludes candidate-relative IDF and sibling
    common-mode subtraction so W9 can train semantic geometry directly.
    """
    if state_tokens.ndim != 3:
        raise ValueError("state_tokens must be [B,S,D]")
    if definition_tokens.ndim != 4:
        raise ValueError("definition_tokens must be [B,K,T,D]")
    if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
        raise ValueError("state_mask mismatch")
    if definition_mask.shape != definition_tokens.shape[:3] or definition_mask.dtype != torch.bool:
        raise ValueError("definition_mask mismatch")
    if state_tokens.shape[0] != definition_tokens.shape[0]:
        raise ValueError("alignment batch mismatch")
    if (state_mask.sum(-1) < 1).any():
        raise ValueError("alignment requires state content tokens")
    if (definition_mask.sum(-1) < 1).any():
        raise ValueError("alignment requires definition content tokens")

    if isinstance(scorer, SemanticAlignmentBridgeScorer):
        projected_state = scorer.project_context(state_tokens)
        projected_definition = scorer.project_schema(definition_tokens)
    elif isinstance(scorer, CompetitiveCoarseScorer):
        projected_state = scorer._project(state_tokens)
        projected_definition = scorer._project(definition_tokens)
    else:
        raise TypeError("unsupported W9 scorer type")

    similarity = torch.einsum(
        "bktd,bsd->bkts",
        projected_definition,
        projected_state,
    )

    schema_to_state = similarity.masked_fill(
        ~state_mask[:, None, None, :],
        -1e4,
    ).max(dim=-1).values
    schema_to_state = (
        schema_to_state.masked_fill(~definition_mask, 0.0).sum(dim=-1)
        / definition_mask.sum(dim=-1).clamp_min(1)
    )

    state_to_schema = similarity.masked_fill(
        ~definition_mask[:, :, :, None],
        -1e4,
    ).max(dim=2).values
    state_weights = state_mask[:, None, :].to(state_to_schema.dtype)
    state_to_schema = (
        (state_to_schema * state_weights).sum(dim=-1)
        / state_weights.sum(dim=-1).clamp_min(1.0)
    )

    scores = 0.5 * (schema_to_state + state_to_schema)
    if not torch.isfinite(scores).all():
        raise ValueError("semantic alignment produced non-finite scores")
    return scores


def expected_bridge_parameter_count(mode: str, d_rel: int = 128, rank: int = 16) -> int:
    one = 2 * int(d_rel) * int(rank)
    if mode == "shared":
        return one
    if mode == "asymmetric":
        return 2 * one
    raise ValueError(f"unknown semantic bridge mode: {mode}")
