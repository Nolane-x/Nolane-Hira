from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .competitive import (
    MIN_COVERAGE_WEIGHT,
    CompetitiveCoarseScorer,
    count_competitive_parameters,
)


CONJUNCTIVE_EXTRA_PARAMETER_COUNT = 3
CONJUNCTIVE_TOTAL_PARAMETER_COUNT = 32772
INITIAL_FACTOR_THRESHOLD = 0.0
INITIAL_FACTOR_TEMPERATURE = 1.0
INITIAL_RAW_ALPHA = -4.0
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 10.0
LOG_EPS = 1e-6


@dataclass(frozen=True)
class ConjunctiveEvidenceOutput:
    logits: Tensor
    base_logits: Tensor
    factor_evidence: Tensor
    factor_logits: Tensor
    factor_probabilities: Tensor
    conjunction_log_evidence: Tensor
    conjunction_residual: Tensor


class ConjunctiveEvidenceScorer(nn.Module):
    """Small explicit-AND extension of the production competitive scorer.

    The free-form CompetitiveCoarseScorer remains the base path. The W7
    branch reuses its exact 256->128 projection to score explicit schema
    factors against the already-cached state tokens, then applies a symmetric
    smooth conjunction. Only three new scalar parameters are introduced:
    one shared factor threshold, one shared temperature, and one shared
    non-negative residual gate.

    Factor order is intentionally irrelevant. No role-specific parameter is
    permitted in W7.
    """

    def __init__(
        self,
        base: CompetitiveCoarseScorer | None = None,
    ):
        super().__init__()
        self.base = base or CompetitiveCoarseScorer(
            d_model=256,
            d_rel=128,
        )
        self.factor_threshold = nn.Parameter(
            torch.tensor(float(INITIAL_FACTOR_THRESHOLD))
        )
        self.log_factor_temperature = nn.Parameter(
            torch.tensor(math.log(INITIAL_FACTOR_TEMPERATURE))
        )
        self.raw_conjunction_alpha = nn.Parameter(
            torch.tensor(float(INITIAL_RAW_ALPHA))
        )

    @property
    def d_model(self) -> int:
        return self.base.d_model

    @property
    def d_rel(self) -> int:
        return self.base.d_rel

    def factor_temperature(self) -> Tensor:
        return self.log_factor_temperature.clamp(
            math.log(MIN_TEMPERATURE),
            math.log(MAX_TEMPERATURE),
        ).exp()

    def conjunction_alpha(self) -> Tensor:
        return F.softplus(self.raw_conjunction_alpha)

    def factor_evidence(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        factor_tokens: Tensor,
        factor_token_mask: Tensor,
        factor_present_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one evidence scalar per candidate factor.

        Shapes:
            state_tokens: [B,S,D]
            state_mask: [B,S]
            factor_tokens: [B,K,F,T,D]
            factor_token_mask: [B,K,F,T]
            factor_present_mask: optional [B,K,F]

        The evidence operator is frozen pre-data as:
        mean(token MaxSim to state) + 0.5 * min(token MaxSim to state).

        There is no candidate-relative IDF and no sibling common-mode
        subtraction in the factor branch. Candidate-set coupling therefore
        remains confined to the existing free-form production path.
        """
        if (
            state_tokens.ndim != 3
            or state_tokens.shape[-1] != self.d_model
        ):
            raise ValueError("state_tokens must be [B,S,D]")
        if (
            state_mask.shape != state_tokens.shape[:2]
            or state_mask.dtype != torch.bool
        ):
            raise ValueError("state_mask mismatch")
        if (
            factor_tokens.ndim != 5
            or factor_tokens.shape[-1] != self.d_model
        ):
            raise ValueError("factor_tokens must be [B,K,F,T,D]")
        if (
            factor_token_mask.shape != factor_tokens.shape[:4]
            or factor_token_mask.dtype != torch.bool
        ):
            raise ValueError("factor_token_mask mismatch")
        if factor_tokens.shape[0] != state_tokens.shape[0]:
            raise ValueError("factor/state batch mismatch")
        if (state_mask.sum(-1) < 1).any():
            raise ValueError("factor evidence requires state content tokens")

        batch, candidates, factors, _, _ = factor_tokens.shape
        if factor_present_mask is None:
            factor_present_mask = factor_token_mask.any(dim=-1)
        if (
            factor_present_mask.shape
            != (batch, candidates, factors)
            or factor_present_mask.dtype != torch.bool
        ):
            raise ValueError("factor_present_mask mismatch")
        if (factor_present_mask.sum(-1) < 1).any():
            raise ValueError("every candidate requires at least one factor")
        invalid_present = (
            factor_present_mask
            & (factor_token_mask.sum(dim=-1) < 1)
        )
        if invalid_present.any():
            raise ValueError(
                "present factors require at least one content token"
            )

        projected_state = self.base._project(state_tokens)
        projected_factors = self.base._project(factor_tokens)
        similarity = torch.einsum(
            "bkftd,bsd->bkfts",
            projected_factors,
            projected_state,
        )
        similarity = similarity.masked_fill(
            ~state_mask[:, None, None, None, :],
            -1e4,
        )
        coverage = similarity.max(dim=-1).values

        valid = factor_token_mask.to(coverage.dtype)
        counts = valid.sum(dim=-1).clamp_min(1.0)
        mean_coverage = (coverage * valid).sum(dim=-1) / counts
        min_coverage = coverage.masked_fill(
            ~factor_token_mask,
            1e4,
        ).min(dim=-1).values
        evidence = (
            mean_coverage
            + MIN_COVERAGE_WEIGHT * min_coverage
        )
        evidence = evidence * self.base.scale().to(
            evidence.device,
            evidence.dtype,
        )
        evidence = torch.where(
            factor_present_mask,
            evidence,
            torch.zeros_like(evidence),
        )
        if not torch.isfinite(evidence).all():
            raise ValueError(
                "conjunctive factor evidence produced non-finite values"
            )
        return evidence

    def forward_with_factors(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_tokens: Tensor,
        option_token_ids: Tensor,
        option_mask: Tensor,
        factor_tokens: Tensor,
        factor_token_mask: Tensor,
        factor_present_mask: Tensor | None = None,
    ) -> ConjunctiveEvidenceOutput:
        base_logits = self.base(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_tokens=option_tokens,
            option_token_ids=option_token_ids,
            option_mask=option_mask,
        )
        evidence = self.factor_evidence(
            state_tokens=state_tokens,
            state_mask=state_mask,
            factor_tokens=factor_tokens,
            factor_token_mask=factor_token_mask,
            factor_present_mask=factor_present_mask,
        )

        if factor_present_mask is None:
            factor_present_mask = factor_token_mask.any(dim=-1)
        present = factor_present_mask.to(evidence.dtype)

        temperature = self.factor_temperature().to(
            evidence.device,
            evidence.dtype,
        )
        threshold = self.factor_threshold.to(
            evidence.device,
            evidence.dtype,
        )
        factor_logits = (evidence - threshold) / temperature
        probabilities = torch.sigmoid(factor_logits)
        log_match = torch.log(
            probabilities.clamp_min(LOG_EPS)
        )
        denom = present.sum(dim=-1).clamp_min(1.0)
        log_and = (log_match * present).sum(dim=-1) / denom

        alpha = self.conjunction_alpha().to(
            log_and.device,
            log_and.dtype,
        )
        residual = alpha * log_and
        logits = base_logits + residual

        if not torch.isfinite(logits).all():
            raise ValueError(
                "conjunctive coarse scorer produced non-finite logits"
            )
        return ConjunctiveEvidenceOutput(
            logits=logits,
            base_logits=base_logits,
            factor_evidence=evidence,
            factor_logits=factor_logits,
            factor_probabilities=probabilities,
            conjunction_log_evidence=log_and,
            conjunction_residual=residual,
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
        """Backward-safe free-form path when factor artifacts are absent."""
        return self.base(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_tokens=option_tokens,
            option_token_ids=option_token_ids,
            option_mask=option_mask,
        )


def count_conjunctive_parameters(
    model: ConjunctiveEvidenceScorer,
) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def conjunctive_extra_parameter_count(
    model: ConjunctiveEvidenceScorer,
) -> int:
    return (
        count_conjunctive_parameters(model)
        - count_competitive_parameters(model.base)
    )
