from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .symmetric_semantic import SymmetricSemanticScorer


def _validate_inputs(
    scorer: SymmetricSemanticScorer,
    *,
    state_tokens: Tensor,
    state_mask: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
    option_view_tokens: Tensor,
    option_view_token_mask: Tensor,
    option_view_mask: Tensor,
) -> None:
    if state_tokens.ndim != 3 or state_tokens.shape[-1] != scorer.d_model:
        raise ValueError("state_tokens must be [B,S,D]")
    if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
        raise ValueError("state_mask mismatch")
    if question_tokens.ndim != 3 or question_tokens.shape[-1] != scorer.d_model:
        raise ValueError("question_tokens must be [B,Q,D]")
    if (
        question_mask.shape != question_tokens.shape[:2]
        or question_mask.dtype != torch.bool
    ):
        raise ValueError("question_mask mismatch")
    if (
        option_view_tokens.ndim != 5
        or option_view_tokens.shape[-1] != scorer.d_model
    ):
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
        raise ValueError("triadic scorer requires >=2 options")
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


def _aggregate_triadic(
    triad: Tensor,
    *,
    state_mask: Tensor,
    question_mask: Tensor,
    option_view_token_mask: Tensor,
    option_view_mask: Tensor,
) -> Tensor:
    """Aggregate [B,K,V,T,S,Q] evidence symmetrically over all three token axes."""
    # Per option token: strongest state/question support, then token mean.
    option_support = triad.masked_fill(
        ~state_mask[:, None, None, None, :, None],
        -1e4,
    )
    option_support = option_support.masked_fill(
        ~question_mask[:, None, None, None, None, :],
        -1e4,
    ).amax(dim=(-1, -2))
    option_support = option_support.masked_fill(~option_view_token_mask, 0.0)
    option_counts = option_view_token_mask.sum(-1).clamp_min(1).to(
        option_support.dtype
    )
    option_mean = option_support.sum(-1) / option_counts

    # Per state token: strongest option-token/question support, then state mean.
    state_support = triad.masked_fill(
        ~option_view_token_mask[..., None, None],
        -1e4,
    )
    state_support = state_support.masked_fill(
        ~question_mask[:, None, None, None, None, :],
        -1e4,
    ).amax(dim=(-1, -3))
    state_support = state_support.masked_fill(
        ~state_mask[:, None, None, :],
        0.0,
    )
    state_counts = state_mask.sum(-1).clamp_min(1).to(state_support.dtype)
    state_mean = state_support.sum(-1) / state_counts[:, None, None]

    # Per question token: strongest state/option-token support, then query mean.
    question_support = triad.masked_fill(
        ~option_view_token_mask[..., None, None],
        -1e4,
    )
    question_support = question_support.masked_fill(
        ~state_mask[:, None, None, None, :, None],
        -1e4,
    ).amax(dim=(-2, -3))
    question_support = question_support.masked_fill(
        ~question_mask[:, None, None, :],
        0.0,
    )
    question_counts = question_mask.sum(-1).clamp_min(1).to(
        question_support.dtype
    )
    question_mean = question_support.sum(-1) / question_counts[:, None, None]

    view_scores = (option_mean + state_mean + question_mean) / 3.0
    view_scores = view_scores.masked_fill(~option_view_mask, 0.0)
    view_counts = option_view_mask.sum(-1).clamp_min(1).to(view_scores.dtype)
    logits = view_scores.sum(-1) / view_counts
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("triadic scorer produced non-finite logits")
    return logits


class ParameterFreeTriadicScorer(SymmetricSemanticScorer):
    """S3-A0: direct frozen-W28 coordinate triadic evidence."""

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
        _validate_inputs(
            self,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        state = self._project(state_tokens)
        question = self._project(question_tokens)
        options = self._project(option_view_tokens)

        triad = torch.einsum(
            "bsd,bqd,bkvtd->bkvtsq",
            state,
            question,
            options,
        ) / sqrt(float(self.d_rel))

        return _aggregate_triadic(
            triad,
            state_mask=state_mask,
            question_mask=question_mask,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )


class TriadicCPSemanticScorer(SymmetricSemanticScorer):
    """S3-A learned low-rank CP state-question-option semantic scorer."""

    heads = 2
    factor_rank = 16
    factor_width = heads * factor_rank

    def __init__(self, d_model: int = 256, d_rel: int = 128):
        super().__init__(d_model=d_model, d_rel=d_rel)
        self.state_factor = nn.Linear(self.d_rel, self.factor_width, bias=False)
        self.question_factor = nn.Linear(self.d_rel, self.factor_width, bias=False)
        self.option_factor = nn.Linear(self.d_rel, self.factor_width, bias=False)

        # Shared, symmetric initialization. The training authority fixes the
        # global seed before construction.
        for module in (
            self.state_factor,
            self.question_factor,
            self.option_factor,
        ):
            nn.init.xavier_uniform_(module.weight)

    @property
    def factor_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (
                self.state_factor,
                self.question_factor,
                self.option_factor,
            )
            for p in module.parameters()
        )

    @property
    def factor_trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (
                self.state_factor,
                self.question_factor,
                self.option_factor,
            )
            for p in module.parameters()
            if p.requires_grad
        )

    def freeze_factors(self) -> None:
        for module in (
            self.state_factor,
            self.question_factor,
            self.option_factor,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def load_factor_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected = {
            "state_factor.weight": (self.factor_width, self.d_rel),
            "question_factor.weight": (self.factor_width, self.d_rel),
            "option_factor.weight": (self.factor_width, self.d_rel),
        }
        if set(state_dict) != set(expected):
            raise ValueError("Hira v1 S3 factor checkpoint keys changed")
        own = self.state_dict()
        with torch.no_grad():
            for key, shape in expected.items():
                value = state_dict[key]
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(f"Hira v1 S3 factor shape mismatch: {key}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Hira v1 S3 factor tensor non-finite: {key}")
                own[key].copy_(
                    value.to(device=own[key].device, dtype=own[key].dtype)
                )
        if freeze:
            self.freeze_factors()

    def _factorize(self, x: Tensor, module: nn.Linear) -> Tensor:
        y = module(x)
        return y.view(*y.shape[:-1], self.heads, self.factor_rank)

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
        _validate_inputs(
            self,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        state_rel = self._project(state_tokens)
        question_rel = self._project(question_tokens)
        option_rel = self._project(option_view_tokens)

        state = self._factorize(state_rel, self.state_factor)
        question = self._factorize(question_rel, self.question_factor)
        options = self._factorize(option_rel, self.option_factor)

        # Sum over both CP head and rank dimensions -> [B,K,V,T,S,Q].
        triad = torch.einsum(
            "bshr,bqhr,bkvthr->bkvtsq",
            state,
            question,
            options,
        ) / sqrt(float(self.heads * self.factor_rank))

        return _aggregate_triadic(
            triad,
            state_mask=state_mask,
            question_mask=question_mask,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )


__all__ = [
    "ParameterFreeTriadicScorer",
    "TriadicCPSemanticScorer",
]
