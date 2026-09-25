from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


MAX_SOURCE_FRACTION = 0.25
INITIAL_BETA_FRACTION = 0.025
MIN_ANCHOR_SPREAD = 1e-3
_INITIAL_THETA = math.log(
    (INITIAL_BETA_FRACTION / MAX_SOURCE_FRACTION)
    / (1.0 - INITIAL_BETA_FRACTION / MAX_SOURCE_FRACTION)
)


@dataclass
class AnchorResidualOutput:
    logits: Tensor
    anchor_logits: Tensor
    competitive_residual: Tensor
    relation_residual: Tensor
    competitive_correction: Tensor
    relation_correction: Tensor
    anchor_spread: Tensor
    beta_competitive: Tensor
    beta_relation: Tensor
    bounded: bool


def _median_midpoint(values: Tensor) -> Tensor:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("median values must be a non-empty vector")
    ordered = values.sort().values
    n = int(ordered.numel())
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def robust_anchor_spread(
    anchor_logits: Tensor,
    option_mask: Tensor | None = None,
) -> Tensor:
    """Per-case midpoint-median absolute deviation of valid anchor logits."""
    if anchor_logits.ndim != 2:
        raise ValueError("anchor_logits must be [B,K]")
    if option_mask is None:
        option_mask = torch.ones_like(anchor_logits, dtype=torch.bool)
    if option_mask.shape != anchor_logits.shape or option_mask.dtype != torch.bool:
        raise ValueError("option_mask must be bool [B,K]")
    if (option_mask.sum(-1) < 2).any():
        raise ValueError("every W15 decision requires at least two valid options")

    spreads = []
    for row, mask in zip(anchor_logits, option_mask):
        valid = row[mask]
        median = _median_midpoint(valid)
        mad = _median_midpoint((valid - median).abs())
        spreads.append(mad.clamp_min(MIN_ANCHOR_SPREAD))
    return torch.stack(spreads, dim=0)


def center_valid(x: Tensor, option_mask: Tensor | None = None) -> Tensor:
    if x.ndim != 2:
        raise ValueError("residual tensor must be [B,K]")
    if option_mask is None:
        option_mask = torch.ones_like(x, dtype=torch.bool)
    if option_mask.shape != x.shape or option_mask.dtype != torch.bool:
        raise ValueError("option_mask must be bool [B,K]")
    valid = option_mask.to(x.dtype)
    count = valid.sum(-1, keepdim=True).clamp_min(1.0)
    mean = (x * valid).sum(-1, keepdim=True) / count
    return torch.where(option_mask, x - mean, torch.zeros_like(x))


class AnchorPreservingResidualMixer(nn.Module):
    """Six-parameter W15 anchor-primary typed residual mixer.

    Primitive IDs follow HIRACore:
    0 = choice
    1 = score
    2 = noul

    The bounded path enforces a structural correction cap per source:
    |correction_source| <= 0.25 * robust_anchor_spread.
    """

    def __init__(self, *, bounded: bool = True):
        super().__init__()
        self.bounded = bool(bounded)
        self.theta_competitive = nn.Parameter(
            torch.full((3,), float(_INITIAL_THETA))
        )
        self.theta_relation = nn.Parameter(
            torch.full((3,), float(_INITIAL_THETA))
        )

    def beta_competitive(self) -> Tensor:
        return MAX_SOURCE_FRACTION * torch.sigmoid(self.theta_competitive)

    def beta_relation(self) -> Tensor:
        return MAX_SOURCE_FRACTION * torch.sigmoid(self.theta_relation)

    def forward(
        self,
        *,
        anchor_logits: Tensor,
        d0_anchor_logits: Tensor,
        competitive_logits: Tensor,
        relation_delta: Tensor,
        qtype: Tensor,
        option_mask: Tensor | None = None,
    ) -> AnchorResidualOutput:
        for name, value in (
            ("anchor_logits", anchor_logits),
            ("d0_anchor_logits", d0_anchor_logits),
            ("competitive_logits", competitive_logits),
            ("relation_delta", relation_delta),
        ):
            if value.ndim != 2:
                raise ValueError(f"{name} must be [B,K]")
            if value.shape != anchor_logits.shape:
                raise ValueError(f"{name} shape mismatch")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")

        batch, _ = anchor_logits.shape
        if qtype.shape != (batch,) or qtype.dtype != torch.long:
            raise ValueError("qtype must be torch.long [B]")
        if qtype.numel() and (int(qtype.min()) < 0 or int(qtype.max()) > 2):
            raise ValueError("qtype must be 0(choice),1(score),2(noul)")

        if option_mask is None:
            option_mask = torch.ones_like(anchor_logits, dtype=torch.bool)
        if option_mask.shape != anchor_logits.shape or option_mask.dtype != torch.bool:
            raise ValueError("option_mask must be bool [B,K]")
        if (option_mask.sum(-1) < 2).any():
            raise ValueError("every W15 decision requires at least two valid options")

        competitive_residual = center_valid(
            competitive_logits - d0_anchor_logits,
            option_mask,
        )
        relation_residual = center_valid(relation_delta, option_mask)
        spread = robust_anchor_spread(anchor_logits, option_mask)

        beta_c_all = self.beta_competitive()
        beta_r_all = self.beta_relation()
        beta_c = beta_c_all[qtype]
        beta_r = beta_r_all[qtype]

        if self.bounded:
            denom = spread[:, None]
            competitive_correction = (
                denom
                * beta_c[:, None]
                * torch.tanh(competitive_residual / denom)
            )
            relation_correction = (
                denom
                * beta_r[:, None]
                * torch.tanh(relation_residual / denom)
            )
        else:
            competitive_correction = (
                beta_c[:, None] * competitive_residual
            )
            relation_correction = beta_r[:, None] * relation_residual

        competitive_correction = torch.where(
            option_mask,
            competitive_correction,
            torch.zeros_like(competitive_correction),
        )
        relation_correction = torch.where(
            option_mask,
            relation_correction,
            torch.zeros_like(relation_correction),
        )

        logits = (
            anchor_logits
            + competitive_correction
            + relation_correction
        )
        logits = torch.where(
            option_mask,
            logits,
            torch.full_like(logits, -1e4),
        )
        if not torch.isfinite(logits).all():
            raise ValueError("W15 residual mixer produced non-finite logits")

        return AnchorResidualOutput(
            logits=logits,
            anchor_logits=anchor_logits,
            competitive_residual=competitive_residual,
            relation_residual=relation_residual,
            competitive_correction=competitive_correction,
            relation_correction=relation_correction,
            anchor_spread=spread,
            beta_competitive=beta_c,
            beta_relation=beta_r,
            bounded=self.bounded,
        )


def count_anchor_residual_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "AnchorPreservingResidualMixer",
    "AnchorResidualOutput",
    "INITIAL_BETA_FRACTION",
    "MAX_SOURCE_FRACTION",
    "MIN_ANCHOR_SPREAD",
    "center_valid",
    "count_anchor_residual_parameters",
    "robust_anchor_spread",
]
