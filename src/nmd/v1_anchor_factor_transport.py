from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROLE_TEMPERATURE = 0.10
ANCHOR_SEPARATION_MARGIN = 0.20
ROLE_ANCHOR_WEIGHT = 0.50
VALUE_ANCHOR_WEIGHT = 0.50


@dataclass(frozen=True)
class StateFactorAnchors:
    role: Tensor
    value: Tensor
    role_weights: Tensor
    value_weights: Tensor


@dataclass(frozen=True)
class FactorTransportLoss:
    total: Tensor
    role_total: Tensor
    value_total: Tensor
    role_alignment: Tensor
    role_separation: Tensor
    value_alignment: Tensor
    value_separation: Tensor


def _normalized_masked_weights(weights: Tensor, mask: Tensor) -> Tensor:
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


def extract_state_factor_anchors(
    *,
    projection: nn.Linear,
    state_tokens: Tensor,
    state_mask: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
    role_temperature: float = ROLE_TEMPERATURE,
) -> StateFactorAnchors:
    if state_tokens.ndim != 3 or question_tokens.ndim != 3:
        raise ValueError("S28 state/question tokens must be [B,T,D]")
    if state_mask.shape != state_tokens.shape[:2]:
        raise ValueError("S28 state mask shape changed")
    if question_mask.shape != question_tokens.shape[:2]:
        raise ValueError("S28 question mask shape changed")
    if state_tokens.shape[0] != question_tokens.shape[0]:
        raise ValueError("S28 state/question batch mismatch")
    if role_temperature <= 0:
        raise ValueError("S28 role temperature must be positive")

    state = F.normalize(projection(state_tokens), dim=-1)
    question = F.normalize(projection(question_tokens), dim=-1)

    state_q = torch.einsum("bqd,bsd->bqs", question, state)
    state_q = state_q.masked_fill(~question_mask[..., None], -1e4)
    state_role_score = state_q.amax(dim=1).masked_fill(~state_mask, -1e4)
    state_role = torch.softmax(state_role_score / role_temperature, dim=-1)
    state_role = _normalized_masked_weights(state_role, state_mask)

    role_anchor = F.normalize(
        torch.einsum("bs,bsd->bd", state_role, state),
        dim=-1,
    )

    coefficient = torch.einsum("bsd,bd->bs", state, role_anchor)
    state_content = F.normalize(
        state - coefficient[..., None] * role_anchor[:, None, :],
        dim=-1,
    )
    value_weights = _normalized_masked_weights(
        (1.0 - state_role) * state_mask.to(state_role.dtype),
        state_mask,
    )
    value_anchor = F.normalize(
        torch.einsum("bs,bsd->bd", value_weights, state_content),
        dim=-1,
    )

    if not bool(
        torch.isfinite(role_anchor).all()
        and torch.isfinite(value_anchor).all()
        and torch.isfinite(state_role).all()
        and torch.isfinite(value_weights).all()
    ):
        raise ValueError("S28 anchor extraction became non-finite")

    return StateFactorAnchors(
        role=role_anchor,
        value=value_anchor,
        role_weights=state_role,
        value_weights=value_weights,
    )


def _factor_transport_loss(
    canonical: Tensor,
    paraphrase: Tensor,
    *,
    separation_margin: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if canonical.shape != paraphrase.shape or canonical.ndim != 2:
        raise ValueError("S28 anchor factors must match [B,D]")
    if canonical.shape[0] < 2:
        raise ValueError("S28 anchor transport needs B>=2 for negatives")
    if separation_margin < 0:
        raise ValueError("S28 separation margin must be non-negative")

    canonical = F.normalize(canonical, dim=-1)
    paraphrase = F.normalize(paraphrase, dim=-1)
    cross = canonical @ paraphrase.transpose(0, 1)
    same = cross.diagonal()
    eye = torch.eye(
        canonical.shape[0],
        dtype=torch.bool,
        device=canonical.device,
    )
    strongest_wrong = cross.masked_fill(eye, float("-inf")).amax(-1)

    alignment = (1.0 - same).mean()
    separation = F.relu(
        canonical.new_tensor(separation_margin)
        - (same - strongest_wrong)
    ).mean()
    total = alignment + separation
    return total, alignment, separation


def anchor_level_factor_transport_loss(
    canonical: StateFactorAnchors,
    paraphrase: StateFactorAnchors,
    *,
    separation_margin: float = ANCHOR_SEPARATION_MARGIN,
    role_weight: float = ROLE_ANCHOR_WEIGHT,
    value_weight: float = VALUE_ANCHOR_WEIGHT,
) -> FactorTransportLoss:
    if role_weight != ROLE_ANCHOR_WEIGHT or value_weight != VALUE_ANCHOR_WEIGHT:
        raise ValueError("S28 role/value weights are frozen at 0.50 / 0.50")

    role_total, role_alignment, role_separation = _factor_transport_loss(
        canonical.role,
        paraphrase.role,
        separation_margin=separation_margin,
    )
    value_total, value_alignment, value_separation = _factor_transport_loss(
        canonical.value,
        paraphrase.value,
        separation_margin=separation_margin,
    )
    total = role_weight * role_total + value_weight * value_total

    for value in (
        total,
        role_total,
        value_total,
        role_alignment,
        role_separation,
        value_alignment,
        value_separation,
    ):
        if not bool(torch.isfinite(value)):
            raise ValueError("S28 anchor transport loss became non-finite")

    return FactorTransportLoss(
        total=total,
        role_total=role_total,
        value_total=value_total,
        role_alignment=role_alignment,
        role_separation=role_separation,
        value_alignment=value_alignment,
        value_separation=value_separation,
    )


def anchor_transport_added_parameter_count() -> int:
    return 0


__all__ = [
    "ROLE_TEMPERATURE",
    "ANCHOR_SEPARATION_MARGIN",
    "ROLE_ANCHOR_WEIGHT",
    "VALUE_ANCHOR_WEIGHT",
    "StateFactorAnchors",
    "FactorTransportLoss",
    "extract_state_factor_anchors",
    "anchor_level_factor_transport_loss",
    "anchor_transport_added_parameter_count",
]
