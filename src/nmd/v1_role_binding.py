from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RoleBindingDiagnostics:
    normalized_role_entropy: Tensor
    max_role_weight: Tensor
    normalized_value_entropy: Tensor
    max_value_weight: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "normalized_role_entropy": float(
                self.normalized_role_entropy.detach().cpu()
            ),
            "max_role_weight": float(self.max_role_weight.detach().cpu()),
            "normalized_value_entropy": float(
                self.normalized_value_entropy.detach().cpu()
            ),
            "max_value_weight": float(self.max_value_weight.detach().cpu()),
        }


class RolePreservingEvidenceBinding(nn.Module):
    """Parameter-free S11 role->value late-interaction binding.

    S10 first pooled question-conditioned state evidence into one vector and
    then compared that vector with pooled options. S11 keeps the role anchor
    and value support separate until the final option score:

        question tokens -> role weights over state positions
        role weights -> local value-position weights
        state value tokens <-> option tokens -> late MaxSim
        local value weights * MaxSim -> option score

    The locality kernel is fixed and contains no learned parameters. The
    shared projection remains the only learned relation surface.
    """

    def __init__(
        self,
        *,
        role_temperature: float = 0.10,
        contrastive_temperature: float = 0.10,
        value_window: int = 4,
    ):
        super().__init__()
        if role_temperature <= 0:
            raise ValueError("S11 role temperature must be positive")
        if contrastive_temperature <= 0:
            raise ValueError("S11 contrastive temperature must be positive")
        if int(value_window) < 1:
            raise ValueError("S11 value window must be >=1")
        self.role_temperature = float(role_temperature)
        self.contrastive_temperature = float(contrastive_temperature)
        self.value_window = int(value_window)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _project(self, projection: nn.Linear, x: Tensor) -> Tensor:
        if not isinstance(projection, nn.Linear):
            raise TypeError("S11 binding requires a shared nn.Linear projection")
        if projection.bias is not None:
            raise ValueError("S11 binding requires bias-free shared projection")
        if x.shape[-1] != projection.in_features:
            raise ValueError("S11 projection input mismatch")
        y = F.normalize(projection(x), dim=-1)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("S11 projection produced non-finite values")
        return y

    def _validate(
        self,
        *,
        projection: nn.Linear,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> None:
        if state_tokens.ndim != 3:
            raise ValueError("S11 state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3:
            raise ValueError("S11 question_tokens must be [B,Q,D]")
        if option_view_tokens.ndim != 5:
            raise ValueError("S11 option_view_tokens must be [B,K,V,T,D]")
        if (
            state_tokens.shape[-1] != projection.in_features
            or question_tokens.shape[-1] != projection.in_features
            or option_view_tokens.shape[-1] != projection.in_features
        ):
            raise ValueError("S11 d_model mismatch")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("S11 state_mask mismatch")
        if (
            question_mask.shape != question_tokens.shape[:2]
            or question_mask.dtype != torch.bool
        ):
            raise ValueError("S11 question_mask mismatch")
        if (
            option_view_token_mask.shape != option_view_tokens.shape[:4]
            or option_view_token_mask.dtype != torch.bool
        ):
            raise ValueError("S11 option token mask mismatch")
        if (
            option_view_mask.shape != option_view_tokens.shape[:3]
            or option_view_mask.dtype != torch.bool
        ):
            raise ValueError("S11 option view mask mismatch")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_view_tokens.shape[0]
        ):
            raise ValueError("S11 batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S11 binding requires K>=2")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("S11 binding requires state content")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S11 binding requires question content")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("S11 binding requires active option views")
        if bool(
            ((option_view_token_mask.sum(-1) < 1) & option_view_mask).any()
        ):
            raise ValueError("S11 active option views require content tokens")

    def _local_value_weights(
        self,
        role_weights: Tensor,
        state_mask: Tensor,
    ) -> Tensor:
        batch, state_len = role_weights.shape
        positions = torch.arange(state_len, device=role_weights.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        local = (distance >= 1) & (distance <= self.value_window)
        local = local[None, :, :].expand(batch, -1, -1)
        local = (
            local
            & state_mask[:, :, None]
            & state_mask[:, None, :]
        )

        kernel = local.to(role_weights.dtype)
        row_sum = kernel.sum(-1, keepdim=True)

        # Degenerate one-token states cannot expose a distinct value position.
        # Fall back to the valid anchor itself only for those rows.
        fallback = torch.eye(
            state_len,
            device=role_weights.device,
            dtype=role_weights.dtype,
        )[None, :, :].expand(batch, -1, -1)
        fallback = fallback * state_mask[:, :, None].to(role_weights.dtype)
        kernel = torch.where(row_sum > 0, kernel, fallback)
        kernel = kernel / kernel.sum(-1, keepdim=True).clamp_min(1.0)

        value_weights = torch.einsum("bs,bsv->bv", role_weights, kernel)
        value_weights = value_weights * state_mask.to(value_weights.dtype)
        value_weights = value_weights / value_weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return value_weights

    def forward(
        self,
        *,
        projection: nn.Linear,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> tuple[Tensor, RoleBindingDiagnostics]:
        self._validate(
            projection=projection,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        state = self._project(projection, state_tokens)
        question = self._project(projection, question_tokens)
        options = self._project(projection, option_view_tokens)

        role_similarity = torch.einsum("bqd,bsd->bqs", question, state)
        role_similarity = role_similarity.masked_fill(
            ~question_mask[..., None],
            -1e4,
        )
        role_score = role_similarity.amax(dim=1)
        role_score = role_score.masked_fill(~state_mask, -1e4)
        role_weights = torch.softmax(
            role_score / self.role_temperature,
            dim=-1,
        )
        role_weights = role_weights * state_mask.to(role_weights.dtype)
        role_weights = role_weights / role_weights.sum(-1, keepdim=True).clamp_min(1e-12)

        value_weights = self._local_value_weights(role_weights, state_mask)

        state_option = torch.einsum(
            "bsd,bkvtd->bkvst",
            state,
            options,
        )
        state_option = state_option.masked_fill(
            ~option_view_token_mask[:, :, :, None, :],
            -1e4,
        )
        value_support = state_option.amax(dim=-1)
        value_support = value_support.masked_fill(
            ~state_mask[:, None, None, :],
            0.0,
        )

        view_scores = (
            value_support * value_weights[:, None, None, :]
        ).sum(-1)
        view_scores = view_scores.masked_fill(~option_view_mask, 0.0)
        view_count = option_view_mask.sum(-1).clamp_min(1).to(view_scores.dtype)
        logits = view_scores.sum(-1) / view_count
        logits = logits / self.contrastive_temperature
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S11 binding logits became non-finite")

        eps = torch.finfo(role_weights.dtype).eps
        valid_state_count = state_mask.sum(-1).to(role_weights.dtype)
        entropy_denominator = torch.where(
            valid_state_count > 1,
            valid_state_count.log(),
            torch.ones_like(valid_state_count),
        )

        role_entropy = -(
            role_weights.clamp_min(eps).log() * role_weights
        ).sum(-1) / entropy_denominator
        value_entropy = -(
            value_weights.clamp_min(eps).log() * value_weights
        ).sum(-1) / entropy_denominator

        diagnostics = RoleBindingDiagnostics(
            normalized_role_entropy=role_entropy.mean(),
            max_role_weight=role_weights.max(-1).values.mean(),
            normalized_value_entropy=value_entropy.mean(),
            max_value_weight=value_weights.max(-1).values.mean(),
        )
        return logits, diagnostics


def binding_cross_entropy(logits: Tensor, gold: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError("S11 binding logits must be [B,K>=2]")
    if gold.shape != logits.shape[:1] or gold.dtype != torch.long:
        raise ValueError("S11 binding gold shape/dtype mismatch")
    if gold.numel() and (
        int(gold.min().item()) < 0 or int(gold.max().item()) >= logits.shape[-1]
    ):
        raise ValueError("S11 binding gold index out of range")
    loss = F.cross_entropy(logits, gold)
    if not bool(torch.isfinite(loss)):
        raise ValueError("S11 binding loss became non-finite")
    return loss


def binding_gold_vs_max_wrong_margin(logits: Tensor, gold: Tensor) -> Tensor:
    if logits.ndim != 2 or gold.shape != logits.shape[:1]:
        raise ValueError("S11 binding margin shape mismatch")
    gold_logits = logits.gather(-1, gold[:, None]).squeeze(-1)
    indices = torch.arange(logits.shape[-1], device=logits.device)
    mask = indices[None, :].eq(gold[:, None])
    strongest_wrong = logits.masked_fill(mask, float("-inf")).max(-1).values
    return gold_logits - strongest_wrong


__all__ = [
    "RoleBindingDiagnostics",
    "RolePreservingEvidenceBinding",
    "binding_cross_entropy",
    "binding_gold_vs_max_wrong_margin",
]
