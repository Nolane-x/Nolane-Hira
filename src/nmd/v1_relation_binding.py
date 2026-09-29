from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RelationBindingDiagnostics:
    state_role_normalized_entropy: Tensor
    state_role_max_weight: Tensor
    option_role_normalized_entropy: Tensor
    option_role_max_weight: Tensor
    mean_best_pair_score: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "state_role_normalized_entropy": float(
                self.state_role_normalized_entropy.detach().cpu()
            ),
            "state_role_max_weight": float(
                self.state_role_max_weight.detach().cpu()
            ),
            "option_role_normalized_entropy": float(
                self.option_role_normalized_entropy.detach().cpu()
            ),
            "option_role_max_weight": float(
                self.option_role_max_weight.detach().cpu()
            ),
            "mean_best_pair_score": float(
                self.mean_best_pair_score.detach().cpu()
            ),
        }


class RelationStructuredRoleValueBinding(nn.Module):
    """Parameter-free S12 role/value relation binding.

    S11 transported role evidence through a fixed local token window. S12
    removes token-distance assumptions. It identifies state and option role
    anchors from the query, constructs role-relative token relation vectors,
    then compares explicit state-token <-> option-token pairs.

    No learned parameter is owned by this module.
    """

    def __init__(
        self,
        *,
        role_temperature: float = 0.10,
        contrastive_temperature: float = 0.10,
    ):
        super().__init__()
        if role_temperature <= 0:
            raise ValueError("S12 role temperature must be positive")
        if contrastive_temperature <= 0:
            raise ValueError("S12 contrastive temperature must be positive")
        self.role_temperature = float(role_temperature)
        self.contrastive_temperature = float(contrastive_temperature)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _project(self, projection: nn.Linear, x: Tensor) -> Tensor:
        if not isinstance(projection, nn.Linear):
            raise TypeError("S12 binding requires a shared nn.Linear projection")
        if projection.bias is not None:
            raise ValueError("S12 binding requires bias-free shared projection")
        if x.shape[-1] != projection.in_features:
            raise ValueError("S12 projection input mismatch")
        y = F.normalize(projection(x), dim=-1)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("S12 projection produced non-finite values")
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
            raise ValueError("S12 state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3:
            raise ValueError("S12 question_tokens must be [B,Q,D]")
        if option_view_tokens.ndim != 5:
            raise ValueError("S12 option_view_tokens must be [B,K,V,T,D]")
        if (
            state_tokens.shape[-1] != projection.in_features
            or question_tokens.shape[-1] != projection.in_features
            or option_view_tokens.shape[-1] != projection.in_features
        ):
            raise ValueError("S12 d_model mismatch")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("S12 state_mask mismatch")
        if (
            question_mask.shape != question_tokens.shape[:2]
            or question_mask.dtype != torch.bool
        ):
            raise ValueError("S12 question_mask mismatch")
        if (
            option_view_token_mask.shape != option_view_tokens.shape[:4]
            or option_view_token_mask.dtype != torch.bool
        ):
            raise ValueError("S12 option token mask mismatch")
        if (
            option_view_mask.shape != option_view_tokens.shape[:3]
            or option_view_mask.dtype != torch.bool
        ):
            raise ValueError("S12 option view mask mismatch")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_view_tokens.shape[0]
        ):
            raise ValueError("S12 batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S12 binding requires K>=2")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("S12 binding requires state content")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S12 binding requires question content")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("S12 binding requires active option views")
        if bool(
            ((option_view_token_mask.sum(-1) < 1) & option_view_mask).any()
        ):
            raise ValueError("S12 active option views require content tokens")

    def _state_role(
        self,
        state: Tensor,
        state_mask: Tensor,
        question: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        similarity = torch.einsum("bqd,bsd->bqs", question, state)
        similarity = similarity.masked_fill(~question_mask[..., None], -1e4)
        score = similarity.amax(dim=1).masked_fill(~state_mask, -1e4)
        weights = torch.softmax(score / self.role_temperature, dim=-1)
        weights = weights * state_mask.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        anchor = F.normalize(
            torch.einsum("bs,bsd->bd", weights, state),
            dim=-1,
        )
        return weights, anchor

    def _option_role(
        self,
        options: Tensor,
        option_token_mask: Tensor,
        question: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        similarity = torch.einsum("bqd,bkvtd->bkvqt", question, options)
        similarity = similarity.masked_fill(
            ~question_mask[:, None, None, :, None],
            -1e4,
        )
        score = similarity.amax(dim=-2)
        score = score.masked_fill(~option_token_mask, -1e4)
        weights = torch.softmax(score / self.role_temperature, dim=-1)
        weights = weights * option_token_mask.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        anchor = F.normalize(
            torch.einsum("bkvt,bkvtd->bkvd", weights, options),
            dim=-1,
        )
        return weights, anchor

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
    ) -> tuple[Tensor, RelationBindingDiagnostics]:
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

        state_role_weights, state_anchor = self._state_role(
            state,
            state_mask,
            question,
            question_mask,
        )
        option_role_weights, option_anchor = self._option_role(
            options,
            option_view_token_mask,
            question,
            question_mask,
        )

        state_relation = F.normalize(
            state - state_anchor[:, None, :],
            dim=-1,
        )
        option_relation = F.normalize(
            options - option_anchor[:, :, :, None, :],
            dim=-1,
        )

        direct = torch.einsum(
            "bsd,bkvtd->bkvst",
            state,
            options,
        )
        relation = torch.einsum(
            "bsd,bkvtd->bkvst",
            state_relation,
            option_relation,
        )
        pair_score = 0.5 * (direct + relation)
        pair_score = pair_score.masked_fill(
            ~state_mask[:, None, None, :, None],
            -1e4,
        )
        pair_score = pair_score.masked_fill(
            ~option_view_token_mask[:, :, :, None, :],
            -1e4,
        )
        best_pair = pair_score.amax(dim=(-1, -2))

        role_compatibility = torch.einsum(
            "bd,bkvd->bkv",
            state_anchor,
            option_anchor,
        )
        view_scores = 0.5 * (best_pair + role_compatibility)
        view_scores = view_scores.masked_fill(~option_view_mask, 0.0)
        view_count = option_view_mask.sum(-1).clamp_min(1).to(view_scores.dtype)
        logits = view_scores.sum(-1) / view_count
        logits = logits / self.contrastive_temperature
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S12 binding logits became non-finite")

        eps = torch.finfo(state_role_weights.dtype).eps
        state_count = state_mask.sum(-1).to(state_role_weights.dtype)
        state_den = torch.where(
            state_count > 1,
            state_count.log(),
            torch.ones_like(state_count),
        )
        state_entropy = -(
            state_role_weights.clamp_min(eps).log() * state_role_weights
        ).sum(-1) / state_den

        option_count = option_view_token_mask.sum(-1).to(option_role_weights.dtype)
        option_den = torch.where(
            option_count > 1,
            option_count.log(),
            torch.ones_like(option_count),
        )
        option_entropy = -(
            option_role_weights.clamp_min(eps).log() * option_role_weights
        ).sum(-1) / option_den
        active = option_view_mask.to(option_entropy.dtype)
        option_entropy_mean = (
            option_entropy * active
        ).sum() / active.sum().clamp_min(1.0)
        option_max_mean = (
            option_role_weights.max(-1).values * active
        ).sum() / active.sum().clamp_min(1.0)

        diagnostics = RelationBindingDiagnostics(
            state_role_normalized_entropy=state_entropy.mean(),
            state_role_max_weight=state_role_weights.max(-1).values.mean(),
            option_role_normalized_entropy=option_entropy_mean,
            option_role_max_weight=option_max_mean,
            mean_best_pair_score=(
                best_pair * active
            ).sum() / active.sum().clamp_min(1.0),
        )
        return logits, diagnostics


def relation_binding_cross_entropy(logits: Tensor, gold: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError("S12 binding logits must be [B,K>=2]")
    if gold.shape != logits.shape[:1] or gold.dtype != torch.long:
        raise ValueError("S12 binding gold shape/dtype mismatch")
    if gold.numel() and (
        int(gold.min().item()) < 0 or int(gold.max().item()) >= logits.shape[-1]
    ):
        raise ValueError("S12 binding gold index out of range")
    loss = F.cross_entropy(logits, gold)
    if not bool(torch.isfinite(loss)):
        raise ValueError("S12 binding loss became non-finite")
    return loss


def relation_binding_gold_vs_max_wrong_margin(
    logits: Tensor,
    gold: Tensor,
) -> Tensor:
    if logits.ndim != 2 or gold.shape != logits.shape[:1]:
        raise ValueError("S12 binding margin shape mismatch")
    gold_logits = logits.gather(-1, gold[:, None]).squeeze(-1)
    indices = torch.arange(logits.shape[-1], device=logits.device)
    mask = indices[None, :].eq(gold[:, None])
    strongest_wrong = logits.masked_fill(mask, float("-inf")).max(-1).values
    return gold_logits - strongest_wrong


__all__ = [
    "RelationBindingDiagnostics",
    "RelationStructuredRoleValueBinding",
    "relation_binding_cross_entropy",
    "relation_binding_gold_vs_max_wrong_margin",
]
