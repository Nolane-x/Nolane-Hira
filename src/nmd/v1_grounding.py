from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GroundingDiagnostics:
    normalized_attention_entropy: Tensor
    max_attention_weight: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "normalized_attention_entropy": float(
                self.normalized_attention_entropy.detach().cpu()
            ),
            "max_attention_weight": float(
                self.max_attention_weight.detach().cpu()
            ),
        }


class QueryConditionedStateOptionGrounding(nn.Module):
    """Parameter-free S10 grounding operator over a shared projection.

    The projection is owned by the existing S8/S9 semantic scorer.  This
    module introduces no learned parameters.  It turns question tokens into
    attention over state tokens, pools the selected state evidence, then
    contrasts that evidence against all dynamic options.
    """

    def __init__(
        self,
        *,
        attention_temperature: float = 0.10,
        contrastive_temperature: float = 0.10,
    ):
        super().__init__()
        if attention_temperature <= 0:
            raise ValueError("grounding attention temperature must be positive")
        if contrastive_temperature <= 0:
            raise ValueError("grounding contrastive temperature must be positive")
        self.attention_temperature = float(attention_temperature)
        self.contrastive_temperature = float(contrastive_temperature)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _project(self, projection: nn.Linear, x: Tensor) -> Tensor:
        if not isinstance(projection, nn.Linear):
            raise TypeError("S10 grounding requires a shared nn.Linear projection")
        if projection.bias is not None:
            raise ValueError("S10 grounding requires bias-free shared projection")
        if x.shape[-1] != projection.in_features:
            raise ValueError("S10 grounding projection input mismatch")
        y = F.normalize(projection(x), dim=-1)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("S10 grounding projection produced non-finite values")
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
            raise ValueError("S10 state_tokens must be [B,S,D]")
        if question_tokens.ndim != 3:
            raise ValueError("S10 question_tokens must be [B,Q,D]")
        if option_view_tokens.ndim != 5:
            raise ValueError("S10 option_view_tokens must be [B,K,V,T,D]")
        if (
            state_tokens.shape[-1] != projection.in_features
            or question_tokens.shape[-1] != projection.in_features
            or option_view_tokens.shape[-1] != projection.in_features
        ):
            raise ValueError("S10 grounding d_model mismatch")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("S10 state_mask mismatch")
        if (
            question_mask.shape != question_tokens.shape[:2]
            or question_mask.dtype != torch.bool
        ):
            raise ValueError("S10 question_mask mismatch")
        if (
            option_view_token_mask.shape != option_view_tokens.shape[:4]
            or option_view_token_mask.dtype != torch.bool
        ):
            raise ValueError("S10 option token mask mismatch")
        if (
            option_view_mask.shape != option_view_tokens.shape[:3]
            or option_view_mask.dtype != torch.bool
        ):
            raise ValueError("S10 option view mask mismatch")
        if not (
            state_tokens.shape[0]
            == question_tokens.shape[0]
            == option_view_tokens.shape[0]
        ):
            raise ValueError("S10 grounding batch mismatch")
        if option_view_tokens.shape[1] < 2:
            raise ValueError("S10 grounding requires K>=2")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("S10 grounding requires state content")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S10 grounding requires question content")
        if bool((option_view_mask.sum(-1) < 1).any()):
            raise ValueError("S10 grounding requires active option views")
        if bool(
            (
                (option_view_token_mask.sum(-1) < 1)
                & option_view_mask
            ).any()
        ):
            raise ValueError("S10 active option views require content tokens")

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
    ) -> tuple[Tensor, GroundingDiagnostics]:
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

        relevance = torch.einsum("bqd,bsd->bqs", question, state)
        relevance = relevance / self.attention_temperature
        relevance = relevance.masked_fill(
            ~state_mask[:, None, :],
            -1e4,
        )
        attention = torch.softmax(relevance, dim=-1)
        attention = attention * question_mask[..., None].to(attention.dtype)

        per_question_evidence = torch.einsum(
            "bqs,bsd->bqd",
            attention,
            state,
        )
        q_weights = question_mask.to(per_question_evidence.dtype)[..., None]
        evidence = (
            per_question_evidence * q_weights
        ).sum(1) / q_weights.sum(1).clamp_min(1.0)
        evidence = F.normalize(evidence, dim=-1)

        option_token_weights = option_view_token_mask.to(options.dtype)[..., None]
        option_view_pooled = (
            options * option_token_weights
        ).sum(-2) / option_token_weights.sum(-2).clamp_min(1.0)
        option_view_pooled = F.normalize(option_view_pooled, dim=-1)

        view_weights = option_view_mask.to(options.dtype)[..., None]
        option_pooled = (
            option_view_pooled * view_weights
        ).sum(-2) / view_weights.sum(-2).clamp_min(1.0)
        option_pooled = F.normalize(option_pooled, dim=-1)

        logits = torch.einsum("bd,bkd->bk", evidence, option_pooled)
        logits = logits / self.contrastive_temperature
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S10 grounding logits became non-finite")

        eps = torch.finfo(attention.dtype).eps
        entropy_per_q = -(attention.clamp_min(eps).log() * attention).sum(-1)
        valid_state_count = state_mask.sum(-1).to(attention.dtype)
        entropy_denominator = valid_state_count.log().clamp_min(1.0)
        normalized_entropy = entropy_per_q / entropy_denominator[:, None]
        q_mask_float = question_mask.to(attention.dtype)
        normalized_entropy = (
            normalized_entropy * q_mask_float
        ).sum(-1) / q_mask_float.sum(-1).clamp_min(1.0)

        max_weight_per_q = attention.max(-1).values
        max_weight = (
            max_weight_per_q * q_mask_float
        ).sum(-1) / q_mask_float.sum(-1).clamp_min(1.0)

        diagnostics = GroundingDiagnostics(
            normalized_attention_entropy=normalized_entropy.mean(),
            max_attention_weight=max_weight.mean(),
        )
        return logits, diagnostics


def grounding_cross_entropy(
    logits: Tensor,
    gold: Tensor,
) -> Tensor:
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError("S10 grounding logits must be [B,K>=2]")
    if gold.shape != logits.shape[:1] or gold.dtype != torch.long:
        raise ValueError("S10 grounding gold shape/dtype mismatch")
    if gold.numel() and (
        int(gold.min().item()) < 0 or int(gold.max().item()) >= logits.shape[-1]
    ):
        raise ValueError("S10 grounding gold index out of range")
    loss = F.cross_entropy(logits, gold)
    if not bool(torch.isfinite(loss)):
        raise ValueError("S10 grounding loss became non-finite")
    return loss


def grounding_gold_vs_max_wrong_margin(
    logits: Tensor,
    gold: Tensor,
) -> Tensor:
    if logits.ndim != 2 or gold.shape != logits.shape[:1]:
        raise ValueError("S10 grounding margin shape mismatch")
    gold_logits = logits.gather(-1, gold[:, None]).squeeze(-1)
    indices = torch.arange(logits.shape[-1], device=logits.device)
    mask = indices[None, :].eq(gold[:, None])
    strongest_wrong = logits.masked_fill(mask, float("-inf")).max(-1).values
    return gold_logits - strongest_wrong


__all__ = [
    "GroundingDiagnostics",
    "QueryConditionedStateOptionGrounding",
    "grounding_cross_entropy",
    "grounding_gold_vs_max_wrong_margin",
]
