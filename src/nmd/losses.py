from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


def choice_ce(logits: Tensor, gold: Tensor) -> Tensor:
    return F.cross_entropy(logits, gold)


def distribution_kl(logits: Tensor, teacher_probs: Tensor, temperature: float = 1.0) -> Tensor:
    t = float(temperature)
    teacher = teacher_probs / teacher_probs.sum(-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(
        F.log_softmax(logits / t, dim=-1), teacher, reduction="batchmean"
    ) * (t * t)


def brier_loss(logits: Tensor, gold: Tensor) -> Tensor:
    p = torch.softmax(logits, dim=-1)
    onehot = F.one_hot(gold, num_classes=logits.shape[-1]).to(p.dtype)
    return ((p - onehot) ** 2).sum(-1).mean()


def soft_brier_loss(logits: Tensor, teacher_probs: Tensor) -> Tensor:
    p = torch.softmax(logits, dim=-1)
    teacher = teacher_probs / teacher_probs.sum(-1, keepdim=True).clamp_min(1e-12)
    return ((p - teacher) ** 2).sum(-1).mean()


def ordinal_expected_mae(
    logits: Tensor,
    gold_score: Tensor,
    score_support: Tensor | None = None,
) -> Tensor:
    p = torch.softmax(logits, dim=-1)
    if score_support is None:
        support = torch.arange(
            logits.shape[-1],
            device=logits.device,
            dtype=p.dtype,
        ).unsqueeze(0).expand_as(p)
    else:
        support = score_support.to(
            device=logits.device,
            dtype=p.dtype,
        )
        if support.ndim == 1:
            support = support.unsqueeze(0).expand_as(p)
        if support.shape != p.shape:
            raise ValueError(
                "score_support must be [K] or match logits [B,K]"
            )
    expected = (p * support).sum(-1)
    return (expected - gold_score.to(p.dtype)).abs().mean()


def pairwise_margin_loss(
    logits: Tensor, positive: Tensor, negative: Tensor, margin: float = 0.2
) -> Tensor:
    pos = logits.gather(1, positive[:, None]).squeeze(1)
    neg = logits.gather(1, negative[:, None]).squeeze(1)
    return F.relu(float(margin) - pos + neg).mean()


@dataclass(frozen=True)
class LossWeights:
    hard_ce: float = 1.0
    teacher_kl: float = 0.0
    brier: float = 0.1
    soft_brier: float = 0.0
    ordinal_mae: float = 0.0


def typed_decision_loss(
    logits: Tensor,
    gold: Tensor,
    *,
    teacher_probs: Tensor | None = None,
    gold_score: Tensor | None = None,
    score_support: Tensor | None = None,
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    parts: dict[str, Tensor] = {}
    if weights.hard_ce:
        parts["hard_ce"] = choice_ce(logits, gold) * weights.hard_ce
    if weights.brier:
        parts["brier"] = brier_loss(logits, gold) * weights.brier
    if teacher_probs is not None and weights.teacher_kl:
        parts["teacher_kl"] = distribution_kl(logits, teacher_probs) * weights.teacher_kl
    if teacher_probs is not None and weights.soft_brier:
        parts["soft_brier"] = soft_brier_loss(logits, teacher_probs) * weights.soft_brier
    if gold_score is not None and weights.ordinal_mae:
        parts["ordinal_mae"] = ordinal_expected_mae(
            logits,
            gold_score,
            score_support=score_support,
        ) * weights.ordinal_mae
    if not parts:
        raise ValueError("at least one loss component must be enabled")
    return sum(parts.values()), parts
