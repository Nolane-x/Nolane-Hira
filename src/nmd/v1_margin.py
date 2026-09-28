from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _validate_logits_gold(logits: Tensor, gold: Tensor) -> None:
    if logits.ndim < 2:
        raise ValueError("S9 logits must include a candidate axis")
    if logits.shape[-1] < 2:
        raise ValueError("S9 requires at least two candidates")
    if gold.shape != logits.shape[:-1]:
        raise ValueError("S9 gold shape must match logits without candidate axis")
    if gold.dtype != torch.long:
        raise TypeError("S9 gold indices must be torch.long")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S9 logits contain non-finite values")
    if gold.numel() and (
        int(gold.min().item()) < 0 or int(gold.max().item()) >= logits.shape[-1]
    ):
        raise ValueError("S9 gold index out of range")


def gold_vs_all_negative_margin_loss(
    logits: Tensor,
    gold: Tensor,
    *,
    margin: float = 0.20,
    reduction: str = "mean",
) -> Tensor:
    """Require the gold logit to exceed every incorrect candidate.

    Per row:
        mean_{j != gold} relu(margin - (gold_logit - logit_j))

    This has no learned parameters and treats every wrong option symmetrically.
    """
    _validate_logits_gold(logits, gold)
    if margin <= 0:
        raise ValueError("S9 margin must be positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("unsupported S9 margin reduction")

    k = logits.shape[-1]
    gold_logits = logits.gather(-1, gold.unsqueeze(-1))
    gaps = gold_logits - logits
    candidate_index = torch.arange(k, device=logits.device)
    mask = candidate_index.view(*([1] * gold.ndim), k).ne(gold.unsqueeze(-1))
    losses = F.relu(logits.new_tensor(float(margin)) - gaps)
    row_loss = losses.masked_select(mask).reshape(*gold.shape, k - 1).mean(-1)

    if reduction == "none":
        return row_loss
    if reduction == "sum":
        return row_loss.sum()
    return row_loss.mean()


def gold_vs_max_wrong_margin(logits: Tensor, gold: Tensor) -> Tensor:
    """Signed gold minus strongest-wrong logit margin per decision row."""
    _validate_logits_gold(logits, gold)
    k = logits.shape[-1]
    gold_logits = logits.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    candidate_index = torch.arange(k, device=logits.device)
    mask = candidate_index.view(*([1] * gold.ndim), k).eq(gold.unsqueeze(-1))
    wrong = logits.masked_fill(mask, float("-inf"))
    strongest_wrong = wrong.max(-1).values
    return gold_logits - strongest_wrong


def top1_top2_margin(logits: Tensor) -> Tensor:
    """Absolute separation between the two highest logits per row."""
    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("S9 top1-top2 margin requires K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("S9 logits contain non-finite values")
    top2 = logits.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def margin_satisfaction_rate(
    logits: Tensor,
    gold: Tensor,
    *,
    margin: float = 0.20,
) -> Tensor:
    if margin <= 0:
        raise ValueError("S9 margin must be positive")
    signed = gold_vs_max_wrong_margin(logits, gold)
    return signed.ge(float(margin)).to(torch.float32).mean()


__all__ = [
    "gold_vs_all_negative_margin_loss",
    "gold_vs_max_wrong_margin",
    "margin_satisfaction_rate",
    "top1_top2_margin",
]
