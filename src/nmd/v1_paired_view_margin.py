from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class PairedViewMarginDiagnostics:
    canonical_mean_margin: float
    paraphrase_mean_margin: float
    canonical_violation_rate: float
    paraphrase_violation_rate: float
    both_satisfied_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "canonical_mean_margin": self.canonical_mean_margin,
            "paraphrase_mean_margin": self.paraphrase_mean_margin,
            "canonical_violation_rate": self.canonical_violation_rate,
            "paraphrase_violation_rate": self.paraphrase_violation_rate,
            "both_satisfied_rate": self.both_satisfied_rate,
        }


def _validate(
    canonical_logits: Tensor,
    paraphrase_logits: Tensor,
    gold: Tensor,
    margin: float,
) -> None:
    if canonical_logits.ndim != 2 or paraphrase_logits.ndim != 2:
        raise ValueError("S18 logits must be rank-2 [N,K]")
    if canonical_logits.shape != paraphrase_logits.shape:
        raise ValueError("S18 canonical/paraphrase logits must have equal shape")
    if canonical_logits.shape[1] < 2:
        raise ValueError("S18 requires K>=2")
    if gold.ndim != 1 or gold.shape[0] != canonical_logits.shape[0]:
        raise ValueError("S18 gold must be rank-1 with one label per row")
    if gold.device != canonical_logits.device or paraphrase_logits.device != canonical_logits.device:
        raise ValueError("S18 logits/gold must share device")
    if not canonical_logits.dtype.is_floating_point or paraphrase_logits.dtype != canonical_logits.dtype:
        raise ValueError("S18 logits must share floating dtype")
    if gold.dtype != torch.long:
        raise ValueError("S18 gold must be torch.long")
    if margin <= 0:
        raise ValueError("S18 margin must be positive")
    if not bool(torch.isfinite(canonical_logits).all() and torch.isfinite(paraphrase_logits).all()):
        raise ValueError("S18 logits must be finite")
    if bool(((gold < 0) | (gold >= canonical_logits.shape[1])).any()):
        raise ValueError("S18 gold index out of range")


def gold_vs_hardest_wrong_margin(logits: Tensor, gold: Tensor) -> Tensor:
    if logits.ndim != 2 or gold.ndim != 1 or logits.shape[0] != gold.shape[0]:
        raise ValueError("S18 margin input shape changed")
    n, k = logits.shape
    if k < 2:
        raise ValueError("S18 margin requires K>=2")
    row = torch.arange(n, device=logits.device)
    chosen = logits[row, gold]
    mask = torch.arange(k, device=logits.device)[None, :].eq(gold[:, None])
    wrong = logits.masked_fill(mask, float("-inf")).amax(-1)
    return chosen - wrong


def paired_both_view_margin_loss(
    canonical_logits: Tensor,
    paraphrase_logits: Tensor,
    gold: Tensor,
    *,
    margin: float = 0.20,
) -> tuple[Tensor, Tensor, Tensor, PairedViewMarginDiagnostics]:
    _validate(canonical_logits, paraphrase_logits, gold, margin)

    canonical_margin = gold_vs_hardest_wrong_margin(canonical_logits, gold)
    paraphrase_margin = gold_vs_hardest_wrong_margin(paraphrase_logits, gold)

    target = canonical_logits.new_tensor(margin)
    canonical_hinge = F.relu(target - canonical_margin)
    paraphrase_hinge = F.relu(target - paraphrase_margin)
    loss = 0.5 * (canonical_hinge.mean() + paraphrase_hinge.mean())

    canonical_violation = canonical_margin < target
    paraphrase_violation = paraphrase_margin < target
    both_satisfied = (~canonical_violation) & (~paraphrase_violation)

    diagnostics = PairedViewMarginDiagnostics(
        canonical_mean_margin=float(canonical_margin.detach().mean().cpu()),
        paraphrase_mean_margin=float(paraphrase_margin.detach().mean().cpu()),
        canonical_violation_rate=float(canonical_violation.float().mean().detach().cpu()),
        paraphrase_violation_rate=float(paraphrase_violation.float().mean().detach().cpu()),
        both_satisfied_rate=float(both_satisfied.float().mean().detach().cpu()),
    )
    return loss, canonical_margin, paraphrase_margin, diagnostics


__all__ = [
    "PairedViewMarginDiagnostics",
    "gold_vs_hardest_wrong_margin",
    "paired_both_view_margin_loss",
]
