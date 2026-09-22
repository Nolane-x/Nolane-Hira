from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .structural import tokens


def multiset_recall(premise: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not hypothesis:
        return 0.0
    p = Counter(premise)
    h = Counter(hypothesis)
    overlap = sum(min(n, p.get(tok, 0)) for tok, n in h.items())
    return overlap / sum(h.values())


def ordered_lcs_recall(premise: Sequence[str], hypothesis: Sequence[str]) -> float:
    """Recall of hypothesis tokens captured by LCS(premise, hypothesis)."""
    if not hypothesis:
        return 0.0
    prev = [0] * (len(hypothesis) + 1)
    for p_tok in premise:
        cur = [0]
        for j, h_tok in enumerate(hypothesis, start=1):
            if p_tok == h_tok:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1] / len(hypothesis)


@dataclass(frozen=True)
class NearStructuralExample:
    index: int
    label: int
    multiset_recall: float
    ordered_lcs_recall: float

    @property
    def score(self) -> float:
        return max(self.multiset_recall, self.ordered_lcs_recall)

    @property
    def min_score(self) -> float:
        return min(self.multiset_recall, self.ordered_lcs_recall)


def balanced_near_structural_non_entailment_indices(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    labels: Sequence[int],
    *,
    threshold: float = 0.80,
    min_hypothesis_tokens: int = 3,
    max_per_label: int = 500,
) -> tuple[list[int], dict[str, object]]:
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    if min_hypothesis_tokens <= 0:
        raise ValueError("min_hypothesis_tokens must be positive")
    if max_per_label <= 0:
        raise ValueError("max_per_label must be positive")
    if not (len(premises) == len(hypotheses) == len(labels)):
        raise ValueError("premises/hypotheses/labels length mismatch")

    rows: dict[int, list[NearStructuralExample]] = {1: [], 2: []}
    for i, raw_label in enumerate(labels):
        label = int(raw_label)
        if label not in rows:
            continue
        p = tokens(premises[i])
        h = tokens(hypotheses[i])
        if len(h) < min_hypothesis_tokens:
            continue
        multi = multiset_recall(p, h)
        lcs = ordered_lcs_recall(p, h)
        if multi >= threshold or lcs >= threshold:
            rows[label].append(NearStructuralExample(i, label, multi, lcs))

    available = {label: len(items) for label, items in rows.items()}
    per_label = min(max_per_label, min(available.values(), default=0))
    selected: list[NearStructuralExample] = []
    if per_label > 0:
        for label in (1, 2):
            ranked = sorted(
                rows[label],
                key=lambda x: (-x.score, -x.min_score, x.index),
            )[:per_label]
            selected.extend(ranked)
        selected.sort(key=lambda x: x.index)

    stats = {
        "threshold": float(threshold),
        "min_hypothesis_tokens": int(min_hypothesis_tokens),
        "available_by_label": {str(k): v for k, v in available.items()},
        "selected_per_label": per_label,
        "selected_total": len(selected),
        "mean_max_recall": (
            sum(x.score for x in selected) / len(selected) if selected else 0.0
        ),
        "mean_multiset_recall": (
            sum(x.multiset_recall for x in selected) / len(selected) if selected else 0.0
        ),
        "mean_ordered_lcs_recall": (
            sum(x.ordered_lcs_recall for x in selected) / len(selected)
            if selected else 0.0
        ),
    }
    return [x.index for x in selected], stats


def interpolate_state_dicts(
    base: Mapping[str, Tensor],
    direction: Mapping[str, Tensor],
    alpha: float,
) -> dict[str, Tensor]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if set(base) != set(direction):
        raise ValueError("state dict keys differ")

    out: dict[str, Tensor] = {}
    for key in base:
        a = base[key]
        b = direction[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"state tensor mismatch for {key}")
        if torch.is_floating_point(a):
            out[key] = ((1.0 - alpha) * a + alpha * b).detach().cpu()
        else:
            if not torch.equal(a, b):
                raise ValueError(f"non-floating state differs for {key}")
            out[key] = a.detach().cpu().clone()
    return out
