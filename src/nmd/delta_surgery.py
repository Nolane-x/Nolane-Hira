from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .block_interpolation import atomic_group_for_key
from .interpolation import multiset_recall, ordered_lcs_recall
from .structural import tokens


@dataclass(frozen=True)
class RelationSpan:
    key: str
    start: int
    end: int
    shape: tuple[int, ...]


def relation_layout(state: Mapping[str, Tensor]) -> tuple[RelationSpan, ...]:
    spans: list[RelationSpan] = []
    cursor = 0
    for key in sorted(state):
        if atomic_group_for_key(key) != "relation":
            continue
        value = state[key]
        if not torch.is_floating_point(value):
            raise ValueError(f"relation state must be floating point: {key}")
        n = int(value.numel())
        spans.append(
            RelationSpan(
                key=key,
                start=cursor,
                end=cursor + n,
                shape=tuple(int(x) for x in value.shape),
            )
        )
        cursor += n
    if not spans:
        raise ValueError("relation state is empty")
    return tuple(spans)


def flatten_relation(
    state: Mapping[str, Tensor],
    *,
    layout: Sequence[RelationSpan] | None = None,
) -> Tensor:
    spans = tuple(layout) if layout is not None else relation_layout(state)
    chunks: list[Tensor] = []
    for span in spans:
        if span.key not in state:
            raise ValueError(f"missing relation key {span.key}")
        value = state[span.key]
        if tuple(value.shape) != span.shape:
            raise ValueError(f"relation shape mismatch for {span.key}")
        chunks.append(value.detach().cpu().reshape(-1).to(torch.float64))
    return torch.cat(chunks)


def relation_delta(
    base: Mapping[str, Tensor],
    direction: Mapping[str, Tensor],
    *,
    layout: Sequence[RelationSpan] | None = None,
) -> Tensor:
    spans = tuple(layout) if layout is not None else relation_layout(base)
    return flatten_relation(direction, layout=spans) - flatten_relation(
        base, layout=spans
    )


def flatten_relation_named_tensors(
    tensors: Mapping[str, Tensor],
    *,
    layout: Sequence[RelationSpan],
) -> Tensor:
    chunks: list[Tensor] = []
    for span in layout:
        if span.key not in tensors:
            raise ValueError(f"missing tensor for relation key {span.key}")
        value = tensors[span.key]
        if tuple(value.shape) != span.shape:
            raise ValueError(f"tensor shape mismatch for {span.key}")
        chunks.append(value.detach().cpu().reshape(-1).to(torch.float64))
    return torch.cat(chunks)


def coordinate_scores(
    structural_gradient: Tensor,
    retention_energy: Tensor,
    delta: Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    g = structural_gradient.detach().cpu().to(torch.float64)
    e = retention_energy.detach().cpu().to(torch.float64)
    d = delta.detach().cpu().to(torch.float64)
    if not (g.shape == e.shape == d.shape):
        raise ValueError("gradient/energy/delta shape mismatch")
    if (e < 0).any():
        raise ValueError("retention energy must be non-negative")
    raw_benefit = -g * d
    benefit = raw_benefit.clamp_min(0.0)
    cost = e * d.square()
    positive = (benefit > 0.0) & (d.abs() > 1e-12)
    benefit_only = torch.where(
        positive, benefit, torch.full_like(benefit, float("-inf"))
    )
    benefit_over_cost = torch.where(
        positive,
        benefit / (cost + float(eps)),
        torch.full_like(benefit, float("-inf")),
    )
    return {
        "raw_benefit": raw_benefit,
        "benefit": benefit,
        "cost": cost,
        "positive_mask": positive,
        "benefit_score": benefit_only,
        "benefit_over_cost_score": benefit_over_cost,
    }


def top_fraction_mask(
    score: Tensor,
    positive_mask: Tensor,
    fraction: float,
) -> Tensor:
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    score = score.detach().cpu().to(torch.float64)
    positive_mask = positive_mask.detach().cpu().bool()
    if score.shape != positive_mask.shape or score.ndim != 1:
        raise ValueError("score and positive_mask must be equal 1-D tensors")
    pool = torch.nonzero(positive_mask, as_tuple=False).squeeze(1)
    if pool.numel() == 0:
        raise ValueError("positive-benefit pool is empty")

    # pool is ascending global coordinate order. Stable score sort therefore
    # preserves ascending global index for exact score ties.
    local_order = torch.argsort(
        score[pool], descending=True, stable=True
    )
    ranked = pool[local_order]
    count = max(1, int(math.ceil(fraction * int(pool.numel()))))
    selected = ranked[:count]
    mask = torch.zeros_like(positive_mask)
    mask[selected] = True
    return mask


def nested_masks(masks: Sequence[Tensor]) -> bool:
    if not masks:
        return False
    prev = masks[0].detach().cpu().bool()
    for current in masks[1:]:
        cur = current.detach().cpu().bool()
        if cur.shape != prev.shape:
            return False
        if (prev & ~cur).any():
            return False
        prev = cur
    return True


def apply_relation_mask(
    base: Mapping[str, Tensor],
    direction: Mapping[str, Tensor],
    mask: Tensor,
    *,
    alpha: float,
    layout: Sequence[RelationSpan] | None = None,
) -> dict[str, Tensor]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if set(base) != set(direction):
        raise ValueError("state dict keys differ")
    spans = tuple(layout) if layout is not None else relation_layout(base)
    total = spans[-1].end
    mask = mask.detach().cpu().bool()
    if mask.shape != (total,):
        raise ValueError(f"mask must have shape ({total},)")

    relation_keys = {span.key for span in spans}
    out: dict[str, Tensor] = {
        key: value.detach().cpu().clone() for key, value in base.items()
    }
    for span in spans:
        a = base[span.key].detach().cpu()
        b = direction[span.key].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"state tensor mismatch for {span.key}")
        local = mask[span.start:span.end].reshape(span.shape)
        moved = (a + alpha * (b - a)).to(a.dtype)
        out[span.key] = torch.where(local, moved, a)

    # Fail closed if layout omitted or misclassified relation keys.
    for key in base:
        if atomic_group_for_key(key) == "relation" and key not in relation_keys:
            raise ValueError(f"relation key not covered by layout: {key}")
    return out


def sparse_identity_invariants(
    base: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    mask: Tensor,
    *,
    layout: Sequence[RelationSpan] | None = None,
) -> dict[str, bool]:
    spans = tuple(layout) if layout is not None else relation_layout(base)
    mask = mask.detach().cpu().bool()
    relation_keys = {span.key for span in spans}

    non_relation_identical = True
    for key in base:
        if key not in candidate:
            non_relation_identical = False
            break
        if key not in relation_keys and not torch.equal(
            base[key].detach().cpu(), candidate[key].detach().cpu()
        ):
            non_relation_identical = False
            break

    masked_off_identical = True
    for span in spans:
        a = base[span.key].detach().cpu().reshape(-1)
        c = candidate[span.key].detach().cpu().reshape(-1)
        local = mask[span.start:span.end]
        if not torch.equal(a[~local], c[~local]):
            masked_off_identical = False
            break

    return {
        "non_relation_bit_identical": non_relation_identical,
        "masked_off_relation_bit_identical": masked_off_identical,
    }


@dataclass(frozen=True)
class RankedStructuralWindowExample:
    index: int
    label: int
    score: float
    secondary_score: float


def balanced_ranked_structural_window_indices(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    labels: Sequence[int],
    *,
    start_per_label: int,
    count_per_label: int,
    min_hypothesis_tokens: int = 3,
) -> tuple[list[int], dict[str, object]]:
    """Exact R18/R19 ranking with LCS evaluated only at the primary cutoff.

    ordered-LCS recall can never exceed multiset recall, so the frozen primary
    score max(multiset, LCS) is exactly multiset recall. LCS is only the
    secondary tie-break. We first find the multiset cutoff required to cover
    the requested rank window, then evaluate LCS for every row at or above
    that cutoff. This is mathematically identical to exhaustive LCS ranking
    while avoiding unnecessary LCS work on the large MultiNLI train split.
    """
    if start_per_label < 0:
        raise ValueError("start_per_label must be non-negative")
    if count_per_label <= 0:
        raise ValueError("count_per_label must be positive")
    if min_hypothesis_tokens <= 0:
        raise ValueError("min_hypothesis_tokens must be positive")
    if not (len(premises) == len(hypotheses) == len(labels)):
        raise ValueError("premises/hypotheses/labels length mismatch")

    stop = start_per_label + count_per_label
    primary_rows: dict[int, list[tuple[int, float]]] = {1: [], 2: []}
    for i, raw_label in enumerate(labels):
        label = int(raw_label)
        if label not in primary_rows:
            continue
        p = tokens(premises[i])
        h = tokens(hypotheses[i])
        if len(h) < min_hypothesis_tokens:
            continue
        multi = multiset_recall(p, h)
        primary_rows[label].append((i, multi))

    eligible = {
        label: len(items) for label, items in primary_rows.items()
    }
    if any(n < stop for n in eligible.values()):
        raise ValueError(
            f"not enough ranked structural examples for window: {eligible}"
        )

    selected_by_label: dict[int, list[RankedStructuralWindowExample]] = {}
    selected: list[RankedStructuralWindowExample] = []
    cutoff_by_label: dict[str, float] = {}
    lcs_evaluated_by_label: dict[str, int] = {}

    for label in (1, 2):
        primary_ranked = sorted(
            primary_rows[label], key=lambda row: (-row[1], row[0])
        )
        cutoff = float(primary_ranked[stop - 1][1])
        cutoff_by_label[str(label)] = cutoff
        contenders = [
            (index, multi)
            for index, multi in primary_rows[label]
            if multi >= cutoff
        ]

        exact_rows: list[RankedStructuralWindowExample] = []
        for index, multi in contenders:
            p = tokens(premises[index])
            h = tokens(hypotheses[index])
            lcs = ordered_lcs_recall(p, h)
            if lcs > multi + 1e-12:
                raise RuntimeError(
                    "ordered-LCS recall exceeded multiset recall"
                )
            exact_rows.append(
                RankedStructuralWindowExample(
                    index=index,
                    label=label,
                    score=multi,
                    secondary_score=lcs,
                )
            )
        lcs_evaluated_by_label[str(label)] = len(exact_rows)
        exact_rows.sort(
            key=lambda x: (-x.score, -x.secondary_score, x.index)
        )
        window = exact_rows[start_per_label:stop]
        if len(window) != count_per_label:
            raise RuntimeError("cutoff ranking produced incomplete window")
        selected_by_label[label] = window
        selected.extend(window)

    selected.sort(key=lambda x: x.index)
    stats: dict[str, object] = {
        "eligible_by_label": {str(k): v for k, v in eligible.items()},
        "start_per_label": int(start_per_label),
        "count_per_label": int(count_per_label),
        "selected_total": len(selected),
        "min_hypothesis_tokens": int(min_hypothesis_tokens),
        "primary_cutoff_by_label": cutoff_by_label,
        "lcs_evaluated_by_label": lcs_evaluated_by_label,
        "ranking_optimization": (
            "exact multiset-primary cutoff; LCS only for cutoff contenders"
        ),
        "mean_score_by_label": {
            str(label): sum(x.score for x in items) / len(items)
            for label, items in selected_by_label.items()
        },
        "min_score_by_label": {
            str(label): min(x.score for x in items)
            for label, items in selected_by_label.items()
        },
        "max_score_by_label": {
            str(label): max(x.score for x in items)
            for label, items in selected_by_label.items()
        },
    }
    return [x.index for x in selected], stats
