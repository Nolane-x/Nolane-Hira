from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .interpolation import multiset_recall, ordered_lcs_recall
from .structural import tokens


ATOMIC_GROUPS = (
    "relation",
    "representation",
    "coarse",
    "budget_control",
)

_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "relation": (
        "cross_attn.",
        "cross_ln.",
        "cross_ff.",
        "cross_score.",
        "delta_scale.",
    ),
    "representation": (
        "q_proj.",
        "seg_proj.",
        "opt_proj.",
        "token_proj.",
        "type_emb.",
        "state_ln.",
        "option_ln.",
        "option_token_weight.",
    ),
    "coarse": ("coarse_bias.",),
    "budget_control": ("budget_gate.",),
}

_GROUP_EXACT: dict[str, tuple[str, ...]] = {
    "relation": ("late_scale",),
    "representation": (),
    "coarse": ("coarse_scale",),
    "budget_control": (),
}


def atomic_group_for_key(key: str) -> str:
    matches: list[str] = []
    for group in ATOMIC_GROUPS:
        if key in _GROUP_EXACT[group] or any(
            key.startswith(prefix) for prefix in _GROUP_PREFIXES[group]
        ):
            matches.append(group)
    if len(matches) != 1:
        raise ValueError(f"state key {key!r} maps to {len(matches)} groups: {matches}")
    return matches[0]


def partition_state_keys(state: Mapping[str, Tensor]) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {group: [] for group in ATOMIC_GROUPS}
    for key in state:
        buckets[atomic_group_for_key(key)].append(key)
    return {group: tuple(sorted(keys)) for group, keys in buckets.items()}


def selected_keys_for_groups(
    state: Mapping[str, Tensor], groups: Sequence[str]
) -> tuple[str, ...]:
    unknown = sorted(set(groups) - set(ATOMIC_GROUPS))
    if unknown:
        raise ValueError(f"unknown atomic groups: {unknown}")
    wanted = set(groups)
    return tuple(
        key for key in state if atomic_group_for_key(key) in wanted
    )


def interpolate_selected_groups(
    base: Mapping[str, Tensor],
    direction: Mapping[str, Tensor],
    *,
    groups: Sequence[str],
    alpha: float,
) -> dict[str, Tensor]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if set(base) != set(direction):
        raise ValueError("state dict keys differ")
    selected = set(selected_keys_for_groups(base, groups))

    out: dict[str, Tensor] = {}
    for key in base:
        a = base[key]
        b = direction[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"state tensor mismatch for {key}")
        if key not in selected:
            out[key] = a.detach().cpu().clone()
            continue
        if torch.is_floating_point(a):
            out[key] = ((1.0 - alpha) * a + alpha * b).detach().cpu()
        else:
            if not torch.equal(a, b):
                raise ValueError(f"selected non-floating state differs for {key}")
            out[key] = a.detach().cpu().clone()
    return out


def unselected_keys_are_bit_identical(
    base: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    *,
    groups: Sequence[str],
) -> bool:
    selected = set(selected_keys_for_groups(base, groups))
    return all(
        key in candidate and torch.equal(base[key], candidate[key])
        for key in base
        if key not in selected
    )


@dataclass(frozen=True)
class RankedStructuralExample:
    index: int
    label: int
    multiset_recall: float
    ordered_lcs_recall: float

    @property
    def score(self) -> float:
        return max(self.multiset_recall, self.ordered_lcs_recall)

    @property
    def secondary_score(self) -> float:
        return min(self.multiset_recall, self.ordered_lcs_recall)


def balanced_ranked_structural_non_entailment_indices(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    labels: Sequence[int],
    *,
    per_label: int = 250,
    min_hypothesis_tokens: int = 3,
) -> tuple[list[int], dict[str, object]]:
    if per_label <= 0:
        raise ValueError("per_label must be positive")
    if min_hypothesis_tokens <= 0:
        raise ValueError("min_hypothesis_tokens must be positive")
    if not (len(premises) == len(hypotheses) == len(labels)):
        raise ValueError("premises/hypotheses/labels length mismatch")

    rows: dict[int, list[RankedStructuralExample]] = {1: [], 2: []}
    for i, raw_label in enumerate(labels):
        label = int(raw_label)
        if label not in rows:
            continue
        p = tokens(premises[i])
        h = tokens(hypotheses[i])
        if len(h) < min_hypothesis_tokens:
            continue
        rows[label].append(
            RankedStructuralExample(
                index=i,
                label=label,
                multiset_recall=multiset_recall(p, h),
                ordered_lcs_recall=ordered_lcs_recall(p, h),
            )
        )

    eligible = {label: len(items) for label, items in rows.items()}
    if any(n < per_label for n in eligible.values()):
        raise ValueError(
            f"not enough eligible ranked structural examples: {eligible}"
        )

    selected: list[RankedStructuralExample] = []
    selected_by_label: dict[int, list[RankedStructuralExample]] = {}
    for label in (1, 2):
        ranked = sorted(
            rows[label],
            key=lambda x: (-x.score, -x.secondary_score, x.index),
        )[:per_label]
        selected_by_label[label] = ranked
        selected.extend(ranked)
    selected.sort(key=lambda x: x.index)

    stats: dict[str, object] = {
        "eligible_by_label": {str(k): v for k, v in eligible.items()},
        "selected_per_label": int(per_label),
        "selected_total": len(selected),
        "min_hypothesis_tokens": int(min_hypothesis_tokens),
        "mean_score_by_label": {
            str(label): sum(x.score for x in items) / len(items)
            for label, items in selected_by_label.items()
        },
        "min_score_by_label": {
            str(label): min(x.score for x in items)
            for label, items in selected_by_label.items()
        },
        "mean_secondary_score_by_label": {
            str(label): sum(x.secondary_score for x in items) / len(items)
            for label, items in selected_by_label.items()
        },
    }
    return [x.index for x in selected], stats
