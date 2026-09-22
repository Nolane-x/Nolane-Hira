from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from itertools import cycle
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .hard_negative import anti_entailment_margin_loss, evaluate_repair_slice
from .hira import HIRACore
from .relation_cache import RelationCache, forward_cached, minibatches, multiclass_brier
from .retention import teacher_kl


_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(str(text).lower()))


def is_contiguous_subsequence(premise: Sequence[str], hypothesis: Sequence[str]) -> bool:
    if not hypothesis or len(hypothesis) > len(premise):
        return False
    width = len(hypothesis)
    return any(tuple(premise[i:i + width]) == tuple(hypothesis) for i in range(len(premise) - width + 1))


def is_ordered_subsequence(premise: Sequence[str], hypothesis: Sequence[str]) -> bool:
    if not hypothesis:
        return False
    j = 0
    for tok in premise:
        if tok == hypothesis[j]:
            j += 1
            if j == len(hypothesis):
                return True
    return False


def is_multiset_subset(premise: Sequence[str], hypothesis: Sequence[str]) -> bool:
    if not hypothesis:
        return False
    p = Counter(premise)
    h = Counter(hypothesis)
    return all(p.get(tok, 0) >= n for tok, n in h.items())


def lexical_recall(premise: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not hypothesis:
        return 0.0
    p = Counter(premise)
    h = Counter(hypothesis)
    overlap = sum(min(n, p.get(tok, 0)) for tok, n in h.items())
    return overlap / sum(h.values())


@dataclass(frozen=True)
class StructuralFlags:
    contiguous: bool
    ordered: bool
    multiset_subset: bool
    lexical_recall: float

    @property
    def any_shortcut(self) -> bool:
        return self.contiguous or self.ordered or self.multiset_subset

    @property
    def strength(self) -> tuple[int, int, int, float]:
        return (
            int(self.contiguous),
            int(self.ordered),
            int(self.multiset_subset),
            float(self.lexical_recall),
        )


def structural_flags(premise: str, hypothesis: str) -> StructuralFlags:
    p = tokens(premise)
    h = tokens(hypothesis)
    return StructuralFlags(
        contiguous=is_contiguous_subsequence(p, h),
        ordered=is_ordered_subsequence(p, h),
        multiset_subset=is_multiset_subset(p, h),
        lexical_recall=lexical_recall(p, h),
    )


@dataclass(frozen=True)
class StructuralExample:
    index: int
    label: int
    flags: StructuralFlags


def balanced_structural_non_entailment_indices(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    labels: Sequence[int],
    *,
    max_per_label: int,
) -> tuple[list[int], dict[str, object]]:
    if max_per_label <= 0:
        raise ValueError("max_per_label must be positive")
    if not (len(premises) == len(hypotheses) == len(labels)):
        raise ValueError("premises/hypotheses/labels length mismatch")

    rows: dict[int, list[StructuralExample]] = {1: [], 2: []}
    for i, label_raw in enumerate(labels):
        label = int(label_raw)
        if label not in rows:
            continue
        flags = structural_flags(premises[i], hypotheses[i])
        if flags.any_shortcut:
            rows[label].append(StructuralExample(i, label, flags))

    available = {label: len(items) for label, items in rows.items()}
    per_label = min(max_per_label, min(available.values(), default=0))
    if per_label <= 0:
        raise ValueError("no balanced structural non-entailment examples available")

    selected: list[StructuralExample] = []
    for label in (1, 2):
        items = sorted(
            rows[label],
            key=lambda x: (
                -x.flags.strength[0],
                -x.flags.strength[1],
                -x.flags.strength[2],
                -x.flags.strength[3],
                x.index,
            ),
        )[:per_label]
        selected.extend(items)

    selected.sort(key=lambda x: x.index)
    stats = {
        "available_by_label": {str(k): v for k, v in available.items()},
        "selected_per_label": per_label,
        "selected_total": len(selected),
        "predicate_counts": {
            "contiguous": sum(x.flags.contiguous for x in selected),
            "ordered": sum(x.flags.ordered for x in selected),
            "multiset_subset": sum(x.flags.multiset_subset for x in selected),
        },
        "mean_lexical_recall": (
            sum(x.flags.lexical_recall for x in selected) / len(selected)
            if selected else 0.0
        ),
    }
    return [x.index for x in selected], stats


@dataclass(frozen=True)
class StructuralRetentionConfig:
    epochs: int = 4
    batch_size: int = 96
    lr: float = 1e-4
    hard_replay_ratio: int = 1
    brier_weight: float = 0.1
    margin: float = 0.5
    margin_weight: float = 0.5
    teacher_kl_weight: float = 0.5
    matched_accuracy_floor: float = 0.5606666612625122
    seed: int = 13


def train_structural_candidate(
    hira: HIRACore,
    structural_train: RelationCache,
    replay_train: RelationCache,
    replay_teacher_probs: Tensor,
    matched_validation: RelationCache,
    structural_validation: RelationCache,
    *,
    config: StructuralRetentionConfig,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    for cache in (structural_train, replay_train, matched_validation, structural_validation):
        cache.validate()
    if config.hard_replay_ratio not in (1, 2):
        raise ValueError("R16 hard_replay_ratio must be 1 or 2")
    ref = structural_train.option_embeddings
    for name, cache in (
        ("replay", replay_train),
        ("matched", matched_validation),
        ("structural_validation", structural_validation),
    ):
        if not torch.equal(ref, cache.option_embeddings):
            raise ValueError(f"structural/{name} option embeddings differ")
    if replay_teacher_probs.shape != (
        len(replay_train),
        replay_train.option_embeddings.shape[0],
    ):
        raise ValueError("replay teacher probabilities shape mismatch")

    baseline = {
        "matched": evaluate_repair_slice(hira, matched_validation, batch_size=config.batch_size),
        "structural": evaluate_repair_slice(hira, structural_validation, batch_size=config.batch_size),
    }

    optimizer = torch.optim.AdamW(hira.parameters(), lr=config.lr)
    history: list[dict[str, object]] = []
    best_state = None
    best_key = None
    fallback_state = None
    fallback_key = None

    for epoch in range(config.epochs):
        hira.train()
        hard_batches = list(minibatches(
            len(structural_train), config.batch_size,
            seed=config.seed, epoch=epoch, shuffle=True,
        ))
        replay_batches = list(minibatches(
            len(replay_train), config.batch_size,
            seed=config.seed + 10_000, epoch=epoch, shuffle=True,
        ))
        steps = max(
            (len(hard_batches) + config.hard_replay_ratio - 1) // config.hard_replay_ratio,
            len(replay_batches),
        )
        hard_iter = cycle(hard_batches)
        replay_iter = cycle(replay_batches)
        totals = {
            "loss": 0.0,
            "structural_ce": 0.0,
            "structural_brier": 0.0,
            "structural_margin": 0.0,
            "replay_ce": 0.0,
            "replay_brier": 0.0,
            "teacher_kl": 0.0,
        }

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)

            hard_loss_sum = None
            hard_ce_sum = 0.0
            hard_brier_sum = 0.0
            hard_margin_sum = 0.0
            for _hard in range(config.hard_replay_ratio):
                idx = next(hard_iter)
                out = forward_cached(hira, structural_train, idx)
                labels = structural_train.labels[idx].long()
                ce = F.cross_entropy(out.logits, labels)
                brier = multiclass_brier(out.probabilities, labels)
                margin = anti_entailment_margin_loss(out.logits, labels, margin=config.margin)
                component = ce + config.brier_weight * brier + config.margin_weight * margin
                hard_loss_sum = component if hard_loss_sum is None else hard_loss_sum + component
                hard_ce_sum += float(ce.detach())
                hard_brier_sum += float(brier.detach())
                hard_margin_sum += float(margin.detach())
            hard_loss = hard_loss_sum / config.hard_replay_ratio

            replay_idx = next(replay_iter)
            replay_out = forward_cached(hira, replay_train, replay_idx)
            replay_labels = replay_train.labels[replay_idx].long()
            replay_ce = F.cross_entropy(replay_out.logits, replay_labels)
            replay_brier = multiclass_brier(replay_out.probabilities, replay_labels)
            kl = teacher_kl(replay_out.logits, replay_teacher_probs[replay_idx])
            replay_loss = replay_ce + config.brier_weight * replay_brier

            hard_weight = config.hard_replay_ratio / (config.hard_replay_ratio + 1.0)
            replay_weight = 1.0 / (config.hard_replay_ratio + 1.0)
            loss = (
                hard_weight * hard_loss
                + replay_weight * replay_loss
                + config.teacher_kl_weight * kl
            )
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["structural_ce"] += hard_ce_sum / config.hard_replay_ratio
            totals["structural_brier"] += hard_brier_sum / config.hard_replay_ratio
            totals["structural_margin"] += hard_margin_sum / config.hard_replay_ratio
            totals["replay_ce"] += float(replay_ce.detach())
            totals["replay_brier"] += float(replay_brier.detach())
            totals["teacher_kl"] += float(kl.detach())

        matched = evaluate_repair_slice(hira, matched_validation, batch_size=config.batch_size)
        structural = evaluate_repair_slice(hira, structural_validation, batch_size=config.batch_size)
        eligible = matched["accuracy"] >= config.matched_accuracy_floor
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "matched": matched,
            "structural": structural,
            "eligible": eligible,
            "train": {k: v / steps for k, v in totals.items()},
        }
        history.append(row)

        snapshot = {k: v.detach().cpu().clone() for k, v in hira.state_dict().items()}
        fb_key = (float(matched["accuracy"]), float(structural["non_entailment_accuracy"]))
        if fallback_key is None or fb_key > fallback_key:
            fallback_key = fb_key
            fallback_state = snapshot

        if eligible:
            key = (
                float(structural["non_entailment_accuracy"]),
                float(matched["accuracy"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = snapshot

    selected_eligible = best_state is not None
    if best_state is None:
        if fallback_state is None:
            raise RuntimeError("structural training produced no checkpoint")
        best_state = fallback_state

    hira.load_state_dict(best_state)
    selected = {
        "matched": evaluate_repair_slice(hira, matched_validation, batch_size=config.batch_size),
        "structural": evaluate_repair_slice(hira, structural_validation, batch_size=config.batch_size),
    }
    return history, best_state, {
        "baseline": baseline,
        "selected": selected,
        "selected_eligible": selected_eligible,
    }
