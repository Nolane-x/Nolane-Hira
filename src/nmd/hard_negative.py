from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .hira import HIRACore
from .relation_cache import (
    RelationCache,
    evaluate_cached,
    forward_cached,
    minibatches,
    multiclass_brier,
)


_TOKEN = re.compile(r"[a-z0-9]+")


def lexical_overlap_score(premise: str, hypothesis: str) -> float:
    """Multiset recall of hypothesis tokens found in the premise.

    This is deliberately simple and frozen before R14 training. It is not HANS-aware
    and does not inspect any held-out stress example.
    """
    p = Counter(_TOKEN.findall(str(premise).lower()))
    h = Counter(_TOKEN.findall(str(hypothesis).lower()))
    total = sum(h.values())
    if total == 0:
        return 0.0
    overlap = sum(min(count, p.get(tok, 0)) for tok, count in h.items())
    return overlap / total


@dataclass(frozen=True)
class RankedExample:
    index: int
    label: int
    overlap: float


def rank_by_overlap(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    labels: Sequence[int],
) -> list[RankedExample]:
    if not (len(premises) == len(hypotheses) == len(labels)):
        raise ValueError("premises/hypotheses/labels length mismatch")
    rows = [
        RankedExample(i, int(label), lexical_overlap_score(premises[i], hypotheses[i]))
        for i, label in enumerate(labels)
        if int(label) in (0, 1, 2)
    ]
    rows.sort(key=lambda x: (-x.overlap, x.index))
    return rows


def balanced_top_indices(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    labels: Sequence[int],
    *,
    per_label: int,
) -> tuple[list[int], dict[str, object]]:
    if per_label <= 0:
        raise ValueError("per_label must be positive")
    ranked = rank_by_overlap(premises, hypotheses, labels)
    chosen: dict[int, list[RankedExample]] = {0: [], 1: [], 2: []}
    for row in ranked:
        bucket = chosen[row.label]
        if len(bucket) < per_label:
            bucket.append(row)
        if all(len(v) >= per_label for v in chosen.values()):
            break
    if any(len(v) < per_label for v in chosen.values()):
        raise ValueError("not enough examples for balanced overlap curriculum")

    selected = [x for label in (0, 1, 2) for x in chosen[label]]
    selected.sort(key=lambda x: x.index)
    stats = {
        "per_label": per_label,
        "total": len(selected),
        "mean_overlap_by_label": {
            str(label): sum(x.overlap for x in rows) / len(rows)
            for label, rows in chosen.items()
        },
        "min_overlap_by_label": {
            str(label): min(x.overlap for x in rows)
            for label, rows in chosen.items()
        },
        "max_overlap_by_label": {
            str(label): max(x.overlap for x in rows)
            for label, rows in chosen.items()
        },
    }
    return [x.index for x in selected], stats


def top_non_entailment_indices(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    labels: Sequence[int],
    *,
    per_label: int,
) -> tuple[list[int], dict[str, object]]:
    if per_label <= 0:
        raise ValueError("per_label must be positive")
    ranked = rank_by_overlap(premises, hypotheses, labels)
    chosen: dict[int, list[RankedExample]] = {1: [], 2: []}
    for row in ranked:
        if row.label in chosen and len(chosen[row.label]) < per_label:
            chosen[row.label].append(row)
        if all(len(v) >= per_label for v in chosen.values()):
            break
    if any(len(v) < per_label for v in chosen.values()):
        raise ValueError("not enough non-entailment examples")
    selected = chosen[1] + chosen[2]
    selected.sort(key=lambda x: x.index)
    stats = {
        "per_label": per_label,
        "total": len(selected),
        "mean_overlap_by_label": {
            str(label): sum(x.overlap for x in rows) / len(rows)
            for label, rows in chosen.items()
        },
        "min_overlap_by_label": {
            str(label): min(x.overlap for x in rows)
            for label, rows in chosen.items()
        },
    }
    return [x.index for x in selected], stats


@torch.no_grad()
def evaluate_repair_slice(
    hira: HIRACore,
    cache: RelationCache,
    *,
    batch_size: int = 128,
) -> dict[str, float]:
    metrics = evaluate_cached(hira, cache, batch_size=batch_size)
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}
    for idx in minibatches(len(cache), batch_size, seed=0, epoch=0, shuffle=False):
        out = forward_cached(hira, cache, idx)
        labels = cache.labels[idx].long()
        pred = out.probabilities.argmax(-1)
        for label in (0, 1, 2):
            mask = labels.eq(label)
            total[label] += int(mask.sum())
            correct[label] += int(pred[mask].eq(labels[mask]).sum())
    for label in (0, 1, 2):
        metrics[f"recall_{label}"] = (
            correct[label] / total[label] if total[label] else float("nan")
        )
    ne_total = total[1] + total[2]
    ne_correct = correct[1] + correct[2]
    metrics["non_entailment_accuracy"] = ne_correct / ne_total if ne_total else float("nan")
    return metrics


def anti_entailment_margin_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    margin: float = 0.5,
) -> Tensor:
    """For neutral/contradiction golds, force gold logit above entailment logit."""
    mask = labels.ne(0)
    if not mask.any():
        return logits.sum() * 0.0
    gold = logits[mask].gather(1, labels[mask].unsqueeze(1)).squeeze(1)
    entail = logits[mask, 0]
    return F.relu(float(margin) - gold + entail).mean()


def train_hard_negative_repair(
    hira: HIRACore,
    train: RelationCache,
    matched_validation: RelationCache,
    hard_validation: RelationCache,
    *,
    epochs: int = 6,
    batch_size: int = 96,
    lr: float = 5e-4,
    seed: int = 13,
    brier_weight: float = 0.1,
    margin: float = 0.5,
    margin_weight: float = 0.5,
    matched_accuracy_floor: float = 0.5606666612625122,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, float]]:
    train.validate()
    matched_validation.validate()
    hard_validation.validate()
    if not torch.equal(train.option_embeddings, matched_validation.option_embeddings):
        raise ValueError("train/matched option embeddings differ")
    if not torch.equal(train.option_embeddings, hard_validation.option_embeddings):
        raise ValueError("train/hard option embeddings differ")

    baseline = {
        "matched": evaluate_repair_slice(hira, matched_validation, batch_size=batch_size),
        "hard": evaluate_repair_slice(hira, hard_validation, batch_size=batch_size),
    }

    optimizer = torch.optim.AdamW(hira.parameters(), lr=lr)
    history: list[dict[str, object]] = []
    best_state = None
    best_key = None
    fallback_state = None
    fallback_key = None

    for epoch in range(int(epochs)):
        hira.train()
        loss_sum = 0.0
        ce_sum = 0.0
        brier_sum = 0.0
        margin_sum = 0.0
        seen = 0

        for idx in minibatches(len(train), batch_size, seed=seed, epoch=epoch, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            out = forward_cached(hira, train, idx)
            labels = train.labels[idx].long()
            ce = F.cross_entropy(out.logits, labels)
            brier = multiclass_brier(out.probabilities, labels)
            margin_loss = anti_entailment_margin_loss(out.logits, labels, margin=margin)
            loss = ce + float(brier_weight) * brier + float(margin_weight) * margin_loss
            loss.backward()
            optimizer.step()

            n = len(idx)
            seen += n
            loss_sum += float(loss.detach()) * n
            ce_sum += float(ce.detach()) * n
            brier_sum += float(brier.detach()) * n
            margin_sum += float(margin_loss.detach()) * n

        matched = evaluate_repair_slice(hira, matched_validation, batch_size=batch_size)
        hard = evaluate_repair_slice(hira, hard_validation, batch_size=batch_size)
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(1, seen),
            "train_ce": ce_sum / max(1, seen),
            "train_brier": brier_sum / max(1, seen),
            "train_margin": margin_sum / max(1, seen),
            "matched": matched,
            "hard": hard,
            "eligible": matched["accuracy"] >= matched_accuracy_floor,
        }
        history.append(row)

        snapshot = {
            k: v.detach().cpu().clone()
            for k, v in hira.state_dict().items()
        }
        fallback_candidate = (
            float(matched["accuracy"]),
            float(hard["non_entailment_accuracy"]),
        )
        if fallback_key is None or fallback_candidate > fallback_key:
            fallback_key = fallback_candidate
            fallback_state = snapshot

        if row["eligible"]:
            key = (
                float(hard["non_entailment_accuracy"]),
                float(hard["accuracy"]),
                float(matched["accuracy"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = snapshot

    selected_eligible = best_state is not None
    if best_state is None:
        if fallback_state is None:
            raise RuntimeError("repair training produced no checkpoint")
        best_state = fallback_state
    hira.load_state_dict(best_state)
    selected = {
        "matched": evaluate_repair_slice(hira, matched_validation, batch_size=batch_size),
        "hard": evaluate_repair_slice(hira, hard_validation, batch_size=batch_size),
    }
    return history, best_state, {
        "baseline": baseline,
        "selected": selected,
        "selected_eligible": selected_eligible,
    }
