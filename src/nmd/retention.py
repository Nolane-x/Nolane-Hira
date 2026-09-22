from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Iterable

import torch
from torch import Tensor
import torch.nn.functional as F

from .hard_negative import anti_entailment_margin_loss, evaluate_repair_slice
from .hira import HIRACore
from .relation_cache import (
    RelationCache,
    forward_cached,
    minibatches,
    multiclass_brier,
)


@torch.no_grad()
def cached_teacher_probabilities(
    teacher: HIRACore,
    cache: RelationCache,
    *,
    batch_size: int = 128,
) -> Tensor:
    cache.validate()
    teacher.eval()
    rows = []
    for idx in minibatches(len(cache), batch_size, seed=0, epoch=0, shuffle=False):
        rows.append(forward_cached(teacher, cache, idx).probabilities.cpu())
    return torch.cat(rows)


def teacher_kl(student_logits: Tensor, teacher_probs: Tensor) -> Tensor:
    teacher_probs = teacher_probs / teacher_probs.sum(-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher_probs.to(student_logits.device, student_logits.dtype),
        reduction="batchmean",
    )


@dataclass(frozen=True)
class RetentionConfig:
    epochs: int = 5
    batch_size: int = 96
    lr: float = 2e-4
    brier_weight: float = 0.1
    margin: float = 0.5
    margin_weight: float = 0.5
    teacher_kl_weight: float = 0.5
    matched_accuracy_floor: float = 0.5606666612625122
    seed: int = 13


def train_retention_candidate(
    hira: HIRACore,
    hard_train: RelationCache,
    replay_train: RelationCache,
    replay_teacher_probs: Tensor,
    matched_validation: RelationCache,
    hard_validation: RelationCache,
    *,
    config: RetentionConfig,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    for cache in (hard_train, replay_train, matched_validation, hard_validation):
        cache.validate()
    option_ref = hard_train.option_embeddings
    for name, cache in (
        ("replay", replay_train),
        ("matched", matched_validation),
        ("hard_validation", hard_validation),
    ):
        if not torch.equal(option_ref, cache.option_embeddings):
            raise ValueError(f"hard/{name} option embeddings differ")
    if replay_teacher_probs.shape != (
        len(replay_train),
        replay_train.option_embeddings.shape[0],
    ):
        raise ValueError("replay teacher probabilities shape mismatch")

    baseline = {
        "matched": evaluate_repair_slice(
            hira, matched_validation, batch_size=config.batch_size
        ),
        "hard": evaluate_repair_slice(
            hira, hard_validation, batch_size=config.batch_size
        ),
    }

    optimizer = torch.optim.AdamW(hira.parameters(), lr=config.lr)
    history: list[dict[str, object]] = []
    best_state = None
    best_key = None
    fallback_state = None
    fallback_key = None

    for epoch in range(config.epochs):
        hira.train()
        hard_batches = list(
            minibatches(
                len(hard_train),
                config.batch_size,
                seed=config.seed,
                epoch=epoch,
                shuffle=True,
            )
        )
        replay_batches = list(
            minibatches(
                len(replay_train),
                config.batch_size,
                seed=config.seed + 10_000,
                epoch=epoch,
                shuffle=True,
            )
        )
        steps = max(len(hard_batches), len(replay_batches))
        hard_iter = cycle(hard_batches)
        replay_iter = cycle(replay_batches)

        totals = {
            "loss": 0.0,
            "hard_ce": 0.0,
            "hard_brier": 0.0,
            "hard_margin": 0.0,
            "replay_ce": 0.0,
            "replay_brier": 0.0,
            "teacher_kl": 0.0,
        }

        for _ in range(steps):
            hard_idx = next(hard_iter)
            replay_idx = next(replay_iter)
            optimizer.zero_grad(set_to_none=True)

            hard_out = forward_cached(hira, hard_train, hard_idx)
            hard_labels = hard_train.labels[hard_idx].long()
            hard_ce = F.cross_entropy(hard_out.logits, hard_labels)
            hard_brier = multiclass_brier(hard_out.probabilities, hard_labels)
            hard_margin = anti_entailment_margin_loss(
                hard_out.logits, hard_labels, margin=config.margin
            )
            hard_loss = (
                hard_ce
                + config.brier_weight * hard_brier
                + config.margin_weight * hard_margin
            )

            replay_out = forward_cached(hira, replay_train, replay_idx)
            replay_labels = replay_train.labels[replay_idx].long()
            replay_ce = F.cross_entropy(replay_out.logits, replay_labels)
            replay_brier = multiclass_brier(
                replay_out.probabilities, replay_labels
            )
            kl = teacher_kl(
                replay_out.logits,
                replay_teacher_probs[replay_idx],
            )
            replay_loss = replay_ce + config.brier_weight * replay_brier

            # Equal hard/replay sampling weight, plus an explicit retention anchor.
            loss = 0.5 * hard_loss + 0.5 * replay_loss + config.teacher_kl_weight * kl
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["hard_ce"] += float(hard_ce.detach())
            totals["hard_brier"] += float(hard_brier.detach())
            totals["hard_margin"] += float(hard_margin.detach())
            totals["replay_ce"] += float(replay_ce.detach())
            totals["replay_brier"] += float(replay_brier.detach())
            totals["teacher_kl"] += float(kl.detach())

        matched = evaluate_repair_slice(
            hira, matched_validation, batch_size=config.batch_size
        )
        hard = evaluate_repair_slice(
            hira, hard_validation, batch_size=config.batch_size
        )
        eligible = matched["accuracy"] >= config.matched_accuracy_floor
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "matched": matched,
            "hard": hard,
            "eligible": eligible,
            "train": {k: v / steps for k, v in totals.items()},
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

        if eligible:
            key = (
                float(hard["non_entailment_accuracy"]),
                float(matched["accuracy"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = snapshot

    selected_eligible = best_state is not None
    if best_state is None:
        if fallback_state is None:
            raise RuntimeError("retention training produced no checkpoint")
        best_state = fallback_state

    hira.load_state_dict(best_state)
    selected = {
        "matched": evaluate_repair_slice(
            hira, matched_validation, batch_size=config.batch_size
        ),
        "hard": evaluate_repair_slice(
            hira, hard_validation, batch_size=config.batch_size
        ),
    }
    return history, best_state, {
        "baseline": baseline,
        "selected": selected,
        "selected_eligible": selected_eligible,
    }
