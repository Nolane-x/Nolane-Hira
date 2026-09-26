from __future__ import annotations

from collections import defaultdict
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .atomic_geometry_eval import (
    _batches,
    _factor_logits,
    _prediction_row,
    _selection_key,
    _summarize_factor_predictions,
)
from .compositional_projection_cache import validate_w28_cache
from .compositional_projection_authority import FACTOR_IDS

PROJECTION_TEMPERATURE = 0.07
TRAIN_BATCH_SIZE = 32
PROJECTION_EPOCHS = 8
PROJECTION_LR = 3e-4
PROJECTION_WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
CANDIDATE_SEEDS = {"T0": 2519, "T1": 2539}


def evaluate_projection(cache: dict, projection: Tensor) -> dict[str, object]:
    validate_w28_cache(cache)
    projection = projection.float()
    raw = []
    for case in cache["cases"]:
        pack = cache["schemas"][str(case["domain_id"])]
        state = case["representations"]["severity_tokens"].float()
        logits = {
            factor_id: _factor_logits(
                state,
                pack[f"factor_{factor_id.lower()}"],
                projection,
            )
            for factor_id in FACTOR_IDS
        }
        raw.append(_prediction_row(case, logits))
    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    return {
        "per_domain": {
            domain: _summarize_factor_predictions(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": _summarize_factor_predictions(raw),
        "predictions": {str(row["case_id"]): row for row in raw},
    }


def _loss_for_case(case, pack, projection: Tensor) -> Tensor:
    state = case["representations"]["severity_tokens"].float()
    gold_vector = tuple(int(x) for x in case["factor_vector"])
    losses = []
    for index, factor_id in enumerate(FACTOR_IDS):
        logits = _factor_logits(
            state,
            pack[f"factor_{factor_id.lower()}"],
            projection,
        )
        target = torch.tensor([gold_vector[index]], dtype=torch.long)
        losses.append(
            F.cross_entropy(
                (logits / PROJECTION_TEMPERATURE).unsqueeze(0),
                target,
            )
        )
    return torch.stack(losses).sum()


def train_projection(
    candidate: str,
    train_cache: dict,
    dev_cache: dict,
    w9_projection: Tensor,
):
    if candidate not in CANDIDATE_SEEDS:
        raise ValueError("W28 candidate must be T0 or T1")
    validate_w28_cache(train_cache, expected_partition="train")
    validate_w28_cache(dev_cache, expected_partition="dev")
    if tuple(w9_projection.shape) != (128, 256):
        raise ValueError("W28 frozen W9 projection shape changed")

    seed = CANDIDATE_SEEDS[candidate]
    torch.manual_seed(seed)
    projection = nn.Parameter(w9_projection.detach().float().clone())
    optimizer = torch.optim.AdamW(
        [projection],
        lr=PROJECTION_LR,
        weight_decay=PROJECTION_WEIGHT_DECAY,
    )
    cases = train_cache["cases"]
    history = []
    best_key = None
    best_projection = None
    best_receipt = None

    for epoch in range(1, PROJECTION_EPOCHS + 1):
        running = 0.0
        steps = 0
        for batch_indices in _batches(
            list(range(len(cases))),
            seed=seed,
            epoch=epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for idx in batch_indices:
                case = cases[idx]
                pack = train_cache["schemas"][str(case["domain_id"])]
                batch_losses.append(_loss_for_case(case, pack, projection))
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([projection], GRAD_CLIP)
            optimizer.step()
            running += float(loss.detach()) * len(batch_indices)
            steps += 1

        train_loss = running / len(cases)
        dev_eval = evaluate_projection(dev_cache, projection.detach())
        key = _selection_key(dev_eval["pooled"], epoch, train_loss)
        row = {
            "epoch": epoch,
            "train_factor_ce": train_loss,
            "dev": dev_eval["pooled"],
            "selection_key": list(key),
            "optimizer_steps_this_epoch": steps,
        }
        history.append(row)
        if best_key is None or key > best_key:
            best_key = key
            best_projection = projection.detach().cpu().clone()
            best_receipt = dict(row)

    assert best_projection is not None and best_receipt is not None
    return best_projection, best_receipt, history


def primary_rescue(row, p0) -> bool:
    factors = row["factors"]
    return bool(
        all(float(factors[f]["top1"]) >= 0.85 for f in FACTOR_IDS)
        and all(float(factors[f]["balanced_accuracy"]) >= 0.82 for f in FACTOR_IDS)
        and float(row["factor_vector_top1"]) >= 0.70
        and float(row["composed_severity_top1"]) >= 0.75
        and float(row["invalid_factor_vector_rate"]) <= 0.12
        and float(row["composed_severity_top1"]) - float(p0["composed_severity_top1"]) >= 0.25
        and float(row["factor_probability_mass_max_error"]) <= 1e-6
    )


def replica_rescue(row, p0) -> bool:
    factors = row["factors"]
    return bool(
        all(float(factors[f]["top1"]) >= 0.82 for f in FACTOR_IDS)
        and all(float(factors[f]["balanced_accuracy"]) >= 0.80 for f in FACTOR_IDS)
        and float(row["factor_vector_top1"]) >= 0.65
        and float(row["composed_severity_top1"]) >= 0.70
        and float(row["invalid_factor_vector_rate"]) <= 0.15
        and float(row["composed_severity_top1"]) - float(p0["composed_severity_top1"]) >= 0.20
        and float(row["factor_probability_mass_max_error"]) <= 1e-6
    )


__all__ = [
    "CANDIDATE_SEEDS",
    "GRAD_CLIP",
    "PROJECTION_EPOCHS",
    "PROJECTION_LR",
    "PROJECTION_TEMPERATURE",
    "PROJECTION_WEIGHT_DECAY",
    "TRAIN_BATCH_SIZE",
    "evaluate_projection",
    "primary_rescue",
    "replica_rescue",
    "train_projection",
]
