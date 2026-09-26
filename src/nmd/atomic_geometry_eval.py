from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
import random
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .atomic_geometry_authority import FACTOR_IDS, VIEW_IDS, compose_severity
from .atomic_geometry_cache import validate_w25_cache
from .interface_decomposition_eval import _q0_symmetric

PROJECTION_TEMPERATURE = 0.07
TRAIN_BATCH_SIZE = 32
PROJECTION_EPOCHS = 8
PROBE_EPOCHS = 20
PROJECTION_LR = 3e-4
PROJECTION_WEIGHT_DECAY = 0.01
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

CANDIDATES = ("Q0", "Q1", "T0", "T1")
CANDIDATE_SEEDS = {"Q0": 2609, "Q1": 2617, "T0": 2519, "T1": 2539}


def candidate_seed(candidate: str) -> int:
    try:
        return CANDIDATE_SEEDS[candidate]
    except KeyError as exc:
        raise ValueError(f"unknown W25 candidate: {candidate}") from exc


def _projection_for_raw() -> Tensor:
    return torch.eye(256, dtype=torch.float32)


def _factor_logits(
    state_tokens: Tensor,
    views: Mapping[str, Mapping[str, object]],
    projection: Tensor,
) -> Tensor:
    rows = []
    for view_id in VIEW_IDS:
        view = views[view_id]
        rows.append(
            _q0_symmetric(
                state_tokens.float(),
                view["option_tokens"].float(),
                view["option_content_mask"].bool(),
                projection,
            )
        )
    return torch.stack(rows, dim=0).mean(dim=0)


def _record(logits: Tensor, gold: int) -> dict[str, object]:
    logits = logits.float()
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W25 requires finite 1-D logits")
    rank, mrr, _ = _rank_metrics(logits, int(gold))
    pred = int(logits.argmax())
    probs = torch.softmax(logits, dim=0)
    return {
        "gold": int(gold),
        "pred": pred,
        "correct": pred == int(gold),
        "rank": int(rank),
        "mrr": float(mrr),
        "margin": float(_margin(logits, int(gold))),
        "probability_mass_error": abs(float(probs.sum()) - 1.0),
    }


def _binary_balanced_accuracy(rows: Sequence[Mapping[str, object]]) -> float:
    recalls = []
    for gold in (0, 1):
        subset = [row for row in rows if int(row["gold"]) == gold]
        if not subset:
            return 0.0
        recalls.append(sum(bool(row["correct"]) for row in subset) / len(subset))
    return sum(recalls) / 2.0


def _factor_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "top1": 0.0,
            "balanced_accuracy": 0.0,
            "mrr": 0.0,
            "mean_margin": 0.0,
        }
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / n,
        "balanced_accuracy": _binary_balanced_accuracy(rows),
        "mrr": sum(float(row["mrr"]) for row in rows) / n,
        "mean_margin": sum(float(row["margin"]) for row in rows) / n,
    }


def _summarize_factor_predictions(raw: Sequence[Mapping[str, object]]) -> dict[str, object]:
    n = len(raw)
    factor_rows = {
        factor_id: [row["factors"][factor_id] for row in raw]
        for factor_id in FACTOR_IDS
    }
    vector_top1 = sum(bool(row["vector_correct"]) for row in raw) / max(1, n)
    invalid_rate = sum(bool(row["invalid"]) for row in raw) / max(1, n)
    composed_top1 = sum(bool(row["composed_correct"]) for row in raw) / max(1, n)
    composed_mae = sum(float(row["composed_mae"]) for row in raw) / max(1, n)
    return {
        "case_count": n,
        "factors": {
            factor_id: _factor_summary(factor_rows[factor_id])
            for factor_id in FACTOR_IDS
        },
        "factor_vector_top1": vector_top1,
        "invalid_factor_vector_rate": invalid_rate,
        "composed_severity_top1": composed_top1,
        "composed_severity_mae": composed_mae,
        "factor_probability_mass_max_error": max(
            [
                max(float(v["probability_mass_error"]) for v in row["factors"].values())
                for row in raw
            ]
            or [0.0]
        ),
    }


def _prediction_row(
    case: Mapping[str, object],
    logits_by_factor: Mapping[str, Tensor],
) -> dict[str, object]:
    gold_vector = tuple(int(x) for x in case["factor_vector"])
    factors = {
        factor_id: _record(logits_by_factor[factor_id], gold_vector[index])
        for index, factor_id in enumerate(FACTOR_IDS)
    }
    pred_vector = tuple(int(factors[factor_id]["pred"]) for factor_id in FACTOR_IDS)
    composed = compose_severity(pred_vector)
    gold_severity = int(case["severity"])
    return {
        "case_id": str(case["case_id"]),
        "domain_id": str(case["domain_id"]),
        "severity_gold": gold_severity,
        "factor_gold": gold_vector,
        "factors": factors,
        "factor_pred_vector": pred_vector,
        "vector_correct": pred_vector == gold_vector,
        "composed_severity": composed,
        "composed_correct": composed == gold_severity if composed is not None else False,
        "composed_mae": abs(int(composed) - gold_severity) if composed is not None else 3,
        "invalid": composed is None,
    }


def evaluate_semantic_projection(cache: dict, projection: Tensor) -> dict[str, object]:
    validate_w25_cache(cache)
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

    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
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


def evaluate_raw_a13(cache: dict) -> dict[str, object]:
    return evaluate_semantic_projection(cache, _projection_for_raw())


def _pooled_feature(tokens: Tensor, projection: Tensor | None) -> Tensor:
    x = tokens.float()
    if projection is not None:
        x = F.normalize(F.linear(x, projection.float()), dim=-1)
    pooled = x.mean(dim=0)
    return F.normalize(pooled, dim=0)


class AtomicFactorProbe(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.head = nn.Linear(int(d_in), len(FACTOR_IDS), bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


def probe_parameter_count(probe: nn.Module) -> int:
    return sum(p.numel() for p in probe.parameters())


def _probe_rows(
    probe: AtomicFactorProbe,
    cache: dict,
    projection: Tensor | None,
) -> list[dict[str, object]]:
    rows = []
    probe.eval()
    with torch.inference_mode():
        for case in cache["cases"]:
            feature = _pooled_feature(
                case["representations"]["severity_tokens"],
                projection,
            )
            logits3 = probe(feature)
            factor_logits = {
                factor_id: torch.stack((-logits3[index], logits3[index]))
                for index, factor_id in enumerate(FACTOR_IDS)
            }
            row = _prediction_row(case, factor_logits)
            target = torch.tensor(case["factor_vector"], dtype=torch.float32)
            row["probe_bce"] = float(
                F.binary_cross_entropy_with_logits(logits3.float(), target).detach()
            )
            rows.append(row)
    return rows


def _summarize_probe(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    value = _summarize_factor_predictions(rows)
    value["bce"] = (
        sum(float(row["probe_bce"]) for row in rows) / len(rows)
        if rows else 0.0
    )
    return value


def evaluate_probe(
    probe: AtomicFactorProbe,
    cache: dict,
    projection: Tensor | None,
) -> dict[str, object]:
    validate_w25_cache(cache)
    raw = _probe_rows(probe, cache, projection)
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    return {
        "per_domain": {
            domain: _summarize_probe(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": _summarize_probe(raw),
        "predictions": {str(row["case_id"]): row for row in raw},
    }


def _selection_key(summary: Mapping[str, object], epoch: int, loss: float) -> tuple:
    factors = summary["factors"]
    min_ba = min(float(factors[f]["balanced_accuracy"]) for f in FACTOR_IDS)
    return (
        float(summary["composed_severity_top1"]),
        float(summary["factor_vector_top1"]),
        min_ba,
        -float(summary["invalid_factor_vector_rate"]),
        -float(loss),
        -int(epoch),
    )


def _batches(indices: list[int], *, seed: int, epoch: int) -> list[list[int]]:
    order = list(indices)
    rng = random.Random(seed + epoch * 10007)
    rng.shuffle(order)
    return [order[i : i + TRAIN_BATCH_SIZE] for i in range(0, len(order), TRAIN_BATCH_SIZE)]


def train_probe(
    candidate: str,
    train_cache: dict,
    dev_cache: dict,
    w9_projection: Tensor,
) -> tuple[dict[str, Tensor], dict[str, object], list[dict[str, object]]]:
    if candidate not in {"Q0", "Q1"}:
        raise ValueError("train_probe only supports Q0/Q1")
    validate_w25_cache(train_cache, expected_partition="train")
    validate_w25_cache(dev_cache, expected_partition="dev")

    seed = candidate_seed(candidate)
    torch.manual_seed(seed)
    projection = None if candidate == "Q0" else w9_projection.detach().float()
    d_in = 256 if projection is None else 128
    probe = AtomicFactorProbe(d_in)
    expected = 771 if candidate == "Q0" else 387
    if probe_parameter_count(probe) != expected:
        raise RuntimeError("W25 probe parameter count changed")

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=PROBE_LR,
        weight_decay=PROBE_WEIGHT_DECAY,
    )
    cases = train_cache["cases"]
    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_receipt = None

    for epoch in range(1, PROBE_EPOCHS + 1):
        probe.train()
        running = 0.0
        steps = 0
        for batch_indices in _batches(list(range(len(cases))), seed=seed, epoch=epoch):
            features = []
            labels = []
            for idx in batch_indices:
                case = cases[idx]
                features.append(
                    _pooled_feature(
                        case["representations"]["severity_tokens"],
                        projection,
                    )
                )
                labels.append(torch.tensor(case["factor_vector"], dtype=torch.float32))
            x = torch.stack(features)
            y = torch.stack(labels)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), GRAD_CLIP)
            optimizer.step()
            running += float(loss.detach()) * len(batch_indices)
            steps += 1

        train_loss = running / len(cases)
        dev_eval = evaluate_probe(probe, dev_cache, projection)
        key = _selection_key(dev_eval["pooled"], epoch, train_loss)
        row = {
            "epoch": epoch,
            "train_bce": train_loss,
            "dev": dev_eval["pooled"],
            "selection_key": list(key),
            "optimizer_steps_this_epoch": steps,
        }
        history.append(row)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in probe.state_dict().items()
            }
            best_receipt = deepcopy(row)

    assert best_state is not None and best_receipt is not None
    return best_state, best_receipt, history


def _projection_loss_for_case(
    case: Mapping[str, object],
    pack: Mapping[str, object],
    projection: Tensor,
) -> Tensor:
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
            F.cross_entropy((logits / PROJECTION_TEMPERATURE).unsqueeze(0), target)
        )
    return torch.stack(losses).sum()


def train_projection(
    candidate: str,
    train_cache: dict,
    dev_cache: dict,
    w9_projection: Tensor,
) -> tuple[Tensor, dict[str, object], list[dict[str, object]]]:
    if candidate not in {"T0", "T1"}:
        raise ValueError("train_projection only supports T0/T1")
    validate_w25_cache(train_cache, expected_partition="train")
    validate_w25_cache(dev_cache, expected_partition="dev")
    if tuple(w9_projection.shape) != (128, 256):
        raise ValueError("W25 W9 projection shape changed")

    seed = candidate_seed(candidate)
    torch.manual_seed(seed)
    projection = nn.Parameter(w9_projection.detach().float().clone())
    optimizer = torch.optim.AdamW(
        [projection],
        lr=PROJECTION_LR,
        weight_decay=PROJECTION_WEIGHT_DECAY,
    )
    cases = train_cache["cases"]
    history: list[dict[str, object]] = []
    best_key = None
    best_projection = None
    best_receipt = None

    for epoch in range(1, PROJECTION_EPOCHS + 1):
        running = 0.0
        steps = 0
        for batch_indices in _batches(list(range(len(cases))), seed=seed, epoch=epoch):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for idx in batch_indices:
                case = cases[idx]
                pack = train_cache["schemas"][str(case["domain_id"])]
                batch_losses.append(
                    _projection_loss_for_case(case, pack, projection)
                )
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([projection], GRAD_CLIP)
            optimizer.step()
            running += float(loss.detach()) * len(batch_indices)
            steps += 1

        train_loss = running / len(cases)
        dev_eval = evaluate_semantic_projection(dev_cache, projection.detach())
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
            best_receipt = deepcopy(row)

    assert best_projection is not None and best_receipt is not None
    return best_projection, best_receipt, history


def _adequate_probe(row: Mapping[str, object]) -> bool:
    factors = row["factors"]
    return bool(
        all(float(factors[f]["top1"]) >= 0.90 for f in FACTOR_IDS)
        and all(float(factors[f]["balanced_accuracy"]) >= 0.88 for f in FACTOR_IDS)
        and float(row["factor_vector_top1"]) >= 0.80
        and float(row["composed_severity_top1"]) >= 0.85
        and float(row["invalid_factor_vector_rate"]) <= 0.08
    )


def _primary_rescue(row: Mapping[str, object], p0: Mapping[str, object]) -> bool:
    factors = row["factors"]
    return bool(
        all(float(factors[f]["top1"]) >= 0.85 for f in FACTOR_IDS)
        and all(float(factors[f]["balanced_accuracy"]) >= 0.82 for f in FACTOR_IDS)
        and float(row["factor_vector_top1"]) >= 0.70
        and float(row["composed_severity_top1"]) >= 0.75
        and float(row["invalid_factor_vector_rate"]) <= 0.12
        and float(row["composed_severity_top1"]) - float(p0["composed_severity_top1"]) >= 0.25
    )


def _replica_rescue(row: Mapping[str, object], p0: Mapping[str, object]) -> bool:
    factors = row["factors"]
    return bool(
        all(float(factors[f]["top1"]) >= 0.82 for f in FACTOR_IDS)
        and float(row["factor_vector_top1"]) >= 0.65
        and float(row["composed_severity_top1"]) >= 0.70
        and float(row["invalid_factor_vector_rate"]) <= 0.15
        and float(row["composed_severity_top1"]) - float(p0["composed_severity_top1"]) >= 0.20
    )


def reference_adequate(row: Mapping[str, object]) -> bool:
    factors = row["factors"]
    return bool(
        all(float(factors[f]["top1"]) >= 0.92 for f in FACTOR_IDS)
        and all(float(factors[f]["balanced_accuracy"]) >= 0.90 for f in FACTOR_IDS)
        and float(row["factor_vector_top1"]) >= 0.85
        and float(row["composed_severity_top1"]) >= 0.85
        and float(row["invalid_factor_vector_rate"]) <= 0.05
        and float(row["factor_probability_mass_max_error"]) <= 1e-6
    )


def classify_w25(
    evaluations: Mapping[str, Mapping[str, object]],
    reference: Mapping[str, object],
) -> dict[str, object]:
    domains = ("DY", "DZ")
    per_domain = {}
    for domain in domains:
        rows = {name: evaluations[name]["per_domain"][domain] for name in evaluations}
        ref = reference["per_domain"][domain]
        per_domain[domain] = {
            "reference_adequate": reference_adequate(ref),
            "q0_adequate": _adequate_probe(rows["Q0"]),
            "q1_adequate": _adequate_probe(rows["Q1"]),
            "t0_rescue": _primary_rescue(rows["T0"], rows["P0"]),
            "t1_rescue": _replica_rescue(rows["T1"], rows["P0"]),
            "a0_composed": float(rows["A0"]["composed_severity_top1"]),
            "p0_composed": float(rows["P0"]["composed_severity_top1"]),
            "q0_composed": float(rows["Q0"]["composed_severity_top1"]),
            "q1_composed": float(rows["Q1"]["composed_severity_top1"]),
            "t0_composed": float(rows["T0"]["composed_severity_top1"]),
            "t1_composed": float(rows["T1"]["composed_severity_top1"]),
            "integrity": True,
        }

    if not all(per_domain[d]["reference_adequate"] for d in domains):
        outcome = "W25_REFERENCE_INADEQUATE"
    elif all(per_domain[d]["t0_rescue"] and per_domain[d]["t1_rescue"] for d in domains):
        outcome = "TRAINABLE_PROJECTION_RESCUE"
    elif all(
        per_domain[d]["q0_adequate"]
        and not per_domain[d]["q1_adequate"]
        and per_domain[d]["q1_composed"] <= per_domain[d]["q0_composed"] - 0.15
        for d in domains
    ):
        outcome = "W9_PROJECTION_INFORMATION_LOSS"
    elif all(
        per_domain[d]["q1_adequate"] and per_domain[d]["p0_composed"] < 0.65
        for d in domains
    ):
        outcome = "SEMANTIC_MATCHING_INTERFACE_LIMIT"
    elif all(
        (not per_domain[d]["q0_adequate"])
        and per_domain[d]["t0_composed"] < 0.70
        and per_domain[d]["t1_composed"] < 0.65
        for d in domains
    ):
        outcome = "A13_LINEAR_PROBE_LIMIT"
    else:
        outcome = "ATOMIC_GEOMETRY_DECOMPOSITION_UNRESOLVED"

    return {
        "outcome": outcome,
        "per_domain": per_domain,
        "reference_adequate_domain_count": sum(
            bool(per_domain[d]["reference_adequate"]) for d in domains
        ),
    }


def reference_evaluation_from_scores(
    cases: Sequence[Mapping[str, object]],
    score_lookup: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, object]:
    raw = []
    for case in cases:
        case_id = str(case["case_id"])
        logits_by_factor = {
            factor_id: torch.tensor(score_lookup[case_id][factor_id], dtype=torch.float32)
            for factor_id in FACTOR_IDS
        }
        raw.append(_prediction_row(case, logits_by_factor))

    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
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


__all__ = [
    "AtomicFactorProbe",
    "CANDIDATES",
    "CANDIDATE_SEEDS",
    "GRAD_CLIP",
    "PROBE_EPOCHS",
    "PROBE_LR",
    "PROBE_WEIGHT_DECAY",
    "PROJECTION_EPOCHS",
    "PROJECTION_LR",
    "PROJECTION_TEMPERATURE",
    "PROJECTION_WEIGHT_DECAY",
    "TRAIN_BATCH_SIZE",
    "candidate_seed",
    "classify_w25",
    "evaluate_probe",
    "evaluate_raw_a13",
    "evaluate_semantic_projection",
    "probe_parameter_count",
    "reference_adequate",
    "reference_evaluation_from_scores",
    "train_probe",
    "train_projection",
]
