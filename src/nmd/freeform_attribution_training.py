from __future__ import annotations

import math
import random
from typing import Sequence

import torch
from torch import Tensor

from .competitive import CompetitiveCoarseScorer
from .conjunctive_training import (
    evaluate_w7,
    one_field_pair_margin_loss,
    typed_case_loss,
)
from .hira import HIRACore
from .typed_competitive_cache import SCORER_PARAMETER_COUNT


CANDIDATES = (
    "frozen-w6e-control",
    "typed-only-retune",
    "pair-only-retune",
    "typed-plus-pair-primary",
    "typed-plus-pair-replica",
)

EPOCHS = 6
LR = 3e-4
WEIGHT_DECAY = 0.01
PAIR_MARGIN_WEIGHT = 0.25

SEEDS = {
    "frozen-w6e-control": 0,
    "typed-only-retune": 2003,
    "pair-only-retune": 2011,
    "typed-plus-pair-primary": 2017,
    "typed-plus-pair-replica": 2027,
}


def candidate_seed(candidate: str) -> int:
    if candidate not in SEEDS:
        raise ValueError(f"unknown W7b candidate: {candidate}")
    return int(SEEDS[candidate])


def expected_trainable_parameters(candidate: str) -> int:
    if candidate == "frozen-w6e-control":
        return 0
    if candidate in CANDIDATES:
        return SCORER_PARAMETER_COUNT
    raise ValueError(f"unknown W7b candidate: {candidate}")


def configure_trainability(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
) -> list[Tensor]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown W7b candidate: {candidate}")
    for parameter in hira.parameters():
        parameter.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(candidate != "frozen-w6e-control")
    if candidate == "frozen-w6e-control":
        return []
    parameters = [p for p in scorer.parameters() if p.requires_grad]
    count = sum(p.numel() for p in parameters)
    if count != SCORER_PARAMETER_COUNT:
        raise RuntimeError(
            f"W7b scorer parameter contract changed: {count} != {SCORER_PARAMETER_COUNT}"
        )
    return parameters


def loss_mode(candidate: str) -> tuple[bool, bool]:
    if candidate == "typed-only-retune":
        return True, False
    if candidate == "pair-only-retune":
        return False, True
    if candidate in {"typed-plus-pair-primary", "typed-plus-pair-replica"}:
        return True, True
    if candidate == "frozen-w6e-control":
        return False, False
    raise ValueError(f"unknown W7b candidate: {candidate}")


def dev_selection_key(metrics: dict[str, object], epoch: int) -> tuple:
    diagnosis = metrics["diagnosis_per_k"]
    primitive = metrics["primitive_accuracy"]
    return (
        -float(diagnosis["64"]["final_top1"]),
        -float(diagnosis["32"]["final_top1"]),
        -float(metrics["one_field_k2_coarse_accuracy"]),
        float(metrics["mean_k64_pair_loss_count_coarse"]),
        -float(primitive["choice"]),
        -float(metrics["accuracy"]),
        int(epoch),
    )


def train_w7b_candidate(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    train_cases: Sequence[dict],
    dev_cases: Sequence[dict],
    *,
    epochs: int = EPOCHS,
) -> tuple[list[dict[str, object]], dict[str, Tensor], dict[str, object]]:
    if candidate == "frozen-w6e-control":
        raise ValueError("frozen W7b control is not trainable")

    parameters = configure_trainability(candidate, hira, scorer)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    use_typed, use_pair = loss_mode(candidate)
    seed = candidate_seed(candidate)

    history: list[dict[str, object]] = []
    best_key = None
    best_state = None
    best_metrics = None

    for epoch in range(1, int(epochs) + 1):
        hira.eval()
        scorer.train()
        order = list(range(len(train_cases)))
        random.Random(seed + epoch).shuffle(order)

        typed_sum = 0.0
        pair_sum = 0.0
        total_sum = 0.0

        for index in order:
            case = train_cases[index]
            optimizer.zero_grad(set_to_none=True)

            typed_loss = typed_case_loss(hira, scorer, case)
            pair_loss = one_field_pair_margin_loss(scorer, case)

            parts = []
            if use_typed:
                parts.append(typed_loss)
            if use_pair:
                parts.append(PAIR_MARGIN_WEIGHT * pair_loss)
            if not parts:
                raise RuntimeError("W7b trainable candidate has no loss")
            loss = torch.stack([part for part in parts]).sum()
            loss.backward()
            optimizer.step()

            typed_sum += float(typed_loss.detach())
            pair_sum += float(pair_loss.detach())
            total_sum += float(loss.detach())

        metrics = evaluate_w7(hira, scorer, dev_cases)
        metrics["epoch"] = epoch
        metrics["mean_train_typed_loss"] = typed_sum / len(train_cases)
        metrics["mean_train_pair_loss"] = pair_sum / len(train_cases)
        metrics["mean_train_total_loss"] = total_sum / len(train_cases)
        metrics["typed_loss_enabled"] = use_typed
        metrics["pair_loss_enabled"] = use_pair
        history.append(metrics)

        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in scorer.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("W7b candidate produced no DEV checkpoint")
    scorer.load_state_dict(best_state, strict=True)
    return history, best_state, best_metrics


def _k64(metrics: dict[str, object]) -> float:
    return float(metrics["diagnosis_per_k"]["64"]["final_top1"])


def _k32(metrics: dict[str, object]) -> float:
    return float(metrics["diagnosis_per_k"]["32"]["final_top1"])


def _one_field(metrics: dict[str, object]) -> float:
    return float(metrics["one_field_k2_coarse_accuracy"])


def _pair_loss(metrics: dict[str, object]) -> float:
    return float(metrics["mean_k64_pair_loss_count_coarse"])


def _relative_reduction(before: float, after: float) -> float:
    if before <= 0:
        return 1.0 if after <= 0 else -math.inf
    return (before - after) / before


def replication_primary_gates(
    primary: dict[str, object],
    frozen: dict[str, object],
) -> dict[str, bool]:
    primitive = primary["primitive_accuracy"]
    return {
        "overall": float(primary["accuracy"]) >= 0.85,
        "choice": float(primitive["choice"]) >= 0.80,
        "k32": _k32(primary) >= 0.75,
        "k64": _k64(primary) >= 0.60,
        "one_field": _one_field(primary) >= 0.90,
        "pair_loss": _pair_loss(primary) <= 3.0,
        "probability_mass": float(primary["probability_mass_max_error"]) <= 1e-6,
        "state_once": float(primary["source_state_encodes_per_case"]) == 1.0,
        "k64_gain_vs_frozen": _k64(primary) - _k64(frozen) >= 0.30,
        "one_field_gain_vs_frozen": _one_field(primary) - _one_field(frozen) >= 0.12,
        "pair_loss_reduction_vs_frozen": (
            _relative_reduction(_pair_loss(frozen), _pair_loss(primary)) >= 0.50
        ),
    }


def replica_gates(replica: dict[str, object]) -> dict[str, bool]:
    return {
        "k64": _k64(replica) >= 0.55,
        "one_field": _one_field(replica) >= 0.87,
        "overall": float(replica["accuracy"]) >= 0.82,
    }


def typed_retune_dominant(
    typed: dict[str, object],
    combo: dict[str, object],
) -> bool:
    return (
        abs(_k64(typed) - _k64(combo)) <= 0.05
        and abs(_one_field(typed) - _one_field(combo)) <= 0.03
        and _relative_reduction(_pair_loss(typed), _pair_loss(combo)) < 0.15
    )


def pair_margin_contributor(
    typed: dict[str, object],
    pair: dict[str, object],
    combo: dict[str, object],
    frozen: dict[str, object],
) -> bool:
    gain_signal = (
        _k64(combo) - _k64(typed) >= 0.05
        or _one_field(combo) - _one_field(typed) >= 0.04
    )
    reduction = _relative_reduction(_pair_loss(typed), _pair_loss(combo)) >= 0.20
    pair_independent = (
        _k64(pair) - _k64(frozen) >= 0.10
        or _one_field(pair) - _one_field(frozen) >= 0.08
    )
    overall_ok = float(combo["accuracy"]) >= float(typed["accuracy"]) - 0.03
    return gain_signal and reduction and pair_independent and overall_ok


def pair_margin_primary(
    typed: dict[str, object],
    pair: dict[str, object],
    combo: dict[str, object],
) -> bool:
    return (
        abs(_k64(pair) - _k64(combo)) <= 0.05
        and abs(_one_field(pair) - _one_field(combo)) <= 0.03
        and _k64(combo) - _k64(typed) >= 0.08
    )


def attribution_verdict(
    confirm_as: dict[str, dict[str, object]],
    confirm_at: dict[str, dict[str, object]],
) -> tuple[str, dict[str, object]]:
    domains = {"AS": confirm_as, "AT": confirm_at}
    details: dict[str, object] = {
        "replication_primary": {},
        "replica": {},
        "attribution": {},
    }

    replication_pass = True
    typed_dominant_both = True
    contributor_both = True
    pair_primary_both = True

    for domain, rows in domains.items():
        frozen = rows["frozen-w6e-control"]
        typed = rows["typed-only-retune"]
        pair = rows["pair-only-retune"]
        combo = rows["typed-plus-pair-primary"]
        replica = rows["typed-plus-pair-replica"]

        pg = replication_primary_gates(combo, frozen)
        rg = replica_gates(replica)
        td = typed_retune_dominant(typed, combo)
        pc = pair_margin_contributor(typed, pair, combo, frozen)
        pp = pair_margin_primary(typed, pair, combo)

        details["replication_primary"][domain] = pg
        details["replica"][domain] = rg
        details["attribution"][domain] = {
            "typed_retune_dominant": td,
            "pair_margin_causal_contributor": pc,
            "pair_margin_primary": pp,
        }

        replication_pass = replication_pass and all(pg.values()) and all(rg.values())
        typed_dominant_both = typed_dominant_both and td
        contributor_both = contributor_both and pc
        pair_primary_both = pair_primary_both and pp

    if not replication_pass:
        details["stable_attribution"] = None
        details["replication_pass"] = False
        return "FREEFORM_RETUNE_NONREPLICATING", details

    classes = []
    if typed_dominant_both:
        classes.append("TYPED_RETUNE_DOMINANT")
    if contributor_both:
        classes.append("PAIR_MARGIN_CAUSAL_CONTRIBUTOR")
    if pair_primary_both:
        classes.append("PAIR_MARGIN_PRIMARY")

    details["replication_pass"] = True
    details["stable_attribution"] = classes[0] if len(classes) == 1 else None
    details["active_stable_classes"] = classes

    if len(classes) == 1:
        return "FREEFORM_RETUNE_REPLICATION_ATTRIBUTED", details
    return "FREEFORM_RETUNE_REPLICATION_UNATTRIBUTED", details
