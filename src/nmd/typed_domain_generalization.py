from __future__ import annotations

import random
from typing import Sequence

import torch
from torch import Tensor, nn

from .competitive import CompetitiveCoarseScorer
from .hira import HIRACore
from .typed_competitive_cache import (
    HIRA_PARAMETER_COUNT,
    JOINT_PARAMETER_COUNT,
    SCORER_PARAMETER_COUNT,
    _loss_cached_case,
    evaluate_w6b_cases,
)


CANDIDATES = (
    "frozen-w6b-control",
    "single-source-scorer-only",
    "multi-source-scorer-only",
    "multi-source-joint",
)
TRAINABLE_CANDIDATES = CANDIDATES[1:]
GLOBAL_SEED = 1201
EPOCHS = 6
LR = 3e-4
WEIGHT_DECAY = 0.01


def _internal_mode(candidate: str) -> str:
    if candidate in {
        "single-source-scorer-only",
        "multi-source-scorer-only",
    }:
        return "competitive-w5i-scorer-only"
    if candidate == "multi-source-joint":
        return "competitive-w5i-joint"
    if candidate == "frozen-w6b-control":
        return "frozen"
    raise ValueError(f"unknown W6d candidate: {candidate}")


def configure_w6d_trainability(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
) -> list[Tensor]:
    mode = _internal_mode(candidate)
    if mode == "frozen":
        for parameter in hira.parameters():
            parameter.requires_grad_(False)
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        return []
    if mode == "competitive-w5i-scorer-only":
        for parameter in hira.parameters():
            parameter.requires_grad_(False)
        for parameter in scorer.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in hira.parameters():
            parameter.requires_grad_(True)
        for parameter in scorer.parameters():
            parameter.requires_grad_(True)
    return [
        parameter
        for parameter in [*hira.parameters(), *scorer.parameters()]
        if parameter.requires_grad
    ]


def expected_parameter_counts(candidate: str) -> tuple[int, int]:
    if candidate == "frozen-w6b-control":
        return 0, JOINT_PARAMETER_COUNT
    if candidate in {
        "single-source-scorer-only",
        "multi-source-scorer-only",
    }:
        return SCORER_PARAMETER_COUNT, JOINT_PARAMETER_COUNT
    if candidate == "multi-source-joint":
        return JOINT_PARAMETER_COUNT, JOINT_PARAMETER_COUNT
    raise ValueError(f"unknown W6d candidate: {candidate}")


def dev_selection_key(
    metrics: dict[str, object],
    epoch: int,
) -> tuple:
    primitive = metrics["primitive_accuracy"]
    return (
        -float(metrics["accuracy"]),
        -float(metrics["diagnosis_per_k"]["64"]["accuracy"]),
        -float(primitive["choice"]),
        -float(primitive["score"]),
        -float(primitive["noul"]),
        float(metrics["hard_brier"]),
        float(metrics["score_mae"]),
        float(metrics["soft_ece"]),
        int(epoch),
    )


def train_w6d_candidate(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    train_cases: Sequence[dict],
    dev_cases: Sequence[dict],
    *,
    epochs: int = EPOCHS,
    seed: int = GLOBAL_SEED,
) -> tuple[
    list[dict[str, object]],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, object],
]:
    if candidate not in TRAINABLE_CANDIDATES:
        raise ValueError("frozen control is not a trainable W6d candidate")

    parameters = configure_w6d_trainability(candidate, hira, scorer)
    expected_trainable, _ = expected_parameter_counts(candidate)
    if sum(parameter.numel() for parameter in parameters) != expected_trainable:
        raise RuntimeError("W6d trainable parameter contract changed")

    optimizer = torch.optim.AdamW(
        parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, object]] = []
    best_key = None
    best_hira = None
    best_scorer = None
    best_metrics = None

    for epoch in range(1, int(epochs) + 1):
        if candidate in {
            "single-source-scorer-only",
            "multi-source-scorer-only",
        }:
            hira.eval()
        else:
            hira.train()
        scorer.train()

        order = list(range(len(train_cases)))
        random.Random(int(seed) + epoch).shuffle(order)
        loss_sum = 0.0
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_cached_case(
                hira,
                scorer,
                train_cases[index],
                competitive=True,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())

        metrics = evaluate_w6b_cases(
            hira,
            scorer,
            dev_cases,
            competitive=True,
        )
        metrics["epoch"] = epoch
        metrics["mean_train_case_loss"] = (
            loss_sum / max(1, len(train_cases))
        )
        history.append(metrics)
        key = dev_selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_hira = {
                name: tensor.detach().cpu().clone()
                for name, tensor in hira.state_dict().items()
            }
            best_scorer = {
                name: tensor.detach().cpu().clone()
                for name, tensor in scorer.state_dict().items()
            }

    if best_hira is None or best_scorer is None or best_metrics is None:
        raise RuntimeError("W6d candidate produced no frozen checkpoint")
    hira.load_state_dict(best_hira, strict=True)
    scorer.load_state_dict(best_scorer, strict=True)
    return history, best_hira, best_scorer, best_metrics


def semantic_competence(metrics: dict[str, object]) -> dict[str, bool]:
    primitive = metrics["primitive_accuracy"]
    return {
        "overall_accuracy": float(metrics["accuracy"]) >= 0.65,
        "choice_accuracy": float(primitive["choice"]) >= 0.65,
        "score_accuracy": float(primitive["score"]) >= 0.60,
        "diagnosis_k64_accuracy": (
            float(metrics["diagnosis_per_k"]["64"]["accuracy"]) >= 0.55
        ),
        "noul_accuracy": float(primitive["noul"]) >= 0.65,
        "probability_mass": (
            float(metrics["probability_mass_max_error"]) <= 1e-6
        ),
        "state_once": (
            float(metrics["source_state_encodes_per_case"]) == 1.0
        ),
    }


def diversity_gates(
    multi: dict[str, object],
    single: dict[str, object],
) -> dict[str, bool]:
    mp = multi["primitive_accuracy"]
    sp = single["primitive_accuracy"]
    return {
        "overall_gain": (
            float(multi["accuracy"]) - float(single["accuracy"]) >= 0.05
        ),
        "k64_gain": (
            float(multi["diagnosis_per_k"]["64"]["accuracy"])
            - float(single["diagnosis_per_k"]["64"]["accuracy"])
            >= 0.05
        ),
        "choice_gain": (
            float(mp["choice"]) - float(sp["choice"]) >= 0.03
        ),
        "score_nonregression": (
            float(mp["score"]) >= float(sp["score"]) - 0.02
        ),
    }


def joint_localization_gates(
    joint: dict[str, object],
    multi: dict[str, object],
) -> dict[str, bool]:
    return {
        "joint_competence": all(semantic_competence(joint).values()),
        "material_gain": (
            float(joint["accuracy"]) - float(multi["accuracy"]) >= 0.05
            or float(joint["diagnosis_per_k"]["64"]["accuracy"])
            - float(multi["diagnosis_per_k"]["64"]["accuracy"])
            >= 0.05
        ),
    }


def confirm_verdict(
    *,
    single: dict[str, object],
    multi: dict[str, object],
    joint: dict[str, object],
) -> tuple[
    str,
    dict[str, bool],
    dict[str, bool],
    dict[str, bool],
]:
    competence = semantic_competence(multi)
    diversity = diversity_gates(multi, single)
    joint_gates = joint_localization_gates(joint, multi)

    if all(competence.values()) and all(diversity.values()):
        verdict = "MULTISOURCE_SCORER_GENERALIZATION_RESCUE"
    elif (
        not all(competence.values())
        and all(joint_gates.values())
    ):
        verdict = "JOINT_ADAPTATION_GENERALIZATION_RESCUE"
    else:
        overall_gain = float(multi["accuracy"]) - float(single["accuracy"])
        k64_gain = (
            float(multi["diagnosis_per_k"]["64"]["accuracy"])
            - float(single["diagnosis_per_k"]["64"]["accuracy"])
        )
        if overall_gain >= 0.03 or k64_gain >= 0.03:
            verdict = "MULTISOURCE_GENERALIZATION_PARTIAL"
        else:
            verdict = "GENERALIZATION_FAIL"
    return verdict, competence, diversity, joint_gates
