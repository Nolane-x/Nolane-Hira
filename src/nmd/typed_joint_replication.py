from __future__ import annotations

from typing import Sequence

from torch import Tensor

from .competitive import CompetitiveCoarseScorer
from .hira import HIRACore
from .typed_domain_generalization import (
    dev_selection_key,
    expected_parameter_counts as w6d_expected_parameter_counts,
    semantic_competence,
    train_w6d_candidate,
)
from .typed_joint_replication_authority import (
    PRIMARY_TRAIN_SEED,
    REPLICA_TRAIN_SEED,
)


CANDIDATES = (
    "frozen-w6b-control",
    "multi-source-scorer-only",
    "multi-source-joint-primary",
    "multi-source-joint-replica",
)
TRAINABLE_CANDIDATES = CANDIDATES[1:]
EPOCHS = 6


def internal_w6d_candidate(candidate: str) -> str:
    if candidate == "frozen-w6b-control":
        return "frozen-w6b-control"
    if candidate == "multi-source-scorer-only":
        return "multi-source-scorer-only"
    if candidate in {
        "multi-source-joint-primary",
        "multi-source-joint-replica",
    }:
        return "multi-source-joint"
    raise ValueError(f"unknown W6e candidate: {candidate}")


def candidate_seed(candidate: str) -> int:
    if candidate in {
        "multi-source-scorer-only",
        "multi-source-joint-primary",
    }:
        return PRIMARY_TRAIN_SEED
    if candidate == "multi-source-joint-replica":
        return REPLICA_TRAIN_SEED
    if candidate == "frozen-w6b-control":
        return 0
    raise ValueError(f"unknown W6e candidate: {candidate}")


def expected_parameter_counts(candidate: str) -> tuple[int, int]:
    return w6d_expected_parameter_counts(internal_w6d_candidate(candidate))


def train_w6e_candidate(
    candidate: str,
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    train_cases: Sequence[dict],
    dev_cases: Sequence[dict],
) -> tuple[
    list[dict[str, object]],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, object],
]:
    if candidate not in TRAINABLE_CANDIDATES:
        raise ValueError("frozen control is not a trainable W6e candidate")
    return train_w6d_candidate(
        internal_w6d_candidate(candidate),
        hira,
        scorer,
        train_cases,
        dev_cases,
        epochs=EPOCHS,
        seed=candidate_seed(candidate),
    )


def joint_causal_gates(
    joint: dict[str, object],
    scorer_only: dict[str, object],
) -> dict[str, bool]:
    jp = joint["primitive_accuracy"]
    sp = scorer_only["primitive_accuracy"]
    return {
        "overall_gain": (
            float(joint["accuracy"]) - float(scorer_only["accuracy"]) >= 0.03
        ),
        "k64_gain": (
            float(joint["diagnosis_per_k"]["64"]["accuracy"])
            - float(scorer_only["diagnosis_per_k"]["64"]["accuracy"])
            >= 0.05
        ),
        "choice_nonregression": (
            float(jp["choice"]) >= float(sp["choice"]) - 0.02
        ),
        "score_nonregression": (
            float(jp["score"]) >= float(sp["score"]) - 0.02
        ),
    }


def _full_joint_pass(
    joint: dict[str, object],
    scorer_only: dict[str, object],
) -> tuple[bool, dict[str, bool], dict[str, bool]]:
    competence = semantic_competence(joint)
    causal = joint_causal_gates(joint, scorer_only)
    return all(competence.values()) and all(causal.values()), competence, causal


def replication_verdict(
    *,
    confirm_l: dict[str, dict[str, object]],
    confirm_m: dict[str, dict[str, object]],
) -> tuple[str, dict[str, object]]:
    domains = {"L": confirm_l, "M": confirm_m}
    primary_passes: dict[str, bool] = {}
    primary_competence: dict[str, dict[str, bool]] = {}
    primary_causal: dict[str, dict[str, bool]] = {}
    replica_competence: dict[str, dict[str, bool]] = {}
    replica_no_catastrophic: dict[str, bool] = {}
    primary_material: dict[str, bool] = {}

    for domain_id, paths in domains.items():
        scorer = paths["multi-source-scorer-only"]
        primary = paths["multi-source-joint-primary"]
        replica = paths["multi-source-joint-replica"]

        full, competence, causal = _full_joint_pass(primary, scorer)
        primary_passes[domain_id] = full
        primary_competence[domain_id] = competence
        primary_causal[domain_id] = causal
        replica_competence[domain_id] = semantic_competence(replica)
        replica_no_catastrophic[domain_id] = (
            float(replica["accuracy"]) >= float(primary["accuracy"]) - 0.10
        )
        primary_material[domain_id] = (
            float(primary["accuracy"]) - float(scorer["accuracy"]) >= 0.03
            or float(primary["diagnosis_per_k"]["64"]["accuracy"])
            - float(scorer["diagnosis_per_k"]["64"]["accuracy"])
            >= 0.05
        )

    replica_pass_count = sum(
        all(gates.values()) for gates in replica_competence.values()
    )
    rescue = (
        all(primary_passes.values())
        and replica_pass_count >= 1
        and all(replica_no_catastrophic.values())
    )
    if rescue:
        verdict = "JOINT_GENERALIZATION_REPLICATION_RESCUE"
    elif sum(primary_passes.values()) == 1 or all(primary_material.values()):
        verdict = "JOINT_GENERALIZATION_REPLICATION_PARTIAL"
    else:
        verdict = "JOINT_GENERALIZATION_REPLICATION_FAIL"

    return verdict, {
        "primary_full_pass": primary_passes,
        "primary_competence": primary_competence,
        "primary_causal_gates": primary_causal,
        "primary_material_signal": primary_material,
        "replica_competence": replica_competence,
        "replica_no_catastrophic_regression": replica_no_catastrophic,
        "replica_competence_pass_count": replica_pass_count,
    }
