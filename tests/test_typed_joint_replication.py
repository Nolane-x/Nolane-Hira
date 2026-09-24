from nmd.typed_joint_replication import (
    CANDIDATES,
    candidate_seed,
    expected_parameter_counts,
    internal_w6d_candidate,
    joint_causal_gates,
    replication_verdict,
)
from nmd.typed_joint_replication_authority import (
    PRIMARY_TRAIN_SEED,
    REPLICA_TRAIN_SEED,
)


def metric(
    *,
    overall: float,
    choice: float,
    noul: float,
    score: float,
    k64: float,
):
    return {
        "accuracy": overall,
        "primitive_accuracy": {
            "choice": choice,
            "noul": noul,
            "score": score,
        },
        "diagnosis_per_k": {
            "8": {"accuracy": k64},
            "16": {"accuracy": k64},
            "32": {"accuracy": k64},
            "64": {"accuracy": k64},
        },
        "probability_mass_max_error": 1e-7,
        "source_state_encodes_per_case": 1.0,
    }


def test_w6e_candidate_mapping_parameter_counts_and_seeds_are_frozen():
    assert CANDIDATES == (
        "frozen-w6b-control",
        "multi-source-scorer-only",
        "multi-source-joint-primary",
        "multi-source-joint-replica",
    )
    assert internal_w6d_candidate("frozen-w6b-control") == (
        "frozen-w6b-control"
    )
    assert internal_w6d_candidate("multi-source-scorer-only") == (
        "multi-source-scorer-only"
    )
    assert internal_w6d_candidate("multi-source-joint-primary") == (
        "multi-source-joint"
    )
    assert internal_w6d_candidate("multi-source-joint-replica") == (
        "multi-source-joint"
    )

    assert expected_parameter_counts("frozen-w6b-control") == (0, 454928)
    assert expected_parameter_counts("multi-source-scorer-only") == (
        32769, 454928
    )
    assert expected_parameter_counts("multi-source-joint-primary") == (
        454928, 454928
    )
    assert expected_parameter_counts("multi-source-joint-replica") == (
        454928, 454928
    )

    assert candidate_seed("multi-source-scorer-only") == PRIMARY_TRAIN_SEED
    assert candidate_seed("multi-source-joint-primary") == PRIMARY_TRAIN_SEED
    assert candidate_seed("multi-source-joint-replica") == REPLICA_TRAIN_SEED
    assert PRIMARY_TRAIN_SEED != REPLICA_TRAIN_SEED


def test_joint_causal_gates_use_frozen_w6e_thresholds():
    scorer = metric(
        overall=0.70,
        choice=0.72,
        noul=0.80,
        score=0.80,
        k64=0.40,
    )
    joint = metric(
        overall=0.74,
        choice=0.71,
        noul=0.84,
        score=0.79,
        k64=0.47,
    )
    gates = joint_causal_gates(joint, scorer)
    assert gates == {
        "overall_gain": True,
        "k64_gain": True,
        "choice_nonregression": True,
        "score_nonregression": True,
    }


def test_replication_rescue_requires_primary_on_both_domains():
    scorer = metric(
        overall=0.70,
        choice=0.70,
        noul=0.75,
        score=0.75,
        k64=0.45,
    )
    primary = metric(
        overall=0.78,
        choice=0.77,
        noul=0.80,
        score=0.80,
        k64=0.62,
    )
    replica = metric(
        overall=0.76,
        choice=0.75,
        noul=0.78,
        score=0.79,
        k64=0.58,
    )

    verdict, details = replication_verdict(
        confirm_l={
            "multi-source-scorer-only": scorer,
            "multi-source-joint-primary": primary,
            "multi-source-joint-replica": replica,
        },
        confirm_m={
            "multi-source-scorer-only": scorer,
            "multi-source-joint-primary": primary,
            "multi-source-joint-replica": replica,
        },
    )
    assert verdict == "JOINT_GENERALIZATION_REPLICATION_RESCUE"
    assert details["primary_full_pass"] == {"L": True, "M": True}
    assert details["replica_competence_pass_count"] == 2


def test_replica_cannot_rescue_failing_primary():
    scorer = metric(
        overall=0.70,
        choice=0.70,
        noul=0.75,
        score=0.75,
        k64=0.45,
    )
    primary_fail = metric(
        overall=0.66,
        choice=0.68,
        noul=0.72,
        score=0.70,
        k64=0.50,
    )
    replica_good = metric(
        overall=0.80,
        choice=0.80,
        noul=0.82,
        score=0.81,
        k64=0.65,
    )

    verdict, details = replication_verdict(
        confirm_l={
            "multi-source-scorer-only": scorer,
            "multi-source-joint-primary": primary_fail,
            "multi-source-joint-replica": replica_good,
        },
        confirm_m={
            "multi-source-scorer-only": scorer,
            "multi-source-joint-primary": primary_fail,
            "multi-source-joint-replica": replica_good,
        },
    )
    assert verdict != "JOINT_GENERALIZATION_REPLICATION_RESCUE"
    assert details["replica_competence_pass_count"] == 2


def test_one_domain_primary_pass_is_partial():
    scorer = metric(
        overall=0.70,
        choice=0.70,
        noul=0.75,
        score=0.75,
        k64=0.45,
    )
    primary_pass = metric(
        overall=0.78,
        choice=0.77,
        noul=0.80,
        score=0.80,
        k64=0.62,
    )
    primary_fail = metric(
        overall=0.68,
        choice=0.68,
        noul=0.70,
        score=0.70,
        k64=0.53,
    )
    replica = metric(
        overall=0.74,
        choice=0.73,
        noul=0.76,
        score=0.77,
        k64=0.57,
    )

    verdict, details = replication_verdict(
        confirm_l={
            "multi-source-scorer-only": scorer,
            "multi-source-joint-primary": primary_pass,
            "multi-source-joint-replica": replica,
        },
        confirm_m={
            "multi-source-scorer-only": scorer,
            "multi-source-joint-primary": primary_fail,
            "multi-source-joint-replica": replica,
        },
    )
    assert verdict == "JOINT_GENERALIZATION_REPLICATION_PARTIAL"
    assert details["primary_full_pass"] == {"L": True, "M": False}
