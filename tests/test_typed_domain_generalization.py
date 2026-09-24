import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.typed_domain_generalization import (
    CANDIDATES,
    configure_w6d_trainability,
    confirm_verdict,
    dev_selection_key,
    diversity_gates,
    expected_parameter_counts,
    semantic_competence,
)


def fixture(
    *,
    accuracy=0.7,
    choice=0.7,
    score=0.65,
    noul=0.7,
    k64=0.6,
):
    return {
        "accuracy": accuracy,
        "primitive_accuracy": {
            "choice": choice,
            "score": score,
            "noul": noul,
        },
        "diagnosis_per_k": {
            "8": {"accuracy": k64},
            "16": {"accuracy": k64},
            "32": {"accuracy": k64},
            "64": {"accuracy": k64},
        },
        "hard_brier": 0.4,
        "score_mae": 0.4,
        "soft_ece": 0.1,
        "probability_mass_max_error": 1e-7,
        "source_state_encodes_per_case": 1.0,
    }


def test_w6d_candidate_parameter_contracts():
    for candidate in CANDIDATES:
        hira = HIRACore(d_model=256, dropout=0.0)
        scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
        params = configure_w6d_trainability(candidate, hira, scorer)
        expected_trainable, expected_total = expected_parameter_counts(candidate)
        assert sum(p.numel() for p in params) == expected_trainable
        assert (
            sum(p.numel() for p in hira.parameters())
            + sum(p.numel() for p in scorer.parameters())
        ) == expected_total

    hira = HIRACore(d_model=256, dropout=0.0)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    configure_w6d_trainability("single-source-scorer-only", hira, scorer)
    assert not any(p.requires_grad for p in hira.parameters())
    assert all(p.requires_grad for p in scorer.parameters())


def test_dev_key_prioritizes_semantic_accuracy_before_calibration():
    base = fixture()
    better_k64 = fixture(k64=0.61)
    better_overall = fixture(accuracy=0.71, k64=0.55)
    assert dev_selection_key(better_overall, 6) < dev_selection_key(base, 1)
    assert dev_selection_key(better_k64, 6) < dev_selection_key(base, 1)


def test_multisource_rescue_requires_competence_and_diversity():
    single = fixture(accuracy=0.66, choice=0.66, score=0.62, noul=0.66, k64=0.56)
    multi = fixture(accuracy=0.72, choice=0.71, score=0.64, noul=0.70, k64=0.63)
    joint = fixture(accuracy=0.73, choice=0.72, score=0.65, noul=0.71, k64=0.64)
    verdict, competence, diversity, _ = confirm_verdict(
        single=single, multi=multi, joint=joint
    )
    assert verdict == "MULTISOURCE_SCORER_GENERALIZATION_RESCUE"
    assert all(competence.values())
    assert all(diversity.values())


def test_joint_rescue_localizes_failure_beyond_scorer():
    single = fixture(accuracy=0.50, choice=0.50, score=0.50, noul=0.60, k64=0.40)
    multi = fixture(accuracy=0.56, choice=0.57, score=0.52, noul=0.62, k64=0.47)
    joint = fixture(accuracy=0.70, choice=0.70, score=0.64, noul=0.68, k64=0.60)
    verdict, competence, _, joint_gates = confirm_verdict(
        single=single, multi=multi, joint=joint
    )
    assert not all(competence.values())
    assert all(joint_gates.values())
    assert verdict == "JOINT_ADAPTATION_GENERALIZATION_RESCUE"


def test_partial_and_fail_are_not_promoted():
    single = fixture(accuracy=0.50, choice=0.50, score=0.50, noul=0.60, k64=0.40)
    partial = fixture(accuracy=0.54, choice=0.54, score=0.50, noul=0.60, k64=0.44)
    weak_joint = fixture(accuracy=0.55, choice=0.55, score=0.52, noul=0.60, k64=0.45)
    verdict, _, _, _ = confirm_verdict(
        single=single, multi=partial, joint=weak_joint
    )
    assert verdict == "MULTISOURCE_GENERALIZATION_PARTIAL"

    fail = fixture(accuracy=0.51, choice=0.51, score=0.50, noul=0.60, k64=0.41)
    verdict, _, _, _ = confirm_verdict(
        single=single, multi=fail, joint=weak_joint
    )
    assert verdict == "GENERALIZATION_FAIL"


def test_gate_helpers_have_frozen_keys():
    assert set(semantic_competence(fixture())) == {
        "overall_accuracy",
        "choice_accuracy",
        "score_accuracy",
        "diagnosis_k64_accuracy",
        "noul_accuracy",
        "probability_mass",
        "state_once",
    }
    assert set(diversity_gates(fixture(), fixture())) == {
        "overall_gain",
        "k64_gain",
        "choice_gain",
        "score_nonregression",
    }
