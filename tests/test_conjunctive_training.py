import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.conjunctive_coarse import ConjunctiveEvidenceScorer
from nmd.conjunctive_training import (
    CANDIDATES,
    absolute_competence_gates,
    candidate_seed,
    conjunctive_verdict,
    configure_trainability,
    dev_selection_key,
    expected_trainable_parameters,
    primary_causal_gates,
    replica_robustness_gates,
)
from nmd.hira import HIRACore


def _metrics(
    *,
    overall=0.84,
    diagnosis=0.82,
    k32=0.82,
    k64=0.72,
    noul=0.80,
    score=0.83,
    one_field=0.90,
    pair_loss=4.0,
):
    return {
        "accuracy": overall,
        "diagnosis_accuracy": diagnosis,
        "primitive_accuracy": {
            "choice": 0.82,
            "noul": noul,
            "score": score,
        },
        "diagnosis_per_k": {
            "8": {"final_top1": 0.95},
            "16": {"final_top1": 0.90},
            "32": {"final_top1": k32},
            "64": {"final_top1": k64},
        },
        "hard_brier": 0.20,
        "probability_mass_max_error": 1e-7,
        "source_state_encodes_per_case": 1.0,
        "one_field_k2_coarse_accuracy": one_field,
        "mean_k64_pair_loss_count_coarse": pair_loss,
        "mean_k64_pair_loss_count_final": pair_loss,
    }


def test_candidate_seeds_and_parameter_budgets_are_frozen():
    assert CANDIDATES == (
        "frozen-w6e-control",
        "freeform-retune-control",
        "conjunctive-primary",
        "conjunctive-replica",
    )
    assert candidate_seed("freeform-retune-control") == 1901
    assert candidate_seed("conjunctive-primary") == 1907
    assert candidate_seed("conjunctive-replica") == 1913
    assert expected_trainable_parameters("frozen-w6e-control") == 0
    assert expected_trainable_parameters("freeform-retune-control") == 32769
    assert expected_trainable_parameters("conjunctive-primary") == 32772
    assert expected_trainable_parameters("conjunctive-replica") == 32772


def test_hira_is_frozen_and_candidate_parameter_counts_are_exact():
    hira = HIRACore(d_model=256, dropout=0.0)
    freeform = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    params = configure_trainability(
        "freeform-retune-control",
        hira,
        freeform,
    )
    assert sum(p.numel() for p in params) == 32769
    assert all(not p.requires_grad for p in hira.parameters())

    hira2 = HIRACore(d_model=256, dropout=0.0)
    conjunctive = ConjunctiveEvidenceScorer(
        base=CompetitiveCoarseScorer(d_model=256, d_rel=128)
    )
    params2 = configure_trainability(
        "conjunctive-primary",
        hira2,
        conjunctive,
    )
    assert sum(p.numel() for p in params2) == 32772
    assert all(not p.requires_grad for p in hira2.parameters())


def test_absolute_causal_and_replica_gates():
    freeform = _metrics(
        overall=0.84,
        diagnosis=0.80,
        k32=0.74,
        k64=0.55,
        one_field=0.80,
        pair_loss=8.0,
    )
    primary = _metrics(
        overall=0.83,
        diagnosis=0.82,
        k32=0.82,
        k64=0.70,
        one_field=0.88,
        pair_loss=5.0,
    )
    replica = _metrics(
        overall=0.81,
        diagnosis=0.80,
        k32=0.75,
        k64=0.64,
        one_field=0.84,
        pair_loss=5.5,
    )
    assert all(absolute_competence_gates(primary).values())
    assert all(primary_causal_gates(primary, freeform).values())
    assert all(replica_robustness_gates(replica, freeform).values())


def test_verdict_rescue_partial_and_fail_are_not_conflated():
    freeform = _metrics(
        overall=0.84,
        k32=0.74,
        k64=0.55,
        one_field=0.80,
        pair_loss=8.0,
    )
    primary = _metrics(
        overall=0.83,
        k32=0.82,
        k64=0.70,
        one_field=0.88,
        pair_loss=5.0,
    )
    replica = _metrics(
        overall=0.81,
        k32=0.75,
        k64=0.64,
        one_field=0.84,
        pair_loss=5.5,
    )
    strong = {
        "frozen-w6e-control": freeform,
        "freeform-retune-control": freeform,
        "conjunctive-primary": primary,
        "conjunctive-replica": replica,
    }
    verdict, details = conjunctive_verdict(strong, strong)
    assert verdict == "CONJUNCTIVE_COARSE_RESCUE"
    assert details["full_pass"] is True

    partial_primary = _metrics(
        overall=0.82,
        k32=0.75,
        k64=0.61,
        one_field=0.81,
        pair_loss=7.0,
    )
    partial = {
        **strong,
        "conjunctive-primary": partial_primary,
        "conjunctive-replica": _metrics(k64=0.55),
    }
    verdict, details = conjunctive_verdict(partial, partial)
    assert verdict == "CONJUNCTIVE_COARSE_PARTIAL"
    assert details["full_pass"] is False
    assert details["partial_pass"] is True

    fail_primary = _metrics(
        overall=0.82,
        k64=0.58,
        one_field=0.81,
        pair_loss=7.5,
    )
    fail = {
        **strong,
        "conjunctive-primary": fail_primary,
        "conjunctive-replica": _metrics(k64=0.54),
    }
    verdict, details = conjunctive_verdict(fail, fail)
    assert verdict == "CONJUNCTIVE_COARSE_FAIL"
    assert details["partial_pass"] is False


def test_dev_selection_is_k64_then_k32_then_choice_then_overall():
    base = _metrics()
    better_k64 = _metrics(k64=0.73, k32=0.70)
    assert dev_selection_key(better_k64, 6) < dev_selection_key(base, 1)
