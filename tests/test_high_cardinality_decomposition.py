from __future__ import annotations

from nmd.high_cardinality_decomposition import (
    MIXED,
    UNRESOLVED,
    architecture_classification,
    checkpoint_stability,
    diagnostic_stability,
)


def _base_metrics(**overrides):
    row = {
        "k64_final_top1": 0.50,
        "global_error_has_pair_loss_rate": 0.50,
        "all_pair_win_but_k64_fail_rate": 0.05,
        "winner_reversal_rate": 0.05,
        "winner_k2_final_to_k64_final_sign_reversal_rate": 0.05,
        "relation_damage_given_coarse_correct_rate": 0.05,
        "global_error_winner_beats_gold_k2_coarse_rate": 0.50,
        "relation_rescue_top1_gain": 0.05,
        "oracle_final_top1": 0.99,
        "k64_coarse_top1": 0.55,
        "coarse_correct_case_rate": 0.55,
        "relation_damage_rate": 0.05,
        "relation_rescue_rate": 0.05,
    }
    row.update(overrides)
    return row


def test_pairwise_multiplicity_gate():
    result = architecture_classification(
        _base_metrics(
            global_error_has_pair_loss_rate=0.86,
            all_pair_win_but_k64_fail_rate=0.07,
            winner_reversal_rate=0.10,
            relation_damage_given_coarse_correct_rate=0.08,
            global_error_winner_beats_gold_k2_coarse_rate=0.60,
        )
    )
    assert result["classification"] == "PAIRWISE_MULTIPLICITY_LIMIT"


def test_set_context_reversal_gate():
    result = architecture_classification(
        _base_metrics(
            global_error_has_pair_loss_rate=0.60,
            all_pair_win_but_k64_fail_rate=0.22,
            winner_reversal_rate=0.31,
            winner_k2_final_to_k64_final_sign_reversal_rate=0.25,
        )
    )
    assert result["classification"] == "SET_CONTEXT_RANK_REVERSAL"


def test_coarse_conjunction_gate():
    result = architecture_classification(
        _base_metrics(
            global_error_winner_beats_gold_k2_coarse_rate=0.78,
            relation_rescue_top1_gain=0.04,
            oracle_final_top1=0.98,
        )
    )
    assert result["classification"] == "COARSE_CONJUNCTION_LIMIT"


def test_relation_damage_gate():
    result = architecture_classification(
        _base_metrics(
            k64_final_top1=0.61,
            k64_coarse_top1=0.72,
            coarse_correct_case_rate=0.72,
            relation_damage_given_coarse_correct_rate=0.21,
            oracle_final_top1=0.92,
            relation_damage_rate=0.23,
            relation_rescue_rate=0.10,
        )
    )
    assert result["classification"] == "RELATION_DECOMPOSITION_DAMAGE"


def test_multiple_active_gates_are_mixed_not_posthoc_precedence():
    result = architecture_classification(
        _base_metrics(
            global_error_has_pair_loss_rate=0.90,
            all_pair_win_but_k64_fail_rate=0.05,
            winner_reversal_rate=0.05,
            relation_damage_given_coarse_correct_rate=0.05,
            global_error_winner_beats_gold_k2_coarse_rate=0.80,
            relation_rescue_top1_gain=0.02,
            oracle_final_top1=0.99,
        )
    )
    assert set(result["active_rules"]) == {
        "PAIRWISE_MULTIPLICITY_LIMIT",
        "COARSE_CONJUNCTION_LIMIT",
    }
    assert result["classification"] == MIXED


def test_no_gate_is_unresolved():
    result = architecture_classification(_base_metrics())
    assert result["classification"] == UNRESOLVED


def test_checkpoint_stability_requires_two_domains():
    stable = checkpoint_stability(
        {
            "AD": {"classification": "SET_CONTEXT_RANK_REVERSAL"},
            "AE": {"classification": "SET_CONTEXT_RANK_REVERSAL"},
            "AF": {"classification": UNRESOLVED},
        }
    )
    assert stable == {
        "stable": True,
        "classification": "SET_CONTEXT_RANK_REVERSAL",
        "domains": ["AD", "AE"],
    }

    unstable = checkpoint_stability(
        {
            "AD": {"classification": "SET_CONTEXT_RANK_REVERSAL"},
            "AE": {"classification": UNRESOLVED},
            "AF": {"classification": UNRESOLVED},
        }
    )
    assert unstable["stable"] is False


def test_cross_checkpoint_stability_requires_w6e_and_one_trained_agreement():
    domains_a = {
        "AD": {"classification": "PAIRWISE_MULTIPLICITY_LIMIT"},
        "AE": {"classification": "PAIRWISE_MULTIPLICITY_LIMIT"},
        "AF": {"classification": UNRESOLVED},
    }
    unresolved = {
        "AD": {"classification": UNRESOLVED},
        "AE": {"classification": UNRESOLVED},
        "AF": {"classification": UNRESOLVED},
    }

    out = diagnostic_stability(
        {
            "w6e-joint-primary": domains_a,
            "w6h-projection-retune": domains_a,
            "w6h-semantic-adapter": unresolved,
        }
    )
    assert out["outcome"] == "STABLE_HIGH_CARDINALITY_ARCHITECTURE"
    assert out["stable_classification"] == "PAIRWISE_MULTIPLICITY_LIMIT"
    assert out["rescue_lane_authorized"] is True


def test_cross_checkpoint_opposition_is_mixed():
    multiplicity = {
        "AD": {"classification": "PAIRWISE_MULTIPLICITY_LIMIT"},
        "AE": {"classification": "PAIRWISE_MULTIPLICITY_LIMIT"},
        "AF": {"classification": UNRESOLVED},
    }
    reversal = {
        "AD": {"classification": "SET_CONTEXT_RANK_REVERSAL"},
        "AE": {"classification": "SET_CONTEXT_RANK_REVERSAL"},
        "AF": {"classification": UNRESOLVED},
    }

    out = diagnostic_stability(
        {
            "w6e-joint-primary": multiplicity,
            "w6h-projection-retune": multiplicity,
            "w6h-semantic-adapter": reversal,
        }
    )
    assert out["outcome"] == MIXED
    assert out["rescue_lane_authorized"] is False


def test_unstable_w6e_blocks_rescue():
    unresolved = {
        "AD": {"classification": UNRESOLVED},
        "AE": {"classification": UNRESOLVED},
        "AF": {"classification": UNRESOLVED},
    }
    stable = {
        "AD": {"classification": "COARSE_CONJUNCTION_LIMIT"},
        "AE": {"classification": "COARSE_CONJUNCTION_LIMIT"},
        "AF": {"classification": UNRESOLVED},
    }

    out = diagnostic_stability(
        {
            "w6e-joint-primary": unresolved,
            "w6h-projection-retune": stable,
            "w6h-semantic-adapter": stable,
        }
    )
    assert out["outcome"] == UNRESOLVED
    assert out["rescue_lane_authorized"] is False
