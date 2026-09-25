from nmd.freeform_attribution_training import (
    CANDIDATES,
    attribution_verdict,
    candidate_seed,
    dev_selection_key,
    expected_trainable_parameters,
    loss_mode,
)


def _m(*, overall=.90, choice=.86, k32=.80, k64=.65, one=.92, pair=2.0):
    return {
        "accuracy": overall,
        "primitive_accuracy": {"choice": choice, "noul": .95, "score": .94},
        "diagnosis_per_k": {
            "8": {"final_top1": .90},
            "16": {"final_top1": .86},
            "32": {"final_top1": k32},
            "64": {"final_top1": k64},
        },
        "one_field_k2_coarse_accuracy": one,
        "mean_k64_pair_loss_count_coarse": pair,
        "probability_mass_max_error": 1e-7,
        "source_state_encodes_per_case": 1.0,
    }


def test_candidate_budget_and_loss_modes_are_frozen():
    assert CANDIDATES == (
        "frozen-w6e-control",
        "typed-only-retune",
        "pair-only-retune",
        "typed-plus-pair-primary",
        "typed-plus-pair-replica",
    )
    assert expected_trainable_parameters("frozen-w6e-control") == 0
    for name in CANDIDATES[1:]:
        assert expected_trainable_parameters(name) == 32769
    assert loss_mode("typed-only-retune") == (True, False)
    assert loss_mode("pair-only-retune") == (False, True)
    assert loss_mode("typed-plus-pair-primary") == (True, True)
    assert loss_mode("typed-plus-pair-replica") == (True, True)
    assert candidate_seed("typed-plus-pair-primary") != candidate_seed("typed-plus-pair-replica")


def test_dev_selection_prioritizes_k64_then_k32_then_one_field():
    base = _m(k64=.60, k32=.80, one=.95)
    better_k64 = _m(k64=.61, k32=.10, one=.10)
    assert dev_selection_key(better_k64, 6) < dev_selection_key(base, 1)

    same_k64_better_k32 = _m(k64=.60, k32=.81, one=.10)
    assert dev_selection_key(same_k64_better_k32, 6) < dev_selection_key(base, 1)

    same_first_two_better_one = _m(k64=.60, k32=.80, one=.96)
    assert dev_selection_key(same_first_two_better_one, 6) < dev_selection_key(base, 1)


def test_attribution_verdict_typed_dominant():
    frozen = _m(k64=.20, one=.72, pair=12.0, overall=.82, choice=.70, k32=.40)
    typed = _m(k64=.66, one=.92, pair=2.2)
    pair = _m(k64=.35, one=.82, pair=8.0, overall=.80)
    combo = _m(k64=.68, one=.93, pair=2.0)
    replica = _m(k64=.62, one=.89, pair=2.5, overall=.86)
    rows = {
        "frozen-w6e-control": frozen,
        "typed-only-retune": typed,
        "pair-only-retune": pair,
        "typed-plus-pair-primary": combo,
        "typed-plus-pair-replica": replica,
    }
    verdict, details = attribution_verdict(rows, rows)
    assert verdict == "FREEFORM_RETUNE_REPLICATION_ATTRIBUTED"
    assert details["stable_attribution"] == "TYPED_RETUNE_DOMINANT"


def test_attribution_nonreplication_is_not_relabelled():
    frozen = _m(k64=.20, one=.72, pair=12.0, overall=.82, choice=.70, k32=.40)
    weak = _m(k64=.50, one=.85, pair=4.0, overall=.84, choice=.78, k32=.70)
    rows = {
        "frozen-w6e-control": frozen,
        "typed-only-retune": weak,
        "pair-only-retune": weak,
        "typed-plus-pair-primary": weak,
        "typed-plus-pair-replica": weak,
    }
    verdict, details = attribution_verdict(rows, rows)
    assert verdict == "FREEFORM_RETUNE_NONREPLICATING"
    assert details["replication_pass"] is False
