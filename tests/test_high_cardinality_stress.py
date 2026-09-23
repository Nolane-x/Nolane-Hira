import math

import torch

from nmd.high_cardinality_stress import (
    COLORS,
    K_VALUES,
    MECHANICAL_CASES,
    OBJECTS,
    SEMANTIC_CASES,
    evaluate_mechanical_stress,
    generate_semantic_key_case,
)
from nmd.hira import HIRACore


def test_mechanical_stress_passes_frozen_invariants_for_small_fixture_count():
    torch.manual_seed(123)
    hira = HIRACore(d_model=256, dropout=0.05)
    hira.eval()

    for k in K_VALUES:
        result = evaluate_mechanical_stress(
            hira,
            k=k,
            cases=4,
        )
        assert result.finite_pass is True
        assert result.budget_min == k
        assert result.budget_max == k
        assert result.tail_mass_max == 0.0
        assert result.probability_mass_max_error <= 1e-6
        assert result.permutation_max_error <= 2e-6
        assert result.argmax_invariant_count == 4
        assert result.repeatability_max_error <= 1e-7
        assert result.option_tensor_bytes_per_case == k * 256 * 4
        assert math.isfinite(result.p50_ms_cpu)
        assert math.isfinite(result.p95_ms_cpu)
        assert result.mechanics_pass is True


def test_semantic_generator_is_deterministic_unique_and_opaque():
    for k in K_VALUES:
        a = generate_semantic_key_case(k=k, case_index=0)
        b = generate_semantic_key_case(k=k, case_index=0)
        assert a == b

        state_text, question, options, gold_index = a
        assert len(options) == k
        assert 0 <= gold_index < k
        assert question == "Which option exactly matches the routing_key?"
        assert len({option.criterion_text for option in options}) == k
        assert [option.option_id for option in options] == [
            f"route-{i:03d}" for i in range(k)
        ]
        assert options[gold_index].criterion_text in state_text


def test_semantic_generator_uses_only_frozen_independent_vocabulary():
    forbidden = {
        "cash withdrawal",
        "bank transfer",
        "card payment",
        "beneficiary",
        "banking intent",
    }
    state_text, question, options, _ = generate_semantic_key_case(
        k=255,
        case_index=127,
    )
    payload = " ".join(
        [state_text, question]
        + [option.criterion_text for option in options]
    ).lower()

    assert all(term not in payload for term in forbidden)
    assert all(
        option.criterion_text.split()[0] in COLORS
        for option in options
    )
    assert all(
        option.criterion_text.split()[1] in OBJECTS
        for option in options
    )


def test_frozen_case_counts_and_k_values_do_not_drift():
    assert K_VALUES == (128, 255)
    assert MECHANICAL_CASES == 64
    assert SEMANTIC_CASES == 128
