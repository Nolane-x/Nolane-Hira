from collections import Counter

import pytest

from nmd.typed_domain_generalization_authority import (
    CONFIRM_F_SEED,
    DEV_E_SEED,
    DOMAIN_A,
    DOMAIN_B,
    DOMAIN_C,
    DOMAIN_D,
    DOMAIN_E,
    DOMAIN_F,
    GLOBAL_SEED,
    SOURCE_A_SEED,
    SOURCE_B_SEED,
    SOURCE_C_SEED,
    SOURCE_D_SEED,
    all_w6d_values,
    domain_value_sets,
    generate_w6d_confirm,
    generate_w6d_dev,
    generate_w6d_multi_train,
    generate_w6d_single_train,
)


def _strata(cases):
    return Counter(
        (case.diagnosis_k, case.severity, case.confidence)
        for case in cases
    )


def _gold_signature(case):
    diagnosis = case.typed.decisions[0]
    return diagnosis.options[diagnosis.gold_index].criterion_text


def test_w6d_seeds_and_counts_are_frozen():
    assert (
        SOURCE_A_SEED,
        SOURCE_B_SEED,
        SOURCE_C_SEED,
        SOURCE_D_SEED,
        DEV_E_SEED,
        CONFIRM_F_SEED,
        GLOBAL_SEED,
    ) == (
        181149,
        181151,
        181153,
        181157,
        182251,
        183353,
        1201,
    )

    single = generate_w6d_single_train()
    multi = generate_w6d_multi_train()
    dev = generate_w6d_dev()

    assert len(single) == 384
    assert len(multi) == 384
    assert len(dev) == 192
    assert sum(len(case.typed.decisions) for case in single) == 1920
    assert sum(len(case.typed.decisions) for case in multi) == 1920
    assert sum(len(case.typed.decisions) for case in dev) == 960

    with pytest.raises(RuntimeError, match="sealed"):
        generate_w6d_confirm()


def test_joint_strata_are_exactly_balanced():
    single = _strata(generate_w6d_single_train())
    multi = _strata(generate_w6d_multi_train())
    dev = _strata(generate_w6d_dev())
    confirm = _strata(generate_w6d_confirm(allow_confirm=True))

    assert len(single) == len(multi) == len(dev) == len(confirm) == 48
    assert set(single.values()) == {8}
    assert set(multi.values()) == {8}
    assert set(dev.values()) == {4}
    assert set(confirm.values()) == {4}


def test_multi_source_budget_is_balanced_and_uses_shared_a_subset():
    single = generate_w6d_single_train()
    multi = generate_w6d_multi_train()

    domains = Counter(case.domain_id for case in multi)
    assert domains == {"A": 96, "B": 96, "C": 96, "D": 96}
    assert {case.split for case in multi} == {"train-multi"}
    assert {case.split for case in single} == {"train-single"}

    single_semantics = {_gold_signature(case) for case in single}
    multi_a_semantics = {
        _gold_signature(case)
        for case in multi
        if case.domain_id == "A"
    }
    assert len(multi_a_semantics) == 96
    assert multi_a_semantics <= single_semantics


def test_domains_have_pairwise_disjoint_field_values_and_templates():
    value_sets = domain_value_sets()
    ids = sorted(value_sets)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            assert value_sets[left].isdisjoint(value_sets[right])

    all_templates = [
        template
        for spec in (DOMAIN_A, DOMAIN_B, DOMAIN_C, DOMAIN_D, DOMAIN_E, DOMAIN_F)
        for template in spec.templates
    ]
    assert len(all_templates) == len(set(all_templates))
    assert len(all_w6d_values()) == 6 * 4 * 24


def test_w6d_values_do_not_overlap_w6b_w6c_or_w5_authority_values():
    from nmd.typed_competitive_authority import all_w6b_values
    from nmd.typed_reliability_authority import all_w6c_values
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab

    prior = (
        set(all_w6b_values())
        | set(all_w6c_values())
        | set(all_w5f_vocab())
        | set(all_w5g_vocab())
        | set(all_w5h_vocab())
        | set(all_w5i_vocab())
    )
    overlap = all_w6d_values() & prior
    assert not overlap, overlap


def test_train_dev_confirm_semantics_and_ids_are_disjoint():
    single = generate_w6d_single_train()
    multi = generate_w6d_multi_train()
    dev = generate_w6d_dev()
    confirm = generate_w6d_confirm(allow_confirm=True)

    for cases in (single, multi, dev, confirm):
        ids = [case.typed.case_id for case in cases]
        semantics = [_gold_signature(case) for case in cases]
        assert len(ids) == len(set(ids))
        assert len(semantics) == len(set(semantics))

    train_semantics = {
        _gold_signature(case)
        for case in [*single, *multi]
    }
    dev_semantics = {_gold_signature(case) for case in dev}
    confirm_semantics = {_gold_signature(case) for case in confirm}
    assert train_semantics.isdisjoint(dev_semantics)
    assert train_semantics.isdisjoint(confirm_semantics)
    assert dev_semantics.isdisjoint(confirm_semantics)


def test_every_case_has_same_five_typed_primitives_and_valid_soft_targets():
    samples = (
        generate_w6d_single_train()[:24]
        + generate_w6d_multi_train()[:24]
        + generate_w6d_dev()[:24]
    )
    for case in samples:
        decisions = case.typed.decisions
        assert [d.question_id for d in decisions] == [
            "diagnosis",
            "response",
            "needs_review",
            "risk",
            "urgency",
        ]
        assert [d.primitive for d in decisions] == [
            "choice",
            "choice",
            "noul",
            "score",
            "score",
        ]
        for decision in decisions:
            probs = decision.gold_probabilities
            assert abs(sum(probs) - 1.0) < 1e-12
            assert probs[decision.gold_index] == max(probs)
