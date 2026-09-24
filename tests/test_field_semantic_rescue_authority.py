from collections import Counter

import pytest

from nmd.field_semantic_rescue_authority import (
    CONFIRM_Y_SEED,
    CONFIRM_Z_SEED,
    DEV_X_SEED,
    DOMAIN_T,
    DOMAIN_U,
    DOMAIN_V,
    DOMAIN_W,
    DOMAIN_X,
    DOMAIN_Y,
    DOMAIN_Z,
    GLOBAL_SEED,
    SOURCE_T_SEED,
    SOURCE_U_SEED,
    SOURCE_V_SEED,
    SOURCE_W_SEED,
    all_w6h_values,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w6h_confirm,
    generate_w6h_dev,
    generate_w6h_train,
)


def _strata(cases):
    return Counter(
        (case.diagnosis_k, case.severity, case.confidence)
        for case in cases
    )


def test_w6h_seeds_counts_and_confirm_seal_are_frozen():
    assert (
        SOURCE_T_SEED,
        SOURCE_U_SEED,
        SOURCE_V_SEED,
        SOURCE_W_SEED,
        DEV_X_SEED,
        CONFIRM_Y_SEED,
        CONFIRM_Z_SEED,
        GLOBAL_SEED,
    ) == (
        221227,
        221231,
        221237,
        221239,
        222347,
        223451,
        224557,
        1601,
    )

    train = generate_w6h_train()
    dev = generate_w6h_dev()
    assert len(train) == 384
    assert len(dev) == 192
    assert sum(len(case.typed.decisions) for case in train) == 1920
    assert sum(len(case.typed.decisions) for case in dev) == 960

    with pytest.raises(RuntimeError, match="sealed"):
        generate_w6h_confirm("Y")
    with pytest.raises(RuntimeError, match="sealed"):
        generate_w6h_confirm("Z")


def test_w6h_train_and_dev_joint_strata_are_balanced():
    train = generate_w6h_train()
    dev = generate_w6h_dev()
    assert Counter(case.domain_id for case in train) == {
        "T": 96,
        "U": 96,
        "V": 96,
        "W": 96,
    }
    train_strata = _strata(train)
    dev_strata = _strata(dev)
    assert len(train_strata) == len(dev_strata) == 48
    assert set(train_strata.values()) == {8}
    assert set(dev_strata.values()) == {4}


def test_w6h_domains_are_pairwise_disjoint_in_values_templates_and_roles():
    values = domain_value_sets()
    templates = domain_template_sets()
    roles = domain_role_sets()
    ids = sorted(values)
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            assert values[left].isdisjoint(values[right])
            assert templates[left].isdisjoint(templates[right])
            assert roles[left].isdisjoint(roles[right])


def test_w6h_values_are_fresh_against_w5_through_w6g():
    from nmd.high_k_localization_authority import all_w6f_values
    from nmd.second_order_localization_authority import all_w6g_values
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd.typed_competitive_authority import all_w6b_values
    from nmd.typed_domain_generalization_authority import all_w6d_values
    from nmd.typed_joint_replication_authority import all_w6e_values
    from nmd.typed_reliability_authority import all_w6c_values

    prior = (
        set(all_w5f_vocab())
        | set(all_w5g_vocab())
        | set(all_w5h_vocab())
        | set(all_w5i_vocab())
        | set(all_w6b_values())
        | set(all_w6c_values())
        | set(all_w6d_values())
        | set(all_w6e_values())
        | set(all_w6f_values())
        | set(all_w6g_values())
    )
    overlap = all_w6h_values() & prior
    assert not overlap, overlap


def test_w6h_case_ids_are_unique_and_prefix_scoped():
    rows = [*generate_w6h_train(), *generate_w6h_dev()]
    ids = [case.typed.case_id for case in rows]
    assert len(ids) == len(set(ids))
    assert all(case_id.startswith("w6h-") for case_id in ids)


def test_w6h_each_case_has_same_five_typed_decisions():
    rows = generate_w6h_train()[:64] + generate_w6h_dev()[:64]
    for case in rows:
        decisions = case.typed.decisions
        assert [decision.question_id for decision in decisions] == [
            "diagnosis",
            "response",
            "needs_review",
            "risk",
            "urgency",
        ]
        assert [decision.primitive for decision in decisions] == [
            "choice",
            "choice",
            "noul",
            "score",
            "score",
        ]
        for decision in decisions:
            assert abs(sum(decision.gold_probabilities) - 1.0) < 1e-12
            assert decision.gold_probabilities[decision.gold_index] == max(
                decision.gold_probabilities
            )


def test_domain_registry_is_exactly_t_to_z():
    assert {
        DOMAIN_T.domain_id,
        DOMAIN_U.domain_id,
        DOMAIN_V.domain_id,
        DOMAIN_W.domain_id,
        DOMAIN_X.domain_id,
        DOMAIN_Y.domain_id,
        DOMAIN_Z.domain_id,
    } == {"T", "U", "V", "W", "X", "Y", "Z"}
