from collections import Counter

import pytest

from nmd.conjunctive_authority import (
    CONFIRM_AL_SEED,
    CONFIRM_AM_SEED,
    DEV_AK_SEED,
    DOMAIN_AG,
    DOMAIN_AH,
    DOMAIN_AI,
    DOMAIN_AJ,
    DOMAIN_AK,
    DOMAIN_AL,
    DOMAIN_AM,
    FREEFORM_SEED,
    CONJUNCTIVE_PRIMARY_SEED,
    CONJUNCTIVE_REPLICA_SEED,
    SOURCE_AG_SEED,
    SOURCE_AH_SEED,
    SOURCE_AI_SEED,
    SOURCE_AJ_SEED,
    diagnosis_distance_histogram,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w7_confirm,
    generate_w7_dev,
    generate_w7_train,
)


def _strata(rows):
    return Counter(
        (row.diagnosis_k, row.severity, row.confidence)
        for row in rows
    )


def test_w7_seeds_and_split_counts_are_frozen():
    assert (
        SOURCE_AG_SEED,
        SOURCE_AH_SEED,
        SOURCE_AI_SEED,
        SOURCE_AJ_SEED,
        DEV_AK_SEED,
        CONFIRM_AL_SEED,
        CONFIRM_AM_SEED,
        FREEFORM_SEED,
        CONJUNCTIVE_PRIMARY_SEED,
        CONJUNCTIVE_REPLICA_SEED,
    ) == (
        251347,
        251353,
        251359,
        251363,
        252461,
        253567,
        254671,
        1901,
        1907,
        1913,
    )
    train = generate_w7_train()
    dev = generate_w7_dev()
    assert len(train) == 384
    assert len(dev) == 192
    assert Counter(row.domain_id for row in train) == {
        "AG": 96,
        "AH": 96,
        "AI": 96,
        "AJ": 96,
    }
    assert Counter(row.domain_id for row in dev) == {"AK": 192}
    assert set(_strata(train).values()) == {8}
    assert set(_strata(dev).values()) == {4}


def test_confirm_domains_are_sealed_by_default():
    with pytest.raises(RuntimeError, match="sealed"):
        generate_w7_confirm("AL")
    with pytest.raises(RuntimeError, match="sealed"):
        generate_w7_confirm("AM")


def test_w7_domains_are_pairwise_disjoint():
    values = domain_value_sets()
    templates = domain_template_sets()
    roles = domain_role_sets()
    ids = sorted(values)
    assert ids == ["AG", "AH", "AI", "AJ", "AK", "AL", "AM"]
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            assert values[left].isdisjoint(values[right])
            assert templates[left].isdisjoint(templates[right])
            assert roles[left].isdisjoint(roles[right])


def test_w7_k64_distance_anatomy_is_exact():
    rows = [row for row in generate_w7_dev() if row.diagnosis_k == 64]
    assert len(rows) == 48
    for row in rows:
        assert diagnosis_distance_histogram(row) == {
            0: 1,
            1: 12,
            2: 20,
            3: 15,
            4: 16,
        }


def test_freeform_text_and_explicit_factors_are_identical():
    rows = generate_w7_train()[:32] + generate_w7_dev()[:32]
    for row in rows:
        diagnosis = row.typed.decisions[0]
        assert len(diagnosis.options) == len(row.diagnosis_factors)
        assert len(diagnosis.options) == len(row.diagnosis_signatures)
        for option, factors in zip(
            diagnosis.options,
            row.diagnosis_factors,
        ):
            assert option.criterion_text == "; ".join(factors)
            assert len(factors) == 4


def test_all_cases_have_five_typed_decisions_and_unique_w7_ids():
    rows = generate_w7_train() + generate_w7_dev()
    ids = [row.typed.case_id for row in rows]
    assert len(ids) == len(set(ids))
    assert all(case_id.startswith("w7-") for case_id in ids)
    for row in rows:
        decisions = row.typed.decisions
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


def test_domain_registry_is_exactly_ag_to_am():
    assert {
        DOMAIN_AG.domain_id,
        DOMAIN_AH.domain_id,
        DOMAIN_AI.domain_id,
        DOMAIN_AJ.domain_id,
        DOMAIN_AK.domain_id,
        DOMAIN_AL.domain_id,
        DOMAIN_AM.domain_id,
    } == {"AG", "AH", "AI", "AJ", "AK", "AL", "AM"}
