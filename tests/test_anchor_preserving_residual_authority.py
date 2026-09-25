from __future__ import annotations

from collections import Counter

import pytest

from nmd.anchor_preserving_residual_authority import (
    CONFIRM_DOMAINS,
    DEV_DOMAIN,
    K_VALUES,
    TRAIN_DOMAINS,
    all_w15_text_atoms,
    generate_w15_confirm,
    generate_w15_dev,
    generate_w15_train,
)


def _assert_case_contract(case):
    assert case.domain_id in {*TRAIN_DOMAINS, DEV_DOMAIN, *CONFIRM_DOMAINS}
    assert case.diagnosis_k in K_VALUES
    assert len(case.typed.decisions) == 5
    assert [d.question_id for d in case.typed.decisions] == [
        "diagnosis",
        "response",
        "needs_review",
        "risk",
        "urgency",
    ]
    assert [d.primitive for d in case.typed.decisions] == [
        "choice",
        "choice",
        "noul",
        "score",
        "score",
    ]
    assert len(case.option_definitions) == 5

    for decision, definitions in zip(
        case.typed.decisions,
        case.option_definitions,
    ):
        assert len(definitions) == len(decision.options)
        for option, paraphrases in zip(decision.options, definitions):
            assert len(paraphrases) == 3
            assert len(set(paraphrases)) == 3
            assert option.criterion_text == paraphrases[0]

    assert len(case.typed.decisions[0].options) == case.diagnosis_k
    assert len(case.typed.decisions[1].options) == 4
    assert len(case.typed.decisions[2].options) == 2
    assert len(case.typed.decisions[3].options) == 4
    assert len(case.typed.decisions[4].options) == 4

    for decision in case.typed.decisions:
        assert len(decision.gold_probabilities) == len(decision.options)
        assert abs(sum(decision.gold_probabilities) - 1.0) <= 1e-9
        assert max(range(len(decision.gold_probabilities)), key=decision.gold_probabilities.__getitem__) == decision.gold_index


def test_w15_train_shape_balance_and_typed_contracts():
    rows = generate_w15_train()
    assert len(rows) == 384
    assert Counter(row.domain_id for row in rows) == {
        "BZ": 96,
        "CA": 96,
        "CB": 96,
        "CC": 96,
    }
    for domain in TRAIN_DOMAINS:
        domain_rows = [row for row in rows if row.domain_id == domain]
        assert Counter(row.diagnosis_k for row in domain_rows) == {
            4: 32,
            8: 32,
            16: 32,
        }
    for row in rows:
        assert row.split == "train"
        _assert_case_contract(row)


def test_w15_dev_shape_balance_and_fresh_ids():
    rows = generate_w15_dev()
    assert len(rows) == 96
    assert {row.domain_id for row in rows} == {"CD"}
    assert Counter(row.diagnosis_k for row in rows) == {
        4: 32,
        8: 32,
        16: 32,
    }
    assert len({row.typed.case_id for row in rows}) == 96
    assert all(row.split == "dev-cd" for row in rows)
    for row in rows:
        _assert_case_contract(row)


def test_w15_confirm_capability_is_sealed_by_default():
    for domain in CONFIRM_DOMAINS:
        with pytest.raises(RuntimeError, match="sealed"):
            generate_w15_confirm(domain)


def test_w15_domain_case_ids_and_state_text_are_disjoint():
    train = generate_w15_train()
    dev = generate_w15_dev()
    rows = [*train, *dev]

    ids = [row.typed.case_id for row in rows]
    assert len(ids) == len(set(ids))

    # Every domain has its own subject vocabulary, so complete state strings
    # must not cross domains.
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row.domain_id, set()).add(row.typed.state_text)
    domains = sorted(by_domain)
    for i, left in enumerate(domains):
        for right in domains[i + 1 :]:
            assert by_domain[left].isdisjoint(by_domain[right])


def test_w15_text_atom_manifest_contains_all_schema_views():
    atoms = all_w15_text_atoms()
    assert len(atoms) > 250
    assert all(isinstance(text, str) and text.strip() for text in atoms)
    rows = generate_w15_train()[:8]
    for row in rows:
        for definitions in row.option_definitions:
            for paraphrases in definitions:
                assert set(paraphrases) <= atoms
