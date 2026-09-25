from __future__ import annotations

from collections import defaultdict

import pytest

from nmd.semantic_alignment_authority import (
    ALL_DOMAINS,
    BASES_PER_DOMAIN,
    CONFIRM_DOMAINS,
    DOMAIN_SEEDS,
    INTENTS,
    K_VALUES,
    STATE_VARIANTS_PER_INTENT,
    TRAIN_DOMAINS,
    DEV_DOMAINS,
    VIEW_IDS,
    alignment_training_views,
    generate_w9_domain,
    generate_w9_split,
)


def test_w9_domain_and_seed_contracts():
    assert ALL_DOMAINS == ("AY", "AZ", "BA", "BB", "BC", "BD", "BE")
    assert TRAIN_DOMAINS == ("AY", "AZ", "BA", "BB")
    assert DEV_DOMAINS == ("BC",)
    assert CONFIRM_DOMAINS == ("BD", "BE")
    assert set(DOMAIN_SEEDS) == set(ALL_DOMAINS)
    assert len(set(DOMAIN_SEEDS.values())) == len(ALL_DOMAINS)


def test_every_domain_has_exact_fresh_semantic_shape():
    for domain_id in ALL_DOMAINS:
        intents = INTENTS[domain_id]
        assert len(intents) == 16
        assert len({row.intent_id for row in intents}) == 16
        assert len({row.terse_label for row in intents}) == 16
        assert len({row.natural_definition for row in intents}) == 16
        assert all(len(row.state_texts) == STATE_VARIANTS_PER_INTENT for row in intents)

        for intent in intents:
            label = intent.terse_label.lower()
            assert all(label not in state.lower() for state in intent.state_texts)


def test_domain_views_have_exact_counts_and_paired_identity():
    rows = generate_w9_domain("AY")
    assert len(rows) == BASES_PER_DOMAIN * len(K_VALUES) * len(VIEW_IDS)

    by_base = defaultdict(list)
    for row in rows:
        by_base[row.base_id].append(row)

    assert len(by_base) == BASES_PER_DOMAIN
    for base_rows in by_base.values():
        assert len(base_rows) == len(K_VALUES) * len(VIEW_IDS)
        states = {row.state_text for row in base_rows}
        intents = {row.intent_id for row in base_rows}
        questions = {row.question_text for row in base_rows}
        assert len(states) == len(intents) == len(questions) == 1

        by_k = defaultdict(dict)
        for row in base_rows:
            by_k[row.diagnosis_k][row.view_id] = row

        for k in K_VALUES:
            assert set(by_k[k]) == set(VIEW_IDS)
            label = by_k[k]["label"]
            definition = by_k[k]["definition"]
            assert label.option_ids == definition.option_ids
            assert label.gold_index == definition.gold_index
            assert len(label.option_ids) == k
            assert label.option_ids[label.gold_index] == label.intent_id

        ids4 = set(by_k[4]["definition"].option_ids)
        ids8 = set(by_k[8]["definition"].option_ids)
        ids16 = set(by_k[16]["definition"].option_ids)
        assert ids4 < ids8 < ids16

        order16 = list(by_k[16]["definition"].option_ids)
        for k in (4, 8):
            observed = list(by_k[k]["definition"].option_ids)
            expected = [intent_id for intent_id in order16 if intent_id in set(observed)]
            assert observed == expected


def test_confirm_generation_is_capability_guarded():
    with pytest.raises(PermissionError):
        generate_w9_split("confirm")
    with pytest.raises(PermissionError):
        generate_w9_domain("BD")
    with pytest.raises(PermissionError):
        generate_w9_domain("BE")


def test_train_and_dev_never_materialize_confirm_domains():
    train = generate_w9_split("train")
    dev = generate_w9_split("dev")
    assert {row.domain_id for row in train} == set(TRAIN_DOMAINS)
    assert {row.domain_id for row in dev} == set(DEV_DOMAINS)
    assert not ({row.domain_id for row in train + dev} & set(CONFIRM_DOMAINS))


def test_alignment_training_views_are_definition_k16_once_per_base():
    rows = alignment_training_views()
    assert len(rows) == len(TRAIN_DOMAINS) * BASES_PER_DOMAIN
    assert all(row.view_id == "definition" for row in rows)
    assert all(row.diagnosis_k == 16 for row in rows)
    assert len({row.base_id for row in rows}) == len(rows)
