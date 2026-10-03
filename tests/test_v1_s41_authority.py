from nmd.v1_s41_authority import generate_s41_cases, validate_s41_partitions


def test_s41_partitions_are_frozen_and_disjoint():
    train = generate_s41_cases("train")
    dev = generate_s41_cases("dev")
    validate_s41_partitions(train, dev)
    assert len(train) == 768
    assert len(dev) == 192
    assert len({r.domain for r in train}) == 12
    assert {r.domain for r in train} == {r.domain for r in dev}


def test_s41_k4_opaque_ids_and_paired_gold():
    for row in (*generate_s41_cases("train"), *generate_s41_cases("dev")):
        assert len(row.option_ids) == 4
        assert len(set(row.option_ids)) == 4
        assert row.gold_a != row.gold_b
        assert all("opaque" in x for x in row.option_ids)


def test_s41_train_dev_text_is_exactly_disjoint():
    train = generate_s41_cases("train")
    dev = generate_s41_cases("dev")
    train_text = {
        x
        for r in train
        for x in (
            r.state_a, r.state_b,
            r.question_a1, r.question_a2, r.question_b1, r.question_b2,
            *r.option_texts, *r.option_aliases,
        )
    }
    dev_text = {
        x
        for r in dev
        for x in (
            r.state_a, r.state_b,
            r.question_a1, r.question_a2, r.question_b1, r.question_b2,
            *r.option_texts, *r.option_aliases,
        )
    }
    assert train_text.isdisjoint(dev_text)
