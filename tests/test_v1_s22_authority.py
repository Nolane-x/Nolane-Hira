from nmd.v1_s22_authority import generate_s22_cases, validate_s22_partitions


def test_s22_authority_partition_shape_and_fresh_views():
    train = generate_s22_cases("train")
    dev = generate_s22_cases("dev")
    validate_s22_partitions(train, dev)
    assert len(train) == 768
    assert len(dev) == 192
    assert len({row.domain for row in train}) == 12
    assert len({row.domain for row in dev}) == 12
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(len(set(row.option_ids)) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s22_authority_train_dev_text_isolation():
    train = generate_s22_cases("train")
    dev = generate_s22_cases("dev")
    train_states = {x for row in train for x in (row.state_a, row.state_b)}
    dev_states = {x for row in dev for x in (row.state_a, row.state_b)}
    assert train_states.isdisjoint(dev_states)
    train_q = {x for row in train for x in (row.question_a1,row.question_a2,row.question_b1,row.question_b2)}
    dev_q = {x for row in dev for x in (row.question_a1,row.question_a2,row.question_b1,row.question_b2)}
    assert train_q.isdisjoint(dev_q)
    train_options = {x for row in train for x in (*row.option_texts,*row.option_aliases)}
    dev_options = {x for row in dev for x in (*row.option_texts,*row.option_aliases)}
    assert train_options.isdisjoint(dev_options)


def test_s22_authority_is_deterministic():
    assert generate_s22_cases("train") == generate_s22_cases("train")
    assert generate_s22_cases("dev") == generate_s22_cases("dev")
