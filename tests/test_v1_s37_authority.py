from nmd.v1_s37_authority import generate_s37_cases, validate_s37_partitions


def test_s37_authority_sizes_domains_and_partitions():
    train = generate_s37_cases("train")
    dev = generate_s37_cases("dev")
    validate_s37_partitions(train, dev)
    assert len(train) == 768
    assert len(dev) == 192
    assert len({r.domain for r in train}) == 12
    assert len({r.domain for r in dev}) == 12
    assert all(len(r.option_ids) == 4 for r in (*train, *dev))


def test_s37_authority_is_deterministic():
    a = generate_s37_cases("train")
    b = generate_s37_cases("train")
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]


def test_s37_train_dev_exact_text_partitions_are_disjoint():
    train = generate_s37_cases("train")
    dev = generate_s37_cases("dev")
    ts = {x for r in train for x in (r.state_a, r.state_b)}
    ds = {x for r in dev for x in (r.state_a, r.state_b)}
    tq = {x for r in train for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    dq = {x for r in dev for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}
    to = {x for r in train for x in (*r.option_texts, *r.option_aliases)}
    do = {x for r in dev for x in (*r.option_texts, *r.option_aliases)}
    assert not (ts & ds)
    assert not (tq & dq)
    assert not (to & do)
