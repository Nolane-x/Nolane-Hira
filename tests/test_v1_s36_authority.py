from nmd.v1_s36_authority import generate_s36_cases, validate_s36_partitions


def test_s36_authority_sizes_domains_and_partitions():
    train = generate_s36_cases("train")
    dev = generate_s36_cases("dev")
    validate_s36_partitions(train, dev)
    assert len(train) == 768
    assert len(dev) == 192
    assert len({r.domain for r in train}) == 12
    assert len({r.domain for r in dev}) == 12
    assert all(len(r.option_ids) == 4 for r in (*train, *dev))


def test_s36_authority_is_deterministic():
    a = generate_s36_cases("train")
    b = generate_s36_cases("train")
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
