from nmd.v1_s27_authority import generate_s27_cases, validate_s27_partitions


def test_s27_partition_sizes_domains_and_disjointness():
    train = generate_s27_cases("train")
    dev = generate_s27_cases("dev")
    validate_s27_partitions(train, dev)
    assert len(train) == 768
    assert len(dev) == 192
    assert len({r.domain for r in train}) == 12
    assert len({r.domain for r in dev}) == 12
    assert all(len(r.option_texts) == 4 for r in (*train, *dev))


def test_s27_generation_is_deterministic():
    assert generate_s27_cases("train") == generate_s27_cases("train")
    assert generate_s27_cases("dev") == generate_s27_cases("dev")


def test_s27_uses_opaque_option_ids():
    rows = generate_s27_cases("dev")[:12]
    assert all("opaque" in option_id for row in rows for option_id in row.option_ids)
