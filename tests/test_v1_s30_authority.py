from nmd.v1_s30_authority import generate_s30_cases, validate_s30_partitions


def test_s30_authority_sizes_domains_and_fresh_partition():
    train=generate_s30_cases("train")
    dev=generate_s30_cases("dev")
    validate_s30_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert all(len(r.option_ids)==4 for r in (*train,*dev))


def test_s30_authority_is_deterministic():
    assert generate_s30_cases("train")==generate_s30_cases("train")
    assert generate_s30_cases("dev")==generate_s30_cases("dev")
