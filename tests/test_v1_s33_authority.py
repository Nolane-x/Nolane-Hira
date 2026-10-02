from nmd.v1_s33_authority import generate_s33_cases, validate_s33_partitions


def test_s33_authority_sizes_domains_and_fresh_partition():
    train=generate_s33_cases("train")
    dev=generate_s33_cases("dev")
    validate_s33_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert all(len(r.option_ids)==4 for r in (*train,*dev))


def test_s33_authority_deterministic():
    assert generate_s33_cases("train")==generate_s33_cases("train")
    assert generate_s33_cases("dev")==generate_s33_cases("dev")
