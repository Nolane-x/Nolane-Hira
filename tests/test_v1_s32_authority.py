from nmd.v1_s32_authority import generate_s32_cases, validate_s32_partitions


def test_s32_authority_sizes_domains_and_fresh_partition():
    train=generate_s32_cases("train")
    dev=generate_s32_cases("dev")
    validate_s32_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert all(len(r.option_ids)==4 for r in (*train,*dev))


def test_s32_authority_deterministic():
    assert generate_s32_cases("train")==generate_s32_cases("train")
    assert generate_s32_cases("dev")==generate_s32_cases("dev")
