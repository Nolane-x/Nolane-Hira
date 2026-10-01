from nmd.v1_s29_authority import generate_s29_cases, validate_s29_partitions

def test_s29_authority_sizes_and_fresh_partition():
    train=generate_s29_cases("train")
    dev=generate_s29_cases("dev")
    validate_s29_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert all(len(r.option_ids)==4 for r in (*train,*dev))

def test_s29_authority_deterministic():
    assert generate_s29_cases("train")==generate_s29_cases("train")
    assert generate_s29_cases("dev")==generate_s29_cases("dev")
