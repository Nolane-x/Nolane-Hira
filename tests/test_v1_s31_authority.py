from nmd.v1_s31_authority import generate_s31_cases,validate_s31_partitions


def test_s31_authority_sizes_and_partition():
    train=generate_s31_cases("train")
    dev=generate_s31_cases("dev")
    validate_s31_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert all(len(r.option_ids)==4 for r in (*train,*dev))


def test_s31_authority_deterministic():
    assert generate_s31_cases("train")==generate_s31_cases("train")
    assert generate_s31_cases("dev")==generate_s31_cases("dev")
