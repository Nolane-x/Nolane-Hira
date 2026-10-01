from nmd.v1_s28_authority import generate_s28_cases, validate_s28_partitions


def test_s28_partition_sizes_domains_and_disjointness():
    train=generate_s28_cases("train")
    dev=generate_s28_cases("dev")
    validate_s28_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert len({r.domain for r in dev})==12
    assert all(len(r.option_texts)==4 for r in (*train,*dev))


def test_s28_generation_is_deterministic():
    assert generate_s28_cases("train")==generate_s28_cases("train")
    assert generate_s28_cases("dev")==generate_s28_cases("dev")


def test_s28_option_ids_are_opaque():
    rows=generate_s28_cases("dev")[:12]
    assert all("opaque" in oid for r in rows for oid in r.option_ids)
