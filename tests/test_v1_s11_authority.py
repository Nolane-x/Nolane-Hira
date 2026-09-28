from nmd.v1_s11_authority import generate_s11_cases, validate_s11_partitions


def test_s11_partition_contract_and_fresh_templates():
    train = generate_s11_cases("train")
    dev = generate_s11_cases("dev")
    validate_s11_partitions(train, dev)

    assert len(train) == 768
    assert len(dev) == 192
    assert len({row.domain for row in train}) == 12
    assert len({row.domain for row in dev}) == 12
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))

    assert {row.state_a for row in train}.isdisjoint({row.state_a for row in dev})
    assert {row.question_a1 for row in train}.isdisjoint(
        {row.question_a1 for row in dev}
    )


def test_s11_generation_is_deterministic():
    a = generate_s11_cases("train")
    b = generate_s11_cases("train")
    assert a == b


def test_s11_uses_new_domain_families_not_s10():
    s11 = {row.domain for row in generate_s11_cases("train")}
    s10 = {
        "geothermal_well",
        "satellite_downlink",
        "fermentation_tank",
        "metro_platform",
        "lab_incubator",
        "shipping_container",
        "laser_cutter",
        "river_station",
        "camera_rig",
        "orchard_block",
        "compressor_stage",
        "museum_case",
    }
    assert s11.isdisjoint(s10)
