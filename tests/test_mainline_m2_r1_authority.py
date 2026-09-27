from nmd.mainline_m2_authority import (
    all_m2_high_k_text_atoms,
    generate_m2_high_k_semantic,
)
from nmd.mainline_m2_r1_authority import (
    M2_R1_DEV_CASES_PER_K,
    M2_R1_DEV_PARTITIONS,
    all_m2_r1_dev_text_atoms,
    generate_m2_r1_dev,
)


def test_m2_r1_dev_shape_and_balance():
    rows = generate_m2_r1_dev()
    assert len(rows) == 2 * M2_R1_DEV_CASES_PER_K
    assert M2_R1_DEV_PARTITIONS == {64: "HKH", 128: "HKI"}

    for k, domain in M2_R1_DEV_PARTITIONS.items():
        subset = [row for row in rows if row.k == k]
        assert len(subset) == M2_R1_DEV_CASES_PER_K
        assert {row.domain_id for row in subset} == {domain}
        assert all(row.partition == "r1-dev" for row in subset)
        assert all(len(row.options) == k for row in subset)
        assert all(0 <= row.gold_index < k for row in subset)


def test_m2_r1_dev_ids_and_option_ids_are_unique():
    rows = generate_m2_r1_dev()
    case_ids = [row.case_id for row in rows]
    assert len(case_ids) == len(set(case_ids))
    for row in rows:
        option_ids = [option.option_id for option in row.options]
        assert len(option_ids) == len(set(option_ids))


def test_m2_r1_dev_text_is_exactly_disjoint_from_hka_hkg():
    current = all_m2_r1_dev_text_atoms()
    prior_and_sealed = all_m2_high_k_text_atoms(include_sealed=True)
    assert current.isdisjoint(prior_and_sealed)


def test_m2_r1_dev_is_disjoint_from_reserved_train_states():
    train_states = {
        row.state_text for row in generate_m2_high_k_semantic("train")
    }
    dev_states = {row.state_text for row in generate_m2_r1_dev()}
    assert train_states.isdisjoint(dev_states)


def test_m2_r1_dev_uses_alternate_surface_templates():
    rows = generate_m2_r1_dev()
    assert all("R1" in row.state_text for row in rows)
    assert all(
        option.criterion_text.startswith("Handling rule:")
        for row in rows
        for option in row.options
    )
