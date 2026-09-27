from nmd.mainline_m1_authority import all_m1_authority_text_atoms
from nmd.mainline_m1_r2_authority import (
    R2_CONFIDENCE_BANDS,
    R2_PARTITIONS,
    all_m1_r2_text_atoms,
    generate_m1_r2_authority,
)
from nmd.typed_reliability_authority import all_w6c_values
from nmd.w34_transfer_authority import all_w34_text_atoms


def test_m1_r2_counts_balance_and_disjoint_partitions():
    expected = {"train": 144, "dev": 72}
    seen = set()
    for partition, count in expected.items():
        rows = generate_m1_r2_authority(partition)
        assert len(rows) == count
        ids = {row.case_id for row in rows}
        assert len(ids) == len(rows)
        assert seen.isdisjoint(ids)
        seen |= ids

        for domain in R2_PARTITIONS[partition]:
            subset = [row for row in rows if row.domain_id == domain]
            assert len(subset) == 36
            assert [sum(row.primitive == p for row in subset) for p in ("choice", "score", "noul")] == [12, 12, 12]
            assert {row.confidence_band for row in subset} == set(R2_CONFIDENCE_BANDS)
            assert all(row.is_ood is False for row in subset)


def test_m1_r2_targets_and_typed_values_are_valid():
    for partition in R2_PARTITIONS:
        for row in generate_m1_r2_authority(partition):
            assert len(row.gold_probabilities) == len(row.options)
            assert abs(sum(row.gold_probabilities) - 1.0) <= 1e-12
            assert row.gold_probabilities[row.gold_index] == max(row.gold_probabilities)

            if row.primitive == "noul":
                assert tuple(float(option.value) for option in row.options) == (0.0, 1.0)
            elif row.primitive == "score":
                assert tuple(float(option.value) for option in row.options) == (0.0, 1.0, 2.0)
            else:
                assert all(option.value is None for option in row.options)


def test_m1_r2_text_is_fresh_against_r1_w34_and_w6c():
    current = all_m1_r2_text_atoms()
    assert not (current & all_m1_authority_text_atoms(include_sealed=True))
    assert not (current & all_w34_text_atoms())
    assert not (current & set(all_w6c_values()))


def test_m1_r2_train_dev_states_are_disjoint():
    train = {row.state_text for row in generate_m1_r2_authority("train")}
    dev = {row.state_text for row in generate_m1_r2_authority("dev")}
    assert train.isdisjoint(dev)
