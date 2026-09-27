import pytest

from nmd.mainline_m1_authority import (
    AUTHORITY_PARTITIONS,
    CONFIDENCE_BANDS,
    OOD_KINDS,
    all_m1_authority_text_atoms,
    generate_m1_authority,
)
from nmd.typed_reliability_authority import all_w6c_values
from nmd.w33_transfer_authority import all_w33_text_atoms
from nmd.w34_transfer_authority import all_w34_text_atoms


def test_m1_authority_partition_counts_and_balance():
    expected = {
        "cal_train": 108,
        "cal_dev": 36,
        "selective_confirm": 36,
        "ood_train": 72,
        "ood_dev": 36,
        "ood_confirm": 36,
    }

    for partition, count in expected.items():
        rows = generate_m1_authority(
            partition,
            allow_sealed=partition in {"selective_confirm", "ood_confirm"},
        )
        assert len(rows) == count
        assert all(row.partition == partition for row in rows)

        for domain in AUTHORITY_PARTITIONS[partition]:
            subset = [row for row in rows if row.domain_id == domain]
            assert len(subset) == 36
            assert [sum(row.primitive == p for row in subset) for p in ("choice", "score", "noul")] == [12, 12, 12]

            if partition.startswith("ood_"):
                assert all(row.is_ood for row in subset)
                assert {row.ood_kind for row in subset} == set(OOD_KINDS)
                assert all(row.gold_index is None for row in subset)
                assert all(row.gold_probabilities is None for row in subset)
            else:
                assert all(not row.is_ood for row in subset)
                assert {row.confidence_band for row in subset} == set(CONFIDENCE_BANDS)
                assert all(row.gold_index is not None for row in subset)
                assert all(row.gold_probabilities is not None for row in subset)


def test_m1_sealed_authorities_fail_closed_before_marker():
    with pytest.raises(RuntimeError, match="sealed"):
        generate_m1_authority("selective_confirm")
    with pytest.raises(RuntimeError, match="sealed"):
        generate_m1_authority("ood_confirm")


def test_m1_case_ids_are_disjoint_across_all_partitions():
    seen = set()
    for partition in AUTHORITY_PARTITIONS:
        rows = generate_m1_authority(
            partition,
            allow_sealed=partition in {"selective_confirm", "ood_confirm"},
        )
        ids = {row.case_id for row in rows}
        assert len(ids) == len(rows)
        assert seen.isdisjoint(ids)
        seen |= ids


def test_m1_id_soft_targets_are_valid_and_primitive_options_are_typed():
    for partition in ("cal_train", "cal_dev", "selective_confirm"):
        rows = generate_m1_authority(
            partition,
            allow_sealed=partition == "selective_confirm",
        )
        for row in rows:
            assert row.gold_index is not None
            assert row.gold_probabilities is not None
            assert len(row.gold_probabilities) == len(row.options)
            assert abs(sum(row.gold_probabilities) - 1.0) <= 1e-12
            assert row.gold_probabilities[row.gold_index] == max(row.gold_probabilities)

            if row.primitive == "noul":
                assert tuple(float(option.value) for option in row.options) == (0.0, 1.0)
            elif row.primitive == "score":
                assert tuple(float(option.value) for option in row.options) == (0.0, 1.0, 2.0)
            else:
                assert all(option.value is None for option in row.options)


def test_m1_ood_rows_have_no_fake_gold_decision():
    for partition in ("ood_train", "ood_dev", "ood_confirm"):
        rows = generate_m1_authority(
            partition,
            allow_sealed=partition == "ood_confirm",
        )
        assert all(row.is_ood for row in rows)
        assert all(row.gold_index is None for row in rows)
        assert all(row.gold_probabilities is None for row in rows)
        assert all(row.confidence_band is None for row in rows)


def test_m1_authority_text_is_exactly_fresh_against_recent_exposed_authorities():
    current = all_m1_authority_text_atoms(include_sealed=True)
    assert not (current & all_w34_text_atoms())
    assert not (current & all_w33_text_atoms())
    assert not (current & set(all_w6c_values()))


def test_m1_ood_and_calibration_states_are_disjoint():
    calibration = {
        row.state_text
        for partition in ("cal_train", "cal_dev", "selective_confirm")
        for row in generate_m1_authority(
            partition,
            allow_sealed=partition == "selective_confirm",
        )
    }
    ood = {
        row.state_text
        for partition in ("ood_train", "ood_dev", "ood_confirm")
        for row in generate_m1_authority(
            partition,
            allow_sealed=partition == "ood_confirm",
        )
    }
    assert calibration.isdisjoint(ood)
