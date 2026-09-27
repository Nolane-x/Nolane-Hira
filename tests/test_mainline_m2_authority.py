import pytest

from nmd.mainline_m1_authority import all_m1_authority_text_atoms
from nmd.mainline_m1_r2_authority import all_m1_r2_text_atoms
from nmd.mainline_m2_authority import (
    M2_SEMANTIC_CASES,
    M2_SEMANTIC_PARTITIONS,
    all_m2_high_k_text_atoms,
    generate_m2_high_k_semantic,
)
from nmd.w34_transfer_authority import all_w34_text_atoms


def test_m2_semantic_partition_counts_and_k_are_frozen():
    train = generate_m2_high_k_semantic("train")
    dev = generate_m2_high_k_semantic("dev")
    confirm = generate_m2_high_k_semantic("confirm", allow_sealed=True)

    assert len(train) == 4 * M2_SEMANTIC_CASES["train"] == 64
    assert len(dev) == 2 * M2_SEMANTIC_CASES["dev"] == 32
    assert len(confirm) == M2_SEMANTIC_CASES["confirm"] == 24

    assert {row.k for row in train} == {4, 8, 16, 32}
    assert {row.k for row in dev} == {64, 128}
    assert {row.k for row in confirm} == {255}

    assert M2_SEMANTIC_PARTITIONS == {
        "train": {4: "HKA", 8: "HKB", 16: "HKC", 32: "HKD"},
        "dev": {64: "HKE", 128: "HKF"},
        "confirm": {255: "HKG"},
    }


def test_m2_k255_confirm_fails_closed_before_marker():
    with pytest.raises(RuntimeError, match="remains sealed"):
        generate_m2_high_k_semantic("confirm")


def test_m2_semantic_rows_are_unique_and_well_formed():
    seen_ids = set()
    seen_states = set()

    for partition in M2_SEMANTIC_PARTITIONS:
        rows = generate_m2_high_k_semantic(
            partition,
            allow_sealed=partition == "confirm",
        )
        for row in rows:
            assert row.case_id not in seen_ids
            assert row.state_text not in seen_states
            seen_ids.add(row.case_id)
            seen_states.add(row.state_text)

            assert len(row.options) == row.k
            assert 0 <= row.gold_index < row.k
            assert len({option.option_id for option in row.options}) == row.k
            assert len({option.criterion_text for option in row.options}) == row.k
            assert all(len(option.aliases) == 1 for option in row.options)
            assert all(len(option.exemplars) == 1 for option in row.options)


def test_m2_semantic_exact_text_is_fresh_against_recent_authorities():
    current = all_m2_high_k_text_atoms(include_sealed=True)

    assert not (current & all_w34_text_atoms())
    assert not (current & all_m1_authority_text_atoms(include_sealed=True))
    assert not (current & all_m1_r2_text_atoms())


def test_m2_train_dev_and_confirm_states_are_disjoint():
    train = {
        row.state_text
        for row in generate_m2_high_k_semantic("train")
    }
    dev = {
        row.state_text
        for row in generate_m2_high_k_semantic("dev")
    }
    confirm = {
        row.state_text
        for row in generate_m2_high_k_semantic(
            "confirm",
            allow_sealed=True,
        )
    }

    assert train.isdisjoint(dev)
    assert train.isdisjoint(confirm)
    assert dev.isdisjoint(confirm)
