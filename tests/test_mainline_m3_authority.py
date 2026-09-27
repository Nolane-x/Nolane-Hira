import pytest

from nmd.mainline_m2_authority import all_m2_high_k_text_atoms
from nmd.mainline_m3_authority import (
    M3_PARTITIONS,
    M3_SEALED_PARTITIONS,
    all_m3_text_atoms,
    generate_m3_paired_authority,
)
from nmd.w34_transfer_authority import all_w34_text_atoms


def test_m3_paired_authority_counts_and_balance():
    dev = generate_m3_paired_authority("dev")
    confirm = generate_m3_paired_authority("confirm", allow_sealed=True)

    assert len(dev) == 72
    assert len(confirm) == 36
    assert M3_PARTITIONS == {
        "dev": ("MVA", "MVB"),
        "confirm": ("MVC",),
    }
    assert M3_SEALED_PARTITIONS == {"confirm"}

    for domain in ("MVA", "MVB"):
        subset = [row for row in dev if row.domain_id == domain]
        assert len(subset) == 36
        assert [sum(row.primitive == p for row in subset) for p in ("choice", "score", "noul")] == [12, 12, 12]


def test_m3_confirm_fails_closed_before_marker():
    with pytest.raises(RuntimeError, match="sealed"):
        generate_m3_paired_authority("confirm")


def test_m3_pair_ids_are_unique_and_partition_disjoint():
    dev = generate_m3_paired_authority("dev")
    confirm = generate_m3_paired_authority("confirm", allow_sealed=True)
    dev_ids = {row.pair_id for row in dev}
    confirm_ids = {row.pair_id for row in confirm}

    assert len(dev_ids) == len(dev)
    assert len(confirm_ids) == len(confirm)
    assert dev_ids.isdisjoint(confirm_ids)


def test_m3_en_vi_option_identity_and_values_are_exactly_paired():
    for partition in ("dev", "confirm"):
        rows = generate_m3_paired_authority(
            partition,
            allow_sealed=partition == "confirm",
        )
        for row in rows:
            assert tuple(o.option_id for o in row.en_options) == tuple(
                o.option_id for o in row.vi_options
            )
            assert tuple(o.value for o in row.en_options) == tuple(
                o.value for o in row.vi_options
            )
            assert 0 <= row.gold_index < len(row.en_options)
            assert len(row.en_options) == len(row.vi_options)
            assert row.en_state_text != row.vi_state_text
            assert row.en_question_text != row.vi_question_text


def test_m3_score_and_noul_typed_values_survive_translation():
    rows = generate_m3_paired_authority("dev")
    for row in rows:
        if row.primitive == "score":
            assert tuple(float(o.value) for o in row.en_options) == (0.0, 1.0, 2.0)
            assert tuple(float(o.value) for o in row.vi_options) == (0.0, 1.0, 2.0)
        elif row.primitive == "noul":
            assert tuple(float(o.value) for o in row.en_options) == (0.0, 1.0)
            assert tuple(float(o.value) for o in row.vi_options) == (0.0, 1.0)
        else:
            assert all(o.value is None for o in row.en_options)
            assert all(o.value is None for o in row.vi_options)


def test_m3_text_surface_is_fresh_against_m2_and_w34():
    current = all_m3_text_atoms(include_sealed=True)
    assert not (current & all_m2_high_k_text_atoms(include_sealed=True))
    assert not (current & all_w34_text_atoms())


def test_m3_english_and_vietnamese_surfaces_are_disjoint():
    rows = (
        generate_m3_paired_authority("dev")
        + generate_m3_paired_authority("confirm", allow_sealed=True)
    )
    en = set()
    vi = set()
    for row in rows:
        en.update((row.en_state_text, row.en_question_text))
        vi.update((row.vi_state_text, row.vi_question_text))
        for option in row.en_options:
            en.add(option.criterion_text)
            en.update(option.aliases)
        for option in row.vi_options:
            vi.add(option.criterion_text)
            vi.update(option.aliases)
    assert en.isdisjoint(vi)
