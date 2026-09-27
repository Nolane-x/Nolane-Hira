from nmd.mainline_m2_authority import all_m2_high_k_text_atoms
from nmd.mainline_m3_authority import all_m3_text_atoms
from nmd.mainline_m3_r1_authority import (
    M3_R1_PARTITIONS,
    all_m3_r1_text_atoms,
    generate_m3_r1_authority,
)
from nmd.w34_transfer_authority import all_w34_text_atoms


def test_m3_r1_authority_counts_balance_and_partition_disjointness():
    train = generate_m3_r1_authority("train")
    dev = generate_m3_r1_authority("dev")

    assert M3_R1_PARTITIONS == {
        "train": ("MVD", "MVE", "MVF", "MVG"),
        "dev": ("MVH", "MVI"),
    }
    assert len(train) == 144
    assert len(dev) == 72

    train_ids = {row.pair_id for row in train}
    dev_ids = {row.pair_id for row in dev}
    assert len(train_ids) == len(train)
    assert len(dev_ids) == len(dev)
    assert train_ids.isdisjoint(dev_ids)

    for partition, domains in M3_R1_PARTITIONS.items():
        rows = generate_m3_r1_authority(partition)
        for domain in domains:
            subset = [row for row in rows if row.domain_id == domain]
            assert len(subset) == 36
            assert [
                sum(row.primitive == primitive for row in subset)
                for primitive in ("choice", "score", "noul")
            ] == [12, 12, 12]


def test_m3_r1_en_vi_option_ids_values_and_gold_are_exactly_paired():
    for partition in M3_R1_PARTITIONS:
        for row in generate_m3_r1_authority(partition):
            assert tuple(option.option_id for option in row.en_options) == tuple(
                option.option_id for option in row.vi_options
            )
            assert tuple(option.value for option in row.en_options) == tuple(
                option.value for option in row.vi_options
            )
            assert len(row.en_options) == len(row.vi_options)
            assert 0 <= row.gold_index < len(row.en_options)
            assert (
                row.en_options[row.gold_index].option_id
                == row.vi_options[row.gold_index].option_id
            )
            assert row.en_state_text != row.vi_state_text
            assert row.en_question_text != row.vi_question_text


def test_m3_r1_typed_values_survive_translation():
    for partition in M3_R1_PARTITIONS:
        for row in generate_m3_r1_authority(partition):
            if row.primitive == "score":
                assert tuple(float(option.value) for option in row.en_options) == (
                    0.0,
                    1.0,
                    2.0,
                )
                assert tuple(float(option.value) for option in row.vi_options) == (
                    0.0,
                    1.0,
                    2.0,
                )
            elif row.primitive == "noul":
                assert tuple(float(option.value) for option in row.en_options) == (
                    0.0,
                    1.0,
                )
                assert tuple(float(option.value) for option in row.vi_options) == (
                    0.0,
                    1.0,
                )
            else:
                assert all(option.value is None for option in row.en_options)
                assert all(option.value is None for option in row.vi_options)


def test_m3_r1_text_is_fresh_against_m3a_m2_and_w34():
    current = all_m3_r1_text_atoms()
    assert not (current & all_m3_text_atoms(include_sealed=True))
    assert not (current & all_m2_high_k_text_atoms(include_sealed=True))
    assert not (current & all_w34_text_atoms())


def test_m3_r1_english_and_vietnamese_surfaces_are_disjoint():
    en = set()
    vi = set()
    for partition in M3_R1_PARTITIONS:
        for row in generate_m3_r1_authority(partition):
            en.update((row.en_state_text, row.en_question_text))
            vi.update((row.vi_state_text, row.vi_question_text))
            for option in row.en_options:
                en.add(option.criterion_text)
                en.update(option.aliases)
            for option in row.vi_options:
                vi.add(option.criterion_text)
                vi.update(option.aliases)
    assert en.isdisjoint(vi)
