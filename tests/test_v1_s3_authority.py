from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs, validate_s3_partitions


def test_s3_authority_sizes_domains_and_language():
    train = generate_s3_pairs("train")
    dev = generate_s3_pairs("dev")
    validate_s3_partitions(train, dev)

    assert len(train) == 256
    assert len(dev) == 64
    assert {row.domain for row in train} == {
        "geology_sample",
        "broadcast_station",
        "robot_inventory",
        "water_treatment",
    }
    assert {row.domain for row in dev} == {
        "geology_sample",
        "broadcast_station",
        "robot_inventory",
        "water_treatment",
    }
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s3_authority_is_exact_text_fresh_against_s0_s1_s2():
    s3 = (*generate_s3_pairs("train"), *generate_s3_pairs("dev"))
    prior = (
        *generate_s0_pairs("train"),
        *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"),
        *generate_s1_pairs("dev"),
        *generate_s2_pairs("train"),
        *generate_s2_pairs("dev"),
    )
    assert not ({row.state for row in s3} & {row.state for row in prior})
    assert not (
        {q for row in s3 for q in (row.question_a, row.question_b)}
        & {q for row in prior for q in (row.question_a, row.question_b)}
    )


def test_s3_train_dev_banks_are_separated():
    train = generate_s3_pairs("train")
    dev = generate_s3_pairs("dev")

    assert not ({row.state for row in train} & {row.state for row in dev})
    assert not (
        {q for row in train for q in (row.question_a, row.question_b)}
        & {q for row in dev for q in (row.question_a, row.question_b)}
    )

    train_text = "\n".join(row.state for row in train)
    dev_text = "\n".join(row.state for row in dev)

    for phrase in (
        "classified as basalt",
        "88.4 MHz",
        "aisle A3",
        "membrane ceramic M1",
    ):
        assert phrase in train_text
        assert phrase not in dev_text

    for phrase in (
        "lithology andesite",
        "89.6 MHz",
        "aisle J2",
        "filter element fiber X1",
    ):
        assert phrase in dev_text
        assert phrase not in train_text
