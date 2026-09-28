from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs, validate_s4_partitions


def test_s4_authority_sizes_domains_and_language():
    train = generate_s4_pairs("train")
    dev = generate_s4_pairs("dev")
    validate_s4_partitions(train, dev)

    assert len(train) == 384
    assert len(dev) == 96
    expected = {
        "solar_array",
        "pharmacy_batch",
        "satellite_task",
        "quarry_sample",
        "music_catalog",
        "emergency_drill",
    }
    assert {row.domain for row in train} == expected
    assert {row.domain for row in dev} == expected
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s4_authority_is_exact_text_fresh_against_s0_s3():
    s4 = (*generate_s4_pairs("train"), *generate_s4_pairs("dev"))
    prior = (
        *generate_s0_pairs("train"),
        *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"),
        *generate_s1_pairs("dev"),
        *generate_s2_pairs("train"),
        *generate_s2_pairs("dev"),
        *generate_s3_pairs("train"),
        *generate_s3_pairs("dev"),
    )

    s4_states = {row.state for row in s4}
    prior_states = {row.state for row in prior}
    assert not (s4_states & prior_states)

    s4_questions = {q for row in s4 for q in (row.question_a, row.question_b)}
    prior_questions = {
        q for row in prior for q in (row.question_a, row.question_b)
    }
    assert not (s4_questions & prior_questions)


def test_s4_train_dev_banks_and_templates_are_separated():
    train = generate_s4_pairs("train")
    dev = generate_s4_pairs("dev")

    assert not ({row.state for row in train} & {row.state for row in dev})
    assert not (
        {q for row in train for q in (row.question_a, row.question_b)}
        & {q for row in dev for q in (row.question_a, row.question_b)}
    )

    train_text = "\n".join(row.state for row in train)
    dev_text = "\n".join(row.state for row in dev)

    train_sentinels = (
        "module type mono-PERC",
        "compound azurin",
        "track polar-3",
        "classified as diorite",
        "lead instrument cello",
        "assembly area sector Alpha",
    )
    dev_sentinels = (
        "CIGS modules",
        "active agent ivarol",
        "trajectory lunar-5",
        "rock type andesite",
        "mandolin as principal instrument",
        "sector India",
    )
    for token in train_sentinels:
        assert token in train_text
        assert token not in dev_text
    for token in dev_sentinels:
        assert token in dev_text
        assert token not in train_text
