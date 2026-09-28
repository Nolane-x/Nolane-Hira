from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs, validate_s2_partitions


def test_s2_authority_sizes_domains_and_language():
    train = generate_s2_pairs("train")
    dev = generate_s2_pairs("dev")
    validate_s2_partitions(train, dev)

    assert len(train) == 256
    assert len(dev) == 64
    assert {row.domain for row in train} == {
        "sensor_registry",
        "recipe_card",
        "flight_clearance",
        "museum_loan",
    }
    assert {row.domain for row in dev} == {
        "sensor_registry",
        "recipe_card",
        "flight_clearance",
        "museum_loan",
    }
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s2_authority_is_exact_text_fresh_against_s0_s1():
    s2 = (*generate_s2_pairs("train"), *generate_s2_pairs("dev"))
    prior = (
        *generate_s0_pairs("train"),
        *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"),
        *generate_s1_pairs("dev"),
    )
    s2_states = {row.state for row in s2}
    prior_states = {row.state for row in prior}
    assert not (s2_states & prior_states)

    s2_questions = {q for row in s2 for q in (row.question_a, row.question_b)}
    prior_questions = {q for row in prior for q in (row.question_a, row.question_b)}
    assert not (s2_questions & prior_questions)


def test_s2_train_dev_banks_are_separated():
    train = generate_s2_pairs("train")
    dev = generate_s2_pairs("dev")

    assert not ({row.state for row in train} & {row.state for row in dev})
    assert not (
        {q for row in train for q in (row.question_a, row.question_b)}
        & {q for row in dev for q in (row.question_a, row.question_b)}
    )

    train_text = "\n".join(row.state for row in train)
    dev_text = "\n".join(row.state for row in dev)

    # Use complete lexical sentinels rather than short substrings.  The
    # previous "ion" sentinel falsely matched the unrelated word
    # "calibration" in DEV and stopped the authority before optimizer/DEV
    # exposure.
    for token in (
        "sensor family ion",
        "2026-01-14",
        "features cardamom",
        "runway 04",
        "Arbor Institute",
    ):
        assert token in train_text
        assert token not in dev_text
    for token in (
        "unit as infrared",
        "2026-09-03",
        "names saffron",
        "runway 05",
        "Iris Institute",
    ):
        assert token in dev_text
        assert token not in train_text
