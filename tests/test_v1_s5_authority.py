from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import generate_s5_pairs, validate_s5_partitions


def test_s5_authority_sizes_domains_views_and_language():
    train = generate_s5_pairs("train")
    dev = generate_s5_pairs("dev")
    validate_s5_partitions(train, dev)

    assert len(train) == 512
    assert len(dev) == 128
    expected = {
        "battery_pack",
        "weather_station",
        "textile_batch",
        "cargo_drone",
        "aquaculture_feed",
        "telescope_schedule",
        "ceramic_kiln",
        "data_center",
    }
    assert {row.domain for row in train} == expected
    assert {row.domain for row in dev} == expected
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_texts) == 4 for row in (*train, *dev))
    assert all(len(row.option_aliases) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s5_authority_is_exact_text_fresh_against_s0_s4():
    s5 = (*generate_s5_pairs("train"), *generate_s5_pairs("dev"))
    prior = (
        *generate_s0_pairs("train"),
        *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"),
        *generate_s1_pairs("dev"),
        *generate_s2_pairs("train"),
        *generate_s2_pairs("dev"),
        *generate_s3_pairs("train"),
        *generate_s3_pairs("dev"),
        *generate_s4_pairs("train"),
        *generate_s4_pairs("dev"),
    )

    s5_states = {row.state for row in s5}
    prior_states = {row.state for row in prior}
    assert not (s5_states & prior_states)

    s5_questions = {q for row in s5 for q in (row.question_a, row.question_b)}
    prior_questions = {
        q for row in prior for q in (row.question_a, row.question_b)
    }
    assert not (s5_questions & prior_questions)


def test_s5_train_dev_banks_and_templates_are_separated():
    train = generate_s5_pairs("train")
    dev = generate_s5_pairs("dev")

    assert not ({row.state for row in train} & {row.state for row in dev})
    assert not (
        {q for row in train for q in (row.question_a, row.question_b)}
        & {q for row in dev for q in (row.question_a, row.question_b)}
    )

    train_text = "\n".join(row.state for row in train)
    dev_text = "\n".join(row.state for row in dev)

    train_sentinels = (
        "cell chemistry LFP",
        "ridge North",
        "fiber merino",
        "corridor Aster",
        "protein source spirulina",
        "object M31",
        "body porcelain",
        "compute pod Alder",
    )
    dev_sentinels = (
        "electrochemistry LMFP",
        "ridge South",
        "material tencel",
        "corridor Iris",
        "ingredient duckweed",
        "source M33",
        "ceramic type majolica",
        "pod Maple",
    )
    for token in train_sentinels:
        assert token in train_text
        assert token not in dev_text
    for token in dev_sentinels:
        assert token in dev_text
        assert token not in train_text
