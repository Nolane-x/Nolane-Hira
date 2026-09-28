from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import generate_s5_pairs
from nmd.v1_s6_authority import generate_s6_pairs, validate_s6_partitions


def test_s6_authority_sizes_domains_language_and_views():
    train = generate_s6_pairs("train")
    dev = generate_s6_pairs("dev")
    validate_s6_partitions(train, dev)

    assert len(train) == 768
    assert len(dev) == 192
    assert len({row.domain for row in train}) == 12
    assert len({row.domain for row in dev}) == 12
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(len(row.option_texts) == 4 for row in (*train, *dev))
    assert all(len(row.option_aliases) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s6_train_dev_exact_banks_are_separated():
    train = generate_s6_pairs("train")
    dev = generate_s6_pairs("dev")

    assert not ({row.state for row in train} & {row.state for row in dev})
    assert not (
        {q for row in train for q in (row.question_a, row.question_b)}
        & {q for row in dev for q in (row.question_a, row.question_b)}
    )
    assert not ({text for row in train for text in row.option_texts}
                & {text for row in dev for text in row.option_texts})
    assert not ({text for row in train for text in row.option_aliases}
                & {text for row in dev for text in row.option_aliases})


def test_s6_exact_text_fresh_against_s0_through_s5():
    current = (*generate_s6_pairs("train"), *generate_s6_pairs("dev"))
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
        *generate_s5_pairs("train"),
        *generate_s5_pairs("dev"),
    )

    current_states = {row.state for row in current}
    prior_states = {row.state for row in prior}
    assert not (current_states & prior_states)

    current_questions = {
        q for row in current for q in (row.question_a, row.question_b)
    }
    prior_questions = {
        q for row in prior for q in (row.question_a, row.question_b)
    }
    assert not (current_questions & prior_questions)


def test_s6_split_specific_lexical_sentinels():
    train_text = "\n".join(row.state for row in generate_s6_pairs("train"))
    dev_text = "\n".join(row.state for row in generate_s6_pairs("dev"))

    for token in (
        "fixed-angle",
        "Syrah",
        "ozonesonde",
        "monocrystalline",
        "T1-weighted",
        "argon",
    ):
        assert token in train_text
        assert token not in dev_text

    for token in (
        "elutriation",
        "Mourvedre",
        "dewpoint sensor",
        "IBC",
        "ASL",
        "neon",
    ):
        assert token in dev_text
        assert token not in train_text
