from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import generate_s5_pairs
from nmd.v1_s6_authority import generate_s6_pairs
from nmd.v1_s7_authority import generate_s7_pairs
from nmd.v1_s8_authority import generate_s8_cases, validate_s8_partitions


def _s8_states(rows):
    return {
        text
        for row in rows
        for text in (row.state_a, row.state_b)
    }


def _s8_questions(rows):
    return {
        text
        for row in rows
        for text in (
            row.question_a1,
            row.question_a2,
            row.question_b1,
            row.question_b2,
        )
    }


def test_s8_authority_sizes_views_and_domains():
    train = generate_s8_cases("train")
    dev = generate_s8_cases("dev")
    validate_s8_partitions(train, dev)

    assert len(train) == 768
    assert len(dev) == 192
    assert len({row.domain for row in train}) == 12
    assert len({row.domain for row in dev}) == 12
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(row.state_a != row.state_b for row in (*train, *dev))
    assert all(row.question_a1 != row.question_a2 for row in (*train, *dev))
    assert all(row.question_b1 != row.question_b2 for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(len(row.option_texts) == 4 for row in (*train, *dev))
    assert all(len(row.option_aliases) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s8_train_dev_all_wording_views_are_disjoint():
    train = generate_s8_cases("train")
    dev = generate_s8_cases("dev")

    assert not (_s8_states(train) & _s8_states(dev))
    assert not (_s8_questions(train) & _s8_questions(dev))

    # Option lexical banks are also split-specific even though the primary
    # scientific freshness boundary is state/question wording.
    assert not (
        {x for row in train for x in row.option_texts}
        & {x for row in dev for x in row.option_texts}
    )
    assert not (
        {x for row in train for x in row.option_aliases}
        & {x for row in dev for x in row.option_aliases}
    )


def test_s8_exact_state_question_fresh_against_s0_through_s7():
    current = (*generate_s8_cases("train"), *generate_s8_cases("dev"))
    prior = (
        *generate_s0_pairs("train"), *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"), *generate_s1_pairs("dev"),
        *generate_s2_pairs("train"), *generate_s2_pairs("dev"),
        *generate_s3_pairs("train"), *generate_s3_pairs("dev"),
        *generate_s4_pairs("train"), *generate_s4_pairs("dev"),
        *generate_s5_pairs("train"), *generate_s5_pairs("dev"),
        *generate_s6_pairs("train"), *generate_s6_pairs("dev"),
        *generate_s7_pairs("train"), *generate_s7_pairs("dev"),
    )
    prior_states = {row.state for row in prior}
    prior_questions = {
        q
        for row in prior
        for q in (row.question_a, row.question_b)
    }

    assert not (_s8_states(current) & prior_states)
    assert not (_s8_questions(current) & prior_questions)


def test_s8_train_and_dev_use_distinct_template_families():
    train = generate_s8_cases("train")
    dev = generate_s8_cases("dev")

    assert all(row.state_a.startswith("S8 ") for row in train)
    assert all(row.state_b.startswith("For ") for row in train)
    assert all(row.state_a.startswith("The S8 dossier") for row in dev)
    assert all("Separately" in row.state_b for row in dev)

    assert all(row.question_a1.startswith("In S8 ") for row in train)
    assert all(row.question_b1.startswith("In S8 ") for row in train)
    assert all(row.question_a1.startswith("From the S8 dossier") for row in dev)
    assert all(row.question_b1.startswith("From the S8 dossier") for row in dev)


def test_s8_split_lexical_sentinels():
    train_text = "\n".join(
        [row.state_a for row in generate_s8_cases("train")]
        + [row.state_b for row in generate_s8_cases("train")]
    )
    dev_text = "\n".join(
        [row.state_a for row in generate_s8_cases("dev")]
        + [row.state_b for row in generate_s8_cases("dev")]
    )

    for token in (
        "Blend Alder",
        "chirp linear",
        "orris",
        "strain gauge",
        "millet",
        "cargo bike",
    ):
        assert token in train_text
        assert token not in dev_text

    for token in (
        "Blend Iris",
        "barker-coded",
        "immortelle",
        "MEMS gyro",
        "teff",
        "autonomous cart",
    ):
        assert token in dev_text
        assert token not in train_text
