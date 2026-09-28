from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import generate_s5_pairs
from nmd.v1_s6_authority import generate_s6_pairs
from nmd.v1_s7_authority import generate_s7_pairs
from nmd.v1_s8_authority import generate_s8_cases
from nmd.v1_s9_authority import generate_s9_cases
from nmd.v1_s10_authority import generate_s10_cases, validate_s10_partitions


def _state_texts(rows):
    values = set()
    for row in rows:
        if hasattr(row, "state"):
            values.add(row.state)
        else:
            values.update((row.state_a, row.state_b))
    return values


def _question_texts(rows):
    values = set()
    for row in rows:
        if hasattr(row, "question_a"):
            values.update((row.question_a, row.question_b))
        else:
            values.update((
                row.question_a1,
                row.question_a2,
                row.question_b1,
                row.question_b2,
            ))
    return values


def test_s10_authority_sizes_domains_and_views():
    train = generate_s10_cases("train")
    dev = generate_s10_cases("dev")
    validate_s10_partitions(train, dev)

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


def test_s10_train_dev_exact_banks_are_disjoint():
    train = generate_s10_cases("train")
    dev = generate_s10_cases("dev")

    assert not (_state_texts(train) & _state_texts(dev))
    assert not (_question_texts(train) & _question_texts(dev))
    assert not (
        {text for row in train for text in row.option_texts}
        & {text for row in dev for text in row.option_texts}
    )
    assert not (
        {text for row in train for text in row.option_aliases}
        & {text for row in dev for text in row.option_aliases}
    )


def test_s10_exact_text_fresh_against_s0_through_s9():
    current = (*generate_s10_cases("train"), *generate_s10_cases("dev"))
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
        *generate_s6_pairs("train"),
        *generate_s6_pairs("dev"),
        *generate_s7_pairs("train"),
        *generate_s7_pairs("dev"),
        *generate_s8_cases("train"),
        *generate_s8_cases("dev"),
        *generate_s9_cases("train"),
        *generate_s9_cases("dev"),
    )
    assert not (_state_texts(current) & _state_texts(prior))
    assert not (_question_texts(current) & _question_texts(prior))


def test_s10_split_specific_lexical_sentinels():
    train_text = "\n".join(row.state_a for row in generate_s10_cases("train"))
    dev_text = "\n".join(row.state_a for row in generate_s10_cases("dev"))

    # Use complete context-bearing sentinels. Short lexical fragments can
    # legitimately occur inside unrelated split values (for example QPSK is
    # a substring of the DEV value pi/4-DQPSK).
    for token in (
        "lists working fluid isobutane together with",
        "lists modulation QPSK together with",
        "lists culture strain strain A17 together with",
        "lists lighting mode warm LED together with",
        "lists impeller alloy 7075 aluminum together with",
        "lists cultivar Gala together with",
    ):
        assert token in train_text
        assert token not in dev_text

    for token in (
        "working fluid field contains R1233zd, while",
        "modulation field contains 128QAM, while",
        "culture strain field contains strain J14, while",
        "lighting mode field contains RGBW LED, while",
        "impeller alloy field contains Inconel 625, while",
        "cultivar field contains Jazz, while",
    ):
        assert token in dev_text
        assert token not in train_text
