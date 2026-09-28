from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs
from nmd.v1_s2_authority import generate_s2_pairs
from nmd.v1_s3_authority import generate_s3_pairs
from nmd.v1_s4_authority import generate_s4_pairs
from nmd.v1_s5_authority import generate_s5_pairs
from nmd.v1_s6_authority import generate_s6_pairs
from nmd.v1_s7_authority import generate_s7_pairs, validate_s7_partitions


def test_s7_authority_sizes_and_domains():
    train = generate_s7_pairs("train")
    dev = generate_s7_pairs("dev")
    validate_s7_partitions(train, dev)

    assert len(train) == 768
    assert len(dev) == 192
    assert len({row.domain for row in train}) == 12
    assert len({row.domain for row in dev}) == 12
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(len(row.option_texts) == 4 for row in (*train, *dev))
    assert all(len(row.option_aliases) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s7_train_dev_exact_text_is_separated():
    train = generate_s7_pairs("train")
    dev = generate_s7_pairs("dev")

    assert not ({row.state for row in train} & {row.state for row in dev})
    assert not (
        {q for row in train for q in (row.question_a, row.question_b)}
        & {q for row in dev for q in (row.question_a, row.question_b)}
    )
    assert not (
        {x for row in train for x in row.option_texts}
        & {x for row in dev for x in row.option_texts}
    )


def test_s7_exact_fresh_against_s0_through_s6():
    current = (*generate_s7_pairs("train"), *generate_s7_pairs("dev"))
    prior = (
        *generate_s0_pairs("train"), *generate_s0_pairs("dev"),
        *generate_s1_pairs("train"), *generate_s1_pairs("dev"),
        *generate_s2_pairs("train"), *generate_s2_pairs("dev"),
        *generate_s3_pairs("train"), *generate_s3_pairs("dev"),
        *generate_s4_pairs("train"), *generate_s4_pairs("dev"),
        *generate_s5_pairs("train"), *generate_s5_pairs("dev"),
        *generate_s6_pairs("train"), *generate_s6_pairs("dev"),
    )

    assert not (
        {row.state for row in current}
        & {row.state for row in prior}
    )
    assert not (
        {q for row in current for q in (row.question_a, row.question_b)}
        & {q for row in prior for q in (row.question_a, row.question_b)}
    )


def test_s7_split_specific_sentinels():
    train_text = "\n".join(row.state for row in generate_s7_pairs("train"))
    dev_text = "\n".join(row.state for row in generate_s7_pairs("dev"))

    for token in (
        "andesite",
        "Syrah" if False else "Bourbon",
        "SMF-28",
        "reverse osmosis",
        "panchromatic",
        "sourdough",
    ):
        assert token in train_text
        assert token not in dev_text

    for token in (
        "serpentinite",
        "Pink Bourbon",
        "hollow-core",
        "capacitive deionization",
        "polarimetric SAR",
        "panettone",
    ):
        assert token in dev_text
        assert token not in train_text
