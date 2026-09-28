from nmd.v1_s0_authority import generate_s0_pairs
from nmd.v1_s1_authority import generate_s1_pairs, validate_s1_partitions


def test_s1_authority_sizes_domains_and_language():
    train = generate_s1_pairs("train")
    dev = generate_s1_pairs("dev")
    validate_s1_partitions(train, dev)

    assert len(train) == 256
    assert len(dev) == 64
    assert {row.domain for row in train} == {
        "shipment_manifest",
        "lab_specimen",
        "network_node",
        "festival_access",
    }
    assert {row.domain for row in dev} == {
        "shipment_manifest",
        "lab_specimen",
        "network_node",
        "festival_access",
    }
    assert all(row.language == "en" for row in (*train, *dev))
    assert all(len(row.option_ids) == 4 for row in (*train, *dev))
    assert all(row.gold_a != row.gold_b for row in (*train, *dev))


def test_s1_authority_is_exact_text_fresh_against_s0():
    s1 = (*generate_s1_pairs("train"), *generate_s1_pairs("dev"))
    s0 = (*generate_s0_pairs("train"), *generate_s0_pairs("dev"))

    s1_states = {row.state for row in s1}
    s0_states = {row.state for row in s0}
    assert not (s1_states & s0_states)

    s1_questions = {q for row in s1 for q in (row.question_a, row.question_b)}
    s0_questions = {q for row in s0 for q in (row.question_a, row.question_b)}
    assert not (s1_questions & s0_questions)


def test_s1_train_dev_templates_and_lexical_banks_are_separated():
    train = generate_s1_pairs("train")
    dev = generate_s1_pairs("dev")

    assert not ({row.state for row in train} & {row.state for row in dev})
    assert not (
        {q for row in train for q in (row.question_a, row.question_b)}
        & {q for row in dev for q in (row.question_a, row.question_b)}
    )

    train_text = "\n".join(row.state for row in train)
    dev_text = "\n".join(row.state for row in dev)

    # Representative bank sentinels are intentionally split.
    for token in ("Orion", "ZX-41", "citrate", "VLAN 112", "Amber"):
        assert token in train_text
        assert token not in dev_text
    for token in ("Aster", "UA-34", "tartrate", "VLAN 135", "Ivory"):
        assert token in dev_text
        assert token not in train_text
