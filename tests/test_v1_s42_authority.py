from scripts.hira_v1_s42_a0_cross_view_relational_geometry import cases as s42_a0_cases
from nmd.v1_s41_authority import generate_s41_cases
from nmd.v1_s42_authority import generate_s42_cases, validate_s42_partitions


def _states(rows):
    return {x for r in rows for x in (r.state_a, r.state_b)}


def _questions(rows):
    return {
        x
        for r in rows
        for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)
    }


def _options(rows):
    return {x for r in rows for x in (*r.option_texts, *r.option_aliases)}


def test_s42_authority_partition_contract():
    train = generate_s42_cases("train")
    dev = generate_s42_cases("dev")
    validate_s42_partitions(train, dev)

    assert len(train) == 768
    assert len(dev) == 192
    assert len({r.domain for r in train}) == 12
    assert {r.domain for r in train} == {r.domain for r in dev}
    assert all(r.language == "en" for r in (*train, *dev))
    assert all(len(r.option_ids) == 4 for r in (*train, *dev))
    assert all(len(set(r.option_ids)) == 4 for r in (*train, *dev))
    assert all(r.gold_a != r.gold_b for r in (*train, *dev))


def test_s42_train_dev_exact_surfaces_are_disjoint():
    train = generate_s42_cases("train")
    dev = generate_s42_cases("dev")

    assert not (_states(train) & _states(dev))
    assert not (_questions(train) & _questions(dev))
    assert not (_options(train) & _options(dev))


def test_s42_authority_has_no_exact_overlap_with_s41_train_dev():
    current = (*generate_s42_cases("train"), *generate_s42_cases("dev"))
    prior = (*generate_s41_cases("train"), *generate_s41_cases("dev"))

    assert not (_states(current) & _states(prior))
    assert not (_questions(current) & _questions(prior))
    assert not (_options(current) & _options(prior))


def test_s42_authority_has_no_exact_overlap_with_s42_a0():
    current = (*generate_s42_cases("train"), *generate_s42_cases("dev"))
    a0 = s42_a0_cases()

    a0_states = {x for r in a0 for x in (r.state_a, r.state_b)}
    a0_questions = {x for r in a0 for x in (r.qa1, r.qa2, r.qb1, r.qb2)}
    a0_options = set()
    for case in a0:
        options, _ga, _gb = case.option_pack()
        for option in options:
            a0_options.add(option.criterion_text)
            a0_options.update(option.aliases)

    assert not (_states(current) & a0_states)
    assert not (_questions(current) & a0_questions)
    assert not (_options(current) & a0_options)


def test_s42_authority_is_deterministic_and_opaque():
    a = generate_s42_cases("train")
    b = generate_s42_cases("train")

    assert a == b
    assert all(
        all("__opaque_" in option_id for option_id in r.option_ids)
        for r in a
    )
