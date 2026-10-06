from nmd.v1_s73_authority import generate_s73_cases, validate_s73_partitions
from nmd.v1_s72_authority import generate_s72_cases


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s73_fresh_authority_is_exact_and_disjoint():
    train=generate_s73_cases("train")
    dev=generate_s73_cases("dev")
    validate_s73_partitions(train,dev)

    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert {r.domain for r in train}=={r.domain for r in dev}

    assert not (_states(train)&_states(dev))
    assert not (_questions(train)&_questions(dev))
    assert not (_options(train)&_options(dev))

    prior=(*generate_s72_cases("train"),*generate_s72_cases("dev"))
    current=(*train,*dev)
    assert not (_states(current)&_states(prior))
    assert not (_questions(current)&_questions(prior))
    assert not (_options(current)&_options(prior))


def test_s73_generation_is_deterministic():
    assert generate_s73_cases("train")==generate_s73_cases("train")
    assert generate_s73_cases("dev")==generate_s73_cases("dev")
