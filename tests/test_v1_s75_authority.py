from nmd.v1_s75_authority import generate_s75_cases, validate_s75_partitions
from nmd.v1_s74_authority import generate_s74_cases


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s75_fresh_authority_is_exact_and_disjoint():
    train=generate_s75_cases("train")
    dev=generate_s75_cases("dev")
    validate_s75_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert {r.domain for r in train}=={r.domain for r in dev}

    assert not (_states(train)&_states(dev))
    assert not (_questions(train)&_questions(dev))
    assert not (_options(train)&_options(dev))

    prior=(*generate_s74_cases("train"),*generate_s74_cases("dev"))
    current=(*train,*dev)
    assert not (_states(current)&_states(prior))
    assert not (_questions(current)&_questions(prior))
    assert not (_options(current)&_options(prior))


def test_s75_generation_is_deterministic():
    assert generate_s75_cases("train")==generate_s75_cases("train")
    assert generate_s75_cases("dev")==generate_s75_cases("dev")
