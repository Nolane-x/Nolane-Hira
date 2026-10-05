from nmd.v1_s59_authority import generate_s59_cases
from nmd.v1_s60_authority import generate_s60_cases, validate_s60_partitions


def _states(rows): return {x for r in rows for x in (r.state_a,r.state_b)}
def _questions(rows): return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
def _options(rows): return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s60_authority_is_fresh_exact_and_s59_disjoint():
    train=generate_s60_cases("train")
    dev=generate_s60_cases("dev")
    validate_s60_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert _states(train).isdisjoint(_states(dev))
    assert _questions(train).isdisjoint(_questions(dev))
    assert _options(train).isdisjoint(_options(dev))

    prior=(*generate_s59_cases("train"),*generate_s59_cases("dev"))
    current=(*train,*dev)
    assert _states(current).isdisjoint(_states(prior))
    assert _questions(current).isdisjoint(_questions(prior))
    assert _options(current).isdisjoint(_options(prior))
