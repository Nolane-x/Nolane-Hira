from nmd.v1_s54_authority import generate_s54_cases
from nmd.v1_s55_authority import generate_s55_cases, validate_s55_partitions


def _states(rows): return {x for r in rows for x in (r.state_a,r.state_b)}
def _questions(rows): return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
def _options(rows): return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s55_authority_sizes_domains_and_internal_freshness():
    train=generate_s55_cases("train")
    dev=generate_s55_cases("dev")
    validate_s55_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert len({r.domain for r in dev})==12
    assert not (_states(train)&_states(dev))
    assert not (_questions(train)&_questions(dev))
    assert not (_options(train)&_options(dev))


def test_s55_authority_has_no_exact_overlap_with_s54():
    current=(*generate_s55_cases("train"),*generate_s55_cases("dev"))
    prior=(*generate_s54_cases("train"),*generate_s54_cases("dev"))
    assert not (_states(current)&_states(prior))
    assert not (_questions(current)&_questions(prior))
    assert not (_options(current)&_options(prior))
