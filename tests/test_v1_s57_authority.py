from nmd.v1_s56_authority import generate_s56_cases
from nmd.v1_s57_authority import generate_s57_cases, validate_s57_partitions


def _states(rows): return {x for r in rows for x in (r.state_a,r.state_b)}
def _questions(rows): return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
def _options(rows): return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s57_authority_sizes_domains_and_internal_freshness():
    train=generate_s57_cases("train")
    dev=generate_s57_cases("dev")
    validate_s57_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert len({r.domain for r in dev})==12
    assert not (_states(train)&_states(dev))
    assert not (_questions(train)&_questions(dev))
    assert not (_options(train)&_options(dev))


def test_s57_authority_has_no_exact_overlap_with_s56():
    current=(*generate_s57_cases("train"),*generate_s57_cases("dev"))
    prior=(*generate_s56_cases("train"),*generate_s56_cases("dev"))
    assert not (_states(current)&_states(prior))
    assert not (_questions(current)&_questions(prior))
    assert not (_options(current)&_options(prior))
