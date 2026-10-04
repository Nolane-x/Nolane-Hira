from nmd.v1_s51_authority import generate_s51_train_cases, generate_s51_dev_cases
from nmd.v1_s52_authority import generate_s52_cases
from nmd.v1_s53_authority import generate_s53_cases, validate_s53_partitions


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s53_authority_sizes_domains_and_internal_freshness():
    train=generate_s53_cases("train")
    dev=generate_s53_cases("dev")
    validate_s53_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert len({r.domain for r in dev})==12
    assert not (_states(train)&_states(dev))
    assert not (_questions(train)&_questions(dev))
    assert not (_options(train)&_options(dev))


def test_s53_has_no_exact_overlap_with_s52_or_s51():
    current=(*generate_s53_cases("train"),*generate_s53_cases("dev"))
    prior52=(*generate_s52_cases("train"),*generate_s52_cases("dev"))
    prior51=(*generate_s51_train_cases(),*generate_s51_dev_cases())
    for prior in (prior52,prior51):
        assert not (_states(current)&_states(prior))
        assert not (_questions(current)&_questions(prior))
        assert not (_options(current)&_options(prior))
