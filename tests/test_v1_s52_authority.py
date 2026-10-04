from nmd.v1_s51_authority import generate_s51_train_cases, generate_s51_dev_cases
from nmd.v1_s52_authority import generate_s52_cases, validate_s52_partitions


def _states(rows): return {x for r in rows for x in (r.state_a,r.state_b)}
def _questions(rows): return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}
def _options(rows): return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s52_authority_sizes_domains_and_internal_freshness():
    train=generate_s52_cases("train"); dev=generate_s52_cases("dev")
    validate_s52_partitions(train,dev)
    assert len(train)==768 and len(dev)==192
    assert len({r.domain for r in train})==12
    assert len({r.domain for r in dev})==12


def test_s52_authority_has_no_exact_overlap_with_s51_train_dev():
    current=(*generate_s52_cases("train"),*generate_s52_cases("dev"))
    prior=(*generate_s51_train_cases(),*generate_s51_dev_cases())
    assert not (_states(current)&_states(prior))
    assert not (_questions(current)&_questions(prior))
    assert not (_options(current)&_options(prior))
