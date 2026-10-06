from nmd.v1_s71_authority import generate_s71_cases
from nmd.v1_s72_authority import generate_s72_cases, validate_s72_partitions


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s72_authority_is_fresh_exact_and_s71_disjoint():
    train=generate_s72_cases("train")
    dev=generate_s72_cases("dev")
    validate_s72_partitions(train,dev)

    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert _states(train).isdisjoint(_states(dev))
    assert _questions(train).isdisjoint(_questions(dev))
    assert _options(train).isdisjoint(_options(dev))

    prior=(*generate_s71_cases("train"),*generate_s71_cases("dev"))
    current=(*train,*dev)
    assert _states(current).isdisjoint(_states(prior))
    assert _questions(current).isdisjoint(_questions(prior))
    assert _options(current).isdisjoint(_options(prior))


def test_s72_authority_does_not_self_import():
    source=open("src/nmd/v1_s72_authority.py",encoding="utf-8").read()
    assert "from .v1_s71_authority import generate_s71_cases" in source
    assert "from .v1_s72_authority" not in source
    assert 'prior=(*generate_s71_cases("train"),*generate_s71_cases("dev"))' in source
