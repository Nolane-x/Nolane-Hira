from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from hira_v1_s44_a0_private_correction_fork import cases as s44_a0_cases
from hira_v1_s45_a0_cross_view_consistent_private_correction import cases as s45_a0_cases
from nmd.v1_s44_authority import generate_s44_cases
from nmd.v1_s45_authority import generate_s45_cases, validate_s45_partitions


def _states(rows):
    return {x for r in rows for x in (r.state_a, r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts, *r.option_aliases)}


def _a0_surfaces(cases):
    states={x for r in cases for x in (r.state_a,r.state_b)}
    questions={x for r in cases for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    options=set()
    for case in cases:
        packed,_ga,_gb=case.option_pack()
        for option in packed:
            options.add(option.criterion_text)
            options.update(option.aliases)
    return states,questions,options


def test_s45_authority_partition_contract():
    train=generate_s45_cases("train")
    dev=generate_s45_cases("dev")
    validate_s45_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert {r.domain for r in train}=={r.domain for r in dev}
    assert all(r.language=="en" for r in (*train,*dev))
    assert all(len(r.option_ids)==4 and len(set(r.option_ids))==4 for r in (*train,*dev))
    assert all(r.gold_a != r.gold_b for r in (*train,*dev))


def test_s45_train_dev_exact_surfaces_are_disjoint():
    train=generate_s45_cases("train")
    dev=generate_s45_cases("dev")
    assert not (_states(train)&_states(dev))
    assert not (_questions(train)&_questions(dev))
    assert not (_options(train)&_options(dev))


def test_s45_authority_has_no_exact_overlap_with_s44_train_dev():
    current=(*generate_s45_cases("train"),*generate_s45_cases("dev"))
    prior=(*generate_s44_cases("train"),*generate_s44_cases("dev"))
    assert not (_states(current)&_states(prior))
    assert not (_questions(current)&_questions(prior))
    assert not (_options(current)&_options(prior))


def test_s45_a0_has_no_exact_overlap_with_s44_a0():
    s45=_a0_surfaces(s45_a0_cases())
    s44=_a0_surfaces(s44_a0_cases())
    assert not (s45[0]&s44[0])
    assert not (s45[1]&s44[1])
    assert not (s45[2]&s44[2])


def test_s45_authority_has_no_exact_overlap_with_s45_a0():
    current=(*generate_s45_cases("train"),*generate_s45_cases("dev"))
    a0_states,a0_questions,a0_options=_a0_surfaces(s45_a0_cases())
    assert not (_states(current)&a0_states)
    assert not (_questions(current)&a0_questions)
    assert not (_options(current)&a0_options)


def test_s45_authority_is_deterministic_and_opaque():
    a=generate_s45_cases("train")
    b=generate_s45_cases("train")
    assert a==b
    assert all(all("__opaque_" in option_id for option_id in r.option_ids) for r in a)
