from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from hira_v1_s44_a0_private_correction_fork import cases as s44_a0_cases
from nmd.v1_s43_authority import generate_s43_cases
from nmd.v1_s44_authority import generate_s44_cases, validate_s44_partitions


def _states(rows):
    return {x for r in rows for x in (r.state_a, r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1, r.question_a2, r.question_b1, r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts, *r.option_aliases)}


def test_s44_authority_partition_contract():
    train=generate_s44_cases("train")
    dev=generate_s44_cases("dev")
    validate_s44_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert {r.domain for r in train}=={r.domain for r in dev}
    assert all(r.language=="en" for r in (*train,*dev))
    assert all(len(r.option_ids)==4 and len(set(r.option_ids))==4 for r in (*train,*dev))
    assert all(r.gold_a != r.gold_b for r in (*train,*dev))


def test_s44_train_dev_exact_surfaces_are_disjoint():
    train=generate_s44_cases("train")
    dev=generate_s44_cases("dev")
    assert not (_states(train)&_states(dev))
    assert not (_questions(train)&_questions(dev))
    assert not (_options(train)&_options(dev))


def test_s44_authority_has_no_exact_overlap_with_s43_train_dev():
    current=(*generate_s44_cases("train"),*generate_s44_cases("dev"))
    prior=(*generate_s43_cases("train"),*generate_s43_cases("dev"))
    assert not (_states(current)&_states(prior))
    assert not (_questions(current)&_questions(prior))
    assert not (_options(current)&_options(prior))


def test_s44_authority_has_no_exact_overlap_with_s44_a0():
    current=(*generate_s44_cases("train"),*generate_s44_cases("dev"))
    a0=s44_a0_cases()
    a0_states={x for r in a0 for x in (r.state_a,r.state_b)}
    a0_questions={x for r in a0 for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    a0_options=set()
    for case in a0:
        options,_ga,_gb=case.option_pack()
        for option in options:
            a0_options.add(option.criterion_text)
            a0_options.update(option.aliases)
    assert not (_states(current)&a0_states)
    assert not (_questions(current)&a0_questions)
    assert not (_options(current)&a0_options)


def test_s44_authority_is_deterministic_and_opaque():
    a=generate_s44_cases("train")
    b=generate_s44_cases("train")
    assert a==b
    assert all(all("__opaque_" in option_id for option_id in r.option_ids) for r in a)
