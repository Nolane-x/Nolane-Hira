from pathlib import Path
import sys

from nmd.v1_s53_authority import generate_s53_cases
from nmd.v1_s54_authority import generate_s54_cases, validate_s54_partitions

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
from hira_v1_s54_a0_joint_state_query_option_interaction import cases as a0_cases


def _states(rows):
    return {x for r in rows for x in (r.state_a,r.state_b)}


def _questions(rows):
    return {x for r in rows for x in (r.question_a1,r.question_a2,r.question_b1,r.question_b2)}


def _options(rows):
    return {x for r in rows for x in (*r.option_texts,*r.option_aliases)}


def test_s54_authority_sizes_domains_and_internal_freshness():
    train=generate_s54_cases("train")
    dev=generate_s54_cases("dev")
    validate_s54_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert len({r.domain for r in dev})==12
    assert not (_states(train)&_states(dev))
    assert not (_questions(train)&_questions(dev))
    assert not (_options(train)&_options(dev))


def test_s54_authority_has_no_exact_overlap_with_s53_train_dev():
    current=(*generate_s54_cases("train"),*generate_s54_cases("dev"))
    prior=(*generate_s53_cases("train"),*generate_s53_cases("dev"))
    assert not (_states(current)&_states(prior))
    assert not (_questions(current)&_questions(prior))
    assert not (_options(current)&_options(prior))


def test_s54_authority_has_no_exact_overlap_with_s54_a0():
    current=(*generate_s54_cases("train"),*generate_s54_cases("dev"))
    cs=_states(current)
    cq=_questions(current)
    co=_options(current)

    a0s=set()
    a0q=set()
    a0o=set()
    for case in a0_cases():
        a0s.update((case.state_a,case.state_b))
        a0q.update((case.qa1,case.qa2,case.qb1,case.qb2))
        opts,_ga,_gb=case.option_pack()
        for o in opts:
            a0o.add(o.criterion_text)
            a0o.update(o.aliases)

    assert not (cs&a0s)
    assert not (cq&a0q)
    assert not (co&a0o)


def test_s54_seed_is_fresh_after_s53():
    source=Path("research/HIRA-V1-S54-CONTRACT.md").read_text(encoding="utf-8")
    a0=Path("scripts/hira_v1_s54_a0_joint_state_query_option_interaction.py").read_text(encoding="utf-8")
    assert "seed **75001**" in source
    assert "SEED=75_001" in a0
