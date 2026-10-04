from pathlib import Path
import sys

from nmd.v1_s51_authority import generate_s51_train_cases, generate_s51_dev_cases
from nmd.v1_s52_authority import generate_s52_cases, validate_s52_partitions

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
from hira_v1_s52_a0_query_relation_canonicalization import cases as s52_a0_cases


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


def test_s52_authority_has_no_exact_overlap_with_s52_a0():
    current=(*generate_s52_cases("train"),*generate_s52_cases("dev"))
    cs=_states(current); cq=_questions(current); co=_options(current)
    a0=s52_a0_cases()
    a0s={x for r in a0 for x in (r.state_a,r.state_b)}
    a0q={x for r in a0 for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    a0o=set()
    for case in a0:
        opts,_ga,_gb=case.option_pack()
        for o in opts:
            a0o.add(o.criterion_text)
            a0o.update(o.aliases)
    assert not (cs&a0s)
    assert not (cq&a0q)
    assert not (co&a0o)
