from pathlib import Path
import sys

from nmd.v1_s49_authority import generate_s49_cases
from nmd.v1_s50_authority import generate_s50_cases, validate_s50_partitions

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
from hira_v1_s50_a0_shared_native_forked_readouts import cases as s50_a0_cases


def _states(r): return (r.state_a,r.state_b)
def _questions(r): return (r.question_a1,r.question_a2,r.question_b1,r.question_b2)
def _options(r): return (*r.option_texts,*r.option_aliases)


def test_s50_authority_sizes_domains_and_internal_freshness():
    train=generate_s50_cases("train")
    dev=generate_s50_cases("dev")
    validate_s50_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert len({r.domain for r in dev})==12
    assert all(r.domain.startswith("shared_native_") for r in (*train,*dev))


def test_s50_authority_has_no_exact_overlap_with_s49_train_dev():
    current=(*generate_s50_cases("train"),*generate_s50_cases("dev"))
    prior=(*generate_s49_cases("train"),*generate_s49_cases("dev"))
    assert not ({x for r in current for x in _states(r)} & {x for r in prior for x in _states(r)})
    assert not ({x for r in current for x in _questions(r)} & {x for r in prior for x in _questions(r)})
    assert not ({x for r in current for x in _options(r)} & {x for r in prior for x in _options(r)})


def test_s50_authority_has_no_exact_overlap_with_s50_a0():
    current=(*generate_s50_cases("train"),*generate_s50_cases("dev"))
    a0=s50_a0_cases()
    cs={x for r in current for x in _states(r)}
    cq={x for r in current for x in _questions(r)}
    co={x for r in current for x in _options(r)}
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
