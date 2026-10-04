from pathlib import Path
import sys

from nmd.v1_s46_authority import generate_s46_cases
from nmd.v1_s47_authority import generate_s47_cases, validate_s47_partitions

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
from hira_v1_s47_a0_ordinal_pairwise_consensus import cases as s47_a0_cases


def _state_texts(row):
    return (row.state_a,row.state_b)


def _question_texts(row):
    return (row.question_a1,row.question_a2,row.question_b1,row.question_b2)


def _option_texts(row):
    return (*row.option_texts,*row.option_aliases)


def test_s47_authority_sizes_domains_and_internal_freshness():
    train=generate_s47_cases("train")
    dev=generate_s47_cases("dev")
    validate_s47_partitions(train,dev)
    assert len(train)==768
    assert len(dev)==192
    assert len({r.domain for r in train})==12
    assert len({r.domain for r in dev})==12


def test_s47_authority_has_no_exact_overlap_with_s46_train_dev():
    current=(*generate_s47_cases("train"),*generate_s47_cases("dev"))
    prior=(*generate_s46_cases("train"),*generate_s46_cases("dev"))

    cs={x for r in current for x in _state_texts(r)}
    ps={x for r in prior for x in _state_texts(r)}
    cq={x for r in current for x in _question_texts(r)}
    pq={x for r in prior for x in _question_texts(r)}
    co={x for r in current for x in _option_texts(r)}
    po={x for r in prior for x in _option_texts(r)}

    assert not (cs&ps)
    assert not (cq&pq)
    assert not (co&po)


def test_s47_authority_has_no_exact_overlap_with_s47_a0():
    current=(*generate_s47_cases("train"),*generate_s47_cases("dev"))
    a0=s47_a0_cases()

    cs={x for r in current for x in _state_texts(r)}
    cq={x for r in current for x in _question_texts(r)}
    co={x for r in current for x in _option_texts(r)}

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
