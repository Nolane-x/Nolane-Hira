from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s47_a0_ordinal_pairwise_consensus import cases as s47_cases
from hira_v1_s48_a0_query_quotient_option_evidence import cases as s48_cases


def _surfaces(cases):
    states={x for r in cases for x in (r.state_a,r.state_b)}
    questions={x for r in cases for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    options=set()
    for case in cases:
        packed,_ga,_gb=case.option_pack()
        for option in packed:
            options.add(option.criterion_text)
            options.update(option.aliases)
    return states,questions,options


def test_s48_a0_script_compiles_and_binds_quotient():
    path=Path("scripts/hira_v1_s48_a0_query_quotient_option_evidence.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S48_A0_QUERY_QUOTIENT_OPTION_EVIDENCE_READY" in source
    assert "QueryQuotientPrivateCorrectionFork" in source
    assert "arbitrary_k255_pass" in source
    assert "orthogonal_nuisance_quotient_max_abs_error" in source
    assert "raw_query_bypass" in source
    assert "_ownership_warmstart_court" in source


def test_s48_a0_rows_are_wholly_new_vs_s47_a0():
    s48=_surfaces(s48_cases())
    s47=_surfaces(s47_cases())
    assert not (s48[0]&s47[0])
    assert not (s48[1]&s47[1])
    assert not (s48[2]&s47[2])
    assert len(s48_cases())==16


def test_s48_a0_workflow_is_marker_gated():
    path=Path(".github/workflows/hira-v1-s48-a0-query-quotient-option-evidence.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s48-query-quotient-option-evidence" in source
    assert "research/HIRA-V1-S48-ENABLE-A0" in source
    assert "scripts/hira_v1_s48_a0_query_quotient_option_evidence.py" in source
    assert "tests/test_v1_query_quotient_private_correction.py" in source
    assert 'assert r["quotient_trainable_parameter_count"]==0' in source
    assert 'assert r["raw_query_bypass"] is False' in source
    assert 'assert o["js_only_native_runtime_gradient_l1"]==0.0' in source
