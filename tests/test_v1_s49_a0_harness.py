from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s48_a0_query_quotient_option_evidence import cases as s48_cases
from hira_v1_s49_a0_query_free_option_identity import cases as s49_cases


def _surfaces(cases):
    states={x for r in cases for x in (r.state_a,r.state_b)}
    questions={x for r in cases for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    options=set()
    for case in cases:
        packed,_ga,_gb=case.option_pack()
        for option in packed:
            options.add(option.criterion_text); options.update(option.aliases)
    return states,questions,options


def test_s49_a0_script_compiles_and_binds_query_free_identity():
    path=Path("scripts/hira_v1_s49_a0_query_free_option_identity.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S49_A0_PRIVATE_QUERY_FREE_OPTION_IDENTITY_READY" in source
    assert "QueryFreeIdentityPrivateCorrectionFork" in source
    assert "identity_api_question_inputs_absent" in source
    assert "arbitrary_k255_pass" in source
    assert "raw_query_live_corrected_logit_max_abs" in source
    assert "_ownership_warmstart_court" in source


def test_s49_a0_rows_are_wholly_new_vs_s48_a0():
    a=_surfaces(s49_cases()); b=_surfaces(s48_cases())
    assert not (a[0]&b[0]); assert not (a[1]&b[1]); assert not (a[2]&b[2])
    assert len(s49_cases())==16


def test_s49_a0_workflow_is_marker_gated():
    path=Path(".github/workflows/hira-v1-s49-a0-query-free-option-identity.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s49-private-query-free-option-identity" in source
    assert "research/HIRA-V1-S49-ENABLE-A0" in source
    assert "scripts/hira_v1_s49_a0_query_free_option_identity.py" in source
    assert "tests/test_v1_query_free_option_identity.py" in source
    assert 'assert r["identity_parameter_count"]==0' in source
    assert 'assert r["identity_api_question_inputs_absent"] is True' in source
    assert 'assert o["js_only_native_runtime_gradient_l1"]==0.0' in source
