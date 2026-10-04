from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s46_a0_robust_three_expert_consensus import cases as s46_cases
from hira_v1_s47_a0_ordinal_pairwise_consensus import cases as s47_cases


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


def test_s47_a0_script_compiles_and_binds_ordinal_consensus():
    path=Path("scripts/hira_v1_s47_a0_ordinal_pairwise_consensus.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S47_A0_ORDINAL_PAIRWISE_CONSENSUS_READY" in source
    assert "OrdinalPairwiseConsensus" in source
    assert "arbitrary_k255_pass" in source
    assert "strict_copeland_dominance_violations" in source
    assert "cycle_private_tiebreak_matches" in source
    assert "actual_shell_legacy_vs_ordinal_max_abs" in source
    assert "_ownership_warmstart_court" in source


def test_s47_a0_rows_are_wholly_new_vs_s46_a0():
    s47=_surfaces(s47_cases())
    s46=_surfaces(s46_cases())
    assert not (s47[0]&s46[0])
    assert not (s47[1]&s46[1])
    assert not (s47[2]&s46[2])
    assert len(s47_cases())==16


def test_s47_a0_workflow_is_marker_gated():
    path=Path(".github/workflows/hira-v1-s47-a0-ordinal-pairwise-consensus.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s47-ordinal-pairwise-consensus" in source
    assert "research/HIRA-V1-S47-ENABLE-A0" in source
    assert "scripts/hira_v1_s47_a0_ordinal_pairwise_consensus.py" in source
    assert "tests/test_v1_ordinal_pairwise_consensus.py" in source
    assert 'assert r["decision_parameter_count"]==0' in source
    assert 'assert r["arbitrary_k255_pass"] is True' in source
    assert 'assert r["strict_copeland_dominance_violations"]==0' in source
    assert 'assert o["js_only_native_runtime_gradient_l1"]==0.0' in source
