from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from hira_v1_s45_a0_cross_view_consistent_private_correction import cases as s45_cases
from hira_v1_s46_a0_robust_three_expert_consensus import cases as s46_cases


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


def test_s46_a0_script_compiles_and_binds_robust_consensus():
    path=Path("scripts/hira_v1_s46_a0_robust_three_expert_consensus.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")

    assert "HIRA_V1_S46_A0_ROBUST_THREE_EXPERT_CONSENSUS_READY" in source
    assert "RobustThreeExpertMedianFusion" in source
    assert "GradientIsolatedFullKEvidenceFusion" in source
    assert "k255_pass" in source
    assert "one_extreme_outlier_max_abs_error" in source
    assert "actual_shell_legacy_vs_robust_max_abs" in source
    assert "_ownership_warmstart_court" in source
    assert "seed=65001" not in source
    assert "seed=66001" not in source


def test_s46_a0_rows_are_wholly_new_vs_s45_a0():
    s46=_surfaces(s46_cases())
    s45=_surfaces(s45_cases())
    assert not (s46[0]&s45[0])
    assert not (s46[1]&s45[1])
    assert not (s46[2]&s45[2])
    assert len(s46_cases())==16


def test_s46_a0_workflow_is_marker_gated():
    path=Path(".github/workflows/hira-v1-s46-a0-robust-three-expert-consensus.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s46-robust-three-expert-consensus" in source
    assert "research/HIRA-V1-S46-ENABLE-A0" in source
    assert "scripts/hira_v1_s46_a0_robust_three_expert_consensus.py" in source
    assert "tests/test_v1_robust_three_expert_consensus.py" in source
    assert 'assert r["fusion_parameter_count"]==0' in source
    assert 'assert r["arbitrary_k255_pass"] is True' in source
    assert 'assert r["one_extreme_outlier_max_abs_error"]<=3e-6' in source
    assert 'assert o["js_only_native_runtime_gradient_l1"]==0.0' in source
