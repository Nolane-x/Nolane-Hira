from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s49_a0_query_free_option_identity import cases as s49_cases
from hira_v1_s50_a0_shared_native_forked_readouts import cases as s50_cases


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


def test_s50_a0_script_compiles_and_binds_shared_cache():
    path=Path("scripts/hira_v1_s50_a0_shared_native_forked_readouts.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S50_A0_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_READY" in source
    assert "freeze_shared_native_evidence" in source
    assert "reference_private_logits" in source
    assert "treatment_private_logits" in source
    assert "branch_order_replay_max_abs_error" in source
    assert "private_optimizer_native_parameter_count" in source
    assert "native_training_arms_in_private_phase" in source


def test_s50_a0_rows_are_wholly_new_vs_s49_a0():
    s50=_surfaces(s50_cases())
    s49=_surfaces(s49_cases())
    assert len(s50_cases())==16
    assert not (s50[0]&s49[0])
    assert not (s50[1]&s49[1])
    assert not (s50[2]&s49[2])


def test_s50_a0_workflow_is_marker_gated():
    path=Path(".github/workflows/hira-v1-s50-a0-shared-native-forked-readouts.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s50-shared-native-forked-readouts" in source
    assert "research/HIRA-V1-S50-ENABLE-A0" in source
    assert "scripts/hira_v1_s50_a0_shared_native_forked_readouts.py" in source
    assert "tests/test_v1_shared_native_private_readouts.py" in source
    assert 'assert r["private_optimizer_native_parameter_count"]==0' in source
    assert 'assert r["branch_order_replay_max_abs_error"]==0.0' in source
    assert 'assert r["cache_tensors_require_grad_count"]==0' in source
