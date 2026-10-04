from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s51_a0_persisted_native_authority import cases


def test_s51_a0_script_compiles_and_is_mechanical_only():
    path=Path("scripts/hira_v1_s51_a0_persisted_native_authority.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S51_A0_PERSISTED_NATIVE_AUTHORITY_READY" in source
    assert "save_native_authority" in source
    assert "load_native_authority" in source
    assert "tamper_rejected" in source
    assert "branch_order_replay_max_abs_error" in source
    assert "used_for_model_selection" in source
    assert len(cases())==16


def test_s51_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s51-a0-persisted-native-authority.yml").read_text(encoding="utf-8")
    assert "feat/hira-v1-s51-persisted-native-authority" in source
    assert "research/HIRA-V1-S51-ENABLE-A0" in source
    assert "scripts/hira_v1_s51_a0_persisted_native_authority.py" in source
    assert "tests/test_v1_persisted_native_authority.py" in source
    assert "tests/test_v1_s51_phase_a_isolation.py" in source
    assert "tests/test_v1_s51_phase_b_isolation.py" in source
    assert 'assert r["tamper_rejected"] is True' in source
    assert 'assert r["branch_order_replay_max_abs_error"]==0.0' in source
