from pathlib import Path


def test_s49_treatment_continuation_compiles_and_never_runs_reference():
    path=Path("scripts/hira_v1_s49_treatment_continuation.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "37178974852" in source
    assert "reference_rerun_performed" in source
    assert "treatment_only_mechanical_continuation_authorized" in source
    assert "s49._run_treatment(" in source
    assert "s49._run_reference(" not in source
    assert "observed!=frozen_hashes" in source
    assert "all_24_epoch_runtime_state_sha256_equal" in source
    assert "constructor_compatibility_fix_only" in source
    assert "second_full_s49_run_performed" in source


def test_s49_treatment_continuation_workflow_is_marker_gated():
    path=Path(".github/workflows/hira-v1-s49-treatment-only-continuation.yml")
    source=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s49-private-query-free-option-identity" in source
    assert "research/HIRA-V1-S49-ENABLE-TREATMENT-CONTINUATION" in source
    assert "scripts/hira_v1_s49_treatment_continuation.py" in source
    assert "research/HIRA-V1-S49-RECOVERED-REFERENCE.json" in source
    assert "run-id: 37178164136" in source
    assert 'assert r["reference_rerun_performed"] is False' in source
    assert 'assert traj["all_24_epoch_runtime_state_sha256_equal"] is True' in source
