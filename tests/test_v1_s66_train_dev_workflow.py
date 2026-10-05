from pathlib import Path


def test_s66_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s66-per-view-responsibility-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S66-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37329441437" in source
    assert "11353806431" in source
    assert "sha256:61d76394e79457f26251468fea6e73be53e0507d979aa64d5b24c67f18562c39" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s66_train_dev.py" in source
    assert 'assert cv["reference_gate_trainable_parameters"]==60' in source
    assert 'assert cv["treatment_gate_trainable_parameters"]==60' in source
    assert 'assert cv["added_treatment_parameters"]==0' in source
    assert 'assert cv["reference_shared_pair_target"] is True' in source
    assert 'assert cv["treatment_per_view_responsibility"] is True' in source
    assert 'assert cv["single_view_inference_both_arms"] is True' in source
    assert 'assert cv["alpha_probe"]==0.35' in source
    assert 'assert cv["target_tolerance"]==1e-8' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
