from pathlib import Path


def test_s55_train_dev_workflow_is_marker_gated_and_exact_a0_pinned():
    source=Path(".github/workflows/hira-v1-s55-learned-joint-relation-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S55-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37211474503" in source
    assert "11306837784" in source
    assert "sha256:66ad8bbc2a2046ba00fdc17b60bef991dc246cd2182268edf3965ea91e29b812" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s55_train_dev.py" in source
    assert 'assert branch["parameter_surface"]["private_trainable_parameters"]==180224' in source
    assert 'assert branch["parameter_surface"]["learned_joint_trainable_parameters"]==65536' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
