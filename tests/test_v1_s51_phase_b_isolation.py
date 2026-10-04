from pathlib import Path


def test_s51_phase_b_has_no_native_training_path():
    path=Path("scripts/hira_v1_s51_private_court.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")

    assert "load_native_authority" in source
    assert "generate_s51_train_cases" in source
    assert "generate_s51_dev_cases" in source
    assert "_train_shared_native" not in source
    assert "build_hira_v1_s17_norm_balanced_core" not in source
    assert "load_hira_v0_m4_bundle" not in source
    assert "norm_balanced_gradient_update" not in source
    assert '"native_optimizer_constructed":False' in source
    assert '"native_training_performed":False' in source
    assert '"native_retraining_performed":False' in source


def test_s51_phase_b_is_artifact_and_train_manifest_bound():
    source=Path("scripts/hira_v1_s51_private_court.py").read_text(encoding="utf-8")
    assert "--authority-checkpoint" in source
    assert "--authority-receipt" in source
    assert "--authority-train-manifest" in source
    assert "verify_file_sha256" in source
    assert "checkpoint_file_sha256" in source
    assert "train_manifest_sha256" in source
    assert "loaded runtime authority hash changed" in source
    assert "loaded logical native digest changed" in source
