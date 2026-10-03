from pathlib import Path


def test_s39_matched_trainer_compiles_and_binds_canonical_a0():
    path = Path("scripts/hira_v1_s39_train_dev.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "SEED = 60001" in source
    assert (
        "from hira_v1_s39_a0_gradient_isolated_bilinear import cases as s39_a0_cases"
        in source
    )
    assert "hira_v1_s39_a0_full_bilinear_readout" not in source
    assert '"same_epoch_counterfactual"' in source
    assert "delta_treatment_minus_same_epoch_control" in source


def test_s39_matched_workflow_is_bound_to_branch_marker_and_seed():
    path = Path(
        ".github/workflows/"
        "hira-v1-s39-matched-gradient-isolated-bilinear-train-dev.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s39-gradient-isolated-bilinear" in source
    assert 'research/HIRA-V1-S39-ENABLE-TRAIN-DEV' in source
    assert 'assert r["seed"]==60001' in source
    assert 'assert r["seed"]==59001' not in source
    assert "scripts/hira_v1_s39_train_dev.py" in source
    assert "scripts/hira_v1_s39_a0_gradient_isolated_bilinear.py" in source
    assert 'cf=r["same_epoch_counterfactual"]' in source
