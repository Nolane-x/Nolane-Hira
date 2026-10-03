from pathlib import Path


def test_s40_matched_trainer_compiles_and_freezes_authority():
    path = Path("scripts/hira_v1_s40_train_dev.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert 'SEED = 61001' in source
    assert 'READOUT_PARAMETER_COUNT = 65_536' in source
    assert 'signature_anchor_loss' in source
    assert 'project_runtime_gradient_against_reference_anchor' in source
    assert 'w_gradient_projected": False' in source
    assert '"coefficient": None' in source
    assert 'HIRA_V1_S40_TRAIN_DEV_RECEIPT=' in source


def test_s40_matched_trainer_uses_preupdate_reference_signatures():
    source = Path("scripts/hira_v1_s40_train_dev.py").read_text(encoding="utf-8")
    i_det = source.index("ref_sig_c_det = ref_sig_c.detach()")
    i_treatment = source.index(") = _treatment_losses(treatment, rows, readout)")
    i_ref_step = source.index("ref_opt.step()")
    assert i_det < i_treatment < i_ref_step


def test_s40_matched_workflow_contract_file_exists_when_staged():
    path = Path(".github/workflows/hira-v1-s40-matched-reference-anchored-train-dev.yml")
    if not path.exists():
        return
    source = path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s40-reference-anchored-coadaptation" in source
    assert "research/HIRA-V1-S40-ENABLE-TRAIN-DEV" in source
    assert 'assert r["seed"]==61001' in source
