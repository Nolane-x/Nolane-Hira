from pathlib import Path


def test_s44_a0_script_compiles_and_binds_private_fork():
    path = Path("scripts/hira_v1_s44_a0_private_correction_fork.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "HIRA_V1_S44_A0_PRIVATE_CORRECTION_REPRESENTATION_READY" in source
    assert "S44_A0_PRIVATE_CORRECTION_REPRESENTATION_ONLY" in source
    assert "PrivateCorrectionRepresentationFork" in source
    assert "s39_binding_ce_only" in source
    assert "BINDING_COEFFICIENT" in source
    assert "warm_start" in source
    assert "matched_native_one_step_parameter_max_abs" in source
    assert "private_vs_w_only_residual_max_abs" in source
    assert "seed=65001" in source
    assert "seed=65141" in source

    assert "S43-A0" not in source
    assert "s43_a0_only" not in source
    assert "64001" not in source
    assert "64141" not in source


def test_s44_a0_uses_wholly_new_local_semantic_rows():
    source = Path("scripts/hira_v1_s44_a0_private_correction_fork.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "quantum polariton recorder",
        "picosecond strain camera",
        "molecular chirality radar",
        "ultracold rotation tile",
        "neutron spin holograph",
        "attosecond charge router",
    ):
        assert token in source

    assert 'case_id=f"a0-s44-' in source
    assert 'domain="s44_a0_only"' in source


def test_s44_a0_workflow_is_marker_gated_and_ownership_verified():
    path = Path(
        ".github/workflows/hira-v1-s44-a0-private-correction-fork.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s44-private-correction-fork" in source
    assert "research/HIRA-V1-S44-ENABLE-A0" in source
    assert "scripts/hira_v1_s44_a0_private_correction_fork.py" in source
    assert "tests/test_v1_private_correction_fork.py" in source
    assert 'assert r["adapter_a_parameter_count"]==32768' in source
    assert 'assert r["adapter_b_parameter_count"]==16384' in source
    assert 'assert r["private_adapter_parameter_count"]==49152' in source
    assert 'assert r["bilinear_parameter_count"]==65536' in source
    assert 'assert r["correction_parameter_count"]==114688' in source
    assert 'assert r["treatment_total_trainable_parameter_count"]==163840' in source
    assert 'assert r["matched_native_one_step_parameter_max_abs"]==0.0' in source
    assert 'warm=r["warm_start"]' in source
    assert 'assert warm[2]["adapter_a_gradient_l1"]>0.0' in source
    assert 'assert r["private_vs_w_only_residual_max_abs"]>1e-7' in source
