from pathlib import Path


def test_s45_a0_script_compiles_and_binds_cross_view_consistency():
    path = Path("scripts/hira_v1_s45_a0_cross_view_consistent_private_correction.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "HIRA_V1_S45_A0_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_READY" in source
    assert "S45_A0_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_ONLY" in source
    assert "PrivateCorrectionRepresentationFork" in source
    assert "s44_binding_ce_plus_cross_view_js" in source
    assert "BINDING_COEFFICIENT" in source
    assert "INVARIANCE_COEFFICIENT" in source
    assert "_js_divergence" in source
    assert "identical_corrected_logit_js" in source
    assert "nonidentical_probe_js" in source
    assert "js_only_adapter_a_gradient_l1" in source
    assert "js_only_adapter_b_gradient_l1" in source
    assert "js_only_w_gradient_l1" in source
    assert "js_only_native_runtime_gradient_l1" in source
    assert "seed=66001" in source
    assert "seed=66141" in source

    assert "S44-A0 quantum polariton recorder" not in source
    assert "seed=65001, train=False" not in source
    assert "seed=65141, train=True" not in source


def test_s45_a0_uses_wholly_new_local_semantic_rows():
    source = Path(
        "scripts/hira_v1_s45_a0_cross_view_consistent_private_correction.py"
    ).read_text(encoding="utf-8")
    for token in (
        "quantum Hall velocimeter",
        "femtosecond magnetostriction camera",
        "cryogenic phonon gyroscope",
        "topological acoustic compass",
        "atomic parity vectormeter",
        "nanophotonic recoil balance",
    ):
        assert token in source

    assert 'case_id=f"a0-s45-' in source
    assert 'domain="s45_a0_only"' in source


def test_s45_a0_workflow_is_marker_gated_and_js_verified():
    path = Path(
        ".github/workflows/hira-v1-s45-a0-cross-view-consistent-private-correction.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s45-cross-view-consistent-private-correction" in source
    assert "research/HIRA-V1-S45-ENABLE-A0" in source
    assert "scripts/hira_v1_s45_a0_cross_view_consistent_private_correction.py" in source
    assert 'assert r["adapter_a_parameter_count"]==32768' in source
    assert 'assert r["adapter_b_parameter_count"]==16384' in source
    assert 'assert r["private_adapter_parameter_count"]==49152' in source
    assert 'assert r["bilinear_parameter_count"]==65536' in source
    assert 'assert r["correction_parameter_count"]==114688' in source
    assert 'assert r["treatment_total_trainable_parameter_count"]==163840' in source
    assert 'assert r["cross_view_js_coefficient"]==0.25' in source
    assert 'assert r["matched_native_one_step_parameter_max_abs"]==0.0' in source
    assert 'assert r["identical_corrected_logit_js"]<=1e-12' in source
    assert 'assert r["nonidentical_probe_js"]>0.0' in source
    assert 'assert r["js_only_adapter_a_gradient_l1"]>0.0' in source
    assert 'assert r["js_only_adapter_b_gradient_l1"]>0.0' in source
    assert 'assert r["js_only_w_gradient_l1"]>0.0' in source
    assert 'assert r["js_only_native_runtime_gradient_l1"]==0.0' in source
