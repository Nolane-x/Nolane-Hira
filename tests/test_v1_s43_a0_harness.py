from pathlib import Path


def test_s43_a0_script_compiles_and_binds_relative_gap_anchor():
    path = Path("scripts/hira_v1_s43_a0_cross_view_relative_gap_geometry.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "HIRA_V1_S43_A0_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_READY" in source
    assert "S43_A0_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_FULL_BILINEAR_ONLY" in source
    assert "relative_gap_signature_geometry_anchor" in source
    assert "relative_gap_geometry_loss" in source
    assert "relative_gap_tensor" in source
    assert "relative_gap_geometry_shape" in source
    assert "relative_gap_offdiagonal_count" in source
    assert "relative_gap_diagonal_perturbation_loss" in source
    assert "relative_gap_wrong_option_perturbation_loss" in source
    assert "equal_full_mse_distorted_gap_loss" in source
    assert "relative_gap_shared_option_permutation_error" in source
    assert "seed=64001" in source
    assert "seed=64141" in source

    assert "S42-A0" not in source
    assert "s42_a0_only" not in source
    assert "63001" not in source
    assert "63141" not in source


def test_s43_a0_uses_wholly_new_local_semantic_rows():
    source = Path(
        "scripts/hira_v1_s43_a0_cross_view_relative_gap_geometry.py"
    ).read_text(encoding="utf-8")

    for token in (
        "topological photon relay",
        "cryogenic magnon sampler",
        "attosecond field camera",
        "quantum acoustic router",
        "muonic phase scanner",
        "quantum heat-flow camera",
    ):
        assert token in source

    assert 'case_id=f"a0-s43-' in source
    assert 'domain="s43_a0_only"' in source


def test_s43_a0_workflow_is_marker_gated_and_gap_verified():
    path = Path(
        ".github/workflows/"
        "hira-v1-s43-a0-cross-view-relative-gap-geometry.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s43-cross-view-relative-gap" in source
    assert "research/HIRA-V1-S43-ENABLE-A0" in source
    assert "scripts/hira_v1_s43_a0_cross_view_relative_gap_geometry.py" in source
    assert "tests/test_v1_cross_view_relative_gap_anchor.py" in source
    assert 'assert r["relative_gap_geometry_shape"]==[3,4,4]' in source
    assert 'assert r["relative_gap_offdiagonal_count"]==36' in source
    assert 'assert r["relative_gap_identical_loss"]==0.0' in source
    assert 'assert r["relative_gap_diagonal_perturbation_loss"]>0.0' in source
    assert 'assert r["relative_gap_wrong_option_perturbation_loss"]>0.0' in source
    assert 'assert r["equal_full_mse_absolute_shift_gap_loss"]==0.0' in source
    assert 'assert r["equal_full_mse_distorted_gap_loss"]>0.0' in source
    assert 'assert r["relative_gap_shared_option_permutation_error"]<=1e-7' in source
