from pathlib import Path


def test_s42_a0_script_compiles_and_binds_relational_anchor():
    path = Path("scripts/hira_v1_s42_a0_cross_view_relational_geometry.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert "HIRA_V1_S42_A0_CROSS_VIEW_RELATIONAL_GEOMETRY_READY" in source
    assert "S42_A0_CROSS_VIEW_RELATIONAL_GEOMETRY_FULL_BILINEAR_ONLY" in source
    assert "relational_signature_geometry_anchor" in source
    assert "relational_diagonal_only_perturbation_loss" in source
    assert "relational_offdiagonal_only_perturbation_loss" in source
    assert "relational_shared_option_permutation_error" in source
    assert "seed=63001" in source
    assert "seed=63141" in source

    assert "S41-A0" not in source
    assert "s41_a0_only" not in source
    assert "signature_anchor_loss" not in source
    assert "62001" not in source
    assert "62141" not in source


def test_s42_a0_uses_wholly_new_local_semantic_rows():
    source = Path(
        "scripts/hira_v1_s42_a0_cross_view_relational_geometry.py"
    ).read_text(encoding="utf-8")

    for token in (
        "terahertz phase lattice",
        "phononic vortex camera",
        "ultracold field mapper",
        "quantum strain scanner",
        "coherent muon imager",
        "neutrino timing tile",
    ):
        assert token in source

    assert 'case_id=f"a0-s42-' in source
    assert 'domain="s42_a0_only"' in source


def test_s42_a0_workflow_is_marker_gated_and_full_geometry_verified():
    path = Path(
        ".github/workflows/"
        "hira-v1-s42-a0-cross-view-relational-geometry.yml"
    )
    source = path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s42-cross-view-relational-geometry" in source
    assert "research/HIRA-V1-S42-ENABLE-A0" in source
    assert "scripts/hira_v1_s42_a0_cross_view_relational_geometry.py" in source
    assert "tests/test_v1_cross_view_relational_anchor.py" in source
    assert 'assert r["relational_geometry_shape"]==[3,4,4]' in source
    assert 'assert r["relational_diagonal_only_perturbation_loss"]>0.0' in source
    assert 'assert r["relational_offdiagonal_only_perturbation_loss"]>0.0' in source
    assert 'assert r["relational_shared_option_permutation_error"]<=1e-7' in source
