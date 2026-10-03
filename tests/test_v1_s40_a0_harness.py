from pathlib import Path


def test_s40_a0_script_compiles_and_binds_frozen_mechanism():
    path = Path("scripts/hira_v1_s40_a0_reference_anchored_coadaptation.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    assert 'SCHEMA_VERSION = "hira-v1-s40-a0-reference-anchored-projected-coadaptation-v1"' in source
    assert 'OUTCOME = "HIRA_V1_S40_A0_REFERENCE_ANCHORED_PROJECTED_COADAPTATION_READY"' in source
    assert "seed=61001" in source
    assert "project_runtime_gradient_against_reference_anchor" in source
    assert "signature_anchor_loss" in source
    assert '"anchor_coefficient": None' in source
    assert '"projection_epsilon": 1e-12' in source


def test_s40_contract_freezes_parameter_free_projection():
    text = Path("research/HIRA-V1-S40-CONTRACT.md").read_text(encoding="utf-8")
    assert "no anchor coefficient" in text.lower()
    assert "65,536" in text
    assert "seed **61001**" in text
    assert "a dot g_runtime >= 0" in text or "a·g_runtime >= 0" in text
