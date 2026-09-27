from nmd.mainline import (
    HIRA_V0_MAINLINE_M4_VERSION,
    M2_MECHANICS_AUTHORITY,
    HiraV0Manifest,
)


def test_m4_manifest_changes_version_only_not_scientific_maturity():
    manifest = HiraV0Manifest.m4_runtime_provisional().to_dict()

    assert manifest["version"] == HIRA_V0_MAINLINE_M4_VERSION
    assert manifest["typed_runtime"] == "available"
    assert manifest["high_k_mechanics"] == "available"
    assert manifest["high_k_mechanics_authority"] == M2_MECHANICS_AUTHORITY

    assert manifest["semantic_frontend"] == "provisional"
    assert manifest["transfer_core"] == "provisional"
    assert manifest["reliability_ood_abstention"] == "provisional"
    assert manifest["high_k"] == "provisional"
    assert manifest["multilingual"] == "provisional"
    assert manifest["production_ready"] is False
    assert manifest["transfer_core_promoted"] is False
    assert manifest["relation_refinement"] is False
    assert manifest["adaptive_budget"] is False
