from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s53_a0_token_query_option_late_interaction import cases


def test_s53_a0_script_compiles_and_binds_parent_authority():
    path=Path("scripts/hira_v1_s53_a0_token_query_option_late_interaction.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S53_A0_TOKEN_QUERY_OPTION_LATE_INTERACTION_READY" in source
    assert "ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628" in source
    assert "19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916" in source
    assert "pooled_query_bypass_absent" in source
    assert "informative_token_context_sensitivity_max_abs" in source
    assert len(cases())==16


def test_s53_a0_workflow_is_marker_gated_and_parent_artifact_pinned():
    source=Path(".github/workflows/hira-v1-s53-a0-token-query-option-late-interaction.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S53-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "hira-v1-s51-native-authority" in source
    assert "scripts/hira_v1_s53_a0_token_query_option_late_interaction.py" in source
    assert "tests/test_v1_token_query_option_late_interaction.py" in source
    assert 'assert r["reference_correction_parameter_count"]==114688' in source
    assert 'assert r["late_interaction_parameter_count"]==0' in source
    assert 'assert r["pooled_query_bypass_absent"] is True' in source
