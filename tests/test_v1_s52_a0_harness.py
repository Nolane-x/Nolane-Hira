from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))

from hira_v1_s52_a0_query_relation_canonicalization import cases


def test_s52_a0_script_compiles_and_binds_parent_authority():
    path=Path("scripts/hira_v1_s52_a0_query_relation_canonicalization.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "HIRA_V1_S52_A0_QUERY_RELATION_CANONICALIZATION_READY" in source
    assert "ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628" in source
    assert "19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916" in source
    assert "canonicalizer_gradient_l1" in source
    assert "correction_gradient_l1" in source
    assert len(cases())==16


def test_s52_a0_workflow_is_marker_gated_and_parent_artifact_pinned():
    source=Path(".github/workflows/hira-v1-s52-a0-query-relation-canonicalization.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S52-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "hira-v1-s51-native-authority" in source
    assert "scripts/hira_v1_s52_a0_query_relation_canonicalization.py" in source
    assert "tests/test_v1_query_relation_canonicalization.py" in source
    assert 'assert r["reference_canonicalizer_parameter_count"]==32768' in source
    assert 'assert r["treatment_private_trainable_parameter_count"]==147456' in source
    assert 'assert a["correction_gradient_l1"]==0.0' in source
