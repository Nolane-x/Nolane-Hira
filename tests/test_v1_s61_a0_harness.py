from pathlib import Path


def test_s61_a0_script_compiles_and_is_fresh_science_free():
    path=Path("scripts/hira_v1_s61_a0_confidence_adaptive_bounded_hybrid.py")
    source=path.read_text(encoding="utf-8")
    compile(source,str(path),"exec")
    assert "S61_GATE_PARAMETER_COUNT" in source
    assert "adaptive_gate_features" in source
    assert "gate_gradient_to_upstream_zero" in source
    assert "fresh_train_dev_exposed" in source
    assert "teacher_dependency" in source
    assert "generate_s61_cases" not in source


def test_s61_a0_workflow_is_marker_gated():
    source=Path(".github/workflows/hira-v1-s61-a0-confidence-adaptive-bounded-hybrid.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S61-ENABLE-A0" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s61_a0_confidence_adaptive_bounded_hybrid.py" in source
    assert 'assert r["gate_parameter_count"]==5' in source
    assert 'assert r["feature_dimension"]==4' in source
    assert 'assert r["fresh_train_dev_exposed"] is False' in source
