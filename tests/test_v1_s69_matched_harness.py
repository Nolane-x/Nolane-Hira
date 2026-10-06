from pathlib import Path


def test_s69_matched_trainer_freezes_controlled_variable():
    source=Path("scripts/hira_v1_s69_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s69_train_dev.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s69-query-gated-interaction-train-dev-v1"' in source
    assert "SEED=90_001" in source
    assert "generate_s69_cases" in source
    assert "validate_s69_partitions" in source
    assert "build_reference_pairwise_representation" in source
    assert "build_query_gated_pairwise_representation" in source
    assert '"gate_context_source_both_arms":' in source
    assert '"exact_s59_reference_representation"' in source
    assert '"second_dev_run_performed":False' in source
    assert '"external_laya_jev_evaluation_opened":False' in source


def test_s69_trainer_imports_canonical_margin_helper():
    source=Path("scripts/hira_v1_s69_train_dev.py").read_text(encoding="utf-8")
    assert "from nmd.v1_evidence_fusion import fused_gold_vs_max_wrong_margin" in source
    assert "v1_fused_primary_relation_detach" not in source
