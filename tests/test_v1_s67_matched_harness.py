from pathlib import Path


def test_s67_trainer_changes_only_supervision_granularity():
    source=Path("scripts/hira_v1_s67_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s67_train_dev.py","exec")
    assert "SEED=88_001" in source
    assert "REFERENCE_GATE_PARAMS=60" in source
    assert "TREATMENT_GATE_PARAMS=60" in source
    assert "per_view_responsibility_loss" in source
    assert "safe_oracle_alpha_loss" in source
    assert '"reference_gate_objective":"per_view_binary_responsibility_bce"' in source
    assert '"treatment_gate_objective":"safe_oracle_alpha_continuous_bce"' in source
    assert '"added_treatment_parameters":0' in source
    assert '"oracle_alpha_lattice":list(S67_ALPHA_LATTICE)' in source
    assert '"oracle_target_levels":list(S67_TARGET_LEVELS)' in source
    assert '"shared_correction_head_trajectory":True' in source
    assert '"single_view_inference_both_arms":True' in source
    assert '"fused_shadow_selection_arm":False' in source
    assert "train_oracle_nonbinary_fraction" in source
    assert "train_oracle_level_fraction" in source
    assert "s17._selection_key" in source
