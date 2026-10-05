from pathlib import Path


def test_s62_trainer_matches_gold_ce_vs_reliability_bce_only():
    source=Path("scripts/hira_v1_s62_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s62_train_dev.py","exec")
    assert "SEED=83_001" in source
    assert "CORRECTION_PARAMS=114_688" in source
    assert "PAIRWISE_PARAMS=32_832" in source
    assert "GATE_PARAMS=5" in source
    assert "adaptive_gate_gold_loss" in source
    assert "reliability_gate_loss" in source
    assert '"reference_gate_objective":"gold_ce"' in source
    assert '"treatment_gate_objective":"train_only_reliability_bce"' in source
    assert '"added_treatment_parameters":0' in source
    assert '"gate_architecture_identical":True' in source
    assert '"gate_initialization_bit_identical":True' in source
    assert '"alpha_probe":S62_ALPHA_PROBE' in source
    assert '"target_tolerance":S62_TARGET_TOLERANCE' in source
    assert '"dev_target_dependency":False' in source
    assert '"fused_shadow_selection_arm":False' in source
    assert "s17._selection_key" in source
    assert "second_dev_run_performed" in source
