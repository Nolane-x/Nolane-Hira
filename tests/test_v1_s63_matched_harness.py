from pathlib import Path


def test_s63_trainer_changes_only_reliability_representation_family():
    source=Path("scripts/hira_v1_s63_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s63_train_dev.py","exec")
    assert "SEED=84_001" in source
    assert "REFERENCE_GATE_PARAMS=5" in source
    assert "TREATMENT_GATE_PARAMS=61" in source
    assert "reliability_gate_loss" in source
    assert "learned_set_reliability_loss" in source
    assert '"reference_gate_family":"s62_four_scalar_reliability_gate"' in source
    assert '"treatment_gate_family":"s63_learned_permutation_invariant_set_reliability_gate"' in source
    assert '"reference_gate_objective":"train_only_reliability_bce"' in source
    assert '"treatment_gate_objective":"train_only_reliability_bce"' in source
    assert '"added_treatment_parameters":56' in source
    assert '"same_reliability_target":True' in source
    assert '"shared_correction_head_trajectory":True' in source
    assert '"fused_shadow_selection_arm":False' in source
    assert "treatment_phi_gradient_ever_live" in source
    assert "s17._selection_key" in source
    assert "second_dev_run_performed" in source
