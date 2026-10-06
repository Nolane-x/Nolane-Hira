from pathlib import Path


def test_s68_trainer_changes_only_pairwise_context_source():
    source=Path("scripts/hira_v1_s68_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s68_train_dev.py","exec")
    assert "SEED=89_001" in source
    assert "PAIRWISE_PARAMS=S68_PAIRWISE_PARAMETER_COUNT" in source
    assert "GATE_PARAMS=S64_GATE_PARAMETER_COUNT" in source
    assert "ContextModulatedPairwiseHead" in source
    assert "use_joint_context=False" in source
    assert "use_joint_context=True" in source
    assert "per_view_responsibility_loss" in source
    assert '"added_treatment_pairwise_parameters":0' in source
    assert '"pairwise_parameter_initialization_bit_identical":True' in source
    assert '"gate_objective_both_arms":' in source
    assert '"per_view_counterfactual_responsibility_bce"' in source
    assert '"shared_correction_trajectory":True' in source
    assert '"oracle_alpha_target_reopened":False' in source
    assert "selected_pairwise_aggregate_delta_treatment_minus_reference" in source
