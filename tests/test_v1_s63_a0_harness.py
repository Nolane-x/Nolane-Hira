from pathlib import Path


def test_s63_a0_harness_is_mechanical_and_predev_only():
    source=Path("scripts/hira_v1_s63_a0_learned_set_reliability_gate.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s63_a0_learned_set_reliability_gate.py","exec")
    assert "SEED=84_001" in source
    assert 'OUTCOME="HIRA_V1_S63_A0_LEARNED_SET_RELIABILITY_GATE_READY"' in source
    assert "S63_GATE_PARAMETER_COUNT" in source
    assert '"W_phi":[8,4]' in source
    assert '"b_phi":[8]' in source
    assert '"w_out":[20]' in source
    assert '"b_out":[]' in source
    assert "option_permutation_alpha_max_abs_error" in source
    assert "surface_affine_alpha_max_abs_error" in source
    assert "post_output_warm_W_phi_gradient_l1" in source
    assert "post_output_warm_b_phi_gradient_l1" in source
    assert '"reference_gradient_to_upstream_zero":True' in source
    assert '"treatment_gradient_to_upstream_zero":True' in source
    assert '"fresh_train_dev_exposed":False' in source
    assert '"used_for_model_selection":False' in source
    assert '"production_ready_claimed":False' in source
