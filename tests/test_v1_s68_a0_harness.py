from pathlib import Path


def test_s68_a0_harness_freezes_context_modulated_pairwise_mechanics():
    source=Path("scripts/hira_v1_s68_a0_context_modulated_pairwise.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s68_a0_context_modulated_pairwise.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s68-a0-context-modulated-pairwise-v1"' in source
    assert 'OUTCOME="HIRA_V1_S68_A0_CONTEXT_MODULATED_PAIRWISE_READY"' in source
    assert "SEED=89_001" in source
    assert "ContextModulatedPairwiseHead" in source
    assert "use_joint_context=False" in source
    assert "use_joint_context=True" in source
    assert '"added_treatment_parameter_count":0' in source
    assert '"fresh_train_dev_exposed":False' in source
    assert '"actions_artifact_available":False' in source
