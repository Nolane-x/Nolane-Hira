from pathlib import Path


def test_s73_a0_harness_freezes_mechanics():
    source=Path("scripts/hira_v1_s73_a0_counterfactual_safety_veto.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s73_a0_counterfactual_safety_veto.py","exec")
    assert 'SCHEMA_VERSION="hira-v1-s73-a0-counterfactual-safety-veto-v1"' in source
    assert 'OUTCOME="HIRA_V1_S73_A0_COUNTERFACTUAL_SAFETY_VETO_READY"' in source
    assert "SEED=94_001" in source
    assert '"fresh_train_dev_exposed":False' in source
    assert '"candidate_residual_changed":False' in source
    assert "S73_ACCEPT_THRESHOLD" in source
    assert "force_accept=True" in source
    assert "force_accept=False" in source
