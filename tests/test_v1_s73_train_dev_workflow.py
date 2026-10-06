from pathlib import Path


def test_s73_train_dev_workflow_is_marker_gated_and_one_shot():
    path=Path(".github/workflows/hira-v1-s73-counterfactual-safety-veto-train-dev.yml")
    text=path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s73-counterfactual-safety-veto" in text
    assert '"research/HIRA-V1-S73-ENABLE-TRAIN-DEV"' in text
    assert "Run one fresh S73 TRAIN DEV court" in text
    assert "scripts/hira_v1_s73_train_dev.py" in text
    assert "37476327384" in text
    assert "11418978543" in text
    assert "sha256:bab338c55ae37bd322fcc20d838ed3b4250da959e2d7135525ec611da8f9c083" in text
    assert "HIRA_V1_S73_COUNTERFACTUAL_SAFETY_VETO_DEV_COMPLETE" in text
    assert "second_dev_run_performed" in text
    assert "external_laya_jev_evaluation_opened" in text
