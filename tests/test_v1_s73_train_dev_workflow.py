from pathlib import Path
import yaml


def test_s73_train_dev_workflow_is_marker_gated_and_one_shot():
    path=Path(".github/workflows/hira-v1-s73-counterfactual-safety-veto-train-dev.yml")
    data=yaml.safe_load(path.read_text(encoding="utf-8"))
    on=data.get("on",data.get(True))
    push=on["push"]
    assert push["branches"]==["feat/hira-v1-s73-counterfactual-safety-veto"]
    assert push["paths"]==["research/HIRA-V1-S73-ENABLE-TRAIN-DEV"]

    text=path.read_text(encoding="utf-8")
    assert "Run one fresh S73 TRAIN DEV court" in text
    assert "scripts/hira_v1_s73_train_dev.py" in text
    assert "37476327384" in text
    assert "11418978543" in text
    assert "sha256:bab338c55ae37bd322fcc20d838ed3b4250da959e2d7135525ec611da8f9c083" in text
    assert "HIRA_V1_S73_TRAIN_DEV_RECEIPT" in text or "result.json" in text
