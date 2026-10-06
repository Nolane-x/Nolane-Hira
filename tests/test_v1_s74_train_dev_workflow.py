from pathlib import Path


def test_s74_train_dev_workflow_is_marker_gated_and_one_shot():
    path=Path(".github/workflows/hira-v1-s74-direct-set-arbitration-core-train-dev.yml")
    text=path.read_text(encoding="utf-8")

    assert "feat/hira-v1-s74-direct-set-arbitration-core" in text
    assert '"research/HIRA-V1-S74-ENABLE-TRAIN-DEV"' in text
    assert "Run one fresh S74 TRAIN DEV court" in text
    assert "scripts/hira_v1_s74_train_dev.py" in text
    assert "37484291103" in text
    assert "11422648744" in text
    assert "sha256:68620a7c563e5a22325b4415b53020720e37ba05e8f2eb675fef5d83d4089093" in text
    assert "HIRA_V1_S74_DIRECT_SET_ARBITRATION_CORE_DEV_COMPLETE" in text
    assert "second_dev_run_performed" in text
    assert "external_laya_jev_evaluation_opened" in text
