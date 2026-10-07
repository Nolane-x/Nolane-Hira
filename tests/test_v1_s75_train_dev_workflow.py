from pathlib import Path


def test_s75_train_dev_workflow_is_marker_gated_and_one_shot():
    path=Path(".github/workflows/hira-v1-s75-token-level-bidirectional-binding-train-dev.yml")
    text=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s75-token-level-bidirectional-binding" in text
    assert '"research/HIRA-V1-S75-ENABLE-TRAIN-DEV"' in text
    assert "Run one fresh S75 TRAIN DEV court" in text
    assert "scripts/hira_v1_s75_train_dev.py" in text
    assert "37578213725" in text
    assert "11463278233" in text
    assert "sha256:fae3cc55fdbd8ab841da16b0f6ae113dc17d0186f844cbda26fab6a6a70c424e" in text
    assert "HIRA_V1_S75_TOKEN_LEVEL_BIDIRECTIONAL_BINDING_DEV_COMPLETE" in text
    assert "second_dev_run_performed" in text
    assert "external_laya_jev_evaluation_opened" in text
