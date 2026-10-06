from pathlib import Path


def test_s74_a0_workflow_is_marker_gated():
    text=Path(".github/workflows/hira-v1-s74-a0-direct-set-arbitration-core.yml").read_text(encoding="utf-8")
    assert "feat/hira-v1-s74-direct-set-arbitration-core" in text
    assert '"research/HIRA-V1-S74-ENABLE-A0"' in text
    assert "Run one S74 mechanical A0 court" in text
    assert "fresh_train_dev_exposed" in text
