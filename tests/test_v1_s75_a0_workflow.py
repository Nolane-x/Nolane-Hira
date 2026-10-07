from pathlib import Path


def test_s75_a0_workflow_is_marker_gated():
    path=Path(".github/workflows/hira-v1-s75-a0-token-level-bidirectional-binding.yml")
    text=path.read_text(encoding="utf-8")
    assert "feat/hira-v1-s75-token-level-bidirectional-binding" in text
    assert '"research/HIRA-V1-S75-ENABLE-A0"' in text
    assert "Run one S75 mechanical A0 court" in text
    assert "scripts/hira_v1_s75_a0_token_level_bidirectional_binding.py" in text
    assert "HIRA_V1_S75_A0_TOKEN_LEVEL_BIDIRECTIONAL_BINDING_READY" in text
    assert "fresh_train_dev_exposed" in text
