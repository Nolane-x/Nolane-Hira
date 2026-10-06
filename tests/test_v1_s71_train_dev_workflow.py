from pathlib import Path


def test_s71_train_dev_workflow_is_marker_gated_and_authorities_pinned():
    source=Path(".github/workflows/hira-v1-s71-multistat-row-composer-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S71-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37457754519" in source
    assert "11410541387" in source
    assert "sha256:52f38ac6d30177c43205b023114a4192333b3374c98c083aed30fa494aececb5" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s71_train_dev.py" in source
    assert 'assert cv["composer_trainable_parameters_per_arm"]==80' in source
    assert 'assert cv["shared_pairwise_head"] is True' in source
    assert 'assert cv["shared_pairwise_training_trajectory"] is True' in source
    assert 'assert cv["shared_correction_trajectory"] is True' in source
    assert 'assert cv["shared_selection_epoch"] is True' in source
    assert 'assert cv["residual_direction_both_arms"]=="uniform_row_mean"' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
