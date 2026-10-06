from pathlib import Path


def test_s68_train_dev_workflow_is_marker_gated_and_a0_pinned():
    source=Path(".github/workflows/hira-v1-s68-context-modulated-pairwise-train-dev.yml").read_text(encoding="utf-8")
    assert "research/HIRA-V1-S68-ENABLE-TRAIN-DEV" in source
    assert "run-id: 37388612141" in source
    assert "11380079442" in source
    assert "sha256:06cd73759293281780a4ba8e9aa1c375dd95ea1a632061c2351cf21fd5787da5" in source
    assert "run-id: 37192490832" in source
    assert "scripts/hira_v1_s68_train_dev.py" in source
    assert 'assert cv["reference_pairwise_trainable_parameters"]==33344' in source
    assert 'assert cv["treatment_pairwise_trainable_parameters"]==33344' in source
    assert 'assert cv["added_treatment_pairwise_parameters"]==0' in source
    assert 'assert cv["gate_objective_both_arms"]=="per_view_counterfactual_responsibility_bce"' in source
    assert 'assert r["second_dev_run_performed"] is False' in source
    assert 'assert r["external_laya_jev_evaluation_opened"] is False' in source
