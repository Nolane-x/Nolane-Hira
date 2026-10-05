from pathlib import Path


def test_s57_trainer_is_matched_and_gold_protected():
    source=Path("scripts/hira_v1_s57_train_dev.py").read_text(encoding="utf-8")
    compile(source,"scripts/hira_v1_s57_train_dev.py","exec")
    assert "SEED=78_001" in source
    assert "PRIVATE_TRAINABLE=114_688" in source
    assert "REFERENCE_ORDINAL_COEFFICIENT=0.0" in source
    assert "TREATMENT_ORDINAL_COEFFICIENT=0.05" in source
    assert "weighted_pairwise_ordinal_auxiliary" in source
    assert "canonical.gold" in source
    assert "gold_order_protection" in source
    assert "native_optimizer_constructed" in source
    assert "native_training_performed" in source
    assert "second_dev_run_performed" in source
