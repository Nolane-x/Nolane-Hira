from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "hira_v1_s11_train_dev.py"
    spec = importlib.util.spec_from_file_location("hira_v1_s11_train_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(**overrides):
    base = {
        "canonical_paired_both_correct_rate": 0.10,
        "canonical_binding_accuracy": 0.20,
        "canonical_binding_mean_gold_margin": -0.30,
        "cross_view_selected_choice_agreement": 0.40,
        "canonical_question_swap_choice_change_rate": 0.50,
        "canonical_paired_gold_state_support_top2_containment": 0.60,
        "mean_canonical_binding_loss": 1.20,
    }
    base.update(overrides)
    return base


def test_s11_train_contract_constants_are_frozen():
    m = load_module()
    assert m.SEED == 17001
    assert m.EPOCHS == 24
    assert m.BATCH_SIZE == 16
    assert m.LR == 2e-4
    assert m.WEIGHT_DECAY == 0.01
    assert m.GRAD_CLIP == 1.0
    assert m.SWAP_COEFFICIENT == 0.25
    assert m.SWAP_MARGIN == 0.20
    assert m.OPTION_ALIGN_COEFFICIENT == 0.05
    assert m.OPTION_ALIGN_TEMPERATURE == 0.10
    assert m.ROLE_TEMPERATURE == 0.10
    assert m.STATE_TEMPERATURE == 0.10
    assert m.INVARIANCE_COEFFICIENT == 0.25
    assert m.READY == "HIRA_V1_S11_ROLE_VALUE_DEV_READY"
    assert m.FAIL == "HIRA_V1_S11_ROLE_VALUE_DEV_FAIL"


def test_s11_selection_order_prioritizes_paired_correctness_first():
    m = load_module()
    weak = _metrics(canonical_binding_accuracy=0.99)
    paired = _metrics(
        canonical_paired_both_correct_rate=0.11,
        canonical_binding_accuracy=0.01,
    )
    assert m._selection_key(9, paired) > m._selection_key(1, weak)


def test_s11_selection_order_uses_margin_before_cross_view():
    m = load_module()
    margin = _metrics(
        canonical_binding_mean_gold_margin=-0.20,
        cross_view_selected_choice_agreement=0.01,
    )
    cross = _metrics(
        canonical_binding_mean_gold_margin=-0.30,
        cross_view_selected_choice_agreement=0.99,
    )
    assert m._selection_key(9, margin) > m._selection_key(1, cross)


def test_s11_selection_prefers_earlier_epoch_only_after_all_metrics_tie():
    m = load_module()
    metrics = _metrics()
    assert m._selection_key(3, metrics) > m._selection_key(4, metrics)
