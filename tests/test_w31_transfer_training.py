import torch

from nmd.w31_transfer_core import (
    W31_ADAPTER_CHECKPOINT_SCHEMA,
    W31_ADAPTER_PARAMETER_COUNT,
    W31_ADAPTER_RANK,
    load_w31_adapter_checkpoint,
)
from nmd.w31_transfer_eval import (
    ANCHOR_MARGIN_THRESHOLD,
    _selection_key,
    _signed_margin,
)
from nmd.w31_transfer_eval import quality_gate_pass, transfer_gate_summary
from nmd.typed_competitive_cache import file_sha256
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256


def test_w31_signed_margin_contract():
    logits = torch.tensor([0.2, 0.5])
    assert torch.isclose(_signed_margin(logits, 1), torch.tensor(0.3))
    assert torch.isclose(_signed_margin(logits, 0), torch.tensor(-0.3))
    assert ANCHOR_MARGIN_THRESHOLD == 0.08


def test_w31_dev_selection_prioritizes_worst_factor():
    def summary(f0, f1, f2, severity=0.8, invalid=0.0):
        return {
            "factors": {
                "F0": {"balanced_accuracy": f0[0], "top1": f0[1]},
                "F1": {"balanced_accuracy": f1[0], "top1": f1[1]},
                "F2": {"balanced_accuracy": f2[0], "top1": f2[1]},
            },
            "composed_severity_top1": severity,
            "invalid_factor_vector_rate": invalid,
        }

    balanced = summary((0.80, 0.80), (0.80, 0.80), (0.80, 0.80), severity=0.70)
    easy_factor_heavy = summary((0.99, 0.99), (0.99, 0.99), (0.79, 0.99), severity=0.99)
    assert _selection_key(balanced, 4) > _selection_key(easy_factor_heavy, 4)

    same_worst_ba_low_top1 = summary((0.80, 0.75), (0.90, 0.90), (0.90, 0.90))
    same_worst_ba_high_top1 = summary((0.80, 0.80), (0.90, 0.90), (0.90, 0.90))
    assert _selection_key(same_worst_ba_high_top1, 4) > _selection_key(
        same_worst_ba_low_top1,
        4,
    )


def test_w31_checkpoint_contract_roundtrip(tmp_path):
    state = {
        "state_adapter.down.weight": torch.randn(8, 128),
        "state_adapter.up.weight": torch.randn(128, 8),
        "schema_adapter.down.weight": torch.randn(8, 128),
        "schema_adapter.up.weight": torch.randn(128, 8),
    }
    path = tmp_path / "adapter.pt"
    torch.save(
        {
            "schema_version": W31_ADAPTER_CHECKPOINT_SCHEMA,
            "kind": "dual-semantic-adapter",
            "rank": W31_ADAPTER_RANK,
            "parameter_count": W31_ADAPTER_PARAMETER_COUNT,
            "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
            "selected_dev_epoch": 7,
            "adapter_state_dict": state,
        },
        path,
    )
    sha = file_sha256(path)
    loaded, metadata = load_w31_adapter_checkpoint(
        path,
        expected_sha256=sha,
    )
    assert metadata["selected_dev_epoch"] == 7
    assert metadata["parameter_count"] == 4096
    assert metadata["sha256"] == sha
    assert set(loaded) == set(state)
    for key in state:
        assert torch.equal(loaded[key], state[key])


def test_w31_sealed_quality_gate_requires_every_factor():
    good = {
        "factors": {
            "F0": {"top1": 0.91, "balanced_accuracy": 0.89},
            "F1": {"top1": 0.92, "balanced_accuracy": 0.90},
            "F2": {"top1": 0.90, "balanced_accuracy": 0.88},
        },
        "factor_vector_top1": 0.83,
        "composed_severity_top1": 0.84,
        "invalid_factor_vector_rate": 0.04,
        "factor_probability_mass_max_error": 1e-7,
    }
    assert quality_gate_pass(good)
    bad = {
        **good,
        "factors": {
            **good["factors"],
            "F2": {"top1": 0.899, "balanced_accuracy": 0.90},
        },
    }
    assert not quality_gate_pass(bad)


def test_w31_transfer_gate_is_pooled_and_strict():
    baseline = {
        "factors": {
            "F0": {"top1": 0.70},
            "F1": {"top1": 0.72},
            "F2": {"top1": 0.74},
        },
        "composed_severity_top1": 0.50,
    }
    dual = {
        "factors": {
            "F0": {"top1": 0.80},
            "F1": {"top1": 0.83},
            "F2": {"top1": 0.84},
        },
        "composed_severity_top1": 0.66,
    }
    result = transfer_gate_summary(baseline, dual)
    assert result["pass"]
    assert result["severity_delta"] >= 0.15
    assert result["worst_factor_top1_delta"] >= 0.08

    regressed = {
        **dual,
        "factors": {
            "F0": {"top1": 0.80},
            "F1": {"top1": 0.69},
            "F2": {"top1": 0.84},
        },
    }
    assert not transfer_gate_summary(baseline, regressed)["pass"]
