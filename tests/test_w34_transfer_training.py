import torch

from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256
from nmd.w34_transfer_core import (
    W34_CANDIDATE_CHECKPOINT_SCHEMA,
    W34_CANDIDATE_PARAMETER_COUNT,
    W34_CANDIDATE_RANK,
    load_w34_candidate_checkpoint,
)
from nmd.w34_transfer_eval import (
    VECTOR_VALIDITY_COEFFICIENT,
    _selection_key,
    invalid_vector_mass,
    quality_gate_pass,
    transfer_gate_summary,
)


def _binary_logits(value: int) -> torch.Tensor:
    return (
        torch.tensor([8.0, -8.0])
        if value == 0
        else torch.tensor([-8.0, 8.0])
    )


def test_w34_invalid_vector_mass_distinguishes_valid_and_invalid_vectors():
    valid = {
        "F0": _binary_logits(1),
        "F1": _binary_logits(1),
        "F2": _binary_logits(0),
    }
    invalid = {
        "F0": _binary_logits(0),
        "F1": _binary_logits(0),
        "F2": _binary_logits(1),
    }
    assert float(invalid_vector_mass(valid)) < 1e-6
    assert float(invalid_vector_mass(invalid)) > 0.999
    assert VECTOR_VALIDITY_COEFFICIENT == 0.15


def test_w34_dev_selection_prioritizes_worst_factor_quality():
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

    balanced = summary((0.82, 0.82), (0.82, 0.82), (0.82, 0.82), severity=0.70)
    easy_heavy = summary((0.99, 0.99), (0.99, 0.99), (0.81, 0.99), severity=0.99)
    assert _selection_key(balanced, 5) > _selection_key(easy_heavy, 5)

    lower_invalid = summary((0.82, 0.82), (0.82, 0.82), (0.82, 0.82), severity=0.70, invalid=0.01)
    higher_invalid = summary((0.82, 0.82), (0.82, 0.82), (0.82, 0.82), severity=0.70, invalid=0.10)
    assert _selection_key(lower_invalid, 5) > _selection_key(higher_invalid, 5)


def test_w34_candidate_checkpoint_contract_roundtrip(tmp_path):
    state = {
        "state_adapter.down.weight": torch.randn(8, 128),
        "state_adapter.up.weight": torch.randn(128, 8),
        "schema_adapter.down.weight": torch.randn(8, 128),
        "schema_adapter.up.weight": torch.randn(128, 8),
        "interaction_state.weight": torch.randn(8, 128),
        "interaction_schema.weight": torch.randn(8, 128),
        "composition_state.weight": torch.randn(8, 128),
        "composition_schema.weight": torch.randn(8, 128),
    }
    path = tmp_path / "candidate.pt"
    torch.save(
        {
            "schema_version": W34_CANDIDATE_CHECKPOINT_SCHEMA,
            "kind": "coevidence-semantic",
            "rank": W34_CANDIDATE_RANK,
            "parameter_count": W34_CANDIDATE_PARAMETER_COUNT,
            "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
            "selected_dev_epoch": 9,
            "candidate_state_dict": state,
        },
        path,
    )
    sha = file_sha256(path)
    loaded, metadata = load_w34_candidate_checkpoint(
        path,
        expected_sha256=sha,
    )
    assert metadata["selected_dev_epoch"] == 9
    assert metadata["parameter_count"] == 8192
    assert metadata["sha256"] == sha
    assert set(loaded) == set(state)
    for key in state:
        assert torch.equal(loaded[key], state[key])


def _gate_summary(f0, f1, f2, vector, severity, invalid, mass=1e-7):
    return {
        "factors": {
            "F0": {"top1": f0[0], "balanced_accuracy": f0[1]},
            "F1": {"top1": f1[0], "balanced_accuracy": f1[1]},
            "F2": {"top1": f2[0], "balanced_accuracy": f2[1]},
        },
        "factor_vector_top1": vector,
        "composed_severity_top1": severity,
        "invalid_factor_vector_rate": invalid,
        "factor_probability_mass_max_error": mass,
    }


def test_w34_quality_gate_is_exact_preregistered_threshold():
    passing = _gate_summary(
        (0.90, 0.88),
        (0.91, 0.89),
        (0.92, 0.90),
        0.82,
        0.82,
        0.05,
    )
    assert quality_gate_pass(passing)

    failing_f2 = _gate_summary(
        (0.95, 0.95),
        (0.95, 0.95),
        (0.899, 0.95),
        0.90,
        0.90,
        0.0,
    )
    assert not quality_gate_pass(failing_f2)


def test_w34_transfer_gate_requires_all_three_relative_constraints():
    baseline = _gate_summary(
        (0.70, 0.70),
        (0.75, 0.75),
        (0.80, 0.80),
        0.40,
        0.40,
        0.20,
    )
    candidate = _gate_summary(
        (0.82, 0.82),
        (0.85, 0.85),
        (0.88, 0.88),
        0.60,
        0.60,
        0.02,
    )
    result = transfer_gate_summary(baseline, candidate)
    assert result["pass"] is True
    assert result["severity_delta"] >= 0.15
    assert result["worst_factor_top1_delta"] >= 0.08
    assert result["max_factor_top1_regression"] <= 0.02

    regressed = _gate_summary(
        (0.67, 0.82),
        (0.90, 0.90),
        (0.90, 0.90),
        0.70,
        0.70,
        0.0,
    )
    assert transfer_gate_summary(baseline, regressed)["pass"] is False
