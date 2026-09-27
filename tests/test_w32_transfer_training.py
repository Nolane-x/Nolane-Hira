import torch

from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256
from nmd.w32_transfer_core import (
    W32_CANDIDATE_CHECKPOINT_SCHEMA,
    W32_CANDIDATE_PARAMETER_COUNT,
    W32_CANDIDATE_RANK,
    load_w32_candidate_checkpoint,
)
from nmd.w32_transfer_eval import (
    VECTOR_VALIDITY_COEFFICIENT,
    _selection_key,
    invalid_vector_mass,
)


def _binary_logits(value: int) -> torch.Tensor:
    return (
        torch.tensor([8.0, -8.0])
        if value == 0
        else torch.tensor([-8.0, 8.0])
    )


def test_w32_invalid_vector_mass_distinguishes_valid_and_invalid_vectors():
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


def test_w32_dev_selection_prioritizes_worst_factor_quality():
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


def test_w32_candidate_checkpoint_contract_roundtrip(tmp_path):
    state = {
        "state_adapter.down.weight": torch.randn(8, 128),
        "state_adapter.up.weight": torch.randn(128, 8),
        "schema_adapter.down.weight": torch.randn(8, 128),
        "schema_adapter.up.weight": torch.randn(128, 8),
        "interaction_state.weight": torch.randn(8, 128),
        "interaction_schema.weight": torch.randn(8, 128),
    }
    path = tmp_path / "candidate.pt"
    torch.save(
        {
            "schema_version": W32_CANDIDATE_CHECKPOINT_SCHEMA,
            "kind": "interaction-semantic-adapter",
            "rank": W32_CANDIDATE_RANK,
            "parameter_count": W32_CANDIDATE_PARAMETER_COUNT,
            "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
            "selected_dev_epoch": 9,
            "candidate_state_dict": state,
        },
        path,
    )
    sha = file_sha256(path)
    loaded, metadata = load_w32_candidate_checkpoint(
        path,
        expected_sha256=sha,
    )
    assert metadata["selected_dev_epoch"] == 9
    assert metadata["parameter_count"] == 6144
    assert metadata["sha256"] == sha
    assert set(loaded) == set(state)
    for key in state:
        assert torch.equal(loaded[key], state[key])
