from __future__ import annotations

import pytest
import torch

from nmd.nullspace_augmentation import (
    combine_updates,
    first_order_structural_benefit,
    normalize_like,
    normalized_subspace_leakage,
    nullspace_component,
    subspace_coordinates,
)
from nmd.subspace_projection import build_retention_subspace


def _axis_subspace():
    rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    return build_retention_subspace(rows)


def test_nullspace_component_removes_rank_one_axis():
    subspace = _axis_subspace()
    vector = torch.tensor([4.0, 2.0, -3.0], dtype=torch.float64)
    residual, projected = nullspace_component(vector, subspace, rank=1)

    assert torch.allclose(projected, torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(residual, torch.tensor([0.0, 2.0, -3.0], dtype=torch.float64))
    assert normalized_subspace_leakage(residual, subspace, rank=1) < 1e-12


def test_subspace_coordinates_match_axis_projection():
    subspace = _axis_subspace()
    vector = torch.tensor([4.0, 2.0, -3.0], dtype=torch.float64)
    coords = subspace_coordinates(vector, subspace, rank=1)
    assert coords.shape == (1,)
    assert abs(float(coords[0])) == pytest.approx(4.0)


def test_normalize_like_and_combination_are_exact():
    direction = torch.tensor([0.0, 3.0, 4.0], dtype=torch.float64)
    reference = torch.tensor([6.0, 8.0, 0.0], dtype=torch.float64)
    normalized = normalize_like(direction, reference)
    assert float(normalized.norm()) == pytest.approx(float(reference.norm()))

    combined = combine_updates(
        reference,
        normalized,
        alpha=0.25,
        gamma=0.125,
    )
    assert torch.allclose(combined, 0.25 * reference + 0.125 * normalized)


def test_first_order_structural_benefit_sign():
    gradient = torch.tensor([1.0, -2.0], dtype=torch.float64)
    descent = -gradient
    assert first_order_structural_benefit(gradient, descent) > 0.0
    assert first_order_structural_benefit(gradient, gradient) < 0.0


def test_zero_direction_is_rejected():
    subspace = _axis_subspace()
    with pytest.raises(ValueError):
        normalized_subspace_leakage(
            torch.zeros(3, dtype=torch.float64),
            subspace,
            rank=1,
        )
    with pytest.raises(ValueError):
        normalize_like(
            torch.zeros(3, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
        )
