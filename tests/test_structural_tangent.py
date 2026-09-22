import torch

from nmd.structural_tangent import (
    functional_null_structural_tangent,
    normalize_to_reference_l2,
)
from nmd.subspace_projection import build_retention_subspace


def test_functional_null_tangent_removes_rank_one_damage_direction():
    rows = torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    subspace = build_retention_subspace(rows, relative_eigenvalue_floor=1e-12)
    gradient = torch.tensor([-2.0, -3.0, 0.0])

    result = functional_null_structural_tangent(
        gradient,
        subspace,
        rank=1,
    )

    assert torch.allclose(
        result.raw_descent,
        torch.tensor([2.0, 3.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        result.projected_damage,
        torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
        atol=1e-10,
    )
    assert torch.allclose(
        result.null_descent,
        torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64),
        atol=1e-10,
    )
    assert abs(result.retained_energy_fraction - 9.0 / 13.0) < 1e-12
    assert result.leakage_energy_fraction <= 1e-20


def test_normalize_to_reference_l2_matches_reference_norm_and_direction():
    x = torch.tensor([3.0, 4.0], dtype=torch.float64)
    ref = torch.tensor([0.0, 10.0], dtype=torch.float64)

    out = normalize_to_reference_l2(x, ref)

    assert abs(float(out.norm()) - 10.0) < 1e-12
    assert torch.allclose(out / out.norm(), x / x.norm())


def test_functional_null_tangent_rejects_zero_gradient():
    rows = torch.tensor([[1.0, 0.0]])
    subspace = build_retention_subspace(rows)
    try:
        functional_null_structural_tangent(
            torch.zeros(2),
            subspace,
            rank=1,
        )
    except ValueError as exc:
        assert "nonzero" in str(exc)
    else:
        raise AssertionError("expected zero structural gradient to fail")
