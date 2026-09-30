from __future__ import annotations

import torch

from nmd.v1_standardized_triadic_consistency import (
    standardized_triadic_consistency_loss,
)


def test_identical_evidence_has_exact_zero_loss() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0, -1.0], [0.3, -0.5, 0.8, 1.7]])
    loss, a, b, _ = standardized_triadic_consistency_loss(x, x)
    assert float(loss) == 0.0
    assert torch.equal(a, b)


def test_view_swap_is_exactly_symmetric() -> None:
    a = torch.tensor([[1.0, 2.0, 4.0, -1.0]])
    b = torch.tensor([[-2.0, 1.5, 0.0, 3.0]])
    ab, *_ = standardized_triadic_consistency_loss(a, b)
    ba, *_ = standardized_triadic_consistency_loss(b, a)
    assert torch.equal(ab, ba)


def test_option_permutation_is_invariant() -> None:
    a = torch.tensor([[1.0, 2.0, 4.0, -1.0]])
    b = torch.tensor([[-2.0, 1.5, 0.0, 3.0]])
    permutation = torch.tensor([3, 1, 0, 2])
    base, *_ = standardized_triadic_consistency_loss(a, b)
    permuted, *_ = standardized_triadic_consistency_loss(
        a[:, permutation], b[:, permutation]
    )
    assert torch.allclose(base, permuted, atol=0.0, rtol=0.0)


def test_positive_affine_transform_preserves_standardized_evidence() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0, -1.0], [0.3, -0.5, 0.8, 1.7]])
    _, z, _, _ = standardized_triadic_consistency_loss(x, x)
    transformed = 7.0 * x + 13.0
    _, z2, _, _ = standardized_triadic_consistency_loss(transformed, transformed)
    assert torch.allclose(z, z2, atol=2e-6, rtol=2e-6)


def test_mismatched_relative_evidence_gives_gradients_to_both_views() -> None:
    a = torch.tensor([[1.0, 2.0, 4.0, -1.0]], requires_grad=True)
    b = torch.tensor([[1.0, 4.0, 2.0, -1.0]], requires_grad=True)
    loss, *_ = standardized_triadic_consistency_loss(a, b)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0.0
    assert b.grad is not None and float(b.grad.abs().sum()) > 0.0


def test_common_positive_scaling_does_not_suppress_signal() -> None:
    a = torch.tensor([[1.0, 2.0, 4.0, -1.0]])
    b = torch.tensor([[1.0, 4.0, 2.0, -1.0]])
    base, *_ = standardized_triadic_consistency_loss(a, b)
    tiny, *_ = standardized_triadic_consistency_loss(a * 1e-3, b * 1e-3)
    assert torch.allclose(base, tiny, atol=2e-6, rtol=2e-6)


def test_flat_evidence_is_neutral_and_finite() -> None:
    a = torch.full((3, 4), 7.0, requires_grad=True)
    b = torch.full((3, 4), -2.0, requires_grad=True)
    loss, za, zb, diag = standardized_triadic_consistency_loss(a, b)
    assert float(loss) == 0.0
    assert torch.equal(za, torch.zeros_like(za))
    assert torch.equal(zb, torch.zeros_like(zb))
    assert diag.canonical_flat_rate == 1.0
    assert diag.paraphrase_flat_rate == 1.0
    assert bool(torch.isfinite(loss))


def test_nonfinite_input_is_rejected() -> None:
    a = torch.tensor([[1.0, float("nan"), 2.0, 3.0]])
    b = torch.zeros_like(a)
    try:
        standardized_triadic_consistency_loss(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("expected non-finite S20 input to be rejected")
