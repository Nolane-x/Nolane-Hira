import torch

from nmd.v1_evidence_fusion import SymmetricFullKEvidenceFusion
from nmd.v1_reliability_weighted_fusion import (
    ReliabilityWeightedFullKEvidenceFusion,
    standardized_top2_gap,
)


def test_s24_has_zero_parameters():
    assert ReliabilityWeightedFullKEvidenceFusion().parameter_count == 0


def test_s24_expert_swap_symmetry():
    a = torch.tensor([[3.0, 1.0, 0.0, -1.0], [0.1, 0.2, 0.3, 0.4]])
    b = torch.tensor([[0.0, 2.0, 1.0, -1.0], [0.4, 0.3, 0.2, 0.1]])
    f = ReliabilityWeightedFullKEvidenceFusion()
    x, dx = f(a, b)
    y, dy = f(b, a)
    assert torch.allclose(x, y, atol=1e-7, rtol=0.0)
    assert abs(float(dx.primary_weight + dx.relation_weight) - 1.0) <= 1e-7
    assert abs(float(dy.primary_weight + dy.relation_weight) - 1.0) <= 1e-7


def test_s24_option_permutation_equivariance():
    a = torch.tensor([[2.0, -1.0, 0.5, 1.0]])
    b = torch.tensor([[0.1, 3.0, 1.0, -0.5]])
    perm = torch.tensor([2, 0, 3, 1])
    f = ReliabilityWeightedFullKEvidenceFusion()
    base, _ = f(a, b)
    moved, _ = f(a[:, perm], b[:, perm])
    assert torch.allclose(moved, base[:, perm], atol=1e-7, rtol=0.0)


def test_s24_equal_reliability_reduces_to_s14():
    a = torch.tensor([[3.0, 1.0, 0.0, -1.0]])
    b = torch.tensor([[-1.0, 0.0, 1.0, 3.0]])
    s14, _ = SymmetricFullKEvidenceFusion()(a, b)
    s24, d = ReliabilityWeightedFullKEvidenceFusion()(a, b)
    assert torch.allclose(s24, s14, atol=1e-7, rtol=0.0)
    assert torch.allclose(d.primary_weight, torch.tensor(0.5), atol=1e-7)
    assert torch.allclose(d.relation_weight, torch.tensor(0.5), atol=1e-7)


def test_s24_flat_flat_is_neutral_equal_fusion():
    a = torch.full((2, 4), 7.0)
    b = torch.full((2, 4), -2.0)
    out, d = ReliabilityWeightedFullKEvidenceFusion()(a, b)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.allclose(d.primary_weight, torch.tensor(0.5), atol=1e-7)
    assert torch.allclose(d.relation_weight, torch.tensor(0.5), atol=1e-7)


def test_s24_flat_nonflat_is_dominated_by_nonflat_expert():
    a = torch.full((1, 4), 5.0)
    b = torch.tensor([[3.0, 1.0, 0.0, -1.0]])
    out, d = ReliabilityWeightedFullKEvidenceFusion()(a, b)
    assert torch.isfinite(out).all()
    assert float(d.relation_weight) > 0.999
    assert float(d.primary_weight) < 0.001


def test_s24_positive_affine_invariance():
    a = torch.tensor([[2.0, 0.0, -1.0, 1.0]])
    b = torch.tensor([[0.0, 3.0, 1.0, -2.0]])
    f = ReliabilityWeightedFullKEvidenceFusion()
    x, dx = f(a, b)
    y, dy = f(7.0 * a + 13.0, 4.0 * b - 9.0)
    assert torch.allclose(x, y, atol=2e-6, rtol=0.0)
    assert torch.allclose(dx.primary_weight, dy.primary_weight, atol=2e-6)
    assert torch.allclose(dx.relation_weight, dy.relation_weight, atol=2e-6)


def test_s24_relation_is_detached_from_primary_objective():
    a = torch.tensor([[2.0, 0.0, -1.0, 1.0]], requires_grad=True)
    b = torch.tensor([[0.0, 3.0, 1.0, -2.0]], requires_grad=True)
    out, _ = ReliabilityWeightedFullKEvidenceFusion()(a, b)
    out.square().sum().backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0.0
    assert b.grad is None


def test_s24_gap_is_nonnegative():
    z = torch.tensor([[0.0, 1.0, 3.0, 2.0], [-1.0, -1.0, -1.0, -1.0]])
    gap = standardized_top2_gap(z)
    assert torch.equal(gap, torch.tensor([1.0, 0.0]))
