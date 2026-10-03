import torch

from nmd.v1_evidence_fusion import standardize_full_k_evidence
from nmd.v1_robust_three_expert_consensus import RobustThreeExpertMedianFusion


def _inputs(*, batch=5, k=7):
    g = torch.Generator().manual_seed(46000 + k)
    return (
        torch.randn(batch, k, generator=g),
        torch.randn(batch, k, generator=g),
        torch.randn(batch, k, generator=g),
    )


def test_s46_zero_parameter_surface():
    op = RobustThreeExpertMedianFusion()
    assert op.parameter_count == 0
    assert list(op.parameters()) == []


def test_s46_arbitrary_k_three_seven_255():
    op = RobustThreeExpertMedianFusion()
    for k in (3, 7, 255):
        p, n, c = _inputs(k=k)
        fused, diag = op(p, n, c)
        assert tuple(fused.shape) == (5, k)
        assert bool(torch.isfinite(fused).all())
        values = diag.to_dict()
        assert all(0.0 <= x <= 1.0 for x in values.values())


def test_s46_logical_option_permutation_equivariance():
    op = RobustThreeExpertMedianFusion()
    p, n, c = _inputs(batch=3, k=7)
    base, _ = op(p, n, c)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    moved, _ = op(p[:, perm], n[:, perm], c[:, perm])
    assert torch.equal(moved, base[:, perm])


def test_s46_independent_shift_positive_scale_invariance():
    op = RobustThreeExpertMedianFusion()
    p, n, c = _inputs(batch=4, k=7)
    base, _ = op(p, n, c)
    moved, _ = op(
        3.25 * p + 17.0,
        0.75 * n - 9.0,
        8.5 * c + 101.0,
    )
    assert torch.allclose(moved, base, rtol=0.0, atol=3e-6)


def test_s46_all_three_identical_equals_standardized_identity():
    op = RobustThreeExpertMedianFusion()
    p, _n, _c = _inputs(batch=4, k=7)
    fused, _ = op(p, p, p)
    standardized, _ = standardize_full_k_evidence(p, epsilon=1e-6)
    assert torch.equal(fused, standardized)


def test_s46_coordinatewise_median_is_inside_expert_envelope():
    op = RobustThreeExpertMedianFusion()
    p, n, c = _inputs(batch=4, k=7)
    fused, _ = op(p, n, c)
    experts = []
    for x in (p, n, c):
        sx, _ = standardize_full_k_evidence(x, epsilon=1e-6)
        experts.append(sx)
    stack = torch.stack(experts, dim=1)
    assert bool((fused >= stack.amin(dim=1)).all())
    assert bool((fused <= stack.amax(dim=1)).all())


def test_s46_one_extreme_outlier_cannot_move_two_agreeing_experts():
    op = RobustThreeExpertMedianFusion()
    g = torch.Generator().manual_seed(46111)
    agreeing = torch.randn(6, 7, generator=g)
    # Positive scale/shift keeps the agreeing expert standardized-identical.
    agreeing_2 = 5.0 * agreeing + 73.0
    outlier = torch.randn(6, 7, generator=g) * 1e9
    fused, _ = op(agreeing, agreeing_2, outlier)
    expected, _ = standardize_full_k_evidence(agreeing, epsilon=1e-6)
    assert torch.allclose(fused, expected, rtol=0.0, atol=3e-6)


def test_s46_flat_expert_is_finite_and_probability_mass_is_valid():
    op = RobustThreeExpertMedianFusion()
    p, n, _c = _inputs(batch=6, k=7)
    flat = torch.full_like(p, 3.0)
    fused, _ = op(p, n, flat)
    assert bool(torch.isfinite(fused).all())
    probs = torch.softmax(fused, dim=-1)
    assert float((probs.sum(-1) - 1.0).abs().max()) <= 1e-6


def test_s46_is_deterministic():
    op = RobustThreeExpertMedianFusion()
    p, n, c = _inputs(batch=3, k=7)
    a, _ = op(p, n, c)
    b, _ = op(p, n, c)
    assert torch.equal(a, b)
