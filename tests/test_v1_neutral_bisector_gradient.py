import torch

from nmd.v1_norm_balanced_gradient import norm_balanced_gradient_update
from nmd.v1_neutral_bisector_gradient import (
    neutral_bisector_norm_balanced_gradient_update,
)


def _unit(x):
    return x / x.norm()


def test_s23_zero_relation_returns_primary_exactly():
    p = [torch.tensor([1.0, -2.0, 3.0])]
    r = [torch.zeros(3)]
    out, d = neutral_bisector_norm_balanced_gradient_update(p, r)
    assert torch.equal(out[0], p[0])
    assert d.special_case == "relation_zero"


def test_s23_zero_primary_returns_relation_exactly():
    p = [torch.zeros(3)]
    r = [torch.tensor([1.0, -2.0, 3.0])]
    out, d = neutral_bisector_norm_balanced_gradient_update(p, r)
    assert torch.equal(out[0], r[0])
    assert d.special_case == "primary_zero"


def test_s23_both_zero_stays_zero():
    p = [torch.zeros(4)]
    r = [torch.zeros(4)]
    out, d = neutral_bisector_norm_balanced_gradient_update(p, r)
    assert torch.equal(out[0], torch.zeros(4))
    assert d.special_case == "both_zero"


def test_s23_conflict_is_exact_neutral_bisector_without_projection():
    p = [torch.tensor([1.0, -2.0])]
    r = [torch.tensor([-2.0, 1.0])]
    out, d = neutral_bisector_norm_balanced_gradient_update(p, r)
    expected = _unit(_unit(p[0]) + _unit(r[0]))
    assert d.conflict is True
    assert d.normalized_pre_dot < 0.0
    assert d.normalized_post_dot == d.normalized_pre_dot
    assert d.projection_coefficient == 0.0
    assert d.special_case == "neutral_bisector"
    assert torch.allclose(_unit(out[0]), expected, atol=1e-6, rtol=0.0)


def test_s23_exchange_symmetry():
    p = [torch.tensor([1.0, -2.0, 0.5])]
    r = [torch.tensor([-0.8, 1.2, 2.0])]
    a, _ = neutral_bisector_norm_balanced_gradient_update(p, r)
    b, _ = neutral_bisector_norm_balanced_gradient_update(r, p)
    assert torch.allclose(a[0], b[0], atol=1e-7, rtol=0.0)


def test_s23_no_conflict_matches_s17_exact_rule():
    p = [torch.tensor([2.0, 1.0, 0.0])]
    r = [torch.tensor([1.0, 3.0, 0.0])]
    s17, d17 = norm_balanced_gradient_update(p, r)
    s23, d23 = neutral_bisector_norm_balanced_gradient_update(p, r)
    assert d17.conflict is False
    assert d23.conflict is False
    assert torch.allclose(s23[0], s17[0], atol=1e-7, rtol=0.0)


def test_s23_joint_scale_equivariance():
    p = [torch.tensor([1.0, 2.0, -1.0])]
    r = [torch.tensor([-0.5, 1.0, 2.0])]
    a, _ = neutral_bisector_norm_balanced_gradient_update(p, r)
    scale = 7.0
    b, _ = neutral_bisector_norm_balanced_gradient_update(
        [scale * p[0]], [scale * r[0]]
    )
    assert torch.allclose(b[0], scale * a[0], atol=1e-5, rtol=1e-6)


def test_s23_near_opposite_is_finite():
    p = [torch.tensor([1.0, 0.0])]
    r = [torch.tensor([-1.0, 1e-7])]
    out, d = neutral_bisector_norm_balanced_gradient_update(p, r)
    assert d.conflict is True
    assert torch.isfinite(out[0]).all()


def test_s23_rejects_invalid_inputs():
    for p, r in (([], []), ([torch.zeros(2)], [torch.zeros(3)])):
        try:
            neutral_bisector_norm_balanced_gradient_update(p, r)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid S23 gradients must fail")
