import torch

from nmd.v1_norm_balanced_gradient import norm_balanced_gradient_update
from nmd.v1_primary_priority_gradient import (
    primary_priority_norm_balanced_gradient_update,
)


def test_s22_zero_relation_returns_primary_exactly():
    p = [torch.tensor([1.0, -2.0, 3.0])]
    r = [torch.zeros(3)]
    out, d = primary_priority_norm_balanced_gradient_update(p, r)
    assert torch.equal(out[0], p[0])
    assert d.special_case == "relation_zero"


def test_s22_zero_primary_returns_relation_exactly():
    p = [torch.zeros(3)]
    r = [torch.tensor([1.0, -2.0, 3.0])]
    out, d = primary_priority_norm_balanced_gradient_update(p, r)
    assert torch.equal(out[0], r[0])
    assert d.special_case == "primary_zero"


def test_s22_both_zero_stays_zero():
    p = [torch.zeros(4)]
    r = [torch.zeros(4)]
    out, d = primary_priority_norm_balanced_gradient_update(p, r)
    assert torch.equal(out[0], torch.zeros(4))
    assert d.special_case == "both_zero"


def test_s22_conflict_projects_relation_not_primary():
    p = [torch.tensor([1.0, -2.0])]
    r = [torch.tensor([-2.0, 1.0])]
    out, d = primary_priority_norm_balanced_gradient_update(p, r)

    up = p[0] / p[0].norm()
    ur = r[0] / r[0].norm()
    pre = torch.dot(up, ur)
    expected_relation = ur - pre * up
    expected_direction = up + expected_relation
    expected_direction = expected_direction / expected_direction.norm()

    assert d.conflict is True
    assert d.normalized_pre_dot < 0.0
    assert d.normalized_post_dot >= -1e-6
    assert torch.allclose(
        out[0] / out[0].norm(),
        expected_direction,
        atol=1e-6,
        rtol=0.0,
    )

    # The protected primary unit direction itself is not modified.
    recovered_relation = expected_direction * 0.0 + expected_relation
    assert torch.allclose(up, p[0] / p[0].norm(), atol=0.0, rtol=0.0)
    assert not torch.allclose(recovered_relation, ur)


def test_s22_no_conflict_matches_s17_exact_rule():
    p = [torch.tensor([2.0, 1.0, 0.0])]
    r = [torch.tensor([1.0, 3.0, 0.0])]
    s17, d17 = norm_balanced_gradient_update(p, r)
    s22, d22 = primary_priority_norm_balanced_gradient_update(p, r)
    assert d17.conflict is False
    assert d22.conflict is False
    assert torch.allclose(s22[0], s17[0], atol=1e-7, rtol=0.0)


def test_s22_joint_scale_equivariance():
    p = [torch.tensor([1.0, 2.0, -1.0])]
    r = [torch.tensor([-0.5, 1.0, 2.0])]
    a, _ = primary_priority_norm_balanced_gradient_update(p, r)
    scale = 7.0
    b, _ = primary_priority_norm_balanced_gradient_update(
        [scale * p[0]],
        [scale * r[0]],
    )
    assert torch.allclose(b[0], scale * a[0], atol=1e-5, rtol=1e-6)


def test_s22_rejects_invalid_inputs():
    for p, r in (
        ([], []),
        ([torch.zeros(2)], [torch.zeros(3)]),
    ):
        try:
            primary_priority_norm_balanced_gradient_update(p, r)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid S22 gradients must fail")
