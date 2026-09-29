import torch

from nmd.v1_shared_gradient_surgery import (
    apply_gradient_update,
    relation_priority_gradient_surgery,
)


def test_s16_conflicting_primary_is_projected_off_relation_direction():
    primary = [torch.tensor([1.0, -2.0]), torch.tensor([0.5])]
    relation = [torch.tensor([-2.0, 1.0]), torch.tensor([-1.0])]
    projected, combined, diag = relation_priority_gradient_surgery(primary, relation)

    assert diag.conflict is True
    assert diag.pre_dot < 0.0
    assert diag.post_primary_relation_dot >= -1e-6
    assert diag.projection_coefficient < 0.0

    dot = sum(float((a * b).sum()) for a, b in zip(projected, relation))
    assert dot >= -1e-6
    for c, p, r in zip(combined, projected, relation):
        assert torch.allclose(c, p + r, atol=1e-7, rtol=0.0)


def test_s16_nonconflicting_primary_is_bit_identical():
    primary = [torch.tensor([1.0, 2.0]), torch.tensor([-0.5])]
    relation = [torch.tensor([2.0, 1.0]), torch.tensor([-1.0])]
    projected, combined, diag = relation_priority_gradient_surgery(primary, relation)

    assert diag.conflict is False
    assert diag.projection_coefficient == 0.0
    for before, after in zip(primary, projected):
        assert torch.equal(before, after)
    for c, p, r in zip(combined, primary, relation):
        assert torch.equal(c, p + r)


def test_s16_zero_relation_gradient_is_stable():
    primary = [torch.tensor([1.0, -3.0]), torch.tensor([2.0])]
    relation = [torch.zeros(2), torch.zeros(1)]
    projected, combined, diag = relation_priority_gradient_surgery(primary, relation)

    assert diag.conflict is False
    assert diag.relation_norm == 0.0
    assert diag.projection_coefficient == 0.0
    for before, after in zip(primary, projected):
        assert torch.equal(before, after)
    for before, after in zip(primary, combined):
        assert torch.equal(before, after)


def test_s16_projection_is_scale_equivariant_for_joint_gradient_scale():
    primary = [torch.tensor([1.0, -2.0, 3.0])]
    relation = [torch.tensor([-4.0, 1.0, -1.0])]
    p1, c1, d1 = relation_priority_gradient_surgery(primary, relation)
    scale = 7.0
    p2, c2, d2 = relation_priority_gradient_surgery(
        [scale * primary[0]],
        [scale * relation[0]],
    )

    assert d1.conflict is d2.conflict
    assert torch.allclose(p2[0], scale * p1[0], atol=1e-5, rtol=1e-6)
    assert torch.allclose(c2[0], scale * c1[0], atol=1e-5, rtol=1e-6)


def test_s16_surgery_adds_no_parameters_or_state():
    # The operator is a pure function; no nn.Module or trainable state exists.
    primary = [torch.randn(4)]
    relation = [torch.randn(4)]
    projected, combined, diag = relation_priority_gradient_surgery(primary, relation)
    assert len(projected) == 1
    assert len(combined) == 1
    assert isinstance(diag.conflict, bool)


def test_s16_apply_gradient_update_writes_exact_gradients():
    p1 = torch.nn.Parameter(torch.zeros(3))
    p2 = torch.nn.Parameter(torch.zeros(2))
    g1 = torch.tensor([1.0, 2.0, 3.0])
    g2 = torch.tensor([-1.0, 4.0])
    apply_gradient_update([p1, p2], [g1, g2])
    assert torch.equal(p1.grad, g1)
    assert torch.equal(p2.grad, g2)


def test_s16_rejects_bad_inputs():
    try:
        relation_priority_gradient_surgery([], [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty S16 gradients must fail")

    try:
        relation_priority_gradient_surgery(
            [torch.zeros(2)],
            [torch.zeros(3)],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must fail")

    try:
        relation_priority_gradient_surgery(
            [torch.zeros(2)],
            [torch.zeros(2)],
            epsilon=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive epsilon must fail")
