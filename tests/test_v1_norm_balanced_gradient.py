import torch

from nmd.v1_norm_balanced_gradient import norm_balanced_gradient_update


def test_s17_zero_relation_returns_primary_exactly():
    p=[torch.tensor([1.0,-2.0,3.0])]
    r=[torch.zeros(3)]
    out,d=norm_balanced_gradient_update(p,r)
    assert torch.equal(out[0],p[0])
    assert d.special_case=="relation_zero"


def test_s17_zero_primary_returns_relation_exactly():
    p=[torch.zeros(3)]
    r=[torch.tensor([1.0,-2.0,3.0])]
    out,d=norm_balanced_gradient_update(p,r)
    assert torch.equal(out[0],r[0])
    assert d.special_case=="primary_zero"


def test_s17_balances_direction_not_raw_magnitude():
    p=[torch.tensor([100.0,0.0])]
    r=[torch.tensor([0.0,1.0])]
    out,d=norm_balanced_gradient_update(p,r)
    v=out[0]/out[0].norm()
    expected=torch.tensor([1.0,1.0])/torch.sqrt(torch.tensor(2.0))
    assert torch.allclose(v,expected,atol=1e-6,rtol=0.0)
    assert abs(d.combined_norm-50.5)<1e-4


def test_s17_conflict_projection_protects_relation_direction():
    p=[torch.tensor([1.0,-2.0])]
    r=[torch.tensor([-2.0,1.0])]
    out,d=norm_balanced_gradient_update(p,r)
    assert d.conflict is True
    assert d.normalized_post_dot >= -1e-6
    assert torch.isfinite(out[0]).all()


def test_s17_joint_scale_equivariance():
    p=[torch.tensor([1.0,2.0,-1.0])]
    r=[torch.tensor([-0.5,1.0,2.0])]
    a,_=norm_balanced_gradient_update(p,r)
    scale=7.0
    b,_=norm_balanced_gradient_update([scale*p[0]],[scale*r[0]])
    assert torch.allclose(b[0],scale*a[0],atol=1e-5,rtol=1e-6)


def test_s17_same_direction_is_collinear():
    p=[torch.tensor([10.0,-20.0])]
    r=[torch.tensor([1.0,-2.0])]
    out,d=norm_balanced_gradient_update(p,r)
    assert d.conflict is False
    assert torch.allclose(out[0]/out[0].norm(),p[0]/p[0].norm(),atol=1e-6,rtol=0.0)


def test_s17_rejects_invalid_inputs():
    try:
        norm_balanced_gradient_update([],[])
    except ValueError:
        pass
    else:
        raise AssertionError
    try:
        norm_balanced_gradient_update([torch.zeros(2)],[torch.zeros(3)])
    except ValueError:
        pass
    else:
        raise AssertionError
