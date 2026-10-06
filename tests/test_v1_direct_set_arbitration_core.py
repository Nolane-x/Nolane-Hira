import torch

from nmd.v1_direct_set_arbitration_core import (
    S74_FUSED_FEATURE_DIM,
    S74_RELATIONAL_FEATURE_DIM,
    S74_INPUT_DIM,
    S74_PARAMETER_COUNT,
    DirectSetArbitrationCore,
    direct_set_arbitration_features,
    direct_set_arbitration_loss,
)


def _pair(batch,k,seed=1):
    g=torch.Generator().manual_seed(seed)
    raw=torch.randn(batch,k,k,generator=g)
    pair=0.5*(raw-raw.transpose(-1,-2))
    eye=torch.eye(k,dtype=torch.bool).unsqueeze(0)
    return pair.masked_fill(eye,0.0)


def test_s74_parameter_count_and_bit_identical_init():
    a=DirectSetArbitrationCore()
    b=DirectSetArbitrationCore()
    assert a.parameter_count==S74_PARAMETER_COUNT==257
    assert b.parameter_count==257
    sa=a.state_dict_exact()
    sb=b.state_dict_exact()
    assert sa.keys()==sb.keys()
    assert all(torch.equal(sa[k],sb[k]) for k in sa)


def test_s74_reference_zeros_relational_channels_and_treatment_is_live():
    fused=torch.tensor([[2.0,0.3,-1.2,0.8]],dtype=torch.float32)
    pair=_pair(1,4,3)
    ref=direct_set_arbitration_features(fused,pair,use_relational=False)
    trt=direct_set_arbitration_features(fused,pair,use_relational=True)

    assert ref.shape==(1,4,S74_INPUT_DIM)
    assert torch.equal(
        ref[...,S74_FUSED_FEATURE_DIM:],
        torch.zeros_like(ref[...,S74_FUSED_FEATURE_DIM:]),
    )
    assert torch.allclose(
        ref[...,:S74_FUSED_FEATURE_DIM],
        trt[...,:S74_FUSED_FEATURE_DIM],
    )
    assert float(trt[...,S74_FUSED_FEATURE_DIM:].abs().max())>1e-5


def test_s74_permutation_equivariance():
    g=torch.Generator().manual_seed(74)
    fused=torch.randn(5,7,generator=g)
    pair=_pair(5,7,75)
    perm=torch.randperm(7,generator=g)
    inv=torch.argsort(perm)

    core=DirectSetArbitrationCore()
    for use_relational in (False,True):
        base=core(fused,pair,use_relational=use_relational)
        pp=pair[:,perm][:,:,perm]
        got=core(fused[:,perm],pp,use_relational=use_relational)[:,inv]
        assert float((base-got).abs().max())<=2e-6


def test_s74_finite_dynamic_k_and_probability_mass():
    for k in (3,7,255):
        fused=torch.randn(2,k)
        pair=_pair(2,k,100+k)
        core=DirectSetArbitrationCore()
        for use_relational in (False,True):
            logits=core(fused,pair,use_relational=use_relational)
            assert logits.shape==(2,k)
            assert torch.isfinite(logits).all()
            probs=torch.softmax(logits,dim=-1)
            assert float((probs.sum(dim=-1)-1.0).abs().max())<=1e-6


def test_s74_is_direct_core_not_fused_residual():
    fused=torch.tensor([[3.0,1.0,-2.0]],dtype=torch.float32)
    pair=_pair(1,3,9)
    core=DirectSetArbitrationCore()
    with torch.no_grad():
        for p in core.parameters():
            p.zero_()
    logits=core(fused,pair,use_relational=True)
    assert torch.equal(logits,torch.zeros_like(logits))
    assert not torch.equal(logits,fused)


def test_s74_gradients_live_but_upstream_is_detached():
    g=torch.Generator().manual_seed(740)
    fused_c=torch.randn(8,4,generator=g,requires_grad=True)
    fused_p=torch.randn(8,4,generator=g,requires_grad=True)
    pair_c=_pair(8,4,741).requires_grad_(True)
    pair_p=_pair(8,4,742).requires_grad_(True)
    gold=torch.arange(8,dtype=torch.long)%4

    core=DirectSetArbitrationCore()
    loss,_=direct_set_arbitration_loss(
        core,fused_c,fused_p,pair_c,pair_p,gold,use_relational=True
    )
    loss.backward()

    grads=[p.grad for p in core.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(float(g.abs().sum())>0 for g in grads)
    assert fused_c.grad is None
    assert fused_p.grad is None
    assert pair_c.grad is None
    assert pair_p.grad is None


def test_s74_exact_checkpoint_replay():
    fused=torch.randn(4,5)
    pair=_pair(4,5,800)
    a=DirectSetArbitrationCore()
    expected=a(fused,pair,use_relational=True)

    b=DirectSetArbitrationCore()
    b.load_state_dict_exact(a.state_dict_exact(),freeze=True)
    got=b(fused,pair,use_relational=True)
    assert torch.equal(expected,got)
    assert b.trainable_parameter_count==0
