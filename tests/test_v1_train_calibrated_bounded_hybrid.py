import torch

from nmd.v1_train_calibrated_bounded_hybrid import (
    S60_ALPHA_INITIAL,
    S60_ALPHA_MAX,
    TrainCalibratedBoundedHybridComposer,
    composer_gold_loss,
)


def test_s60_parameter_surface_and_initial_alpha():
    c=TrainCalibratedBoundedHybridComposer()
    assert c.parameter_count==1
    assert c.trainable_parameter_count==1
    assert set(dict(c.named_parameters()))=={"a"}
    assert abs(float(c.alpha())-S60_ALPHA_INITIAL)<=1e-7
    assert 0.0<float(c.alpha())<S60_ALPHA_MAX


def test_s60_alpha_zero_override_is_exact_fused_identity():
    c=TrainCalibratedBoundedHybridComposer()
    fused=torch.randn(3,7)
    pair=torch.randn(3,7)
    out=c.compose(fused,pair,alpha_override=0.0)
    assert torch.equal(out,fused)


def test_s60_adversarial_pairwise_magnitude_cannot_break_bound():
    c=TrainCalibratedBoundedHybridComposer()
    fused=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    pair=torch.tensor([[1e20,-1e20,5e19,-5e19]])
    out,d=c.compose(fused,pair,return_diagnostics=True)
    residual=(out-fused).abs()
    assert bool((residual<=d["residual_bound"]+1e-7).all())
    assert float(d["pairwise_bounded_direction_max_abs"])<=1.0


def test_s60_flat_pairwise_produces_zero_residual():
    c=TrainCalibratedBoundedHybridComposer()
    fused=torch.randn(2,4)
    pair=torch.full((2,4),17.0)
    out=c.compose(fused,pair)
    assert torch.equal(out,fused)


def test_s60_option_permutation_equivariance():
    g=torch.Generator().manual_seed(60001)
    c=TrainCalibratedBoundedHybridComposer()
    fused=torch.randn(2,7,generator=g)
    pair=torch.randn(2,7,generator=g)
    base=c.compose(fused,pair)
    perm=torch.tensor([3,0,6,2,1,5,4])
    got=c.compose(fused[:,perm],pair[:,perm])
    assert torch.allclose(got,base[:,perm],atol=1e-6,rtol=0)


def test_s60_offset_and_positive_scale_contract():
    c=TrainCalibratedBoundedHybridComposer()
    fused=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    pair=torch.tensor([[2.0,0.5,-0.2,-1.7]])
    base=c.compose(fused,pair)
    transformed=c.compose(3.0*fused+7.0,5.0*pair-11.0)
    assert torch.allclose(transformed,3.0*base+7.0,atol=2e-6,rtol=0)


def test_s60_composer_gradient_only_reaches_scalar():
    c=TrainCalibratedBoundedHybridComposer()
    fc=torch.tensor([[2.0,0.2,-0.4,-1.0]],requires_grad=True)
    fp=torch.tensor([[1.5,0.1,-0.2,-0.8]],requires_grad=True)
    pc=torch.tensor([[0.3,0.1,-0.1,-0.3]],requires_grad=True)
    pp=torch.tensor([[0.2,0.05,-0.05,-0.2]],requires_grad=True)
    gold=torch.tensor([0],dtype=torch.long)
    loss,_=composer_gold_loss(c,fc,fp,pc,pp,gold)
    grads=torch.autograd.grad(loss,(c.a,fc,fp,pc,pp),allow_unused=True)
    assert grads[0] is not None and float(grads[0].abs().sum())>0.0
    assert all(g is None or float(g.abs().sum())==0.0 for g in grads[1:])


def test_s60_k3_k7_k255_finite_full_mass():
    g=torch.Generator().manual_seed(60002)
    c=TrainCalibratedBoundedHybridComposer()
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g)
        pair=torch.randn(2,k,generator=g)
        out=c.compose(fused,pair)
        assert out.shape==(2,k)
        assert bool(torch.isfinite(out).all())
        probs=torch.softmax(out,dim=-1)
        assert float((probs.sum(-1)-1.0).abs().max())<=1e-6


def test_s60_flat_fused_is_finite():
    c=TrainCalibratedBoundedHybridComposer()
    fused=torch.zeros(2,7)
    pair=torch.randn(2,7)
    out=c.compose(fused,pair)
    assert bool(torch.isfinite(out).all())


def test_s60_checkpoint_roundtrip_exact():
    c=TrainCalibratedBoundedHybridComposer()
    clone=TrainCalibratedBoundedHybridComposer()
    clone.load_state_dict_exact(c.state_dict_exact(),freeze=True)
    fused=torch.randn(2,4)
    pair=torch.randn(2,4)
    assert torch.equal(c.compose(fused,pair),clone.compose(fused,pair))
    assert clone.trainable_parameter_count==0
