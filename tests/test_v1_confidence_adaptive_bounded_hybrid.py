import torch

from nmd.v1_confidence_adaptive_bounded_hybrid import (
    S61_ALPHA_INITIAL,
    S61_ALPHA_MAX,
    S61_GATE_PARAMETER_COUNT,
    ConfidenceAdaptiveBoundedHybridGate,
    adaptive_gate_features,
    adaptive_gate_gold_loss,
)


def test_s61_parameter_surface_initialization_and_alpha():
    gate=ConfidenceAdaptiveBoundedHybridGate()
    assert gate.parameter_count==S61_GATE_PARAMETER_COUNT==5
    assert gate.trainable_parameter_count==5
    assert set(dict(gate.named_parameters()))=={"w","b"}
    assert gate.w.shape==(4,)
    assert gate.b.ndim==0
    assert torch.equal(gate.w,torch.zeros_like(gate.w))
    fused=torch.randn(6,7)
    pair=torch.randn(6,7)
    alpha=gate.alpha(fused,pair)
    assert torch.allclose(alpha,torch.full_like(alpha,S61_ALPHA_INITIAL),atol=1e-7,rtol=0)


def test_s61_features_are_detached_finite_and_agreement_is_signed():
    fused=torch.tensor([[3.,1.,0.,-1.],[3.,1.,0.,-1.]],requires_grad=True)
    pair=torch.tensor([[2.,1.,0.,-1.],[0.,3.,1.,-1.]],requires_grad=True)
    x=adaptive_gate_features(fused,pair)
    assert x.shape==(2,4)
    assert x.requires_grad is False
    assert bool(torch.isfinite(x).all())
    assert float(x[0,2])==1.0
    assert float(x[1,2])==-1.0


def test_s61_feature_offset_scale_and_option_permutation_invariance():
    g=torch.Generator().manual_seed(61001)
    fused=torch.randn(3,7,generator=g)
    pair=torch.randn(3,7,generator=g)
    base=adaptive_gate_features(fused,pair)
    transformed=adaptive_gate_features(3.0*fused+7.0,5.0*pair-11.0)
    assert torch.allclose(base,transformed,atol=2e-6,rtol=0)
    perm=torch.tensor([3,0,6,2,1,5,4])
    permuted=adaptive_gate_features(fused[:,perm],pair[:,perm])
    assert torch.allclose(base,permuted,atol=1e-6,rtol=0)


def test_s61_flat_logits_are_finite():
    gate=ConfidenceAdaptiveBoundedHybridGate()
    fused=torch.zeros(2,7)
    pair=torch.zeros(2,7)
    x=adaptive_gate_features(fused,pair)
    out=gate.compose(fused,pair)
    assert bool(torch.isfinite(x).all())
    assert bool(torch.isfinite(out).all())
    assert torch.equal(out,fused)


def test_s61_manual_weights_make_alpha_adaptive_and_bounded():
    gate=ConfidenceAdaptiveBoundedHybridGate()
    with torch.no_grad():
        gate.w.copy_(torch.tensor([-1.5,1.2,0.8,0.6]))
    fused=torch.tensor([[4.,1.,0.,-1.],[1.1,1.0,0.,-1.]])
    pair=torch.tensor([[4.,1.,0.,-1.],[-1.,4.,0.,1.]])
    alpha=gate.alpha(fused,pair).squeeze(-1)
    assert float(alpha[0])!=float(alpha[1])
    assert bool((alpha>0).all())
    assert bool((alpha<S61_ALPHA_MAX).all())


def test_s61_zero_override_exact_identity_and_adversarial_bound():
    gate=ConfidenceAdaptiveBoundedHybridGate()
    fused=torch.tensor([[4.,1.,-1.,-3.]])
    pair=torch.tensor([[1e20,-1e20,5e19,-5e19]])
    identity=gate.compose(fused,pair,alpha_override=0.0)
    assert torch.equal(identity,fused)
    out,d=gate.compose(fused,pair,return_diagnostics=True)
    residual=(out-fused).abs()
    assert bool((residual<=d["residual_bound"]+1e-7).all())
    assert float(d["pairwise_bounded_direction_max_abs"])<=1.0


def test_s61_gate_gradients_reach_only_w_b():
    gate=ConfidenceAdaptiveBoundedHybridGate()
    fc=torch.tensor([[2.,0.2,-0.4,-1.0],[0.4,0.3,0.2,-1.0]],requires_grad=True)
    fp=torch.tensor([[1.5,0.1,-0.2,-0.8],[0.35,0.25,0.1,-0.9]],requires_grad=True)
    pc=torch.tensor([[0.3,0.1,-0.1,-0.3],[-0.2,0.5,0.1,-0.4]],requires_grad=True)
    pp=torch.tensor([[0.2,0.05,-0.05,-0.2],[-0.1,0.4,0.2,-0.5]],requires_grad=True)
    gold=torch.tensor([0,1],dtype=torch.long)
    loss,_=adaptive_gate_gold_loss(gate,fc,fp,pc,pp,gold)
    grads=torch.autograd.grad(loss,(gate.w,gate.b,fc,fp,pc,pp),allow_unused=True)
    assert grads[0] is not None and float(grads[0].abs().sum())>0
    assert grads[1] is not None and float(grads[1].abs().sum())>0
    assert all(g is None or float(g.abs().sum())==0 for g in grads[2:])


def test_s61_k3_k7_k255_full_mass_and_permutation():
    g=torch.Generator().manual_seed(61002)
    gate=ConfidenceAdaptiveBoundedHybridGate()
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g)
        pair=torch.randn(2,k,generator=g)
        out=gate.compose(fused,pair)
        assert out.shape==(2,k)
        assert float((torch.softmax(out,-1).sum(-1)-1).abs().max())<=1e-6
        perm=torch.randperm(k,generator=g)
        got=gate.compose(fused[:,perm],pair[:,perm])
        assert torch.allclose(got,out[:,perm],atol=1e-6,rtol=0)


def test_s61_checkpoint_roundtrip_exact():
    gate=ConfidenceAdaptiveBoundedHybridGate()
    with torch.no_grad():
        gate.w.copy_(torch.tensor([0.1,-0.2,0.3,-0.4]))
        gate.b.add_(0.05)
    clone=ConfidenceAdaptiveBoundedHybridGate()
    clone.load_state_dict_exact(gate.state_dict_exact(),freeze=True)
    fused=torch.randn(2,4)
    pair=torch.randn(2,4)
    assert torch.equal(gate.compose(fused,pair),clone.compose(fused,pair))
    assert clone.trainable_parameter_count==0
