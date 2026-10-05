import torch

from nmd.v1_confidence_adaptive_bounded_hybrid import ConfidenceAdaptiveBoundedHybridGate
from nmd.v1_learned_set_reliability_gate import (
    S63_ALPHA_INITIAL,
    S63_GATE_PARAMETER_COUNT,
    S63_OPTION_HIDDEN_DIM,
    S63_OPTION_INPUT_DIM,
    S63_POOLED_DIM,
    S63_REPRESENTATION_DIM,
    LearnedSetReliabilityGate,
    learned_set_option_features,
    learned_set_reliability_loss,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    reliability_gate_loss,
    train_only_reliability_target,
)


def test_s63_exact_capacity_shapes_and_initial_alpha():
    gate=LearnedSetReliabilityGate()
    assert gate.parameter_count==S63_GATE_PARAMETER_COUNT==61
    assert gate.trainable_parameter_count==61
    params=dict(gate.named_parameters())
    assert set(params)=={"W_phi","b_phi","w_out","b_out"}
    assert tuple(params["W_phi"].shape)==(S63_OPTION_HIDDEN_DIM,S63_OPTION_INPUT_DIM)==(8,4)
    assert tuple(params["b_phi"].shape)==(8,)
    assert tuple(params["w_out"].shape)==(S63_REPRESENTATION_DIM,)==(20,)
    assert params["b_out"].ndim==0
    assert torch.equal(gate.w_out,torch.zeros_like(gate.w_out))

    fused=torch.tensor([[2.0,0.5,-1.0,0.0],[0.1,0.2,0.3,0.4]])
    pair=torch.tensor([[1.0,-0.5,0.0,2.0],[3.0,1.0,-2.0,0.0]])
    alpha=gate.alpha(fused,pair)
    assert torch.allclose(alpha,torch.full_like(alpha,S63_ALPHA_INITIAL),atol=1e-7,rtol=0)


def test_s63_representation_dims_and_permutation_invariance():
    g=torch.Generator().manual_seed(6306301)
    gate=LearnedSetReliabilityGate()
    fused=torch.randn(5,7,generator=g)
    pair=torch.randn(5,7,generator=g)
    option=learned_set_option_features(fused,pair)
    assert option.shape==(5,7,4)
    rep=gate.representation(fused,pair)
    assert rep.shape==(5,20)

    perm=torch.randperm(7,generator=g)
    assert torch.allclose(
        gate.alpha(fused,pair),
        gate.alpha(fused[:,perm],pair[:,perm]),
        atol=1e-6,rtol=0,
    )


def test_s63_affine_invariance_flat_finite_and_identity():
    gate=LearnedSetReliabilityGate()
    with torch.no_grad():
        gate.w_out.copy_(torch.linspace(-0.2,0.2,20))
    fused=torch.tensor([[4.0,1.0,0.0,-1.0],[1.1,1.0,0.0,-1.0]])
    pair=torch.tensor([[4.0,1.0,0.0,-1.0],[-1.0,4.0,0.0,1.0]])
    base=gate.alpha(fused,pair)
    affine=gate.alpha(3*fused+7,5*pair-11)
    assert torch.allclose(base,affine,atol=2e-5,rtol=0)

    flat=gate.alpha(torch.zeros(2,7),torch.zeros(2,7))
    assert torch.isfinite(flat).all()
    ident=gate.compose(fused,pair,alpha_override=0.0)
    assert torch.equal(ident,fused)


def test_s63_bounded_full_k_and_upstream_detach():
    gate=LearnedSetReliabilityGate()
    g=torch.Generator().manual_seed(6306302)
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g,requires_grad=True)
        pair=torch.randn(2,k,generator=g,requires_grad=True)
        out,diag=gate.compose(fused,pair,return_diagnostics=True)
        assert out.shape==(2,k)
        assert torch.isfinite(out).all()
        assert float((torch.softmax(out,-1).sum(-1)-1).abs().max())<=1e-6
        assert bool((diag["residual_max_abs"]<=diag["residual_bound"]+1e-7).all())
        loss=out.square().mean()
        grads=torch.autograd.grad(loss,(fused,pair),allow_unused=True)
        assert grads==(None,None)


def test_s63_same_s62_target_and_staged_gradient_path():
    reference=ConfidenceAdaptiveBoundedHybridGate()
    treatment=LearnedSetReliabilityGate()
    fused_c=torch.tensor([
        [-1.3768910,3.2005668,-3.3355374],
        [-0.4818169,3.1734812,0.6368278],
    ])
    fused_p=torch.tensor([
        [1.6032288,-2.4625099,1.4818633],
        [0.0850223,1.7102296,2.0455709],
    ])
    pair_c=torch.tensor([
        [1.7417690,-0.2375816,0.0183677],
        [1.8026907,-0.3109073,-3.2117627],
    ])
    pair_p=torch.tensor([
        [1.1653450,1.4833535,0.8274329],
        [-0.0134042,-0.6213452,-1.9260790],
    ])
    gold=torch.tensor([0,2],dtype=torch.long)

    y,_=train_only_reliability_target(fused_c,fused_p,pair_c,pair_p,gold)
    assert y.tolist()==[1.0,0.0]

    ref_loss,ref_diag=reliability_gate_loss(reference,fused_c,fused_p,pair_c,pair_p,gold)
    trt_loss,trt_diag=learned_set_reliability_loss(treatment,fused_c,fused_p,pair_c,pair_p,gold)
    assert float(ref_diag["positive_fraction"])==float(trt_diag["positive_fraction"])

    params=(treatment.W_phi,treatment.b_phi,treatment.w_out,treatment.b_out)
    grads=torch.autograd.grad(trt_loss,params,allow_unused=True)
    assert grads[2] is not None and float(grads[2].abs().sum())>0
    assert grads[3] is not None and float(grads[3].abs().sum())>0
    assert grads[0] is not None and float(grads[0].abs().sum())==0.0
    assert grads[1] is not None and float(grads[1].abs().sum())==0.0

    clone=LearnedSetReliabilityGate()
    opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
    warm_loss,_=learned_set_reliability_loss(clone,fused_c,fused_p,pair_c,pair_p,gold)
    opt.zero_grad(set_to_none=True)
    warm_loss.backward()
    opt.step()
    second_loss,_=learned_set_reliability_loss(clone,fused_c,fused_p,pair_c,pair_p,gold)
    phi_grads=torch.autograd.grad(second_loss,(clone.W_phi,clone.b_phi),allow_unused=True)
    assert all(g is not None and float(g.abs().sum())>0 for g in phi_grads)
