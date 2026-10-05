import torch

from nmd.v1_contextual_reliability_gate import (
    S64_ALPHA_INITIAL,
    S64_CONTEXT_DIM,
    S64_CONTEXT_PROJECTION_SEED,
    S64_ENCODER_INIT_SEED,
    S64_GATE_PARAMETER_COUNT,
    S64_OPTION_HIDDEN_DIM,
    S64_OPTION_INPUT_DIM,
    S64_POOLED_DIM,
    S64_REPRESENTATION_DIM,
    ContextInjectedReliabilityGate,
    contextual_option_features,
    contextual_reliability_loss,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    train_only_reliability_target,
)


def _surfaces():
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
    return fused_c,fused_p,pair_c,pair_p,gold


def test_s64_exact_matched_capacity_init_and_context_buffer():
    reference=ContextInjectedReliabilityGate(use_context=False)
    treatment=ContextInjectedReliabilityGate(use_context=True)

    assert reference.parameter_count==treatment.parameter_count==S64_GATE_PARAMETER_COUNT==60
    assert reference.trainable_parameter_count==treatment.trainable_parameter_count==60

    rparams=dict(reference.named_parameters())
    tparams=dict(treatment.named_parameters())
    assert set(rparams)==set(tparams)=={"W_phi","b_phi","w_out","b_out"}
    assert tuple(rparams["W_phi"].shape)==(S64_OPTION_HIDDEN_DIM,S64_OPTION_INPUT_DIM)==(5,8)
    assert tuple(rparams["b_phi"].shape)==(5,)
    assert tuple(rparams["w_out"].shape)==(S64_REPRESENTATION_DIM,)==(14,)
    assert rparams["b_out"].ndim==0

    for key in rparams:
        assert torch.equal(rparams[key],tparams[key])
    assert torch.equal(reference.context_projection,treatment.context_projection)
    assert reference.context_projection.shape==(S64_CONTEXT_DIM,512)
    assert not reference.context_projection.requires_grad
    assert not treatment.context_projection.requires_grad

    fused=torch.tensor([[2.0,0.5,-1.0,0.0],[0.1,0.2,0.3,0.4]])
    pair=torch.tensor([[1.0,-0.5,0.0,2.0],[3.0,1.0,-2.0,0.0]])
    context=torch.randn(2,4,512,generator=torch.Generator().manual_seed(6406401))
    ra=reference.alpha(fused,pair,context)
    ta=treatment.alpha(fused,pair,context)
    assert torch.allclose(ra,torch.full_like(ra,S64_ALPHA_INITIAL),atol=1e-7,rtol=0)
    assert torch.allclose(ta,torch.full_like(ta,S64_ALPHA_INITIAL),atol=1e-7,rtol=0)
    assert S64_CONTEXT_PROJECTION_SEED==64064
    assert S64_ENCODER_INIT_SEED==64164


def test_s64_reference_zero_context_treatment_context_non_degenerate():
    g=torch.Generator().manual_seed(6406402)
    fused=torch.randn(3,7,generator=g)
    pair=torch.randn(3,7,generator=g)
    context=torch.randn(3,7,512,generator=g)

    reference=ContextInjectedReliabilityGate(use_context=False)
    treatment=ContextInjectedReliabilityGate(use_context=True)

    rin=reference.option_input(fused,pair,context)
    tin=treatment.option_input(fused,pair,context)
    assert rin.shape==tin.shape==(3,7,8)
    assert torch.equal(rin[...,4:],torch.zeros_like(rin[...,4:]))
    assert torch.isfinite(tin[...,4:]).all()
    assert float(tin[...,4:].abs().sum())>0.0

    direct=contextual_option_features(context,treatment.context_projection)
    assert direct.shape==(3,7,4)
    assert torch.allclose(tin[...,4:],direct,atol=0,rtol=0)


def test_s64_treatment_permutation_invariance_and_context_sensitivity():
    g=torch.Generator().manual_seed(6406403)
    fused=torch.randn(4,7,generator=g)
    pair=torch.randn(4,7,generator=g)
    context=torch.randn(4,7,512,generator=g)
    perm=torch.randperm(7,generator=g)

    gate=ContextInjectedReliabilityGate(use_context=True)
    with torch.no_grad():
        gate.w_out.copy_(torch.linspace(-0.25,0.25,S64_REPRESENTATION_DIM))

    a=gate.alpha(fused,pair,context)
    ap=gate.alpha(fused[:,perm],pair[:,perm],context[:,perm])
    assert torch.allclose(a,ap,atol=2e-6,rtol=0)

    context2=context.clone()
    context2[:,0,:]=-context2[:,0,:]
    b=gate.alpha(fused,pair,context2)
    assert float((a-b).abs().max())>1e-7


def test_s64_surface_affine_invariance_flat_finite_and_identity():
    g=torch.Generator().manual_seed(6406404)
    fused=torch.randn(3,7,generator=g)
    pair=torch.randn(3,7,generator=g)
    context=torch.randn(3,7,512,generator=g)

    gate=ContextInjectedReliabilityGate(use_context=True)
    with torch.no_grad():
        gate.w_out.copy_(torch.linspace(-0.2,0.2,S64_REPRESENTATION_DIM))

    base=gate.alpha(fused,pair,context)
    affine=gate.alpha(3.0*fused+7.0,5.0*pair-11.0,context)
    assert torch.allclose(base,affine,atol=2e-5,rtol=0)

    flat=gate.alpha(torch.zeros(2,7),torch.zeros(2,7),torch.zeros(2,7,512))
    assert torch.isfinite(flat).all()

    ident=gate.compose(fused,pair,context,alpha_override=0.0)
    assert torch.equal(ident,fused)


def test_s64_bounded_full_k_and_all_upstream_inputs_detached():
    gate=ContextInjectedReliabilityGate(use_context=True)
    g=torch.Generator().manual_seed(6406405)
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g,requires_grad=True)
        pair=torch.randn(2,k,generator=g,requires_grad=True)
        context=torch.randn(2,k,512,generator=g,requires_grad=True)
        out,diag=gate.compose(fused,pair,context,return_diagnostics=True)
        assert out.shape==(2,k)
        assert torch.isfinite(out).all()
        assert float((torch.softmax(out,-1).sum(-1)-1).abs().max())<=1e-6
        assert bool((diag["residual_max_abs"]<=diag["residual_bound"]+1e-7).all())
        loss=out.square().mean()
        grads=torch.autograd.grad(loss,(fused,pair,context),allow_unused=True)
        assert grads==(None,None,None)


def test_s64_matched_target_and_staged_gradient_path():
    fused_c,fused_p,pair_c,pair_p,gold=_surfaces()
    g=torch.Generator().manual_seed(6406406)
    context_c=torch.randn(2,3,512,generator=g)
    context_p=torch.randn(2,3,512,generator=g)

    y,_=train_only_reliability_target(fused_c,fused_p,pair_c,pair_p,gold)
    assert y.tolist()==[1.0,0.0]

    reference=ContextInjectedReliabilityGate(use_context=False)
    treatment=ContextInjectedReliabilityGate(use_context=True)

    rloss,rdiag=contextual_reliability_loss(
        reference,fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold
    )
    tloss,tdiag=contextual_reliability_loss(
        treatment,fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold
    )
    assert float(rdiag["positive_fraction"])==float(tdiag["positive_fraction"])

    for gate,loss in ((reference,rloss),(treatment,tloss)):
        params=(gate.W_phi,gate.b_phi,gate.w_out,gate.b_out)
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        assert grads[2] is not None and float(grads[2].abs().sum())>0
        assert grads[3] is not None and float(grads[3].abs().sum())>0
        assert grads[0] is not None and float(grads[0].abs().sum())==0.0
        assert grads[1] is not None and float(grads[1].abs().sum())==0.0

    clone=ContextInjectedReliabilityGate(use_context=True)
    opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
    warm,_=contextual_reliability_loss(
        clone,fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold
    )
    opt.zero_grad(set_to_none=True)
    warm.backward()
    opt.step()
    second,_=contextual_reliability_loss(
        clone,fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold
    )
    phi=torch.autograd.grad(second,(clone.W_phi,clone.b_phi),allow_unused=True)
    assert all(x is not None and float(x.abs().sum())>0 for x in phi)
