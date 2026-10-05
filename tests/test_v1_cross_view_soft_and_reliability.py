import torch

from nmd.v1_contextual_reliability_gate import ContextInjectedReliabilityGate
from nmd.v1_cross_view_soft_and_reliability import (
    cross_view_soft_and_logit,
    cross_view_soft_and_reliability_loss,
)
from nmd.v1_reliability_supervised_adaptive_gate import train_only_reliability_target


def _batch():
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
    g=torch.Generator().manual_seed(6506501)
    context_c=torch.randn(2,3,512,generator=g)
    context_p=torch.randn(2,3,512,generator=g)
    return fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold


def test_s65_soft_and_is_exact_min_and_pair_symmetric():
    zc=torch.tensor([-3.0,-0.2,0.7,4.0])
    zp=torch.tensor([-2.0,0.4,-1.3,1.0])
    got=cross_view_soft_and_logit(zc,zp)
    expected=torch.minimum(zc,zp)
    assert torch.allclose(got,expected,atol=1e-7,rtol=0)
    assert torch.equal(got,cross_view_soft_and_logit(zp,zc))


def test_s65_soft_and_gradient_routes_through_lower_view_only_when_unequal():
    zc=torch.tensor([-1.0,2.0],requires_grad=True)
    zp=torch.tensor([1.0,-2.0],requires_grad=True)
    out=cross_view_soft_and_logit(zc,zp).sum()
    gc,gp=torch.autograd.grad(out,(zc,zp))
    assert torch.equal(gc,torch.tensor([1.0,0.0]))
    assert torch.equal(gp,torch.tensor([0.0,1.0]))


def test_s65_matched_gate_capacity_and_identical_initialization():
    reference=ContextInjectedReliabilityGate(use_context=True)
    treatment=ContextInjectedReliabilityGate(use_context=True)
    assert reference.parameter_count==treatment.parameter_count==60
    rs=reference.parameter_state_dict_exact()
    ts=treatment.parameter_state_dict_exact()
    assert rs.keys()==ts.keys()
    assert all(torch.equal(rs[k],ts[k]) for k in rs)
    assert torch.equal(reference.context_projection,treatment.context_projection)


def test_s65_exact_s62_target_and_no_extra_objective_parameters():
    fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold=_batch()
    y,_=train_only_reliability_target(
        fused_c,fused_p,pair_c,pair_p,gold
    )
    assert y.tolist()==[1.0,0.0]

    gate=ContextInjectedReliabilityGate(use_context=True)
    before=set(dict(gate.named_parameters()))
    loss,diag=cross_view_soft_and_reliability_loss(
        gate,
        fused_c,fused_p,
        pair_c,pair_p,
        context_c,context_p,
        gold,
    )
    after=set(dict(gate.named_parameters()))
    assert before==after=={"W_phi","b_phi","w_out","b_out"}
    assert gate.parameter_count==60
    assert torch.isfinite(loss)
    assert int(diag["target_count"])==2
    assert int(diag["target_positive_count"])==1


def test_s65_loss_detaches_all_upstream_inputs():
    fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold=_batch()
    fused_c.requires_grad_(True)
    fused_p.requires_grad_(True)
    pair_c.requires_grad_(True)
    pair_p.requires_grad_(True)
    context_c.requires_grad_(True)
    context_p.requires_grad_(True)

    gate=ContextInjectedReliabilityGate(use_context=True)
    loss,_=cross_view_soft_and_reliability_loss(
        gate,
        fused_c,fused_p,
        pair_c,pair_p,
        context_c,context_p,
        gold,
    )
    grads=torch.autograd.grad(
        loss,
        (fused_c,fused_p,pair_c,pair_p,context_c,context_p),
        allow_unused=True,
    )
    assert grads==(None,None,None,None,None,None)


def test_s65_single_view_inference_path_is_unchanged():
    fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold=_batch()
    gate=ContextInjectedReliabilityGate(use_context=True)
    with torch.no_grad():
        gate.w_out.copy_(torch.linspace(-0.2,0.2,14))

    single_before=gate.compose(fused_c,pair_c,context_c)
    _loss,_=cross_view_soft_and_reliability_loss(
        gate,
        fused_c,fused_p,
        pair_c,pair_p,
        context_c,context_p,
        gold,
    )
    single_after=gate.compose(fused_c,pair_c,context_c)
    assert torch.equal(single_before,single_after)


def test_s65_staged_gradient_path_and_full_k_mechanics():
    fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold=_batch()
    gate=ContextInjectedReliabilityGate(use_context=True)
    params=(gate.W_phi,gate.b_phi,gate.w_out,gate.b_out)
    loss,_=cross_view_soft_and_reliability_loss(
        gate,
        fused_c,fused_p,
        pair_c,pair_p,
        context_c,context_p,
        gold,
    )
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    assert grads[2] is not None and float(grads[2].abs().sum())>0
    assert grads[3] is not None and float(grads[3].abs().sum())>0
    assert grads[0] is not None and float(grads[0].abs().sum())==0.0
    assert grads[1] is not None and float(grads[1].abs().sum())==0.0

    clone=ContextInjectedReliabilityGate(use_context=True)
    opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
    warm,_=cross_view_soft_and_reliability_loss(
        clone,
        fused_c,fused_p,
        pair_c,pair_p,
        context_c,context_p,
        gold,
    )
    opt.zero_grad(set_to_none=True)
    warm.backward()
    opt.step()
    second,_=cross_view_soft_and_reliability_loss(
        clone,
        fused_c,fused_p,
        pair_c,pair_p,
        context_c,context_p,
        gold,
    )
    phi=torch.autograd.grad(second,(clone.W_phi,clone.b_phi),allow_unused=True)
    assert all(g is not None and float(g.abs().sum())>0 for g in phi)

    generator=torch.Generator().manual_seed(6506502)
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=generator)
        pair=torch.randn(2,k,generator=generator)
        context=torch.randn(2,k,512,generator=generator)
        out,diag=gate.compose(
            fused,pair,context,return_diagnostics=True
        )
        assert out.shape==(2,k)
        assert float((torch.softmax(out,-1).sum(-1)-1).abs().max())<=1e-6
        assert bool((diag["residual_max_abs"]<=diag["residual_bound"]+1e-7).all())
        assert torch.equal(
            gate.compose(fused,pair,context,alpha_override=0.0),
            fused,
        )
