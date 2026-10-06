import torch

from nmd.v1_counterfactual_safety_veto import (
    S73_ACCEPT_THRESHOLD,
    S73_FEATURE_DIM,
    S73_PARAMETER_COUNT,
    CounterfactualSafetyVeto,
    counterfactual_safety_loss,
    counterfactual_safety_targets,
    safety_features,
)


def _pair_matrix(batch:int,k:int)->torch.Tensor:
    g=torch.Generator().manual_seed(73001+k)
    raw=torch.randn(batch,k,k,generator=g)
    pair=raw-raw.transpose(-1,-2)
    pair=pair-torch.diag_embed(pair.diagonal(dim1=-2,dim2=-1))
    return pair


def test_s73_predictor_parameter_count_and_initial_all_accept():
    predictor=CounterfactualSafetyVeto()
    assert S73_FEATURE_DIM==8
    assert S73_PARAMETER_COUNT==9
    assert S73_ACCEPT_THRESHOLD==0.5
    assert predictor.parameter_count==9
    x=torch.randn(7,8)
    assert torch.equal(predictor.probability(x),torch.full((7,),0.5))
    assert bool(predictor.accept(x).all())


def test_s73_hard_veto_outputs_only_exact_endpoints():
    predictor=CounterfactualSafetyVeto()
    fused=torch.randn(5,4)
    candidate=torch.randn(5,4)
    features=torch.randn(5,8)

    out_accept=predictor.apply(fused,candidate,features,force_accept=True)
    out_veto=predictor.apply(fused,candidate,features,force_accept=False)
    assert torch.equal(out_accept,candidate)
    assert torch.equal(out_veto,fused)


def test_s73_features_finite_and_option_permutation_invariant():
    b,k=6,7
    g=torch.Generator().manual_seed(73002)
    fused=torch.randn(b,k,generator=g)
    candidate=fused+0.05*torch.randn(b,k,generator=g)
    pair=_pair_matrix(b,k)
    direction=torch.tanh(torch.randn(b,k,generator=g))
    alpha=torch.rand(b,1,generator=g)*0.35

    base=safety_features(fused,candidate,pair,direction,alpha)
    perm=torch.tensor([6,2,0,5,1,4,3])
    pp=pair[:,perm][:,:,perm]
    got=safety_features(
        fused[:,perm],
        candidate[:,perm],
        pp,
        direction[:,perm],
        alpha,
    )
    assert bool(torch.isfinite(base).all())
    assert float((base-got).abs().max())<=2e-6


def test_s73_features_have_no_upstream_gradient():
    b,k=4,4
    fused=torch.randn(b,k,requires_grad=True)
    candidate=torch.randn(b,k,requires_grad=True)
    pair=_pair_matrix(b,k).requires_grad_(True)
    direction=torch.randn(b,k,requires_grad=True)
    alpha=torch.rand(b,1,requires_grad=True)
    features=safety_features(fused,candidate,pair,direction,alpha)
    assert not features.requires_grad


def test_s73_counterfactual_targets_are_train_only_detached():
    fused_c=torch.tensor([[3.0,0.0],[0.0,3.0]],requires_grad=True)
    fused_p=torch.tensor([[2.9,0.1],[0.1,2.9]],requires_grad=True)
    cand_c=torch.tensor([[3.2,-0.2],[-0.1,3.1]],requires_grad=True)
    cand_p=torch.tensor([[3.1,-0.1],[-0.2,3.2]],requires_grad=True)
    gold=torch.tensor([0,1])
    yc,yp,diag=counterfactual_safety_targets(
        fused_c,fused_p,cand_c,cand_p,gold
    )
    assert yc.shape==(2,) and yp.shape==(2,)
    assert not yc.requires_grad and not yp.requires_grad
    assert "canonical_safe_count" in diag


def test_s73_loss_trains_predictor_only():
    predictor=CounterfactualSafetyVeto()
    fc=torch.randn(8,8)
    fp=torch.randn(8,8)
    yc=torch.randint(0,2,(8,),dtype=torch.float32)
    yp=torch.randint(0,2,(8,),dtype=torch.float32)
    loss,_=counterfactual_safety_loss(predictor,fc,fp,yc,yp)
    loss.backward()
    assert predictor.weight.grad is not None
    assert predictor.bias.grad is not None


def test_s73_arbitrary_k_features():
    for k in (3,7,255):
        b=2
        fused=torch.randn(b,k)
        candidate=fused+0.01*torch.randn(b,k)
        pair=_pair_matrix(b,k)
        direction=torch.tanh(torch.randn(b,k))
        alpha=torch.full((b,1),0.1)
        features=safety_features(fused,candidate,pair,direction,alpha)
        assert features.shape==(b,8)
        assert bool(torch.isfinite(features).all())
