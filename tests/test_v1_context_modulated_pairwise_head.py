import torch

from nmd.v1_context_modulated_pairwise_head import (
    S68_CONTEXT_DIMENSION,
    S68_CONTEXT_PROJECTION_SEED,
    S68_MODULATION_PARAMETER_COUNT,
    S68_MODULATION_SCALE,
    S68_PAIRWISE_PARAMETER_COUNT,
    ContextModulatedPairwiseHead,
    context_modulated_pairwise_loss,
)
from nmd.v1_explicit_pairwise_decision_head import ExplicitPairwiseDecisionHead


def _rep(batch=3,k=7,seed=6806801):
    g=torch.Generator().manual_seed(seed)
    return torch.randn(batch,k,512,generator=g)


def test_s68_exact_matched_capacity_and_initial_s59_identity():
    reference=ContextModulatedPairwiseHead(use_joint_context=False)
    treatment=ContextModulatedPairwiseHead(use_joint_context=True)
    s59=ExplicitPairwiseDecisionHead()

    assert S68_CONTEXT_DIMENSION==8
    assert S68_CONTEXT_PROJECTION_SEED==68068
    assert S68_MODULATION_SCALE==0.5
    assert S68_MODULATION_PARAMETER_COUNT==512
    assert S68_PAIRWISE_PARAMETER_COUNT==33344
    assert reference.parameter_count==treatment.parameter_count==33344
    assert reference.trainable_parameter_count==treatment.trainable_parameter_count==33344

    rparams=dict(reference.named_parameters())
    tparams=dict(treatment.named_parameters())
    assert set(rparams)==set(tparams)=={"A","u","G"}
    for key in rparams:
        assert torch.equal(rparams[key],tparams[key])
    assert torch.equal(reference.A,s59.A)
    assert torch.equal(reference.u,s59.u)
    assert torch.equal(reference.G,torch.zeros_like(reference.G))
    assert torch.equal(treatment.G,torch.zeros_like(treatment.G))
    assert torch.equal(reference.context_projection,treatment.context_projection)
    assert not reference.context_projection.requires_grad
    assert not treatment.context_projection.requires_grad

    rep=_rep()
    for head in (reference,treatment):
        assert torch.equal(head.modulation(rep),torch.ones(rep.shape[0],64))
        assert torch.allclose(head.pairwise_logits(rep),s59.pairwise_logits(rep),atol=0,rtol=0)
        assert torch.allclose(head.aggregate_logits(rep),s59.aggregate_logits(rep),atol=0,rtol=0)
        a,_=head.select_with_tiebreak(rep)
        b,_=s59.select_with_tiebreak(rep)
        assert torch.equal(a,b)


def test_s68_context_source_is_controlled_and_non_degenerate():
    rep=_rep(batch=4,k=5,seed=6806802)
    reference=ContextModulatedPairwiseHead(use_joint_context=False)
    treatment=ContextModulatedPairwiseHead(use_joint_context=True)

    zr=reference.context_feature(rep)
    zt=treatment.context_feature(rep)
    assert zr.shape==zt.shape==(4,8)
    assert torch.isfinite(zr).all()
    assert torch.isfinite(zt).all()
    assert float(zr.abs().sum())>0.0
    assert float(zt.abs().sum())>0.0
    assert float((zr-zt).abs().max())>1e-7

    ablated=rep.clone()
    ablated[...,256:]=0.0
    assert torch.allclose(
        reference.context_feature(ablated),
        treatment.context_feature(ablated),
        atol=0,
        rtol=0,
    )


def test_s68_option_permutation_context_invariant_pair_equivariant():
    rep=_rep(batch=2,k=7,seed=6806803)
    perm=torch.tensor([5,2,0,6,3,1,4])
    inv=torch.argsort(perm)

    for use_joint in (False,True):
        head=ContextModulatedPairwiseHead(use_joint_context=use_joint)
        with torch.no_grad():
            head.G.copy_(torch.linspace(-0.2,0.2,64*8).reshape(64,8))

        z=head.context_feature(rep)
        zp=head.context_feature(rep[:,perm])
        assert torch.allclose(z,zp,atol=1e-7,rtol=0)

        pair=head.pairwise_logits(rep)
        pairp=head.pairwise_logits(rep[:,perm])
        restored=pairp[:,inv][:,:,inv]
        assert torch.allclose(pair,restored,atol=2e-6,rtol=0)

        agg=head.aggregate_logits(rep)
        aggp=head.aggregate_logits(rep[:,perm])[:,inv]
        assert torch.allclose(agg,aggp,atol=2e-6,rtol=0)


def test_s68_antisymmetry_diagonal_and_arbitrary_k():
    for k in (3,7,255):
        rep=_rep(batch=1,k=k,seed=6806800+k)
        for use_joint in (False,True):
            head=ContextModulatedPairwiseHead(use_joint_context=use_joint)
            with torch.no_grad():
                head.G.copy_(torch.randn(64,8,generator=torch.Generator().manual_seed(6806900+k))*0.05)
            pair=head.pairwise_logits(rep)
            assert pair.shape==(1,k,k)
            assert float((pair+pair.transpose(-1,-2)).abs().max())==0.0
            assert float(torch.diagonal(pair,dim1=-2,dim2=-1).abs().max())==0.0
            agg=head.aggregate_logits(rep)
            assert torch.isfinite(agg).all()
            prob=torch.softmax(agg,dim=-1)
            assert float((prob.sum(-1)-1.0).abs().max())<=1e-6


def test_s68_gold_pair_gradients_live_and_upstream_detached():
    rep=_rep(batch=8,k=4,seed=6806804).requires_grad_(True)
    gold=torch.tensor([0,1,2,3,0,1,2,3],dtype=torch.long)

    grads_by_arm={}
    for label,use_joint in (("reference",False),("treatment",True)):
        head=ContextModulatedPairwiseHead(use_joint_context=use_joint)
        loss,diag=context_modulated_pairwise_loss(head,rep,gold)
        grads=torch.autograd.grad(
            loss,
            (head.A,head.u,head.G,rep),
            allow_unused=True,
        )
        assert grads[0] is not None and float(grads[0].abs().sum())>0.0
        assert grads[1] is not None and float(grads[1].abs().sum())>0.0
        assert grads[2] is not None and float(grads[2].abs().sum())>0.0
        assert grads[3] is None
        assert float(diag["antisymmetry_max_abs_error"])==0.0
        assert float(diag["diagonal_max_abs_error"])==0.0
        grads_by_arm[label]=grads[2].detach().clone()

    assert float((grads_by_arm["reference"]-grads_by_arm["treatment"]).abs().sum())>1e-7


def test_s68_checkpoint_replay_exact():
    rep=_rep(batch=2,k=5,seed=6806805)
    head=ContextModulatedPairwiseHead(use_joint_context=True)
    with torch.no_grad():
        head.A.add_(0.001)
        head.u.mul_(0.99)
        head.G.copy_(torch.randn(64,8,generator=torch.Generator().manual_seed(6806806))*0.03)

    expected=head.pairwise_logits(rep)
    state=head.state_dict_exact()

    replay=ContextModulatedPairwiseHead(use_joint_context=True)
    replay.load_state_dict_exact(state,freeze=True)
    actual=replay.pairwise_logits(rep)
    assert torch.equal(expected,actual)
    assert all(not p.requires_grad for p in replay.parameters())
