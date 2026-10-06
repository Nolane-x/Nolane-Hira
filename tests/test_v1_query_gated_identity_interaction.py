import torch

from nmd.v1_explicit_pairwise_decision_head import (
    ExplicitPairwiseDecisionHead,
    S59_PAIRWISE_PARAMETER_COUNT,
    build_pairwise_representation,
    pairwise_head_loss,
)
from nmd.v1_query_gated_identity_interaction import (
    S69_INTERACTION_SCALE,
    S69_REPRESENTATION_DIMENSION,
    S69_REPRESENTATION_PARAMETER_COUNT,
    build_reference_pairwise_representation,
    build_query_gated_context,
    build_query_gated_pairwise_representation,
)


def _sources(batch=4,k=7,seed=6906901):
    g=torch.Generator().manual_seed(seed)
    identity=torch.randn(batch,k,256,generator=g)
    context=torch.randn(batch,k,256,generator=g)
    identity=torch.nn.functional.normalize(identity,dim=-1)
    context=torch.nn.functional.normalize(context,dim=-1)
    return identity,context


def test_s69_reference_is_exact_s59_and_treatment_is_zero_parameter():
    identity,context=_sources()
    ref=build_reference_pairwise_representation(identity,context)
    s59=build_pairwise_representation(identity,context)
    trt=build_query_gated_pairwise_representation(identity,context)

    assert torch.equal(ref,s59)
    assert ref.shape==trt.shape==(4,7,S69_REPRESENTATION_DIMENSION)
    assert S69_REPRESENTATION_DIMENSION==512
    assert S69_REPRESENTATION_PARAMETER_COUNT==0
    assert S69_INTERACTION_SCALE==16.0
    assert torch.isfinite(trt).all()
    assert not ref.requires_grad
    assert not trt.requires_grad
    assert float((ref-trt).abs().max())>1e-6


def test_s69_zero_identity_neutralizes_interaction_to_reference():
    g=torch.Generator().manual_seed(6906902)
    identity=torch.zeros(3,5,256)
    context=torch.randn(3,5,256,generator=g)
    context=torch.nn.functional.normalize(context,dim=-1)

    ref=build_reference_pairwise_representation(identity,context)
    trt=build_query_gated_pairwise_representation(identity,context)
    gated=build_query_gated_context(identity,context)

    assert torch.allclose(gated,context,atol=1e-7,rtol=0)
    assert torch.allclose(ref,trt,atol=1e-7,rtol=0)


def test_s69_option_permutation_equivariance():
    identity,context=_sources(batch=3,k=7,seed=6906903)
    perm=torch.tensor([5,2,6,0,4,1,3])

    for builder in (
        build_reference_pairwise_representation,
        build_query_gated_pairwise_representation,
    ):
        a=builder(identity,context)
        b=builder(identity[:,perm],context[:,perm])
        assert torch.allclose(b,a[:,perm],atol=1e-7,rtol=0)


def test_s69_matched_s59_heads_are_bit_identical():
    ref=ExplicitPairwiseDecisionHead(trainable=True)
    trt=ExplicitPairwiseDecisionHead(trainable=True)
    assert ref.parameter_count==trt.parameter_count==S59_PAIRWISE_PARAMETER_COUNT==32832
    rs=ref.state_dict_exact()
    ts=trt.state_dict_exact()
    assert rs.keys()==ts.keys()
    assert all(torch.equal(rs[k],ts[k]) for k in rs)


def test_s69_pairwise_contracts_and_permutation_equivariance():
    g=torch.Generator().manual_seed(6906904)
    for k in (3,7,255):
        identity=torch.nn.functional.normalize(
            torch.randn(2,k,256,generator=g),dim=-1
        )
        context=torch.nn.functional.normalize(
            torch.randn(2,k,256,generator=g),dim=-1
        )
        perm=torch.randperm(k,generator=g)

        for builder in (
            build_reference_pairwise_representation,
            build_query_gated_pairwise_representation,
        ):
            rep=builder(identity,context)
            rep_p=builder(identity[:,perm],context[:,perm])
            head=ExplicitPairwiseDecisionHead(trainable=True)
            pair=head.pairwise_logits(rep)
            pair_p=head.pairwise_logits(rep_p)
            score=head.aggregate_logits(rep)
            score_p=head.aggregate_logits(rep_p)

            assert torch.equal(pair+pair.transpose(-1,-2),torch.zeros_like(pair))
            assert torch.equal(
                torch.diagonal(pair,dim1=-2,dim2=-1),
                torch.zeros(2,k),
            )
            assert torch.allclose(
                pair_p,pair[:,perm][:,:,perm],atol=2e-6,rtol=0
            )
            assert torch.allclose(score_p,score[:,perm],atol=2e-6,rtol=0)
            assert float((torch.softmax(score,-1).sum(-1)-1).abs().max())<=1e-6


def test_s69_selected_choice_permutation_equivariance_without_ties():
    identity,context=_sources(batch=8,k=7,seed=6906905)
    perm=torch.tensor([4,1,6,0,2,5,3])

    for builder in (
        build_reference_pairwise_representation,
        build_query_gated_pairwise_representation,
    ):
        rep=builder(identity,context)
        head=ExplicitPairwiseDecisionHead(trainable=True)
        winner,diag=head.select_with_tiebreak(rep)
        winner_p,diag_p=head.select_with_tiebreak(rep[:,perm])
        # For random continuous probes top-score ties should not occur.
        assert bool((diag["top_score_tie_count"]==1).all())
        assert bool((diag_p["top_score_tie_count"]==1).all())
        assert torch.equal(perm[winner_p],winner)


def test_s69_pairwise_gradients_live_and_sources_detached():
    identity,context=_sources(batch=16,k=4,seed=6906906)
    identity=identity.detach().requires_grad_(True)
    context=context.detach().requires_grad_(True)
    gold=torch.arange(16,dtype=torch.long)%4

    for builder in (
        build_reference_pairwise_representation,
        build_query_gated_pairwise_representation,
    ):
        head=ExplicitPairwiseDecisionHead(trainable=True)
        rep=builder(identity,context)
        loss,_=pairwise_head_loss(head,rep,gold)
        grads=torch.autograd.grad(
            loss,(head.A,head.u,identity,context),allow_unused=True
        )
        assert grads[0] is not None and float(grads[0].abs().sum())>0
        assert grads[1] is not None and float(grads[1].abs().sum())>0
        assert grads[2] is None
        assert grads[3] is None


def test_s69_head_checkpoint_replay_exact():
    identity,context=_sources(batch=3,k=7,seed=6906907)
    rep=build_query_gated_pairwise_representation(identity,context)

    head=ExplicitPairwiseDecisionHead(trainable=True)
    before=head.pairwise_logits(rep)
    state=head.state_dict_exact()

    replay=ExplicitPairwiseDecisionHead(trainable=True)
    replay.load_state_dict_exact(state,freeze=True)
    after=replay.pairwise_logits(rep)

    assert torch.equal(before,after)
