import torch
import torch.nn.functional as F

from nmd.v1_explicit_pairwise_decision_head import (
    ExplicitPairwiseDecisionHead,
    S59_PAIRWISE_PARAMETER_COUNT,
)
from nmd.v1_token_level_bidirectional_evidence_binding import (
    S75_INTERACTION_SCALE,
    S75_REPRESENTATION_DIMENSION,
    S75_REPRESENTATION_PARAMETER_COUNT,
    S75_TEMPERATURE,
    build_token_level_bidirectional_pairwise_representation,
    token_level_bidirectional_pool,
)


def _fixture(k=4,seed=75001):
    g=torch.Generator().manual_seed(seed+k)
    b,s,v,t,q,d=3,5,2,4,6,256
    identity=F.normalize(torch.randn(b,k,d,generator=g),dim=-1)
    state=torch.randn(b,s,d,generator=g)
    option=torch.randn(b,k,v,t,d,generator=g)
    query=torch.randn(b,q,d,generator=g)
    state_mask=torch.ones(b,s,dtype=torch.bool)
    option_token_mask=torch.ones(b,k,v,t,dtype=torch.bool)
    option_view_mask=torch.ones(b,k,v,dtype=torch.bool)
    question_mask=torch.ones(b,q,dtype=torch.bool)
    return identity,state,state_mask,option,option_token_mask,option_view_mask,query,question_mask


def _rep(k=4,seed=75001):
    x=_fixture(k,seed)
    return build_token_level_bidirectional_pairwise_representation(
        identity=x[0],
        state_tokens=x[1],
        state_mask=x[2],
        option_view_tokens=x[3],
        option_view_token_mask=x[4],
        option_view_mask=x[5],
        question_tokens=x[6],
        question_mask=x[7],
    )


def test_s75_frozen_constants_and_capacity():
    assert S75_REPRESENTATION_DIMENSION==512
    assert S75_REPRESENTATION_PARAMETER_COUNT==0
    assert S75_TEMPERATURE==0.10
    assert S75_INTERACTION_SCALE==16.0
    head=ExplicitPairwiseDecisionHead()
    assert head.parameter_count==S59_PAIRWISE_PARAMETER_COUNT==32832


def test_s75_dynamic_k_and_finite():
    for k in (3,7,255):
        r=_rep(k,75010+k)
        assert r.shape==(3,k,512)
        assert torch.isfinite(r).all()
        assert not r.requires_grad


def test_s75_option_permutation_equivariance():
    x=list(_fixture(7,75100))
    ref=build_token_level_bidirectional_pairwise_representation(
        identity=x[0],state_tokens=x[1],state_mask=x[2],
        option_view_tokens=x[3],option_view_token_mask=x[4],
        option_view_mask=x[5],question_tokens=x[6],question_mask=x[7],
    )
    perm=torch.tensor([5,1,6,0,3,2,4])
    inv=torch.argsort(perm)
    got=build_token_level_bidirectional_pairwise_representation(
        identity=x[0][:,perm],state_tokens=x[1],state_mask=x[2],
        option_view_tokens=x[3][:,perm],option_view_token_mask=x[4][:,perm],
        option_view_mask=x[5][:,perm],question_tokens=x[6],question_mask=x[7],
    )[:,inv]
    assert float((ref-got).abs().max())<=2e-6


def test_s75_option_view_permutation_invariance():
    x=list(_fixture(4,75200))
    ref=build_token_level_bidirectional_pairwise_representation(
        identity=x[0],state_tokens=x[1],state_mask=x[2],
        option_view_tokens=x[3],option_view_token_mask=x[4],
        option_view_mask=x[5],question_tokens=x[6],question_mask=x[7],
    )
    got=build_token_level_bidirectional_pairwise_representation(
        identity=x[0],state_tokens=x[1],state_mask=x[2],
        option_view_tokens=x[3][:,:,torch.tensor([1,0])],
        option_view_token_mask=x[4][:,:,torch.tensor([1,0])],
        option_view_mask=x[5][:,:,torch.tensor([1,0])],
        question_tokens=x[6],question_mask=x[7],
    )
    assert float((ref-got).abs().max())<=2e-6


def test_s75_masked_tokens_cannot_change_representation():
    x=list(_fixture(4,75300))
    x[2][:,-1]=False
    x[4][:,:,:,-1]=False
    x[7][:,-1]=False
    ref=build_token_level_bidirectional_pairwise_representation(
        identity=x[0],state_tokens=x[1],state_mask=x[2],
        option_view_tokens=x[3],option_view_token_mask=x[4],
        option_view_mask=x[5],question_tokens=x[6],question_mask=x[7],
    )
    x[1][:,-1]=1e6
    x[3][:,:,:,-1]=1e6
    x[6][:,-1]=-1e6
    got=build_token_level_bidirectional_pairwise_representation(
        identity=x[0],state_tokens=x[1],state_mask=x[2],
        option_view_tokens=x[3],option_view_token_mask=x[4],
        option_view_mask=x[5],question_tokens=x[6],question_mask=x[7],
    )
    assert float((ref-got).abs().max())<=2e-6


def test_s75_neutral_constant_positive_gate_collapses_to_joint_pooling():
    g=torch.Generator().manual_seed(75400)
    b,k,q,d=2,4,5,256
    query=F.normalize(torch.randn(b,q,d,generator=g),dim=-1)
    question_mask=torch.ones(b,q,dtype=torch.bool)
    state_support=torch.full((b,q),0.25)
    option_support=torch.full((b,k,q),0.35)
    evidence,diag=token_level_bidirectional_pool(
        query,state_support,option_support,question_mask
    )
    expected=F.normalize(query.mean(dim=1),dim=-1)[:,None,:].expand(b,k,d)
    assert float((evidence-expected).abs().max())<=2e-6
    assert torch.all(diag["bidirectional_compatibility"]>0)


def test_s75_representation_has_zero_upstream_gradient():
    x=list(_fixture(4,75500))
    x[0].requires_grad_(True)
    x[1].requires_grad_(True)
    x[3].requires_grad_(True)
    x[6].requires_grad_(True)
    rep=build_token_level_bidirectional_pairwise_representation(
        identity=x[0],state_tokens=x[1],state_mask=x[2],
        option_view_tokens=x[3],option_view_token_mask=x[4],
        option_view_mask=x[5],question_tokens=x[6],question_mask=x[7],
    )
    assert not rep.requires_grad
    assert x[0].grad is None and x[1].grad is None and x[3].grad is None and x[6].grad is None
