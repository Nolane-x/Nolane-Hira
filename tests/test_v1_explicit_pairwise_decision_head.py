import torch

from nmd.v1_explicit_pairwise_decision_head import (
    S59_PAIRWISE_PARAMETER_COUNT,
    ExplicitPairwiseDecisionHead,
    build_pairwise_representation,
    gold_pairwise_loss,
    pairwise_head_loss,
)


def test_s59_parameter_surface_is_exact_and_bias_free():
    head=ExplicitPairwiseDecisionHead()
    assert head.parameter_count==S59_PAIRWISE_PARAMETER_COUNT==32832
    assert set(dict(head.named_parameters()))=={"A","u"}
    assert head.A.shape==(64,512)
    assert head.u.shape==(64,)


def test_s59_pair_matrix_is_antisymmetric_and_diagonal_zero():
    g=torch.Generator().manual_seed(59001)
    rep=torch.randn(3,7,512,generator=g)
    head=ExplicitPairwiseDecisionHead()
    pair=head.pairwise_logits(rep)
    assert float((pair+pair.transpose(-1,-2)).abs().max())==0.0
    assert float(torch.diagonal(pair,dim1=-2,dim2=-1).abs().max())==0.0


def test_s59_option_permutation_equivariance():
    g=torch.Generator().manual_seed(59002)
    rep=torch.randn(2,7,512,generator=g)
    head=ExplicitPairwiseDecisionHead()
    base_pair=head.pairwise_logits(rep)
    base_score=head.aggregate_logits(rep)
    perm=torch.tensor([3,0,6,2,1,5,4])
    pp=head.pairwise_logits(rep[:,perm])
    ps=head.aggregate_logits(rep[:,perm])
    assert torch.allclose(pp,base_pair[:,perm][:,:,perm],atol=1e-6,rtol=0)
    assert torch.allclose(ps,base_score[:,perm],atol=1e-6,rtol=0)


def test_s59_representation_is_detached():
    identity=torch.randn(2,4,256,requires_grad=True)
    context=torch.randn(2,4,256,requires_grad=True)
    rep=build_pairwise_representation(identity,context)
    assert rep.requires_grad is False
    assert rep.shape==(2,4,512)


def test_s59_gold_loss_supervises_only_gold_pairs():
    pair=torch.tensor([[
        [0.0,2.0,1.0,3.0],
        [-2.0,0.0,8.0,-4.0],
        [-1.0,-8.0,0.0,5.0],
        [-3.0,4.0,-5.0,0.0],
    ]])
    gold=torch.tensor([0],dtype=torch.long)
    loss,diag=gold_pairwise_loss(pair,gold)
    flipped,_=gold_pairwise_loss(-pair,gold)
    assert float(loss)<float(flipped)
    assert int(diag["supervised_gold_pair_count"])==3
    assert int(diag["supervised_distractor_pair_count"])==0


def test_s59_gradients_reach_A_and_u_only():
    g=torch.Generator().manual_seed(59003)
    rep=torch.randn(2,4,512,generator=g,requires_grad=True)
    gold=torch.tensor([0,3],dtype=torch.long)
    head=ExplicitPairwiseDecisionHead()
    loss,_=pairwise_head_loss(head,rep,gold)
    grads=torch.autograd.grad(loss,(head.A,head.u,rep),allow_unused=True)
    assert grads[0] is not None and float(grads[0].abs().sum())>0
    assert grads[1] is not None and float(grads[1].abs().sum())>0
    assert grads[2] is None or float(grads[2].abs().sum())==0.0


def test_s59_arbitrary_k_3_7_255_full_k_and_finite():
    g=torch.Generator().manual_seed(59004)
    head=ExplicitPairwiseDecisionHead()
    for k in (3,7,255):
        rep=torch.randn(1,k,512,generator=g)
        pair=head.pairwise_logits(rep)
        score=head.aggregate_logits(rep)
        assert pair.shape==(1,k,k)
        assert score.shape==(1,k)
        assert bool(torch.isfinite(pair).all())
        assert bool(torch.isfinite(score).all())
        p=torch.softmax(score,dim=-1)
        assert float((p.sum(-1)-1).abs().max())<=1e-6


def test_s59_deterministic_tiebreak_uses_representation_key():
    head=ExplicitPairwiseDecisionHead()
    with torch.no_grad():
        head.A.zero_()
        head.u.zero_()
    rep=torch.zeros(1,4,512)
    rep[0,0,0]=1
    rep[0,1,1]=1
    rep[0,2,2]=1
    rep[0,3,3]=1
    w1,d1=head.select_with_tiebreak(rep)
    w2,d2=head.select_with_tiebreak(rep)
    assert torch.equal(w1,w2)
    assert torch.equal(d1["top_score_tie_count"],torch.tensor([4]))
    assert bool((d1["representation_key_tie_count"]>=1).all())


def test_s59_checkpoint_roundtrip_exact():
    head=ExplicitPairwiseDecisionHead()
    clone=ExplicitPairwiseDecisionHead()
    clone.load_state_dict_exact(head.state_dict_exact(),freeze=True)
    g=torch.Generator().manual_seed(59005)
    rep=torch.randn(2,5,512,generator=g)
    assert torch.equal(head.pairwise_logits(rep),clone.pairwise_logits(rep))
    assert clone.trainable_parameter_count==0
