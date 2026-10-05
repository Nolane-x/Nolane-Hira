import torch

from nmd.v1_pairwise_ranking_consistency import (
    detached_anchor_sign,
    pairwise_ordinal_consistency,
    weighted_pairwise_ordinal_auxiliary,
)


def _gold(batch=1,value=0):
    return torch.full((batch,),value,dtype=torch.long)


def test_s57_anchor_sign_is_detached():
    m=torch.tensor([[1.0,-2.0]],requires_grad=True)
    s=detached_anchor_sign(m)
    assert s.requires_grad is False
    assert torch.equal(s,torch.tensor([[1.0,-1.0]]))


def test_s57_matching_strong_ranking_lower_than_sign_flip():
    a=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    b=torch.tensor([[3.5,0.8,-0.8,-2.5]])
    same,_=pairwise_ordinal_consistency(a,b,_gold())
    flip,_=pairwise_ordinal_consistency(a,-b,_gold())
    assert float(same)<float(flip)
    assert float(flip)>0.5


def test_s57_shared_offset_invariance():
    a=torch.tensor([[3.0,1.0,-1.0,0.5]])
    b=torch.tensor([[2.0,-1.0,0.0,1.5]])
    g=_gold()
    base,_=pairwise_ordinal_consistency(a,b,g)
    shifted,_=pairwise_ordinal_consistency(a+13.0,b-9.0,g)
    assert torch.allclose(base,shifted,rtol=0.0,atol=1e-6)


def test_s57_positive_scale_invariance():
    a=torch.tensor([[3.0,1.0,-1.0,0.5]])
    b=torch.tensor([[2.0,-1.0,0.0,1.5]])
    g=_gold()
    base,_=pairwise_ordinal_consistency(a,b,g)
    scaled,_=pairwise_ordinal_consistency(a*4.0,b*0.5,g)
    assert torch.allclose(base,scaled,rtol=0.0,atol=3e-5)


def test_s57_wrong_gold_anchor_is_filtered():
    # gold=0. In view A, distractors outrank the gold.
    a=torch.tensor([[0.0,4.0,2.0]])
    b=torch.tensor([[4.0,1.0,0.0]])
    _loss,diag=pairwise_ordinal_consistency(a,b,_gold())
    assert int(diag["gold_filtered_direction_count"])>0
    assert float(diag["gold_filtered_direction_fraction"])>0.0


def test_s57_correct_gold_anchor_is_retained():
    a=torch.tensor([[4.0,1.0,0.0]])
    b=torch.tensor([[3.0,0.5,-0.5]])
    _loss,diag=pairwise_ordinal_consistency(a,b,_gold())
    assert int(diag["active_directional_anchor_count"])>0
    assert int(diag["gold_filtered_direction_count"])==0


def test_s57_non_gold_pairs_remain_eligible_when_gold_anchor_is_wrong():
    a=torch.tensor([[0.0,4.0,2.0,-2.0]])
    b=torch.tensor([[3.0,2.0,-1.0,-3.0]])
    _loss,diag=pairwise_ordinal_consistency(a,b,_gold())
    assert float(diag["non_gold_active_direction_fraction"])>0.0


def test_s57_flat_logits_have_zero_active_anchors_and_exact_zero_loss():
    a=torch.zeros(2,7,requires_grad=True)
    b=torch.zeros(2,7,requires_grad=True)
    aux,diag=weighted_pairwise_ordinal_auxiliary(
        a,b,_gold(2),coefficient=0.05
    )
    assert float(aux)==0.0
    assert float(diag["active_directional_anchor_fraction"])==0.0
    grads=torch.autograd.grad(aux,(a,b),allow_unused=True)
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)==0.0


def test_s57_option_permutation_equivariance_with_remapped_gold():
    a=torch.tensor([[3.0,0.5,-1.0,1.2]])
    b=torch.tensor([[2.5,-0.4,-0.8,1.8]])
    gold=torch.tensor([0],dtype=torch.long)
    perm=torch.tensor([2,0,3,1])
    new_gold=(perm==gold.item()).nonzero(as_tuple=False).flatten()
    assert new_gold.numel()==1
    p_gold=new_gold.to(torch.long)
    base,_=pairwise_ordinal_consistency(a,b,gold)
    moved,_=pairwise_ordinal_consistency(a[:,perm],b[:,perm],p_gold)
    assert torch.allclose(base,moved,rtol=0.0,atol=1e-7)


def test_s57_reference_auxiliary_exact_zero_and_treatment_gradient_live():
    a=torch.tensor([[3.0,0.5,-1.0,1.2]],requires_grad=True)
    b=torch.tensor([[2.5,-0.4,-0.8,1.8]],requires_grad=True)
    gold=_gold()

    ref,_=weighted_pairwise_ordinal_auxiliary(
        a,b,gold,coefficient=0.0
    )
    assert float(ref)==0.0

    trt,_=weighted_pairwise_ordinal_auxiliary(
        a,b,gold,coefficient=0.05
    )
    grads=torch.autograd.grad(trt,(a,b),allow_unused=True)
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0.0


def test_s57_k_three_seven_255_finite():
    g=torch.Generator().manual_seed(57001)
    for k in (3,7,255):
        a=torch.randn(2,k,generator=g)
        b=torch.randn(2,k,generator=g)
        gold=torch.tensor([0,k-1],dtype=torch.long)
        aux,diag=weighted_pairwise_ordinal_auxiliary(
            a,b,gold,coefficient=0.05
        )
        assert bool(torch.isfinite(aux))
        assert bool(torch.isfinite(diag["ordinal_loss"]))
        assert bool(torch.isfinite(diag["active_directional_anchor_fraction"]))
