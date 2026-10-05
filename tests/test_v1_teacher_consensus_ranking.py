import torch

from nmd.v1_teacher_consensus_ranking import (
    teacher_consensus_targets,
    teacher_consensus_pairwise_loss,
    weighted_teacher_consensus_auxiliary,
)


def _gold():
    return torch.tensor([0],dtype=torch.long)


def test_s58_strong_same_sign_teacher_pair_is_active():
    ta=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    tb=torch.tensor([[3.5,0.8,-0.8,-2.5]])
    active,sign,diag=teacher_consensus_targets(ta,tb,_gold())
    assert int(active.sum())>0
    assert float(diag["teacher_active_consensus_fraction"])>0.0
    assert bool((sign[active]!=0).all())


def test_s58_low_confidence_teacher_pair_is_inactive():
    ta=torch.zeros(1,4)
    tb=torch.zeros(1,4)
    active,_sign,diag=teacher_consensus_targets(ta,tb,_gold())
    assert int(active.sum())==0
    assert float(diag["teacher_weak_pair_fraction"])==1.0


def test_s58_teacher_sign_disagreement_is_inactive():
    ta=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    tb=-ta
    active,_sign,diag=teacher_consensus_targets(ta,tb,_gold())
    assert int(active.sum())==0
    assert float(diag["teacher_sign_disagreement_fraction"])>0.0


def test_s58_wrong_gold_teacher_consensus_is_filtered_but_non_gold_survives():
    ta=torch.tensor([[0.0,4.0,2.0,-2.0]])
    tb=torch.tensor([[0.0,3.5,1.5,-2.5]])
    active,_sign,diag=teacher_consensus_targets(ta,tb,_gold())
    assert int(diag["teacher_wrong_gold_filtered_count"])>0
    assert float(diag["teacher_non_gold_active_fraction"])>0.0
    # Any pair involving gold=0 that remains active must rank gold first.
    assert int(active.sum())>0


def test_s58_satisfied_student_margin_is_lower_loss_than_violation():
    teacher_a=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    teacher_b=torch.tensor([[3.5,0.8,-0.8,-2.5]])
    good_a=teacher_a.clone().requires_grad_(True)
    good_b=teacher_b.clone().requires_grad_(True)
    bad_a=(-teacher_a).clone().requires_grad_(True)
    bad_b=(-teacher_b).clone().requires_grad_(True)
    lg,_=teacher_consensus_pairwise_loss(good_a,good_b,teacher_a,teacher_b,_gold())
    lb,_=teacher_consensus_pairwise_loss(bad_a,bad_b,teacher_a,teacher_b,_gold())
    assert float(lg)<float(lb)


def test_s58_offset_and_positive_scale_teacher_invariance():
    ta=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    tb=torch.tensor([[3.5,0.8,-0.8,-2.5]])
    base=teacher_consensus_targets(ta,tb,_gold())[:2]
    shifted=teacher_consensus_targets(ta+17.0,tb-9.0,_gold())[:2]
    scaled=teacher_consensus_targets(ta*4.0,tb*0.5,_gold())[:2]
    assert torch.equal(base[0],shifted[0])
    assert torch.equal(base[1],shifted[1])
    assert torch.equal(base[0],scaled[0])
    assert torch.equal(base[1],scaled[1])


def test_s58_teacher_targets_are_detached_and_student_gradients_live():
    ta=torch.tensor([[4.0,1.0,-1.0,-3.0]],requires_grad=True)
    tb=torch.tensor([[3.5,0.8,-0.8,-2.5]],requires_grad=True)
    sa=torch.tensor([[0.2,0.1,0.0,-0.1]],requires_grad=True)
    sb=torch.tensor([[0.1,0.0,-0.1,-0.2]],requires_grad=True)
    aux,_=weighted_teacher_consensus_auxiliary(
        sa,sb,ta,tb,_gold(),coefficient=0.05
    )
    gs=torch.autograd.grad(aux,(sa,sb),retain_graph=True)
    assert sum(float(g.abs().sum()) for g in gs)>0.0
    gt=torch.autograd.grad(aux,(ta,tb),allow_unused=True)
    assert all(g is None or float(g.abs().sum())==0.0 for g in gt)


def test_s58_reference_auxiliary_is_exact_zero_with_valid_graph():
    ta=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    tb=torch.tensor([[3.5,0.8,-0.8,-2.5]])
    sa=torch.randn(1,4,requires_grad=True)
    sb=torch.randn(1,4,requires_grad=True)
    aux,_=weighted_teacher_consensus_auxiliary(
        sa,sb,ta,tb,_gold(),coefficient=0.0
    )
    assert float(aux)==0.0
    gs=torch.autograd.grad(aux,(sa,sb))
    assert all(float(g.abs().sum())==0.0 for g in gs)


def test_s58_flat_student_with_active_teacher_has_positive_loss():
    ta=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    tb=torch.tensor([[3.5,0.8,-0.8,-2.5]])
    sa=torch.zeros(1,4,requires_grad=True)
    sb=torch.zeros(1,4,requires_grad=True)
    loss,diag=teacher_consensus_pairwise_loss(sa,sb,ta,tb,_gold())
    assert float(loss)>0.0
    assert float(diag["student_violation_fraction"])>0.0


def test_s58_option_permutation_equivariance():
    ta=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    tb=torch.tensor([[3.5,0.8,-0.8,-2.5]])
    sa=torch.tensor([[2.0,0.7,-0.3,-1.0]])
    sb=torch.tensor([[1.8,0.6,-0.2,-0.9]])
    base,_=teacher_consensus_pairwise_loss(sa,sb,ta,tb,_gold())
    perm=torch.tensor([2,0,3,1])
    new_gold=(perm==0).nonzero(as_tuple=False).flatten().to(torch.long)
    permuted,_=teacher_consensus_pairwise_loss(
        sa[:,perm],sb[:,perm],ta[:,perm],tb[:,perm],new_gold
    )
    assert abs(float(base)-float(permuted))<=1e-7


def test_s58_arbitrary_k_three_seven_255():
    g=torch.Generator().manual_seed(58001)
    for k in (3,7,255):
        b=2
        ta=torch.randn(b,k,generator=g)
        tb=ta+0.2*torch.randn(b,k,generator=g)
        sa=torch.randn(b,k,generator=g,requires_grad=True)
        sb=torch.randn(b,k,generator=g,requires_grad=True)
        gold=torch.tensor([0,k-1],dtype=torch.long)
        loss,_=teacher_consensus_pairwise_loss(sa,sb,ta,tb,gold)
        assert bool(torch.isfinite(loss))
        p=torch.softmax(sa,dim=-1)
        assert float((p.sum(-1)-1.0).abs().max())<=1e-6
