import torch

from nmd.v1_contextual_reliability_gate import ContextInjectedReliabilityGate
from nmd.v1_per_view_responsibility import (
    per_view_responsibility_targets,
    per_view_responsibility_loss,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    train_only_reliability_target,
)


def _random_batch(batch=2048,k=3,seed=6606601):
    g=torch.Generator().manual_seed(seed)
    fc=2.0*torch.randn(batch,k,generator=g)
    fp=2.0*torch.randn(batch,k,generator=g)
    pc=2.0*torch.randn(batch,k,generator=g)
    pp=2.0*torch.randn(batch,k,generator=g)
    gold=torch.randint(0,k,(batch,),generator=g)
    return fc,fp,pc,pp,gold


def test_s66_per_view_targets_are_detached_and_cover_all_responsibility_states():
    fc,fp,pc,pp,gold=_random_batch()
    yc,yp,diag=per_view_responsibility_targets(fc,fp,pc,pp,gold)

    assert yc.shape==yp.shape==(2048,)
    assert not yc.requires_grad
    assert not yp.requires_grad
    assert set(yc.unique().tolist())=={0.0,1.0}
    assert set(yp.unique().tolist())=={0.0,1.0}

    states=set(zip(yc.tolist(),yp.tolist()))
    assert states=={(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)}
    assert float(diag["target_disagreement_fraction"])>0.0


def test_s66_view_swap_equivariance():
    fc,fp,pc,pp,gold=_random_batch(batch=257,k=4,seed=6606602)
    yc,yp,_=per_view_responsibility_targets(fc,fp,pc,pp,gold)
    syc,syp,_=per_view_responsibility_targets(fp,fc,pp,pc,gold)
    assert torch.equal(yc,syp)
    assert torch.equal(yp,syc)


def test_s66_correctness_and_stability_veto_are_both_required():
    fc,fp,pc,pp,gold=_random_batch(batch=4096,k=4,seed=6606603)
    yc,yp,diag=per_view_responsibility_targets(fc,fp,pc,pp,gold)

    c_expected=(
        diag["canonical_correctness_safe"]
        &diag["canonical_stability_better"]
    ).to(yc.dtype)
    p_expected=(
        diag["paraphrase_correctness_safe"]
        &diag["paraphrase_stability_better"]
    ).to(yp.dtype)

    assert torch.equal(yc,c_expected)
    assert torch.equal(yp,p_expected)

    assert bool((~diag["canonical_correctness_safe"]).any())
    assert bool((~diag["paraphrase_correctness_safe"]).any())
    assert bool((~diag["canonical_stability_better"]).any())
    assert bool((~diag["paraphrase_stability_better"]).any())


def test_s66_reference_pair_label_and_treatment_responsibility_are_distinct():
    fc,fp,pc,pp,gold=_random_batch(batch=2048,k=3,seed=6606604)
    pair,_=train_only_reliability_target(fc,fp,pc,pp,gold)
    yc,yp,_=per_view_responsibility_targets(fc,fp,pc,pp,gold)

    assert bool((yc!=yp).any())
    assert bool((yc!=pair).any() | (yp!=pair).any())


def test_s66_loss_has_no_upstream_gradient_and_gate_gradient_lives():
    fc,fp,pc,pp,gold=_random_batch(batch=32,k=4,seed=6606605)
    fc.requires_grad_(True)
    fp.requires_grad_(True)
    pc.requires_grad_(True)
    pp.requires_grad_(True)

    g=torch.Generator().manual_seed(6606606)
    rc=torch.randn(32,4,512,generator=g,requires_grad=True)
    rp=torch.randn(32,4,512,generator=g,requires_grad=True)

    gate=ContextInjectedReliabilityGate(use_context=True)
    loss,_=per_view_responsibility_loss(
        gate,fc,fp,pc,pp,rc,rp,gold
    )

    grads=torch.autograd.grad(
        loss,
        (gate.W_phi,gate.b_phi,gate.w_out,gate.b_out,fc,fp,pc,pp,rc,rp),
        allow_unused=True,
    )
    assert grads[2] is not None and float(grads[2].abs().sum())>0
    assert grads[3] is not None and float(grads[3].abs().sum())>0
    assert grads[0] is not None and float(grads[0].abs().sum())==0.0
    assert grads[1] is not None and float(grads[1].abs().sum())==0.0
    assert grads[4:]==(None,None,None,None,None,None)


def test_s66_staged_phi_gradient_becomes_live():
    fc,fp,pc,pp,gold=_random_batch(batch=64,k=4,seed=6606607)
    g=torch.Generator().manual_seed(6606608)
    rc=torch.randn(64,4,512,generator=g)
    rp=torch.randn(64,4,512,generator=g)

    gate=ContextInjectedReliabilityGate(use_context=True)
    opt=torch.optim.SGD([gate.w_out,gate.b_out],lr=0.1)
    warm,_=per_view_responsibility_loss(
        gate,fc,fp,pc,pp,rc,rp,gold
    )
    opt.zero_grad(set_to_none=True)
    warm.backward()
    opt.step()

    second,_=per_view_responsibility_loss(
        gate,fc,fp,pc,pp,rc,rp,gold
    )
    phi=torch.autograd.grad(second,(gate.W_phi,gate.b_phi),allow_unused=True)
    assert all(x is not None and float(x.abs().sum())>0 for x in phi)
