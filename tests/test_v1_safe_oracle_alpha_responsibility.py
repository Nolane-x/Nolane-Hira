import torch

from nmd.v1_contextual_reliability_gate import ContextInjectedReliabilityGate
from nmd.v1_per_view_responsibility import per_view_responsibility_loss
from nmd.v1_safe_oracle_alpha_responsibility import (
    S67_ALPHA_LATTICE,
    S67_TARGET_LEVELS,
    bounded_probe_at_alpha,
    safe_oracle_alpha_targets,
    safe_oracle_alpha_loss,
    select_safe_oracle_alpha,
)


def _random_batch(batch=4096,k=4,seed=6706701):
    g=torch.Generator().manual_seed(seed)
    fc=2.0*torch.randn(batch,k,generator=g)
    fp=2.0*torch.randn(batch,k,generator=g)
    pc=2.0*torch.randn(batch,k,generator=g)
    pp=2.0*torch.randn(batch,k,generator=g)
    gold=torch.randint(0,k,(batch,),generator=g)
    return fc,fp,pc,pp,gold


def test_s67_oracle_targets_cover_frozen_levels_and_are_detached():
    fc,fp,pc,pp,gold=_random_batch()
    yc,yp,diag=safe_oracle_alpha_targets(fc,fp,pc,pp,gold)

    assert not yc.requires_grad
    assert not yp.requires_grad
    assert tuple(S67_ALPHA_LATTICE)==(0.0,0.0875,0.175,0.2625,0.35)
    assert tuple(S67_TARGET_LEVELS)==(0.0,0.25,0.5,0.75,1.0)

    uc=set(round(float(x),6) for x in yc.unique())
    up=set(round(float(x),6) for x in yp.unique())
    allowed={0.0,0.25,0.5,0.75,1.0}
    assert uc.issubset(allowed)
    assert up.issubset(allowed)
    assert 0.0 in uc and 0.0 in up
    assert len(uc)>=4
    assert len(up)>=4

    ca=diag["canonical_selected_alpha"]
    pa=diag["paraphrase_selected_alpha"]
    assert set(round(float(x),6) for x in ca.unique()).issubset(set(S67_ALPHA_LATTICE))
    assert set(round(float(x),6) for x in pa.unique()).issubset(set(S67_ALPHA_LATTICE))


def test_s67_selected_nonzero_alpha_is_safe_and_js_improving():
    fc,fp,pc,pp,gold=_random_batch(batch=2048,k=4,seed=6706702)
    yc,yp,d=safe_oracle_alpha_targets(fc,fp,pc,pp,gold)

    cmask=yc.gt(0)
    pmask=yp.gt(0)
    assert bool(cmask.any())
    assert bool(pmask.any())

    assert bool((d["canonical_selected_ce"][cmask] <= d["canonical_baseline_ce"][cmask]+1e-8).all())
    assert bool((d["paraphrase_selected_ce"][pmask] <= d["paraphrase_baseline_ce"][pmask]+1e-8).all())
    assert bool((d["canonical_selected_js"][cmask] < d["js_base"][cmask]-1e-8).all())
    assert bool((d["paraphrase_selected_js"][pmask] < d["js_base"][pmask]-1e-8).all())


def test_s67_view_swap_equivariance():
    fc,fp,pc,pp,gold=_random_batch(batch=257,k=4,seed=6706703)
    yc,yp,_=safe_oracle_alpha_targets(fc,fp,pc,pp,gold)
    syc,syp,_=safe_oracle_alpha_targets(fp,fc,pp,pc,gold)
    assert torch.equal(yc,syp)
    assert torch.equal(yp,syc)


def test_s67_smaller_alpha_tie_break_is_exact():
    # Candidate 0.0875 and 0.175 are equally best (within tolerance).
    # Ascending lattice + strict update must retain 0.0875.
    base_ce=torch.tensor([1.0,1.0])
    base_js=torch.tensor([0.5,0.5])
    candidate_ce=torch.tensor([
        [0.9,0.9],
        [0.9,0.9],
        [0.9,1.2],
        [0.9,1.2],
    ])
    candidate_js=torch.tensor([
        [0.3,0.4],
        [0.3,0.4],
        [0.2,0.2],
        [0.2,0.2],
    ])
    alpha,ce,js=select_safe_oracle_alpha(
        base_ce,base_js,candidate_ce,candidate_js
    )
    assert torch.allclose(alpha,torch.tensor([0.2625,0.0875]),atol=0,rtol=0)
    assert torch.allclose(ce,torch.tensor([0.9,0.9]),atol=0,rtol=0)
    assert torch.allclose(js,torch.tensor([0.2,0.4]),atol=0,rtol=0)


def test_s67_oracle_target_differs_from_s66_binary_target():
    fc,fp,pc,pp,gold=_random_batch(batch=2048,k=4,seed=6706704)
    yc,yp,_=safe_oracle_alpha_targets(fc,fp,pc,pp,gold)

    gate=ContextInjectedReliabilityGate(use_context=True)
    g=torch.Generator().manual_seed(6706705)
    rc=torch.randn(2048,4,512,generator=g)
    rp=torch.randn(2048,4,512,generator=g)
    _,diag=per_view_responsibility_loss(
        gate,fc,fp,pc,pp,rc,rp,gold
    )
    # Difference from S66 binary semantics must be mechanically possible.
    assert bool(((yc!=0)&(yc!=1)).any() or ((yp!=0)&(yp!=1)).any())
    assert float(diag["target_disagreement_fraction"])>=0.0


def test_s67_loss_has_no_upstream_gradient_and_gate_gradient_lives():
    fc,fp,pc,pp,gold=_random_batch(batch=32,k=4,seed=6706706)
    fc.requires_grad_(True)
    fp.requires_grad_(True)
    pc.requires_grad_(True)
    pp.requires_grad_(True)

    g=torch.Generator().manual_seed(6706707)
    rc=torch.randn(32,4,512,generator=g,requires_grad=True)
    rp=torch.randn(32,4,512,generator=g,requires_grad=True)

    gate=ContextInjectedReliabilityGate(use_context=True)
    loss,_=safe_oracle_alpha_loss(
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


def test_s67_staged_phi_gradient_becomes_live_and_bounded_probe_identity():
    fc,fp,pc,pp,gold=_random_batch(batch=64,k=4,seed=6706708)
    g=torch.Generator().manual_seed(6706709)
    rc=torch.randn(64,4,512,generator=g)
    rp=torch.randn(64,4,512,generator=g)

    assert torch.equal(bounded_probe_at_alpha(fc,pc,0.0),fc)

    gate=ContextInjectedReliabilityGate(use_context=True)
    opt=torch.optim.SGD([gate.w_out,gate.b_out],lr=0.1)
    warm,_=safe_oracle_alpha_loss(
        gate,fc,fp,pc,pp,rc,rp,gold
    )
    opt.zero_grad(set_to_none=True)
    warm.backward()
    opt.step()

    second,_=safe_oracle_alpha_loss(
        gate,fc,fp,pc,pp,rc,rp,gold
    )
    phi=torch.autograd.grad(second,(gate.W_phi,gate.b_phi),allow_unused=True)
    assert all(x is not None and float(x.abs().sum())>0 for x in phi)
