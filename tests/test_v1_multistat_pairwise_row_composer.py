import torch

from nmd.v1_multistat_pairwise_row_composer import (
    S71_PARAMETER_COUNT,
    MultiStatPairwiseRowComposer,
    multistat_responsibility_loss,
    normalized_pairwise_row_statistics,
    pairwise_row_mean,
    pairwise_row_statistics,
)


def _batch(batch=64,k=4,seed=7107101):
    g=torch.Generator().manual_seed(seed)
    fused_c=2*torch.randn(batch,k,generator=g)
    fused_p=2*torch.randn(batch,k,generator=g)
    raw_c=torch.randn(batch,k,k,generator=g)
    raw_p=torch.randn(batch,k,k,generator=g)
    pair_c=raw_c-raw_c.transpose(-1,-2)
    pair_p=raw_p-raw_p.transpose(-1,-2)
    eye=torch.eye(k,dtype=torch.bool).unsqueeze(0)
    pair_c=pair_c.masked_fill(eye,0)
    pair_p=pair_p.masked_fill(eye,0)
    ctx_c=torch.randn(batch,k,512,generator=g)
    ctx_p=torch.randn(batch,k,512,generator=g)
    gold=torch.randint(0,k,(batch,),generator=g)
    return fused_c,fused_p,pair_c,pair_p,ctx_c,ctx_p,gold


def test_s71_parameter_match_and_bit_identical_init():
    ref=MultiStatPairwiseRowComposer(use_multistat=False)
    trt=MultiStatPairwiseRowComposer(use_multistat=True)

    assert ref.parameter_count==S71_PARAMETER_COUNT==80
    assert trt.parameter_count==80
    assert ref.trainable_parameter_count==80
    assert trt.trainable_parameter_count==80

    rs=ref.parameter_state_dict_exact()
    ts=trt.parameter_state_dict_exact()
    assert rs.keys()==ts.keys()
    assert all(torch.equal(rs[k],ts[k]) for k in rs)
    assert torch.equal(ref.context_projection,trt.context_projection)


def test_s71_row_statistics_exclude_diagonal_and_reference_zeros_extras():
    _,_,pair,_,ctx,_,_=_batch(batch=3,k=4,seed=7107102)
    stats=pairwise_row_statistics(pair)
    norm=normalized_pairwise_row_statistics(pair)

    assert stats.shape==(3,4,4)
    assert norm.shape==(3,4,4)
    assert torch.allclose(stats[...,0],pair.sum(-1)/3,atol=1e-7,rtol=0)

    ref=MultiStatPairwiseRowComposer(use_multistat=False)
    trt=MultiStatPairwiseRowComposer(use_multistat=True)
    rr=ref.row_channels(pair)
    tt=trt.row_channels(pair)
    assert torch.equal(rr[...,:1],tt[...,:1])
    assert torch.equal(rr[...,1:],torch.zeros_like(rr[...,1:]))
    assert bool((tt[...,1:].abs()>0).any())

    # Changing only diagonal entries must not affect any row statistic.
    diag_changed=pair.clone()
    idx=torch.arange(4)
    diag_changed[:,idx,idx]=12345.0
    assert torch.equal(
        pairwise_row_statistics(pair),
        pairwise_row_statistics(diag_changed),
    )


def test_s71_option_permutation_equivariance_and_alpha_invariance():
    fused_c,_,pair,_,ctx,_,_=_batch(batch=7,k=5,seed=7107103)
    perm=torch.tensor([3,0,4,1,2])
    inv=torch.argsort(perm)

    p_pair=pair[:,perm][:,:,perm]
    p_fused=fused_c[:,perm]
    p_ctx=ctx[:,perm]

    for use in (False,True):
        gate=MultiStatPairwiseRowComposer(use_multistat=use)
        row=gate.row_channels(pair)
        prow=gate.row_channels(p_pair)
        assert torch.allclose(prow[:,inv],row,atol=1e-6,rtol=0)

        a=gate.alpha(fused_c,pair,ctx)
        pa=gate.alpha(p_fused,p_pair,p_ctx)
        assert torch.allclose(a,pa,atol=1e-6,rtol=0)

        out=gate.compose(fused_c,pair,ctx)
        pout=gate.compose(p_fused,p_pair,p_ctx)
        assert torch.allclose(pout[:,inv],out,atol=1e-6,rtol=0)


def test_s71_neutralized_extra_stats_collapse_reference_and_treatment_input():
    fused_c,_,pair,_,ctx,_,_=_batch(batch=8,k=4,seed=7107104)
    ref=MultiStatPairwiseRowComposer(use_multistat=False)
    trt=MultiStatPairwiseRowComposer(use_multistat=True)

    # Mechanical neutralization: treatment extra channels are explicitly zeroed.
    surface_ref=ref.option_input(fused_c,pair,ctx)
    surface_trt=trt.option_input(fused_c,pair,ctx).clone()
    surface_trt[...,9:12]=0
    assert torch.allclose(surface_ref,surface_trt,atol=0,rtol=0)


def test_s71_initial_alpha_identity_bound_and_arbitrary_k():
    g=torch.Generator().manual_seed(7107105)
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g)
        raw=torch.randn(2,k,k,generator=g)
        pair=raw-raw.transpose(-1,-2)
        ctx=torch.randn(2,k,512,generator=g)

        for use in (False,True):
            gate=MultiStatPairwiseRowComposer(use_multistat=use)
            alpha=gate.alpha(fused,pair,ctx)
            assert torch.allclose(alpha,torch.full_like(alpha,0.1),atol=1e-7,rtol=0)

            ident=gate.compose(fused,pair,ctx,alpha_override=0.0)
            assert torch.equal(ident,fused)

            out,d=gate.compose(fused,pair,ctx,return_diagnostics=True)
            assert bool((d["residual_max_abs"]<=d["residual_bound"]+1e-7).all())
            probs=torch.softmax(out,-1)
            assert float((probs.sum(-1)-1).abs().max())<=1e-6


def test_s71_loss_detaches_upstream_and_staged_phi_gradient_lives():
    fused_c,fused_p,pair_c,pair_p,ctx_c,ctx_p,gold=_batch(
        batch=48,k=4,seed=7107106
    )
    for x in (fused_c,fused_p,pair_c,pair_p,ctx_c,ctx_p):
        x.requires_grad_(True)

    gate=MultiStatPairwiseRowComposer(use_multistat=True)
    loss,_=multistat_responsibility_loss(
        gate,fused_c,fused_p,pair_c,pair_p,ctx_c,ctx_p,gold
    )
    grads=torch.autograd.grad(
        loss,
        (
            gate.W_phi,gate.b_phi,gate.w_out,gate.b_out,
            fused_c,fused_p,pair_c,pair_p,ctx_c,ctx_p,
        ),
        allow_unused=True,
    )
    assert grads[2] is not None and float(grads[2].abs().sum())>0
    assert grads[3] is not None and float(grads[3].abs().sum())>0
    assert grads[0] is not None and float(grads[0].abs().sum())==0.0
    assert grads[1] is not None and float(grads[1].abs().sum())==0.0
    assert grads[4:]==(None,None,None,None,None,None)

    # Once the zero-initialized output head has moved, phi gradients must live.
    gate2=MultiStatPairwiseRowComposer(use_multistat=True)
    opt=torch.optim.SGD([gate2.w_out,gate2.b_out],lr=0.1)
    warm,_=multistat_responsibility_loss(
        gate2,
        fused_c.detach(),fused_p.detach(),
        pair_c.detach(),pair_p.detach(),
        ctx_c.detach(),ctx_p.detach(),gold,
    )
    opt.zero_grad(set_to_none=True)
    warm.backward()
    opt.step()

    second,_=multistat_responsibility_loss(
        gate2,
        fused_c.detach(),fused_p.detach(),
        pair_c.detach(),pair_p.detach(),
        ctx_c.detach(),ctx_p.detach(),gold,
    )
    phi=torch.autograd.grad(second,(gate2.W_phi,gate2.b_phi),allow_unused=True)
    assert all(x is not None and float(x.abs().sum())>0 for x in phi)


def test_s71_pairwise_row_mean_matches_exact_uniform_aggregate():
    _,_,pair,_,_,_,_=_batch(batch=11,k=6,seed=7107107)
    expected=pair.sum(-1)/5
    assert torch.allclose(pairwise_row_mean(pair),expected,atol=1e-7,rtol=0)
