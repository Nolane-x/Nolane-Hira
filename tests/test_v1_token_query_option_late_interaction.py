import types

import torch

from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_token_query_option_late_interaction import (
    TokenLevelQueryOptionLateInteraction,
    TokenLateInteractionQueryFreePrivateCorrectionFork,
)


def _inputs(*,batch=2,k=7,q=5,seed=53001):
    g=torch.Generator().manual_seed(seed+k+q)
    native=torch.randn(batch,k,generator=g)
    state=torch.randn(batch,6,256,generator=g)
    state_mask=torch.ones(batch,6,dtype=torch.bool)
    option=torch.randn(batch,k,2,4,256,generator=g)
    option_token_mask=torch.ones(batch,k,2,4,dtype=torch.bool)
    option_view_mask=torch.ones(batch,k,2,dtype=torch.bool)
    question=torch.randn(batch,q,256,generator=g)
    question_mask=torch.ones(batch,q,dtype=torch.bool)
    return dict(
        native_logits=native,
        state_tokens=state,
        state_mask=state_mask,
        option_view_tokens=option,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
        question_tokens=question,
        question_mask=question_mask,
    )


def test_s53_zero_parameter_operator_and_matched_capacity():
    ref=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    trt=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    assert ref.correction_parameter_count==114688
    assert trt.correction_parameter_count==114688
    assert trt.identity_parameter_count==0
    assert trt.late_interaction_parameter_count==0
    rs=ref.correction_state_dict()
    ts=trt.correction_state_dict()
    assert rs.keys()==ts.keys()
    assert all(torch.equal(rs[k],ts[k]) for k in rs)


def test_s53_context_is_per_option_finite_unit_normalized():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    x=_inputs(k=7,q=5,seed=53010)
    logits,identity,context,weights=op.correction_logits_from_state_option(
        **x,return_context=True
    )
    assert tuple(logits.shape)==(2,7)
    assert tuple(identity.shape)==(2,7,256)
    assert tuple(context.shape)==(2,7,256)
    assert tuple(weights.shape)==(2,7,5)
    assert bool(torch.isfinite(context).all())
    assert float((context.norm(dim=-1)-1.0).abs().max())<=1e-5
    assert float((weights.sum(-1)-1.0).abs().max())<=1e-6
    assert float((context[:,0]-context[:,1]).abs().max())>1e-6


def test_s53_query_padding_and_mask_invariance():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    x=_inputs(k=4,q=4,seed=53020)
    a=op.correction_logits_from_state_option(**x)[0]

    g=torch.Generator().manual_seed(53021)
    pad=torch.randn(2,3,256,generator=g)*100.0
    y=dict(x)
    y["question_tokens"]=torch.cat([x["question_tokens"],pad],dim=1)
    y["question_mask"]=torch.cat([
        x["question_mask"],
        torch.zeros(2,3,dtype=torch.bool),
    ],dim=1)
    b=op.correction_logits_from_state_option(**y)[0]
    assert torch.allclose(a,b,rtol=0.0,atol=2e-6)


def test_s53_query_token_permutation_invariance():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    x=_inputs(k=4,q=6,seed=53030)
    a=op.correction_logits_from_state_option(**x)[0]
    perm=torch.tensor([5,1,3,0,4,2])
    y=dict(x)
    y["question_tokens"]=x["question_tokens"][:,perm]
    y["question_mask"]=x["question_mask"][:,perm]
    b=op.correction_logits_from_state_option(**y)[0]
    assert torch.allclose(a,b,rtol=0.0,atol=2e-6)


def test_s53_option_permutation_equivariance():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    x=_inputs(k=7,q=5,seed=53040)
    logits,identity,context,weights=op.correction_logits_from_state_option(
        **x,return_context=True
    )
    perm=torch.tensor([6,2,4,1,5,0,3])
    y=dict(x)
    y["native_logits"]=x["native_logits"][:,perm]
    y["option_view_tokens"]=x["option_view_tokens"][:,perm]
    y["option_view_token_mask"]=x["option_view_token_mask"][:,perm]
    y["option_view_mask"]=x["option_view_mask"][:,perm]
    lp,ip,cp,wp=op.correction_logits_from_state_option(**y,return_context=True)
    assert torch.allclose(lp,logits[:,perm],rtol=0.0,atol=3e-6)
    assert torch.allclose(ip,identity[:,perm],rtol=0.0,atol=3e-6)
    assert torch.allclose(cp,context[:,perm],rtol=0.0,atol=3e-6)
    assert torch.allclose(wp,weights[:,perm],rtol=0.0,atol=3e-6)


def test_s53_k_three_seven_255_full_k_probability_mass():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(53050),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(53051),std=0.01)
    for k in (3,7,255):
        x=_inputs(k=k,q=5,seed=53060)
        logits,_=op.correction_logits_from_state_option(**x)
        assert tuple(logits.shape)==(2,k)
        p=torch.softmax(logits,dim=-1)
        assert float((p.sum(-1)-1.0).abs().max())<=1e-6


def test_s53_treatment_has_no_pooled_query_summary_bypass():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    def _fail(*args,**kwargs):
        raise AssertionError("pooled query_summary must not be called by S53 treatment")
    op.query_summary=types.MethodType(_fail,op)
    x=_inputs(k=4,q=5,seed=53070)
    logits,_=op.correction_logits_from_state_option(**x)
    assert tuple(logits.shape)==(2,4)


def test_s53_informative_token_perturbation_changes_context_and_logits():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(53080),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(53081),std=0.01)
    x=_inputs(k=4,q=5,seed=53082)
    a,_,ca,_=op.correction_logits_from_state_option(**x,return_context=True)
    y=dict(x)
    y["question_tokens"]=x["question_tokens"].clone()
    y["question_tokens"][:,0]+=4.0
    b,_,cb,_=op.correction_logits_from_state_option(**y,return_context=True)
    assert float((ca-cb).abs().max())>1e-5
    assert float((a-b).abs().max())>1e-6


def test_s53_all_masked_query_rejected():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    x=_inputs(k=4,q=5,seed=53090)
    x["question_mask"]=torch.zeros_like(x["question_mask"])
    try:
        op.correction_logits_from_state_option(**x)
    except ValueError as exc:
        assert "at least one active query token" in str(exc)
    else:
        raise AssertionError("S53 accepted an all-masked query")
