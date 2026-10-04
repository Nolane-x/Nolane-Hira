import types

import torch

from nmd.v1_token_query_option_late_interaction import (
    TokenLateInteractionQueryFreePrivateCorrectionFork,
)
from nmd.v1_joint_state_query_option_interaction import (
    JointStateQueryOptionLateInteraction,
    JointStateQueryOptionPrivateCorrectionFork,
)


def _inputs(*,batch=2,k=7,q=5,s=6,v=2,t=4,seed=54001):
    g=torch.Generator().manual_seed(seed+k+q+s+t)
    native=torch.randn(batch,k,generator=g)
    state=torch.randn(batch,s,256,generator=g)
    state_mask=torch.ones(batch,s,dtype=torch.bool)
    option=torch.randn(batch,k,v,t,256,generator=g)
    option_token_mask=torch.ones(batch,k,v,t,dtype=torch.bool)
    option_view_mask=torch.ones(batch,k,v,dtype=torch.bool)
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


def _activate(op,seed=54010):
    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(seed),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(seed+1),std=0.01)


def test_s54_zero_parameter_operator_and_exact_s53_capacity():
    ref=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    trt=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    assert ref.correction_parameter_count==114688
    assert trt.correction_parameter_count==114688
    assert trt.identity_parameter_count==0
    assert trt.joint_interaction_parameter_count==0
    rs=ref.correction_state_dict()
    ts=trt.correction_state_dict()
    assert rs.keys()==ts.keys()
    assert all(torch.equal(rs[k],ts[k]) for k in rs)


def test_s54_context_shape_weights_supports_and_normalization():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    x=_inputs(k=7,q=5,seed=54020)
    logits,identity,context,weights,state_support,option_support=(
        op.correction_logits_from_state_option(**x,return_context=True)
    )
    assert tuple(logits.shape)==(2,7)
    assert tuple(identity.shape)==(2,7,256)
    assert tuple(context.shape)==(2,7,256)
    assert tuple(weights.shape)==(2,7,5)
    assert tuple(state_support.shape)==(2,5)
    assert tuple(option_support.shape)==(2,7,5)
    assert float((weights.sum(-1)-1.0).abs().max())<=1e-6
    assert float((context.norm(dim=-1)-1.0).abs().max())<=1e-6
    assert bool(torch.isfinite(context).all())


def test_s54_state_query_option_padding_mask_invariance():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    _activate(op,54030)
    x=_inputs(k=4,q=4,s=5,t=3,seed=54031)
    a=op.correction_logits_from_state_option(**x)[0]

    y=dict(x)
    g=torch.Generator().manual_seed(54032)
    y["state_tokens"]=torch.cat([
        x["state_tokens"],torch.randn(2,2,256,generator=g)*100
    ],dim=1)
    y["state_mask"]=torch.cat([
        x["state_mask"],torch.zeros(2,2,dtype=torch.bool)
    ],dim=1)

    y["question_tokens"]=torch.cat([
        x["question_tokens"],torch.randn(2,2,256,generator=g)*100
    ],dim=1)
    y["question_mask"]=torch.cat([
        x["question_mask"],torch.zeros(2,2,dtype=torch.bool)
    ],dim=1)

    pad=torch.randn(2,4,2,2,256,generator=g)*100
    y["option_view_tokens"]=torch.cat([x["option_view_tokens"],pad],dim=3)
    y["option_view_token_mask"]=torch.cat([
        x["option_view_token_mask"],
        torch.zeros(2,4,2,2,dtype=torch.bool),
    ],dim=3)

    b=op.correction_logits_from_state_option(**y)[0]
    assert torch.allclose(a,b,rtol=0.0,atol=3e-6)


def test_s54_state_query_option_token_and_view_permutation_invariance():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    _activate(op,54040)
    x=_inputs(k=4,q=5,s=6,v=2,t=4,seed=54041)
    a=op.correction_logits_from_state_option(**x)[0]

    y=dict(x)
    sp=torch.tensor([5,2,0,4,1,3])
    qp=torch.tensor([4,1,3,0,2])
    tp=torch.tensor([3,0,2,1])
    vp=torch.tensor([1,0])
    y["state_tokens"]=x["state_tokens"][:,sp]
    y["state_mask"]=x["state_mask"][:,sp]
    y["question_tokens"]=x["question_tokens"][:,qp]
    y["question_mask"]=x["question_mask"][:,qp]
    y["option_view_tokens"]=x["option_view_tokens"][:,:,vp][:,:,:,tp]
    y["option_view_token_mask"]=x["option_view_token_mask"][:,:,vp][:,:,:,tp]
    y["option_view_mask"]=x["option_view_mask"][:,:,vp]

    b=op.correction_logits_from_state_option(**y)[0]
    assert torch.allclose(a,b,rtol=0.0,atol=4e-6)


def test_s54_option_permutation_equivariance():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    _activate(op,54050)
    x=_inputs(k=7,q=5,seed=54051)
    logits,identity,context,weights,ss,os=op.correction_logits_from_state_option(
        **x,return_context=True
    )
    perm=torch.tensor([6,2,4,1,5,0,3])
    y=dict(x)
    y["native_logits"]=x["native_logits"][:,perm]
    y["option_view_tokens"]=x["option_view_tokens"][:,perm]
    y["option_view_token_mask"]=x["option_view_token_mask"][:,perm]
    y["option_view_mask"]=x["option_view_mask"][:,perm]
    lp,ip,cp,wp,ssp,osp=op.correction_logits_from_state_option(
        **y,return_context=True
    )
    assert torch.allclose(lp,logits[:,perm],rtol=0.0,atol=4e-6)
    assert torch.allclose(ip,identity[:,perm],rtol=0.0,atol=4e-6)
    assert torch.allclose(cp,context[:,perm],rtol=0.0,atol=4e-6)
    assert torch.allclose(wp,weights[:,perm],rtol=0.0,atol=4e-6)
    assert torch.equal(ssp,ss)
    assert torch.allclose(osp,os[:,perm],rtol=0.0,atol=4e-6)


def test_s54_k_three_seven_255_full_k_probability_mass():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    _activate(op,54060)
    for k in (3,7,255):
        x=_inputs(k=k,q=5,seed=54061)
        logits,_=op.correction_logits_from_state_option(**x)
        assert tuple(logits.shape)==(2,k)
        p=torch.softmax(logits,dim=-1)
        assert float((p.sum(-1)-1.0).abs().max())<=1e-6


def test_s54_treatment_has_no_s53_query_option_context_bypass():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    def _fail(*args,**kwargs):
        raise AssertionError("S53 option_conditioned_query_context must not be called")
    op.option_conditioned_query_context=types.MethodType(_fail,op)
    x=_inputs(k=4,q=5,seed=54070)
    logits,_=op.correction_logits_from_state_option(**x)
    assert tuple(logits.shape)==(2,4)


def test_s54_informative_state_query_option_perturbations_change_context_and_logits():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    _activate(op,54080)
    x=_inputs(k=4,q=5,s=6,t=4,seed=54081)
    a,_,ca,*_=op.correction_logits_from_state_option(**x,return_context=True)

    ys=dict(x)
    ys["state_tokens"]=x["state_tokens"].clone()
    ys["state_tokens"][:,0]+=8.0*x["question_tokens"][:,0]
    bs,_,cs,*_=op.correction_logits_from_state_option(**ys,return_context=True)

    yq=dict(x)
    yq["question_tokens"]=x["question_tokens"].clone()
    yq["question_tokens"][:,0]+=4.0
    bq,_,cq,*_=op.correction_logits_from_state_option(**yq,return_context=True)

    yo=dict(x)
    yo["option_view_tokens"]=x["option_view_tokens"].clone()
    yo["option_view_tokens"][:,0,0,0]+=8.0*x["question_tokens"][:,0]
    bo,_,co,*_=op.correction_logits_from_state_option(**yo,return_context=True)

    assert float((ca-cs).abs().max())>1e-6
    assert float((a-bs).abs().max())>1e-7
    assert float((ca-cq).abs().max())>1e-6
    assert float((a-bq).abs().max())>1e-7
    assert float((ca-co).abs().max())>1e-6
    assert float((a-bo).abs().max())>1e-7


def test_s54_all_masked_state_query_or_option_rejected():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)

    # The inherited query-free identity validator may reject state/option
    # emptiness before the S54 joint operator sees it. The contract requires
    # rejection, not ownership of the exact error string.
    for field,seed in (
        ("state",54090),
        ("query",54091),
        ("option",54092),
    ):
        x=_inputs(k=4,q=5,seed=seed)
        if field=="state":
            x["state_mask"]=torch.zeros_like(x["state_mask"])
        elif field=="query":
            x["question_mask"]=torch.zeros_like(x["question_mask"])
        else:
            x["option_view_token_mask"][:,0]=False
        try:
            op.correction_logits_from_state_option(**x)
        except ValueError:
            pass
        else:
            raise AssertionError(f"S54 accepted all-masked {field}")


def test_s54_deterministic_replay_exact():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    _activate(op,54100)
    x=_inputs(k=7,q=6,seed=54101)
    a=op.correction_logits_from_state_option(**x,return_context=True)
    b=op.correction_logits_from_state_option(**x,return_context=True)
    for xa,xb in zip(a,b):
        assert torch.equal(xa,xb)


def test_s54_joint_operator_detaches_cached_inputs():
    interaction=JointStateQueryOptionLateInteraction()
    x=_inputs(k=4,q=5,seed=54110)
    state=x["state_tokens"].requires_grad_(True)
    option=x["option_view_tokens"].requires_grad_(True)
    query=x["question_tokens"].requires_grad_(True)
    context=interaction(
        state_tokens=state,
        state_mask=x["state_mask"],
        option_view_tokens=option,
        option_view_token_mask=x["option_view_token_mask"],
        option_view_mask=x["option_view_mask"],
        question_tokens=query,
        question_mask=x["question_mask"],
    )
    assert context.requires_grad is False
    assert state.grad is None
    assert option.grad is None
    assert query.grad is None
