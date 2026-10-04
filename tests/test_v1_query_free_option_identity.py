import inspect

import torch
import torch.nn.functional as F

from nmd.v1_query_free_option_identity import (
    QueryFreeIdentityPrivateCorrectionFork,
    QueryFreeStateOptionIdentity,
)


def _inputs(*, batch=2, k=7, views=2, s=5, t=4):
    g=torch.Generator().manual_seed(49000+k)
    state=torch.randn(batch,s,256,generator=g)
    state_mask=torch.ones(batch,s,dtype=torch.bool)
    options=torch.randn(batch,k,views,t,256,generator=g)
    option_mask=torch.ones(batch,k,views,t,dtype=torch.bool)
    view_mask=torch.ones(batch,k,views,dtype=torch.bool)
    return state,state_mask,options,option_mask,view_mask


def test_s49_identity_has_zero_parameters_and_no_question_api():
    op=QueryFreeStateOptionIdentity()
    assert op.parameter_count==0
    assert list(op.parameters())==[]
    params=inspect.signature(op.forward).parameters
    assert "question_tokens" not in params
    assert "question_mask" not in params


def test_s49_arbitrary_k_three_seven_255():
    op=QueryFreeStateOptionIdentity()
    for k in (3,7,255):
        st,sm,opt,om,vm=_inputs(k=k)
        out=op(
            state_tokens=st,state_mask=sm,
            option_view_tokens=opt,
            option_view_token_mask=om,
            option_view_mask=vm,
        )
        assert tuple(out.shape)==(2,k,256)
        assert bool(torch.isfinite(out).all())


def test_s49_option_permutation_equivariance_exact_with_tolerance():
    op=QueryFreeStateOptionIdentity()
    st,sm,opt,om,vm=_inputs(batch=2,k=7)
    base=op(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )
    perm=torch.tensor([4,0,6,2,1,5,3])
    moved=op(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt[:,perm],
        option_view_token_mask=om[:,perm],
        option_view_mask=vm[:,perm],
    )
    assert torch.allclose(moved,base[:,perm],rtol=0.0,atol=2e-6)


def test_s49_state_token_permutation_invariance():
    op=QueryFreeStateOptionIdentity()
    st,sm,opt,om,vm=_inputs(batch=2,k=7)
    base=op(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )
    perm=torch.tensor([3,0,4,1,2])
    moved=op(
        state_tokens=st[:,perm],state_mask=sm[:,perm],
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )
    assert torch.allclose(moved,base,rtol=0.0,atol=2e-6)


def test_s49_option_token_permutation_invariance():
    op=QueryFreeStateOptionIdentity()
    st,sm,opt,om,vm=_inputs(batch=2,k=7,t=4)
    base=op(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )
    perm=torch.tensor([2,0,3,1])
    moved=op(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt[:,:,:,perm],
        option_view_token_mask=om[:,:,:,perm],
        option_view_mask=vm,
    )
    assert torch.allclose(moved,base,rtol=0.0,atol=2e-6)


def test_s49_option_view_permutation_invariance():
    op=QueryFreeStateOptionIdentity()
    st,sm,opt,om,vm=_inputs(batch=2,k=7,views=2)
    base=op(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )
    moved=op(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt[:,:,torch.tensor([1,0])],
        option_view_token_mask=om[:,:,torch.tensor([1,0])],
        option_view_mask=vm[:,:,torch.tensor([1,0])],
    )
    assert torch.allclose(moved,base,rtol=0.0,atol=2e-6)


def test_s49_masked_padding_is_invariant():
    op=QueryFreeStateOptionIdentity()
    st,sm,opt,om,vm=_inputs(batch=1,k=4,s=5,t=4)
    sm[:,4]=False
    om[:,:,:,3]=False
    st2=st.clone(); st2[:,4]=1e6
    opt2=opt.clone(); opt2[:,:,:,3]=-1e6
    a=op(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )
    b=op(
        state_tokens=st2,state_mask=sm,
        option_view_tokens=opt2,option_view_token_mask=om,option_view_mask=vm,
    )
    assert torch.allclose(a,b,rtol=0.0,atol=2e-6)


def test_s49_distinct_controlled_options_have_distinct_identity():
    op=QueryFreeStateOptionIdentity()
    state=torch.zeros(1,3,256)
    state[0,0,0]=1.; state[0,1,1]=1.; state[0,2,2]=1.
    sm=torch.ones(1,3,dtype=torch.bool)
    options=torch.zeros(1,3,1,2,256)
    options[0,0,0,0,0]=1.; options[0,0,0,1,3]=1.
    options[0,1,0,0,1]=1.; options[0,1,0,1,4]=1.
    options[0,2,0,0,2]=1.; options[0,2,0,1,5]=1.
    om=torch.ones(1,3,1,2,dtype=torch.bool)
    vm=torch.ones(1,3,1,dtype=torch.bool)
    ids=op(
        state_tokens=state,state_mask=sm,
        option_view_tokens=options,option_view_token_mask=om,option_view_mask=vm,
    )
    sims=torch.matmul(F.normalize(ids,dim=-1),F.normalize(ids,dim=-1).transpose(-1,-2))
    assert float((1.0-sims[0,0,1]).abs())>1e-3
    assert float((1.0-sims[0,0,2]).abs())>1e-3


def test_s49_private_capacity_unchanged_and_raw_query_is_live():
    op=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    assert op.identity_parameter_count==0
    assert op.correction_parameter_count==114688

    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(49101),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(49102),std=0.01)

    st,sm,opt,om,vm=_inputs(batch=1,k=4)
    native=torch.zeros(2,4)
    # repeat state/options to match two relation queries.
    st2=st.repeat_interleave(2,dim=0)
    sm2=sm.repeat_interleave(2,dim=0)
    opt2=opt.repeat_interleave(2,dim=0)
    om2=om.repeat_interleave(2,dim=0)
    vm2=vm.repeat_interleave(2,dim=0)
    q=torch.zeros(2,1,256)
    q[0,0,0]=1.0
    q[1,0,1]=1.0
    qm=torch.ones(2,1,dtype=torch.bool)

    corrected,identity=op.correction_logits_from_state_option(
        native_logits=native,
        state_tokens=st2,state_mask=sm2,
        option_view_tokens=opt2,
        option_view_token_mask=om2,
        option_view_mask=vm2,
        question_tokens=q,question_mask=qm,
    )
    assert torch.equal(identity[0],identity[1])
    assert not torch.equal(corrected[0],corrected[1])


def test_s49_probability_mass_valid():
    op=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(49201),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(49202),std=0.01)
    st,sm,opt,om,vm=_inputs(batch=2,k=7)
    native=torch.randn(2,7,generator=torch.Generator().manual_seed(49203))
    q=torch.randn(2,3,256,generator=torch.Generator().manual_seed(49204))
    qm=torch.ones(2,3,dtype=torch.bool)
    corrected,_=op.correction_logits_from_state_option(
        native_logits=native,
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        question_tokens=q,question_mask=qm,
    )
    probs=torch.softmax(corrected,dim=-1)
    assert float((probs.sum(-1)-1.0).abs().max())<=1e-6


def test_s49_private_correction_accepts_exact_s45_constructor_contract():
    op=QueryFreeIdentityPrivateCorrectionFork(
        native_dimension=256,
        hidden_dimension=64,
        query_norm_epsilon=1e-12,
        private_norm_epsilon=1e-12,
        residual_scale=1.0,
        adapter_seed=65044,
        train_correction=True,
        pair_temperature=0.10,
    )
    assert op.identity_parameter_count==0
    assert op.correction_parameter_count==114688
    assert op.native_dimension==256
    assert op.hidden_dimension==64
    assert op.query_norm_epsilon==1e-12
    assert op.private_norm_epsilon==1e-12
    assert op.residual_scale==1.0
    assert op.adapter_seed==65044
