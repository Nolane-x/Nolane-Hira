import torch
from torch import nn

from nmd.v1_entropic_relation_transport import QueryConditionedEntropicRelationTransport


def _identity(d=4):
    p=nn.Linear(d,d,bias=False)
    with torch.no_grad():
        p.weight.copy_(torch.eye(d))
    return p


def _controlled(question_index=0):
    d=4
    state=torch.tensor([[
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
    ]])
    sm=torch.tensor([[True,True]])
    q=torch.zeros(1,1,d)
    q[0,0,question_index]=1.0
    qm=torch.tensor([[True]])
    options=torch.tensor([[
        [[ [1.,0.,0.,0.],[0.,0.,1.,0.] ]],
        [[ [0.,1.,0.,0.],[0.,0.,0.,1.] ]],
    ]])
    om=torch.ones(1,2,1,2,dtype=torch.bool)
    ovm=torch.ones(1,2,1,dtype=torch.bool)
    return state,sm,q,qm,options,om,ovm


def _run(question_index=0):
    op=QueryConditionedEntropicRelationTransport()
    state,sm,q,qm,options,om,ovm=_controlled(question_index)
    return op.forward_with_components(
        projection=_identity(),
        state_tokens=state,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=options,
        option_view_token_mask=om,
        option_view_mask=ovm,
    )


def test_s34_zero_parameter_and_query_switch():
    op=QueryConditionedEntropicRelationTransport()
    assert op.parameter_count==0
    la,sa,da,ca=_run(0)
    lb,sb,db,cb=_run(1)
    assert la.shape==(1,2)
    assert sa.shape==(1,2,4)
    assert int(la.argmax(-1).item())==0
    assert int(lb.argmax(-1).item())==1
    assert not torch.equal(ca["state_marginal"],cb["state_marginal"])
    assert not torch.equal(ca["option_marginal"],cb["option_marginal"])
    assert not torch.equal(ca["transport_plan"],cb["transport_plan"])
    assert torch.isfinite(sa).all() and torch.isfinite(sb).all()
    assert float(da.max_row_marginal_residual)<1e-3
    assert float(da.max_column_marginal_residual)<1e-5


def test_s34_state_token_permutation_equivariance():
    op=QueryConditionedEntropicRelationTransport()
    state,sm,q,qm,options,om,ovm=_controlled(0)
    p=_identity()
    a=op(
        projection=p,state_tokens=state,state_mask=sm,
        question_tokens=q,question_mask=qm,
        option_view_tokens=options,option_view_token_mask=om,
        option_view_mask=ovm,
    )
    perm=torch.tensor([1,0])
    b=op(
        projection=p,state_tokens=state[:,perm],state_mask=sm[:,perm],
        question_tokens=q,question_mask=qm,
        option_view_tokens=options,option_view_token_mask=om,
        option_view_mask=ovm,
    )
    assert torch.allclose(a[0],b[0],atol=1e-6,rtol=0)
    assert torch.allclose(a[1],b[1],atol=1e-6,rtol=0)


def test_s34_option_token_permutation_equivariance():
    op=QueryConditionedEntropicRelationTransport()
    state,sm,q,qm,options,om,ovm=_controlled(0)
    p=_identity()
    a=op(
        projection=p,state_tokens=state,state_mask=sm,
        question_tokens=q,question_mask=qm,
        option_view_tokens=options,option_view_token_mask=om,
        option_view_mask=ovm,
    )
    perm=torch.tensor([1,0])
    b=op(
        projection=p,state_tokens=state,state_mask=sm,
        question_tokens=q,question_mask=qm,
        option_view_tokens=options[:,:,:,perm],
        option_view_token_mask=om[:,:,:,perm],
        option_view_mask=ovm,
    )
    assert torch.allclose(a[0],b[0],atol=1e-6,rtol=0)
    assert torch.allclose(a[1],b[1],atol=1e-6,rtol=0)


def test_s34_logical_option_permutation_equivariance():
    op=QueryConditionedEntropicRelationTransport()
    state,sm,q,qm,options,om,ovm=_controlled(0)
    p=_identity()
    a=op(
        projection=p,state_tokens=state,state_mask=sm,
        question_tokens=q,question_mask=qm,
        option_view_tokens=options,option_view_token_mask=om,
        option_view_mask=ovm,
    )
    perm=torch.tensor([1,0])
    b=op(
        projection=p,state_tokens=state,state_mask=sm,
        question_tokens=q,question_mask=qm,
        option_view_tokens=options[:,perm],
        option_view_token_mask=om[:,perm],
        option_view_mask=ovm[:,perm],
    )
    assert torch.allclose(a[0][:,perm],b[0],atol=1e-6,rtol=0)
    assert torch.allclose(a[1][:,perm],b[1],atol=1e-6,rtol=0)


def test_s34_masked_padding_invariance_and_degenerate_finite():
    op=QueryConditionedEntropicRelationTransport()
    state,sm,q,qm,options,om,ovm=_controlled(0)
    p=_identity()
    a=op(
        projection=p,state_tokens=state,state_mask=sm,
        question_tokens=q,question_mask=qm,
        option_view_tokens=options,option_view_token_mask=om,
        option_view_mask=ovm,
    )
    padded_state=torch.cat([state,torch.randn(1,2,4)],dim=1)
    padded_sm=torch.cat([sm,torch.zeros(1,2,dtype=torch.bool)],dim=1)
    padded_options=torch.cat([options,torch.randn(1,2,1,2,4)],dim=3)
    padded_om=torch.cat([om,torch.zeros(1,2,1,2,dtype=torch.bool)],dim=3)
    b=op(
        projection=p,state_tokens=padded_state,state_mask=padded_sm,
        question_tokens=q,question_mask=qm,
        option_view_tokens=padded_options,
        option_view_token_mask=padded_om,option_view_mask=ovm,
    )
    assert torch.allclose(a[0],b[0],atol=1e-6,rtol=0)
    assert torch.allclose(a[1],b[1],atol=1e-6,rtol=0)

    one_state=state[:,:1]
    one_sm=sm[:,:1]
    dup_options=options.clone()
    dup_options[:,:,:,:,]=dup_options[:,:,:,0:1,:]
    out=op(
        projection=p,state_tokens=one_state,state_mask=one_sm,
        question_tokens=q,question_mask=qm,
        option_view_tokens=dup_options,option_view_token_mask=om,
        option_view_mask=ovm,
    )
    assert torch.isfinite(out[0]).all()
    assert torch.isfinite(out[1]).all()
