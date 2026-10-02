import torch
from torch import nn

from nmd.v1_native_relation_geometry import NativeA13RelationCanonicalizer
from nmd.v1_relation_canonicalization import CrossViewRelationCanonicalizer


def _fixture(seed=123):
    g=torch.Generator().manual_seed(seed)
    state=torch.randn(2,5,256,generator=g)
    question=torch.randn(2,3,256,generator=g)
    options=torch.randn(2,4,2,4,256,generator=g)
    state_mask=torch.tensor([[1,1,1,1,0],[1,1,1,0,0]],dtype=torch.bool)
    question_mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
    option_token_mask=torch.ones(2,4,2,4,dtype=torch.bool)
    option_token_mask[:, :, 1, -1]=False
    option_view_mask=torch.ones(2,4,2,dtype=torch.bool)
    return state,state_mask,question,question_mask,options,option_token_mask,option_view_mask


def _native(op,data):
    s,sm,q,qm,o,om,ovm=data
    return op(
        state_tokens=s,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=o,
        option_view_token_mask=om,
        option_view_mask=ovm,
    )


def test_s35_zero_parameter_and_signature_width():
    op=NativeA13RelationCanonicalizer()
    logits,sig,_=_native(op,_fixture())
    assert op.parameter_count==0
    assert logits.shape==(2,4)
    assert sig.shape==(2,4,256)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(sig).all()


def test_s35_exact_s13_algorithm_under_identity_projection():
    data=_fixture(456)
    identity=nn.Linear(256,256,bias=False)
    with torch.no_grad():
        identity.weight.copy_(torch.eye(256))
    control=CrossViewRelationCanonicalizer()
    native=NativeA13RelationCanonicalizer()
    s,sm,q,qm,o,om,ovm=data
    a_logits,a_sig,_=control(
        projection=identity,
        state_tokens=s,
        state_mask=sm,
        question_tokens=q,
        question_mask=qm,
        option_view_tokens=o,
        option_view_token_mask=om,
        option_view_mask=ovm,
    )
    b_logits,b_sig,_=_native(native,data)
    assert torch.equal(a_logits,b_logits)
    assert torch.equal(a_sig,b_sig)


def test_s35_state_question_option_token_permutation_equivariance():
    op=NativeA13RelationCanonicalizer()
    data=_fixture(789)
    base_logits,base_sig,_=_native(op,data)
    s,sm,q,qm,o,om,ovm=data

    ps=torch.tensor([3,1,0,4,2])
    pq=torch.tensor([2,0,1])
    pt=torch.tensor([2,0,3,1])
    perm=(
        s[:,ps],sm[:,ps],
        q[:,pq],qm[:,pq],
        o[:,:,:,:, :][:,:,:,pt],om[:,:,:,pt],ovm,
    )
    logits,sig,_=_native(op,perm)
    assert torch.allclose(logits,base_logits,atol=1e-6,rtol=0)
    assert torch.allclose(sig,base_sig,atol=1e-6,rtol=0)


def test_s35_logical_option_permutation_equivariance():
    op=NativeA13RelationCanonicalizer()
    data=_fixture(321)
    base_logits,base_sig,_=_native(op,data)
    s,sm,q,qm,o,om,ovm=data
    p=torch.tensor([2,0,3,1])
    logits,sig,_=_native(op,(s,sm,q,qm,o[:,p],om[:,p],ovm[:,p]))
    assert torch.allclose(logits,base_logits[:,p],atol=1e-6,rtol=0)
    assert torch.allclose(sig,base_sig[:,p],atol=1e-6,rtol=0)


def test_s35_masked_padding_invariance():
    op=NativeA13RelationCanonicalizer()
    data=_fixture(654)
    base_logits,base_sig,_=_native(op,data)
    s,sm,q,qm,o,om,ovm=data
    s2=torch.cat([s,torch.randn(2,2,256)],dim=1)
    sm2=torch.cat([sm,torch.zeros(2,2,dtype=torch.bool)],dim=1)
    q2=torch.cat([q,torch.randn(2,2,256)],dim=1)
    qm2=torch.cat([qm,torch.zeros(2,2,dtype=torch.bool)],dim=1)
    o2=torch.cat([o,torch.randn(2,4,2,2,256)],dim=3)
    om2=torch.cat([om,torch.zeros(2,4,2,2,dtype=torch.bool)],dim=3)
    logits,sig,_=_native(op,(s2,sm2,q2,qm2,o2,om2,ovm))
    assert torch.allclose(logits,base_logits,atol=1e-6,rtol=0)
    assert torch.allclose(sig,base_sig,atol=1e-6,rtol=0)


def test_s35_degenerate_native_geometry_finite():
    op=NativeA13RelationCanonicalizer()
    x=torch.ones(1,1,256)
    q=torch.ones(1,1,256)
    o=torch.ones(1,2,1,1,256)
    logits,sig,_=op(
        state_tokens=x,
        state_mask=torch.ones(1,1,dtype=torch.bool),
        question_tokens=q,
        question_mask=torch.ones(1,1,dtype=torch.bool),
        option_view_tokens=o,
        option_view_token_mask=torch.ones(1,2,1,1,dtype=torch.bool),
        option_view_mask=torch.ones(1,2,1,dtype=torch.bool),
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(sig).all()
