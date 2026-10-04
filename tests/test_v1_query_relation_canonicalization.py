import torch
import torch.nn.functional as F

from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_query_relation_canonicalization import (
    CanonicalizedQueryFreeIdentityPrivateCorrectionFork,
    QueryRelationCanonicalizer,
    paired_relation_code_terms,
    weighted_relation_code_auxiliary,
)


def _query(batch=3,tokens=5,seed=52001):
    g=torch.Generator().manual_seed(seed)
    q=torch.randn(batch,tokens,256,generator=g)
    m=torch.ones(batch,tokens,dtype=torch.bool)
    return q,m


def _state_option(batch=2,k=7,seed=52002):
    g=torch.Generator().manual_seed(seed+k)
    state=torch.randn(batch,6,256,generator=g)
    state_mask=torch.ones(batch,6,dtype=torch.bool)
    option=torch.randn(batch,k,2,4,256,generator=g)
    token_mask=torch.ones(batch,k,2,4,dtype=torch.bool)
    view_mask=torch.ones(batch,k,2,dtype=torch.bool)
    native=torch.randn(batch,k,generator=g)
    return state,state_mask,option,token_mask,view_mask,native


def test_s52_canonicalizer_capacity_and_zero_init_identity():
    op=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    assert op.correction_parameter_count==114688
    assert op.canonicalizer_parameter_count==32768
    assert op.private_trainable_parameter_count==147456
    assert op.identity_parameter_count==0
    q,m=_query()
    raw=op.raw_query_summary(question_tokens=q,question_mask=m)
    code=op.query_summary(question_tokens=q,question_mask=m)
    assert torch.allclose(code,raw,rtol=0.0,atol=1e-7)


def test_s52_reference_treatment_initialization_bit_identical():
    a=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    b=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    for pa,pb in zip(a.private_parameters(),b.private_parameters()):
        assert torch.equal(pa,pb)


def test_s52_canonicalizer_changes_query_without_raw_bypass():
    op=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    q,m=_query(batch=2,seed=52003)
    raw=op.raw_query_summary(question_tokens=q,question_mask=m)
    with torch.no_grad():
        op.query_canonicalizer.adapter_b.normal_(
            generator=torch.Generator().manual_seed(52004),std=0.02
        )
    code=op.query_summary(question_tokens=q,question_mask=m)
    assert float((code-raw).abs().max())>1e-6

    s,sm,o,tm,vm,native=_state_option(batch=2,k=4,seed=52005)
    corrected,identity=op.correction_logits_from_state_option(
        native_logits=native,
        state_tokens=s,
        state_mask=sm,
        option_view_tokens=o,
        option_view_token_mask=tm,
        option_view_mask=vm,
        question_tokens=q,
        question_mask=m,
    )
    assert tuple(corrected.shape)==(2,4)
    assert tuple(identity.shape)==(2,4,256)


def test_s52_k_three_seven_255_full_k_probability_mass():
    op=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(52010),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(52011),std=0.01)
        op.query_canonicalizer.adapter_b.normal_(generator=torch.Generator().manual_seed(52012),std=0.01)

    for k in (3,7,255):
        s,sm,o,tm,vm,native=_state_option(batch=2,k=k,seed=52020)
        q,m=_query(batch=2,seed=52030+k)
        logits,_=op.correction_logits_from_state_option(
            native_logits=native,
            state_tokens=s,
            state_mask=sm,
            option_view_tokens=o,
            option_view_token_mask=tm,
            option_view_mask=vm,
            question_tokens=q,
            question_mask=m,
        )
        assert tuple(logits.shape)==(2,k)
        assert bool(torch.isfinite(logits).all())
        p=torch.softmax(logits,dim=-1)
        assert float((p.sum(-1)-1.0).abs().max())<=1e-6


def test_s52_relation_auxiliary_has_same_and_separation_signal():
    g=torch.Generator().manual_seed(52100)
    base=torch.randn(4,256,generator=g)
    a1=F.normalize(base,dim=-1)
    a2=F.normalize(base+0.25*torch.randn(4,256,generator=g),dim=-1)
    b1=F.normalize(base+0.10*torch.randn(4,256,generator=g),dim=-1)
    b2=F.normalize(base+0.30*torch.randn(4,256,generator=g),dim=-1)
    total,same,sep=paired_relation_code_terms(a1=a1,a2=a2,b1=b1,b2=b2)
    assert bool(torch.isfinite(total))
    assert float(same)>0.0
    assert float(sep)>0.0
    assert float(total)>0.0


def test_s52_reference_auxiliary_is_exact_zero():
    g=torch.Generator().manual_seed(52110)
    q=[F.normalize(torch.randn(3,256,generator=g),dim=-1) for _ in range(4)]
    weighted,diag=weighted_relation_code_auxiliary(
        a1=q[0],a2=q[1],b1=q[2],b2=q[3],coefficient=0.0
    )
    assert float(weighted)==0.0
    assert diag["coefficient"]==0.0


def test_s52_auxiliary_gradient_reaches_only_canonicalizer_not_correction():
    op=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    q1,m1=_query(batch=2,seed=52120)
    q2,m2=_query(batch=2,seed=52121)
    q3,m3=_query(batch=2,seed=52122)
    q4,m4=_query(batch=2,seed=52123)
    c1=op.query_summary(question_tokens=q1,question_mask=m1)
    c2=op.query_summary(question_tokens=q2,question_mask=m2)
    c3=op.query_summary(question_tokens=q3,question_mask=m3)
    c4=op.query_summary(question_tokens=q4,question_mask=m4)
    aux,_=weighted_relation_code_auxiliary(
        a1=c1,a2=c2,b1=c3,b2=c4,coefficient=0.10
    )
    params=op.canonicalizer_parameters()+op.correction_parameters()
    grads=torch.autograd.grad(aux,params,allow_unused=True)
    cg=grads[:2]
    rg=grads[2:]
    assert any(g is not None and float(g.abs().sum())>0 for g in cg)
    assert all(g is None or float(g.abs().sum())==0.0 for g in rg)


def test_s52_distinct_relation_probe_does_not_collapse():
    a=torch.zeros(2,256); b=torch.zeros(2,256)
    a[:,0]=1.0; b[:,1]=1.0
    total,same,sep=paired_relation_code_terms(a1=a,a2=a,b1=b,b2=b)
    assert float(same)==0.0
    assert float(sep)==0.0
    assert float(total)==0.0


def test_s52_raw_query_summary_matches_parent_s44_source():
    op=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    q,m=_query(batch=2,seed=52140)
    raw=op.raw_query_summary(question_tokens=q,question_mask=m)
    parent=PrivateCorrectionRepresentationFork.query_summary(
        op,question_tokens=q,question_mask=m
    )
    assert torch.equal(raw,parent)


def test_s52_private_state_roundtrip_includes_canonicalizer():
    src=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    with torch.no_grad():
        src.adapter_b.normal_(generator=torch.Generator().manual_seed(52200),std=0.01)
        src.bilinear_weight.normal_(generator=torch.Generator().manual_seed(52201),std=0.01)
        src.query_canonicalizer.adapter_b.normal_(generator=torch.Generator().manual_seed(52202),std=0.01)

    state=src.private_state_dict()
    assert set(state)=={
        "adapter_a",
        "adapter_b",
        "bilinear_weight",
        "query_canonicalizer.adapter_a",
        "query_canonicalizer.adapter_b",
    }

    dst=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    dst.load_private_state_dict(state,freeze=True)
    for a,b in zip(src.private_parameters(),dst.private_parameters()):
        assert torch.equal(a.detach().cpu(),b.detach().cpu())
    assert all(not p.requires_grad for p in dst.private_parameters())
