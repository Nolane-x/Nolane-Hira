import torch
import torch.nn.functional as F

from nmd.v1_query_quotient_private_correction import QueryQuotientPrivateCorrectionFork


def _tokens(v):
    return v[:,None,:].clone()


def _mask(batch):
    return torch.ones(batch,1,dtype=torch.bool)


def _structured_signatures(batch=2,k=7):
    s=torch.zeros(batch,k,256)
    for i in range(k):
        s[:,i,0]=float(i)-float(k-1)/2.0
        s[:,i,1]=float((i*i)%5)-2.0
        s[:,i,2]=float((i*3)%7)-3.0
    return F.normalize(s+1e-3,dim=-1)


def test_s48_capacity_exactly_matches_s45():
    op=QueryQuotientPrivateCorrectionFork(train_correction=True)
    assert op.adapter_a_parameter_count==32768
    assert op.adapter_b_parameter_count==16384
    assert op.private_adapter_parameter_count==49152
    assert op.bilinear_parameter_count==65536
    assert op.correction_parameter_count==114688


def test_s48_k_three_seven_255_full_k():
    op=QueryQuotientPrivateCorrectionFork(train_correction=False)
    for k in (3,7,255):
        s=_structured_signatures(batch=2,k=k)
        q=torch.zeros(2,256); q[:,0]=1.0; q[:,1]=0.25
        native=torch.randn(2,k,generator=torch.Generator().manual_seed(48000+k))
        out=op.correction_logits(
            native_logits=native,
            signatures=s,
            question_tokens=_tokens(q),
            question_mask=_mask(2),
        )
        quotient,_=op.query_quotient(
            signatures=s,
            question_tokens=_tokens(q),
            question_mask=_mask(2),
        )
        assert tuple(out.shape)==(2,k)
        assert tuple(quotient.shape)==(2,256)
        assert bool(torch.isfinite(out).all())
        assert bool(torch.isfinite(quotient).all())


def test_s48_option_permutation_leaves_quotient_and_permutes_logits():
    op=QueryQuotientPrivateCorrectionFork(train_correction=False)
    s=_structured_signatures(batch=2,k=7)
    q=torch.zeros(2,256); q[:,0]=1.0; q[:,2]=0.2
    native=torch.randn(2,7,generator=torch.Generator().manual_seed(48111))
    base_q,_=op.query_quotient(
        signatures=s,question_tokens=_tokens(q),question_mask=_mask(2)
    )
    base=op.correction_logits(
        native_logits=native,signatures=s,
        question_tokens=_tokens(q),question_mask=_mask(2),
    )
    perm=torch.tensor([4,0,6,2,1,5,3])
    moved_q,_=op.query_quotient(
        signatures=s[:,perm],question_tokens=_tokens(q),question_mask=_mask(2)
    )
    moved=op.correction_logits(
        native_logits=native[:,perm],signatures=s[:,perm],
        question_tokens=_tokens(q),question_mask=_mask(2),
    )
    assert torch.allclose(moved_q,base_q,rtol=0.0,atol=2e-6)
    assert torch.allclose(moved,base[:,perm],rtol=0.0,atol=2e-6)


def test_s48_orthogonal_query_nuisance_is_quotiented_out():
    op=QueryQuotientPrivateCorrectionFork(train_correction=False)
    s=torch.zeros(1,4,256)
    s[0,:,0]=torch.tensor([-3.,-1.,1.,3.])
    s[0,:,1]=torch.tensor([-1.,2.,-2.,1.])
    q=torch.zeros(1,256); q[0,0]=1.0; q[0,1]=0.5
    nuisance=q.clone(); nuisance[0,200]=9.0
    base,_=op.query_quotient(
        signatures=s,question_tokens=_tokens(q),question_mask=_mask(1)
    )
    moved,_=op.query_quotient(
        signatures=s,question_tokens=_tokens(nuisance),question_mask=_mask(1)
    )
    assert torch.allclose(moved,base,rtol=0.0,atol=2e-6)


def test_s48_relation_relevant_direction_remains_and_distinct_directions_separate():
    op=QueryQuotientPrivateCorrectionFork(train_correction=False)
    s=torch.zeros(1,4,256)
    s[0,:,0]=torch.tensor([-3.,-1.,1.,3.])
    s[0,:,1]=torch.tensor([-2.,2.,-1.,1.])
    q0=torch.zeros(1,256); q0[0,0]=1.0
    q1=torch.zeros(1,256); q1[0,1]=1.0
    z0,_=op.query_quotient(signatures=s,question_tokens=_tokens(q0),question_mask=_mask(1))
    z1,_=op.query_quotient(signatures=s,question_tokens=_tokens(q1),question_mask=_mask(1))
    assert float(z0.norm())>0.99
    assert float(z1.norm())>0.99
    assert float(F.cosine_similarity(z0,z1).abs())<0.99


def test_s48_zero_option_difference_subspace_returns_exact_zero():
    op=QueryQuotientPrivateCorrectionFork(train_correction=False)
    s=torch.ones(2,7,256)
    q=torch.randn(2,256,generator=torch.Generator().manual_seed(48222))
    quotient,_=op.query_quotient(
        signatures=s,question_tokens=_tokens(q),question_mask=_mask(2)
    )
    assert torch.equal(quotient,torch.zeros_like(quotient))
    native=torch.randn(2,7,generator=torch.Generator().manual_seed(48223))
    corrected=op.correction_logits(
        native_logits=native,signatures=s,
        question_tokens=_tokens(q),question_mask=_mask(2),
    )
    assert torch.equal(corrected,native)


def test_s48_treatment_has_no_raw_query_bypass_after_quotient():
    op=QueryQuotientPrivateCorrectionFork(train_correction=True)
    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(48301),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(48302),std=0.01)
    s=torch.zeros(1,4,256)
    s[0,:,0]=torch.tensor([-3.,-1.,1.,3.])
    s[0,:,1]=torch.tensor([-1.,2.,-2.,1.])
    native=torch.randn(1,4,generator=torch.Generator().manual_seed(48303))
    q=torch.zeros(1,256); q[0,0]=1.0; q[0,1]=0.5
    nuisance=q.clone(); nuisance[0,201]=12.0
    base=op.correction_logits(
        native_logits=native,signatures=s,
        question_tokens=_tokens(q),question_mask=_mask(1),
    )
    moved=op.correction_logits(
        native_logits=native,signatures=s,
        question_tokens=_tokens(nuisance),question_mask=_mask(1),
    )
    assert torch.allclose(moved,base,rtol=0.0,atol=2e-6)


def test_s48_probability_mass_valid():
    op=QueryQuotientPrivateCorrectionFork(train_correction=True)
    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(48401),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(48402),std=0.01)
    s=_structured_signatures(batch=4,k=7)
    native=torch.randn(4,7,generator=torch.Generator().manual_seed(48403))
    q=torch.randn(4,256,generator=torch.Generator().manual_seed(48404))
    out=op.correction_logits(
        native_logits=native,signatures=s,
        question_tokens=_tokens(q),question_mask=_mask(4),
    )
    probs=torch.softmax(out,dim=-1)
    assert float((probs.sum(-1)-1.0).abs().max())<=1e-6
