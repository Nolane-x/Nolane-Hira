import torch

from nmd.v1_ordinal_pairwise_consensus import OrdinalPairwiseConsensus


def _inputs(*, batch=5, k=7):
    g=torch.Generator().manual_seed(47000+k)
    return (
        torch.randn(batch,k,generator=g),
        torch.randn(batch,k,generator=g),
        torch.randn(batch,k,generator=g),
    )


def test_s47_zero_parameter_surface():
    op=OrdinalPairwiseConsensus()
    assert op.parameter_count==0
    assert list(op.parameters())==[]


def test_s47_arbitrary_k_three_seven_255():
    op=OrdinalPairwiseConsensus()
    for k in (3,7,255):
        p,n,c=_inputs(k=k)
        fused,diag=op(p,n,c)
        assert tuple(fused.shape)==(5,k)
        assert bool(torch.isfinite(fused).all())
        d=diag.to_dict()
        assert all(0.0<=v<=1.0 for v in d.values())


def test_s47_option_permutation_equivariance_exact():
    op=OrdinalPairwiseConsensus()
    p,n,c=_inputs(batch=4,k=7)
    base,_=op(p,n,c)
    perm=torch.tensor([4,0,6,2,1,5,3])
    moved,_=op(p[:,perm],n[:,perm],c[:,perm])
    assert torch.equal(moved,base[:,perm])


def test_s47_independent_positive_affine_invariance_exact():
    op=OrdinalPairwiseConsensus()
    p,n,c=_inputs(batch=4,k=7)
    base,_=op(p,n,c)
    moved,_=op(3.25*p+17.0,0.75*n-9.0,8.5*c+101.0)
    assert torch.equal(moved,base)


def test_s47_strictly_increasing_nonlinear_rank_transform_invariance():
    op=OrdinalPairwiseConsensus()
    p,n,c=_inputs(batch=4,k=7)
    base,_=op(p,n,c)
    moved,_=op(p.pow(3),n.pow(3),c.pow(3))
    assert torch.equal(moved,base)


def test_s47_extreme_magnitude_blowup_does_not_change_decision():
    op=OrdinalPairwiseConsensus()
    p,n,c=_inputs(batch=4,k=7)
    base,_=op(p,n,c)
    moved,_=op(p,n*1e20,c)
    assert torch.equal(moved,base)


def test_s47_two_identical_experts_dominate_one_adversarial_expert():
    op=OrdinalPairwiseConsensus()
    agreed=torch.tensor([[7.,6.,5.,4.,3.,2.,1.]])
    adversarial=-agreed
    fused,_=op(agreed,agreed,adversarial)
    assert int(fused.argmax(-1).item())==0
    copeland,private,votes=op.components(agreed,agreed,adversarial)
    agreed_pair=torch.sign(agreed[:,:,None]-agreed[:,None,:])
    assert torch.equal(torch.sign(votes),agreed_pair)
    assert int(copeland.argmax(-1).item())==0


def test_s47_private_tiebreak_cannot_overturn_strict_copeland_advantage():
    op=OrdinalPairwiseConsensus()
    p,n,c=_inputs(batch=10,k=7)
    copeland,private,_=op.components(p,n,c)
    fused,_=op(p,n,c)
    base=13
    for b in range(fused.shape[0]):
        for i in range(7):
            for j in range(7):
                if float(copeland[b,i]-copeland[b,j])>=1.0:
                    assert float(fused[b,i]-fused[b,j])>0.0
                    assert abs(float(private[b,i]-private[b,j]))<=12.0
    assert base==2*7-1


def test_s47_private_ordinal_tiebreak_is_deterministic():
    op=OrdinalPairwiseConsensus()
    # Majority cycle gives equal Copeland scores; corrected order resolves it.
    p=torch.tensor([[3.,2.,1.]])
    n=torch.tensor([[1.,3.,2.]])
    c=torch.tensor([[2.,1.,3.]])
    copeland,private,_=op.components(p,n,c)
    assert int(torch.unique(copeland).numel())==1
    fused_a,diag_a=op(p,n,c)
    fused_b,diag_b=op(p,n,c)
    assert torch.equal(fused_a,fused_b)
    assert int(fused_a.argmax(-1).item())==int(private.argmax(-1).item())
    assert diag_a.to_dict()==diag_b.to_dict()


def test_s47_probability_mass_is_valid():
    op=OrdinalPairwiseConsensus()
    p,n,c=_inputs(batch=6,k=7)
    fused,_=op(p,n,c)
    probs=torch.softmax(fused,dim=-1)
    assert float((probs.sum(-1)-1.0).abs().max())<=1e-6
