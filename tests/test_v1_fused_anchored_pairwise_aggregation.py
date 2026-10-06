import torch

from nmd.v1_fused_anchored_pairwise_aggregation import (
    S70_AGGREGATION_PARAMETER_COUNT,
    S70_SOFTMAX_TEMPERATURE,
    fused_anchored_pairwise_aggregate,
    fused_anchored_pairwise_weights,
    uniform_pairwise_aggregate,
)


def _antisymmetric_pair(batch:int,k:int,seed:int)->torch.Tensor:
    g=torch.Generator().manual_seed(seed)
    raw=torch.randn(batch,k,k,generator=g)
    pair=0.5*(raw-raw.transpose(-1,-2))
    eye=torch.eye(k,dtype=torch.bool).unsqueeze(0)
    return pair.masked_fill(eye,0.0)


def test_s70_uniform_matches_s59_row_mean_on_valid_pair_matrix():
    pair=_antisymmetric_pair(3,7,7001)
    fused=torch.randn(3,7,generator=torch.Generator().manual_seed(7002))
    got=uniform_pairwise_aggregate(pair,fused)
    expected=pair.sum(dim=-1)/6.0
    assert torch.equal(got,expected)
    assert S70_AGGREGATION_PARAMETER_COUNT==0
    assert S70_SOFTMAX_TEMPERATURE==1.0


def test_s70_uniform_fused_prior_collapses_treatment_to_reference():
    for k in (3,7,255):
        pair=_antisymmetric_pair(2,k,7100+k)
        fused=torch.zeros(2,k)
        ref=uniform_pairwise_aggregate(pair,fused)
        trt=fused_anchored_pairwise_aggregate(pair,fused)
        assert torch.allclose(ref,trt,atol=2e-7,rtol=0)


def test_s70_weights_ignore_diagonal_and_normalize_opponents():
    fused=torch.tensor([[3.0,1.0,-2.0,0.5],[0.0,0.0,0.0,0.0]])
    w=fused_anchored_pairwise_weights(fused)
    assert w.shape==(2,4,4)
    assert torch.allclose(w.sum(dim=-1),torch.ones(2,4),atol=1e-7,rtol=0)
    diag=torch.diagonal(w,dim1=-2,dim2=-1)
    assert torch.equal(diag,torch.zeros_like(diag))
    assert bool((w>=0).all())


def test_s70_pair_diagonal_never_contributes():
    pair=_antisymmetric_pair(2,4,7201)
    fused=torch.randn(2,4,generator=torch.Generator().manual_seed(7202))
    base_ref=uniform_pairwise_aggregate(pair,fused)
    base_trt=fused_anchored_pairwise_aggregate(pair,fused)

    changed=pair.clone()
    for j in range(4):
        changed[:,j,j]=1000.0+float(j)
    ref=uniform_pairwise_aggregate(changed,fused)
    trt=fused_anchored_pairwise_aggregate(changed,fused)

    assert torch.equal(ref,base_ref)
    assert torch.equal(trt,base_trt)


def test_s70_permutation_equivariance():
    g=torch.Generator().manual_seed(7301)
    pair=_antisymmetric_pair(4,7,7302)
    fused=torch.randn(4,7,generator=g)
    perm=torch.randperm(7,generator=g)

    ref=uniform_pairwise_aggregate(pair,fused)
    trt=fused_anchored_pairwise_aggregate(pair,fused)

    pp=pair[:,perm][:,:,perm]
    fp=fused[:,perm]
    refp=uniform_pairwise_aggregate(pp,fp)
    trtp=fused_anchored_pairwise_aggregate(pp,fp)

    assert torch.allclose(refp,ref[:,perm],atol=1e-7,rtol=0)
    assert torch.allclose(trtp,trt[:,perm],atol=2e-7,rtol=0)


def test_s70_detaches_pair_and_fused_inputs():
    pair=_antisymmetric_pair(2,4,7401).requires_grad_(True)
    fused=torch.randn(2,4,generator=torch.Generator().manual_seed(7402),requires_grad=True)

    ref=uniform_pairwise_aggregate(pair,fused)
    trt=fused_anchored_pairwise_aggregate(pair,fused)

    assert not ref.requires_grad
    assert not trt.requires_grad


def test_s70_finite_arbitrary_k():
    for k in (3,7,255):
        pair=_antisymmetric_pair(2,k,7500+k)
        fused=torch.randn(2,k,generator=torch.Generator().manual_seed(7600+k))
        ref=uniform_pairwise_aggregate(pair,fused)
        trt=fused_anchored_pairwise_aggregate(pair,fused)
        assert ref.shape==trt.shape==(2,k)
        assert torch.isfinite(ref).all()
        assert torch.isfinite(trt).all()
        assert float((torch.softmax(ref,-1).sum(-1)-1).abs().max())<=1e-6
        assert float((torch.softmax(trt,-1).sum(-1)-1).abs().max())<=1e-6
