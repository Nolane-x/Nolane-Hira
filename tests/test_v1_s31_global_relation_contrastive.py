import pytest
import torch
import torch.nn.functional as F

from nmd.v1_s31_global_relation_contrastive import (
    global_cross_case_relation_contrastive_loss,
)


def _orthogonal(q=4,k=4,d=8):
    canonical=torch.zeros(q,k,d)
    paraphrase=torch.zeros(q,k,d)
    gold=torch.arange(q,dtype=torch.long)%k
    for i in range(q):
        canonical[i,gold[i],i]=1.0
        paraphrase[i,gold[i],i]=1.0
        # Fill wrong options with finite nonzero signatures.
        for j in range(k):
            if j!=gold[i]:
                canonical[i,j,(i+j+1)%d]=1.0
                paraphrase[i,j,(i+j+2)%d]=1.0
    return canonical,paraphrase,gold


def test_s31_matched_orthogonal_pairs_have_low_loss():
    c,p,g=_orthogonal()
    total,c2p,p2c=global_cross_case_relation_contrastive_loss(
        c,p,g,temperature=0.10
    )
    assert float(total)<0.001
    assert torch.equal(c2p,p2c)


def test_s31_shuffled_positive_pairing_is_materially_worse():
    c,p,g=_orthogonal()
    good,_,_=global_cross_case_relation_contrastive_loss(c,p,g)
    shuffled=p.roll(1,dims=0)
    bad,_,_=global_cross_case_relation_contrastive_loss(c,shuffled,g)
    assert float(bad)>float(good)+1.0


def test_s31_matched_query_permutation_invariant():
    c,p,g=_orthogonal()
    base,_,_=global_cross_case_relation_contrastive_loss(c,p,g)
    perm=torch.tensor([2,0,3,1])
    got,_,_=global_cross_case_relation_contrastive_loss(
        c[perm],p[perm],g[perm]
    )
    assert torch.allclose(base,got,atol=1e-7,rtol=0)


def test_s31_view_swap_symmetric():
    c,p,g=_orthogonal()
    a,_,_=global_cross_case_relation_contrastive_loss(c,p,g)
    b,_,_=global_cross_case_relation_contrastive_loss(p,c,g)
    assert torch.allclose(a,b,atol=1e-7,rtol=0)


def test_s31_gradients_reach_both_views():
    torch.manual_seed(0)
    c=torch.randn(6,4,12,requires_grad=True)
    p=torch.randn(6,4,12,requires_grad=True)
    g=torch.tensor([0,1,2,3,0,1],dtype=torch.long)
    loss,_,_=global_cross_case_relation_contrastive_loss(c,p,g)
    loss.backward()
    assert c.grad is not None and float(c.grad.abs().sum())>0.0
    assert p.grad is not None and float(p.grad.abs().sum())>0.0
    assert bool(torch.isfinite(c.grad).all())
    assert bool(torch.isfinite(p.grad).all())


def test_s31_requires_multiple_queries():
    c=torch.randn(1,4,8)
    p=torch.randn(1,4,8)
    g=torch.tensor([0])
    with pytest.raises(ValueError,match="Q>=2"):
        global_cross_case_relation_contrastive_loss(c,p,g)
