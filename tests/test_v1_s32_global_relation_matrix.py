import torch

from nmd.v1_s31_global_relation_contrastive import (
    global_cross_case_relation_contrastive_loss,
)
from nmd.v1_s32_global_relation_matrix import (
    global_query_option_relation_contrastive_loss,
)


def _controlled():
    q,k,d=4,4,16
    c=torch.zeros(q,k,d)
    p=torch.zeros(q,k,d)
    for qi in range(q):
        for ki in range(k):
            c[qi,ki,qi*k+ki]=1.0
            p[qi,ki,qi*k+ki]=1.0
    gold=torch.zeros(q,dtype=torch.long)
    return c,p,gold


def test_s32_non_gold_perturbation_distinguishes_gold_only_from_all_option():
    c,p,gold=_controlled()
    gold_good,_,_=global_cross_case_relation_contrastive_loss(
        c,p,gold,temperature=0.10
    )
    all_good,_,_=global_query_option_relation_contrastive_loss(
        c,p,temperature=0.10
    )

    bad=p.clone()
    # Only a non-gold relation is mis-paired. Gold index is 0 for every query.
    bad[0,1]=p[0,2]

    gold_bad,_,_=global_cross_case_relation_contrastive_loss(
        c,bad,gold,temperature=0.10
    )
    all_bad,_,_=global_query_option_relation_contrastive_loss(
        c,bad,temperature=0.10
    )

    assert torch.equal(gold_good,gold_bad)
    assert float(all_bad-all_good)>=0.10


def test_s32_all_option_matched_query_option_permutation_and_view_swap():
    c,p,_gold=_controlled()
    base,_,_=global_query_option_relation_contrastive_loss(c,p,temperature=0.10)

    qperm=torch.tensor([2,0,3,1])
    kperm=torch.tensor([3,1,0,2])
    permuted,_,_=global_query_option_relation_contrastive_loss(
        c[qperm][:,kperm],
        p[qperm][:,kperm],
        temperature=0.10,
    )
    swapped,_,_=global_query_option_relation_contrastive_loss(
        p,c,temperature=0.10
    )

    assert torch.allclose(base,permuted,atol=1e-7,rtol=0)
    assert torch.allclose(base,swapped,atol=1e-7,rtol=0)


def test_s32_all_option_non_gold_gradients_are_live():
    torch.manual_seed(3201)
    q,k,d=5,4,24
    c=torch.randn(q,k,d,requires_grad=True)
    p=torch.randn(q,k,d,requires_grad=True)
    gold=torch.zeros(q,dtype=torch.long)
    loss,_,_=global_query_option_relation_contrastive_loss(c,p,temperature=0.10)
    loss.backward()

    assert c.grad is not None and p.grad is not None
    mask=torch.ones(q,k,dtype=torch.bool)
    mask[torch.arange(q),gold]=False
    c_non=float(c.grad[mask].abs().sum())
    p_non=float(p.grad[mask].abs().sum())
    assert c_non>0.0
    assert p_non>0.0
    assert torch.isfinite(c.grad).all()
    assert torch.isfinite(p.grad).all()


def test_s32_rejects_invalid_shapes_and_temperature():
    c=torch.randn(4,4,8)
    p=torch.randn(4,4,8)
    try:
        global_query_option_relation_contrastive_loss(c,p[:3],temperature=0.10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected shape rejection")

    try:
        global_query_option_relation_contrastive_loss(c,p,temperature=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected temperature rejection")
