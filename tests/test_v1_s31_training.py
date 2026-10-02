import torch

from nmd.v1_s31_training import (
    S31_BINDING_COEFFICIENT,
    S31_GLOBAL_CANONICALIZATION_COEFFICIENT,
    S31_GLOBAL_TEMPERATURE,
    s31_global_relation_block,
)


def test_s31_frozen_coefficients():
    assert S31_BINDING_COEFFICIENT==0.10
    assert S31_GLOBAL_CANONICALIZATION_COEFFICIENT==0.15
    assert S31_GLOBAL_TEMPERATURE==0.10


def test_s31_global_relation_block_gradients():
    torch.manual_seed(4)
    q,k,d=8,4,16
    rc=torch.randn(q,k,requires_grad=True)
    rp=torch.randn(q,k,requires_grad=True)
    sc=torch.randn(q,k,d,requires_grad=True)
    sp=torch.randn(q,k,d,requires_grad=True)
    gold=torch.arange(q,dtype=torch.long)%k
    block,pieces=s31_global_relation_block(rc,rp,sc,sp,gold)
    block.backward()
    assert float(block)>0.0
    assert set(pieces)=={"relation_ce","global_contrastive","global_c2p","global_p2c"}
    for tensor in (rc,rp,sc,sp):
        assert tensor.grad is not None
        assert float(tensor.grad.abs().sum())>0.0
        assert bool(torch.isfinite(tensor.grad).all())
