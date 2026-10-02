import torch

from nmd.v1_s32_training import (
    S32_BINDING_COEFFICIENT,
    S32_GLOBAL_CANONICALIZATION_COEFFICIENT,
    S32_GLOBAL_TEMPERATURE,
    s32_all_option_relation_block,
)


def test_s32_frozen_coefficients():
    assert S32_BINDING_COEFFICIENT==0.10
    assert S32_GLOBAL_CANONICALIZATION_COEFFICIENT==0.15
    assert S32_GLOBAL_TEMPERATURE==0.10


def test_s32_relation_block_is_finite_and_differentiable():
    torch.manual_seed(3202)
    q,k,d=6,4,16
    relation_c=torch.randn(q,k,requires_grad=True)
    relation_p=torch.randn(q,k,requires_grad=True)
    sig_c=torch.randn(q,k,d,requires_grad=True)
    sig_p=torch.randn(q,k,d,requires_grad=True)
    gold=torch.arange(q,dtype=torch.long)%k
    block,pieces=s32_all_option_relation_block(
        relation_c,relation_p,sig_c,sig_p,gold
    )
    assert torch.isfinite(block)
    assert set(pieces)=={
        "relation_ce","global_contrastive","global_c2p","global_p2c"
    }
    block.backward()
    assert relation_c.grad is not None
    assert relation_p.grad is not None
    assert sig_c.grad is not None
    assert sig_p.grad is not None
