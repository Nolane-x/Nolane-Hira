from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .v1_s32_global_relation_matrix import (
    global_query_option_relation_contrastive_loss,
)

S32_BINDING_COEFFICIENT = 0.10
S32_GLOBAL_CANONICALIZATION_COEFFICIENT = 0.15
S32_GLOBAL_TEMPERATURE = 0.10


def s32_all_option_relation_block(
    relation_c: Tensor,
    relation_p: Tensor,
    signature_c: Tensor,
    signature_p: Tensor,
    gold: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    if relation_c.ndim != 2 or relation_p.shape != relation_c.shape:
        raise ValueError("S32 relation logits must share [Q,K]")
    if signature_c.ndim != 3 or signature_p.shape != signature_c.shape:
        raise ValueError("S32 relation signatures must share [Q,K,D]")
    if relation_c.shape[:2] != signature_c.shape[:2]:
        raise ValueError("S32 relation logit/signature shape mismatch")
    if gold.ndim != 1 or gold.shape[0] != relation_c.shape[0]:
        raise ValueError("S32 gold shape mismatch")
    if gold.dtype != torch.long:
        raise ValueError("S32 gold must be torch.long")

    relation_ce = 0.5 * (
        F.cross_entropy(relation_c, gold)
        + F.cross_entropy(relation_p, gold)
    )
    global_loss, c2p, p2c = global_query_option_relation_contrastive_loss(
        signature_c,
        signature_p,
        temperature=S32_GLOBAL_TEMPERATURE,
    )
    block = (
        S32_BINDING_COEFFICIENT * relation_ce
        + S32_GLOBAL_CANONICALIZATION_COEFFICIENT * global_loss
    )
    if not bool(torch.isfinite(block)):
        raise ValueError("S32 all-option relation block became non-finite")
    return block, {
        "relation_ce": relation_ce,
        "global_contrastive": global_loss,
        "global_c2p": c2p,
        "global_p2c": p2c,
    }


__all__=[
    "S32_BINDING_COEFFICIENT",
    "S32_GLOBAL_CANONICALIZATION_COEFFICIENT",
    "S32_GLOBAL_TEMPERATURE",
    "s32_all_option_relation_block",
]
