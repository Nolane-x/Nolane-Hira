"""Logical-option pooling for schemas with multiple descriptions / exemplars.

Candidate budgets operate on logical options, never raw prototype rows. This keeps the
high-cardinality cost tied to the user-visible choice space rather than the number of
supporting exemplars stored per choice.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn

@dataclass
class PrototypePoolOutput:
    option_logits: Tensor          # [B,K]
    best_prototype: Tensor         # [B,K], -1 where no prototype
    prototype_counts: Tensor       # [K]

class LogicalOptionPooler(nn.Module):
    def __init__(self, mode: str = "max", temperature: float = 1.0):
        super().__init__()
        if mode not in {"max", "logmeanexp"}:
            raise ValueError("mode must be max or logmeanexp")
        self.mode=mode
        self.log_temperature=nn.Parameter(torch.tensor(float(temperature)).log(),requires_grad=False)

    def forward(self, prototype_logits: Tensor, prototype_to_option: Tensor, num_options: int) -> PrototypePoolOutput:
        if prototype_logits.ndim != 2: raise ValueError("prototype_logits must be [B,P]")
        if prototype_to_option.ndim != 1 or prototype_to_option.numel()!=prototype_logits.shape[1]:
            raise ValueError("prototype_to_option must be [P]")
        B,P=prototype_logits.shape; K=int(num_options)
        if K<1: raise ValueError("num_options must be positive")
        if (prototype_to_option<0).any() or (prototype_to_option>=K).any():
            raise ValueError("prototype_to_option index out of range")
        outs=[]; best=[]; counts=[]
        temp=self.log_temperature.exp().clamp_min(1e-4)
        for k in range(K):
            idx=(prototype_to_option==k).nonzero(as_tuple=True)[0]
            counts.append(int(idx.numel()))
            if idx.numel()==0:
                outs.append(prototype_logits.new_full((B,),-1e4)); best.append(torch.full((B,),-1,dtype=torch.long,device=prototype_logits.device)); continue
            z=prototype_logits[:,idx]
            mx,arg=z.max(-1); best.append(idx[arg])
            if self.mode=="max": val=mx
            else: val=torch.logsumexp(z/temp,-1)*temp - temp*torch.log(torch.tensor(float(idx.numel()),device=z.device,dtype=z.dtype))
            outs.append(val)
        return PrototypePoolOutput(torch.stack(outs,-1),torch.stack(best,-1),torch.tensor(counts,device=prototype_logits.device))
