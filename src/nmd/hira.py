from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass
class HIRAOutput:
    logits: Tensor
    probabilities: Tensor
    coarse_logits: Tensor
    selected_indices: Tensor
    tail_mass: Tensor

class HIRACore(nn.Module):
    """State-once relation core. Semantic encoding lives outside this module."""
    def __init__(self,d_model:int=256,d_rel:int=128,n_heads:int=4,budget_buckets=(4,8,16,32,64,128,255),dropout:float=.05):
        super().__init__(); self.d_rel=d_rel; self.budget_buckets=tuple(map(int,budget_buckets))
        self.q_proj=nn.Linear(d_model,d_rel,bias=False); self.seg_proj=nn.Linear(d_model,d_rel,bias=False)
        self.opt_proj=nn.Linear(d_model,d_rel,bias=False); self.token_proj=nn.Linear(d_model,d_rel,bias=False)
        self.type_emb=nn.Embedding(3,d_rel); self.state_ln=nn.LayerNorm(d_rel); self.option_ln=nn.LayerNorm(d_rel)
        self.coarse_scale=nn.Parameter(torch.tensor(10.0))
        self.coarse_bias=nn.Sequential(nn.Linear(d_rel*4,d_rel),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_rel,1))
        self.option_token_weight=nn.Sequential(nn.Linear(d_rel,d_rel//2),nn.GELU(),nn.Linear(d_rel//2,1))
        self.late_scale=nn.Parameter(torch.tensor(3.0))
        self.cross_attn=nn.MultiheadAttention(d_rel,n_heads,dropout=dropout,batch_first=True)
        self.cross_ln=nn.LayerNorm(d_rel)
        self.cross_ff=nn.Sequential(nn.Linear(d_rel,d_rel*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_rel*2,d_rel))
        self.cross_score=nn.Sequential(nn.Linear(d_rel*4,d_rel),nn.GELU(),nn.Linear(d_rel,1))
        self.budget_gate=nn.Sequential(nn.Linear(5+d_rel,d_rel),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_rel,len(self.budget_buckets)))
        self.delta_scale=nn.Embedding(3,1); nn.init.ones_(self.delta_scale.weight)

    @staticmethod
    def _softmax(logits:Tensor,mask:Tensor|None)->Tensor:
        if mask is not None: logits=logits.masked_fill(~mask,-1e4)
        return torch.softmax(logits,-1)

    def forward(self,question:Tensor,segments:Tensor,options:Tensor,qtype:Tensor,option_mask:Tensor|None=None,forced_budget:int|None=None)->HIRAOutput:
        q=self.q_proj(question)+self.type_emb(qtype); seg=self.seg_proj(segments)
        attn=torch.softmax(torch.einsum('bd,bsd->bs',q,seg)/(self.d_rel**.5),-1)
        state=self.state_ln(torch.einsum('bs,bsd->bd',attn,seg)+q)
        o=self.option_ln(self.opt_proj(options)); ctx=F.normalize(q+state,dim=-1); on=F.normalize(o,dim=-1)
        c=ctx[:,None,:].expand_as(o); coarse=torch.einsum('bd,bkd->bk',ctx,on)*self.coarse_scale.clamp(.1,100)
        coarse=coarse+self.coarse_bias(torch.cat([o,c,o*c,(o-c).abs()],-1)).squeeze(-1)
        if option_mask is not None: coarse=coarse.masked_fill(~option_mask,-1e4)
        p0=self._softmax(coarse,option_mask); k=coarse.shape[-1]
        budget=k if forced_budget is None else min(int(forced_budget),k)
        idx=coarse.topk(budget,-1).indices
        logits=coarse.clone()  # R7 bootstrap: relation deltas are wired in the next empirical lane.
        p=self._softmax(logits,option_mask)
        reranked=torch.zeros_like(p,dtype=torch.bool); reranked.scatter_(1,idx,True)
        if option_mask is not None: reranked|=~option_mask
        tail=p.masked_fill(reranked,0).sum(-1)
        return HIRAOutput(logits,p,coarse,idx,tail)

def count_parameters(model:nn.Module)->int:
    return sum(p.numel() for p in model.parameters())
