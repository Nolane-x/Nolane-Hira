from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .v1_private_correction_fork import PrivateCorrectionRepresentationFork
from .v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork


class QueryRelationCanonicalizer(nn.Module):
    """S52 compact residual query relation canonicalizer."""

    def __init__(
        self,
        *,
        dimension: int = 256,
        hidden_dimension: int = 64,
        seed: int = 73052,
    ):
        super().__init__()
        if dimension != 256:
            raise ValueError("S52 query dimension is frozen at 256")
        if hidden_dimension != 64:
            raise ValueError("S52 hidden dimension is frozen at 64")
        if seed != 73052:
            raise ValueError("S52 canonicalizer seed is frozen at 73052")
        self.dimension = int(dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.seed = int(seed)

        g=torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        a=torch.randn(
            self.hidden_dimension,
            self.dimension,
            generator=g,
            dtype=torch.float32,
        )*0.02
        b=torch.zeros(
            self.dimension,
            self.hidden_dimension,
            dtype=torch.float32,
        )
        self.adapter_a=nn.Parameter(a)
        self.adapter_b=nn.Parameter(b)

    @property
    def parameter_count(self)->int:
        return int(self.adapter_a.numel()+self.adapter_b.numel())

    def parameters_list(self)->list[nn.Parameter]:
        return [self.adapter_a,self.adapter_b]

    def forward(self, raw_query: Tensor)->Tensor:
        if raw_query.ndim!=2 or raw_query.shape[-1]!=self.dimension:
            raise ValueError("S52 raw query must be [B,256]")
        if not bool(torch.isfinite(raw_query).all()):
            raise ValueError("S52 raw query contains non-finite values")
        a=self.adapter_a.to(dtype=raw_query.dtype)
        b=self.adapter_b.to(dtype=raw_query.dtype)
        hidden=F.gelu(F.linear(raw_query,a,bias=None))
        residual=F.linear(hidden,b,bias=None)
        code=F.normalize(raw_query+residual,dim=-1,eps=1e-12)
        if not bool(torch.isfinite(code).all()):
            raise ValueError("S52 canonical query code produced non-finite values")
        return code


class CanonicalizedQueryFreeIdentityPrivateCorrectionFork(
    QueryFreeIdentityPrivateCorrectionFork
):
    """S52 query-free option identity + canonicalized query readout."""

    def __init__(
        self,
        *,
        native_dimension: int = 256,
        hidden_dimension: int = 64,
        query_norm_epsilon: float = 1e-12,
        private_norm_epsilon: float = 1e-12,
        residual_scale: float = 1.0,
        adapter_seed: int = 65044,
        train_correction: bool = False,
        pair_temperature: float = 0.10,
        canonicalizer_seed: int = 73052,
    ):
        super().__init__(
            native_dimension=native_dimension,
            hidden_dimension=hidden_dimension,
            query_norm_epsilon=query_norm_epsilon,
            private_norm_epsilon=private_norm_epsilon,
            residual_scale=residual_scale,
            adapter_seed=adapter_seed,
            train_correction=train_correction,
            pair_temperature=pair_temperature,
        )
        self.query_canonicalizer=QueryRelationCanonicalizer(
            dimension=native_dimension,
            hidden_dimension=64,
            seed=canonicalizer_seed,
        )
        if not train_correction:
            for p in self.query_canonicalizer.parameters():
                p.requires_grad_(False)

    @property
    def canonicalizer_parameter_count(self)->int:
        return self.query_canonicalizer.parameter_count

    @property
    def private_trainable_parameter_count(self)->int:
        return self.correction_parameter_count+self.canonicalizer_parameter_count

    def canonicalizer_parameters(self)->list[nn.Parameter]:
        return self.query_canonicalizer.parameters_list()

    def private_parameters(self)->list[nn.Parameter]:
        return self.correction_parameters()+self.canonicalizer_parameters()

    def private_state_dict(self)->dict[str,Tensor]:
        state=self.correction_state_dict()
        state["query_canonicalizer.adapter_a"]=self.query_canonicalizer.adapter_a.detach().cpu().clone()
        state["query_canonicalizer.adapter_b"]=self.query_canonicalizer.adapter_b.detach().cpu().clone()
        return state

    def load_private_state_dict(
        self,
        state: dict[str,Tensor],
        *,
        freeze: bool = False,
    )->None:
        required={
            "adapter_a",
            "adapter_b",
            "bilinear_weight",
            "query_canonicalizer.adapter_a",
            "query_canonicalizer.adapter_b",
        }
        if set(state)!=required:
            raise ValueError("S52 private state keys changed")
        self.load_correction_state_dict(
            {
                "adapter_a":state["adapter_a"],
                "adapter_b":state["adapter_b"],
                "bilinear_weight":state["bilinear_weight"],
            },
            freeze=freeze,
        )
        with torch.no_grad():
            self.query_canonicalizer.adapter_a.copy_(
                state["query_canonicalizer.adapter_a"].to(
                    device=self.query_canonicalizer.adapter_a.device,
                    dtype=self.query_canonicalizer.adapter_a.dtype,
                )
            )
            self.query_canonicalizer.adapter_b.copy_(
                state["query_canonicalizer.adapter_b"].to(
                    device=self.query_canonicalizer.adapter_b.device,
                    dtype=self.query_canonicalizer.adapter_b.dtype,
                )
            )
        for p in self.query_canonicalizer.parameters():
            p.requires_grad_(not freeze)

    def raw_query_summary(
        self,
        *,
        question_tokens: Tensor,
        question_mask: Tensor,
    )->Tensor:
        # Explicitly call the S44 implementation so there is one raw-query
        # source and no bypass around the S52 canonicalizer.
        return PrivateCorrectionRepresentationFork.query_summary(
            self,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )

    def query_summary(
        self,
        *,
        question_tokens: Tensor,
        question_mask: Tensor,
    )->Tensor:
        raw=self.raw_query_summary(
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        return self.query_canonicalizer(raw)

    def query_code_pair(
        self,
        *,
        question_tokens: Tensor,
        question_mask: Tensor,
    )->tuple[Tensor,Tensor]:
        raw=self.raw_query_summary(
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        return raw,self.query_canonicalizer(raw)


def paired_relation_code_terms(
    *,
    a1: Tensor,
    a2: Tensor,
    b1: Tensor,
    b2: Tensor,
    separation_ceiling: float = 0.25,
)->tuple[Tensor,Tensor,Tensor]:
    if separation_ceiling != 0.25:
        raise ValueError("S52 separation ceiling is frozen at 0.25")
    for name,value in (("a1",a1),("a2",a2),("b1",b1),("b2",b2)):
        if value.ndim!=2 or value.shape[-1]!=256:
            raise ValueError(f"S52 {name} code must be [B,256]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"S52 {name} code is non-finite")

    a1n=F.normalize(a1,dim=-1)
    a2n=F.normalize(a2,dim=-1)
    b1n=F.normalize(b1,dim=-1)
    b2n=F.normalize(b2,dim=-1)

    same_a=1.0-F.cosine_similarity(a1n,a2n,dim=-1)
    same_b=1.0-F.cosine_similarity(b1n,b2n,dim=-1)
    same=0.5*(same_a+same_b)

    ac=F.normalize(a1n+a2n,dim=-1)
    bc=F.normalize(b1n+b2n,dim=-1)
    cross=F.cosine_similarity(ac,bc,dim=-1)
    separation=F.relu(cross-separation_ceiling)

    same_loss=same.mean()
    separation_loss=separation.mean()
    total=same_loss+separation_loss
    if not all(bool(torch.isfinite(x)) for x in (same_loss,separation_loss,total)):
        raise ValueError("S52 relation-code auxiliary is non-finite")
    return total,same_loss,separation_loss


def weighted_relation_code_auxiliary(
    *,
    a1: Tensor,
    a2: Tensor,
    b1: Tensor,
    b2: Tensor,
    coefficient: float,
)->tuple[Tensor,dict[str,float]]:
    if coefficient not in (0.0,0.10):
        raise ValueError("S52 auxiliary coefficient must be 0.0 or 0.10")
    total,same,separation=paired_relation_code_terms(
        a1=a1,a2=a2,b1=b1,b2=b2,
    )
    weighted=float(coefficient)*total
    return weighted,{
        "coefficient":float(coefficient),
        "same_relation_loss":float(same.detach()),
        "different_relation_hinge":float(separation.detach()),
        "unweighted_total":float(total.detach()),
    }


__all__=[
    "QueryRelationCanonicalizer",
    "CanonicalizedQueryFreeIdentityPrivateCorrectionFork",
    "paired_relation_code_terms",
    "weighted_relation_code_auxiliary",
]
