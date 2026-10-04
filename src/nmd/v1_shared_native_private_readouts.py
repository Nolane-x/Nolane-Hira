from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import torch
from torch import Tensor

from .v1_private_correction_fork import PrivateCorrectionRepresentationFork
from .v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork


def _clone_frozen(tensor: Tensor) -> Tensor:
    out=tensor.detach().clone().contiguous()
    out.requires_grad_(False)
    return out


def _tensor_digest(digest, name: str, tensor: Tensor) -> None:
    value=tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())


@dataclass(frozen=True)
class FrozenSharedNativeEvidence:
    case_ids: tuple[str, ...]
    gold: Tensor
    native_logits: Tensor
    native_signatures: Tensor
    state_tokens: Tensor
    state_mask: Tensor
    option_view_tokens: Tensor
    option_view_token_mask: Tensor
    option_view_mask: Tensor
    question_tokens: Tensor
    question_mask: Tensor

    @property
    def batch_size(self) -> int:
        return len(self.case_ids)

    @property
    def k(self) -> int:
        return int(self.native_logits.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.native_signatures.shape[-1])

    def digest(self) -> str:
        d=sha256()
        d.update(b"hira-v1-s50-shared-native-evidence-v1")
        d.update(json.dumps(list(self.case_ids),ensure_ascii=False,separators=(",",":")).encode("utf-8"))
        for name,tensor in (
            ("gold",self.gold),
            ("native_logits",self.native_logits),
            ("native_signatures",self.native_signatures),
            ("state_tokens",self.state_tokens),
            ("state_mask",self.state_mask),
            ("option_view_tokens",self.option_view_tokens),
            ("option_view_token_mask",self.option_view_token_mask),
            ("option_view_mask",self.option_view_mask),
            ("question_tokens",self.question_tokens),
            ("question_mask",self.question_mask),
        ):
            _tensor_digest(d,name,tensor)
        return d.hexdigest()

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.gold,
            self.native_logits,
            self.native_signatures,
            self.state_tokens,
            self.state_mask,
            self.option_view_tokens,
            self.option_view_token_mask,
            self.option_view_mask,
            self.question_tokens,
            self.question_mask,
        )


def freeze_shared_native_evidence(
    *,
    case_ids: tuple[str, ...],
    gold: Tensor,
    native_logits: Tensor,
    native_signatures: Tensor,
    state_tokens: Tensor,
    state_mask: Tensor,
    option_view_tokens: Tensor,
    option_view_token_mask: Tensor,
    option_view_mask: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
) -> FrozenSharedNativeEvidence:
    if native_logits.ndim!=2:
        raise ValueError("S50 native_logits must be [B,K]")
    b,k=native_logits.shape
    if b<1 or k<2:
        raise ValueError("S50 requires B>=1 and K>=2")
    if len(case_ids)!=b or len(set(case_ids))!=b:
        raise ValueError("S50 case IDs must be unique and match batch")
    if gold.shape!=(b,) or gold.dtype!=torch.long:
        raise ValueError("S50 gold must be int64 [B]")
    if bool(((gold<0)|(gold>=k)).any()):
        raise ValueError("S50 gold index out of range")
    if native_signatures.ndim!=3 or native_signatures.shape[:2]!=(b,k):
        raise ValueError("S50 native_signatures must be [B,K,D]")
    d=native_signatures.shape[-1]
    if d!=256:
        raise ValueError("S50 native dimension is frozen at 256")
    if state_tokens.ndim!=3 or state_tokens.shape[0]!=b or state_tokens.shape[-1]!=d:
        raise ValueError("S50 state_tokens must be [B,S,256]")
    if state_mask.shape!=state_tokens.shape[:2] or state_mask.dtype!=torch.bool:
        raise ValueError("S50 state mask mismatch")
    if option_view_tokens.ndim!=5 or option_view_tokens.shape[0]!=b or option_view_tokens.shape[1]!=k or option_view_tokens.shape[-1]!=d:
        raise ValueError("S50 option_view_tokens must be [B,K,V,T,256]")
    if option_view_token_mask.shape!=option_view_tokens.shape[:4] or option_view_token_mask.dtype!=torch.bool:
        raise ValueError("S50 option token mask mismatch")
    if option_view_mask.shape!=option_view_tokens.shape[:3] or option_view_mask.dtype!=torch.bool:
        raise ValueError("S50 option view mask mismatch")
    if question_tokens.ndim!=3 or question_tokens.shape[0]!=b or question_tokens.shape[-1]!=d:
        raise ValueError("S50 question_tokens must be [B,Q,256]")
    if question_mask.shape!=question_tokens.shape[:2] or question_mask.dtype!=torch.bool:
        raise ValueError("S50 question mask mismatch")
    if bool((state_mask.sum(-1)<1).any()):
        raise ValueError("S50 state content missing")
    if bool((question_mask.sum(-1)<1).any()):
        raise ValueError("S50 question content missing")
    if bool((option_view_mask.sum(-1)<1).any()):
        raise ValueError("S50 option views missing")
    if bool(((option_view_token_mask.sum(-1)<1)&option_view_mask).any()):
        raise ValueError("S50 active option view content missing")
    for name,tensor in (
        ("native_logits",native_logits),
        ("native_signatures",native_signatures),
        ("state_tokens",state_tokens),
        ("option_view_tokens",option_view_tokens),
        ("question_tokens",question_tokens),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"S50 {name} contains non-finite values")

    return FrozenSharedNativeEvidence(
        case_ids=tuple(case_ids),
        gold=_clone_frozen(gold),
        native_logits=_clone_frozen(native_logits),
        native_signatures=_clone_frozen(native_signatures),
        state_tokens=_clone_frozen(state_tokens),
        state_mask=_clone_frozen(state_mask),
        option_view_tokens=_clone_frozen(option_view_tokens),
        option_view_token_mask=_clone_frozen(option_view_token_mask),
        option_view_mask=_clone_frozen(option_view_mask),
        question_tokens=_clone_frozen(question_tokens),
        question_mask=_clone_frozen(question_mask),
    )


def reference_private_logits(
    correction: PrivateCorrectionRepresentationFork,
    evidence: FrozenSharedNativeEvidence,
) -> Tensor:
    return correction.correction_logits(
        native_logits=evidence.native_logits,
        signatures=evidence.native_signatures,
        question_tokens=evidence.question_tokens,
        question_mask=evidence.question_mask,
    )


def treatment_private_logits(
    correction: QueryFreeIdentityPrivateCorrectionFork,
    evidence: FrozenSharedNativeEvidence,
) -> tuple[Tensor, Tensor]:
    return correction.correction_logits_from_state_option(
        native_logits=evidence.native_logits,
        state_tokens=evidence.state_tokens,
        state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
        question_tokens=evidence.question_tokens,
        question_mask=evidence.question_mask,
    )


def correction_initialization_exact(
    reference: PrivateCorrectionRepresentationFork,
    treatment: QueryFreeIdentityPrivateCorrectionFork,
) -> bool:
    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    return rs.keys()==ts.keys() and all(torch.equal(rs[k],ts[k]) for k in rs)


__all__=[
    "FrozenSharedNativeEvidence",
    "freeze_shared_native_evidence",
    "reference_private_logits",
    "treatment_private_logits",
    "correction_initialization_exact",
]
