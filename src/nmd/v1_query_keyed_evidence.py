from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .w33_coevidence_semantic import CoEvidenceSemanticScorer


class ParameterFreeQueryEvidenceScorer(CoEvidenceSemanticScorer):
    """S1-A0: zero-new-parameter query-keyed state evidence extraction.

    The frozen W28 projection embeds question and state content tokens in the
    same relation space. The mean normalized question vector ranks state tokens
    by cosine similarity. The two most relevant original d_model state tokens
    are then presented to the exact frozen W34 co-evidence scorer.

    This baseline changes *which evidence is scored*, not the score after the
    fact. It adds no parameters.
    """

    evidence_slots = 2

    @property
    def added_parameter_count(self) -> int:
        return 0

    def extract_evidence(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.d_model:
            raise ValueError("state_tokens must be [B,S,D]")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("state_mask mismatch")
        if question_tokens.ndim != 3 or question_tokens.shape[-1] != self.d_model:
            raise ValueError("question_tokens must be [B,Q,D]")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("question_mask mismatch")
        if state_tokens.shape[0] != question_tokens.shape[0]:
            raise ValueError("batch mismatch")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("every state requires at least one content token")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("every question requires at least one content token")

        state_rel = F.normalize(self.projection(state_tokens), dim=-1)
        question_rel = F.normalize(self.projection(question_tokens), dim=-1)
        q_weights = question_mask.to(question_rel.dtype)[..., None]
        q = (question_rel * q_weights).sum(1) / q_weights.sum(1).clamp_min(1)
        q = F.normalize(q, dim=-1)

        relevance = torch.einsum("bsd,bd->bs", state_rel, q)
        relevance = relevance.masked_fill(~state_mask, -1e4)

        k = min(self.evidence_slots, state_tokens.shape[1])
        indices = relevance.topk(k=k, dim=-1).indices
        gather = indices[..., None].expand(-1, -1, self.d_model)
        evidence = state_tokens.gather(1, gather)

        if k < self.evidence_slots:
            # Duplicate the strongest valid slot only for the degenerate
            # one-token state case; this does not re-encode the state.
            evidence = torch.cat(
                [evidence, evidence[:, :1].expand(-1, self.evidence_slots - k, -1)],
                dim=1,
            )
        evidence_mask = torch.ones(
            evidence.shape[:2],
            dtype=torch.bool,
            device=evidence.device,
        )
        return evidence, evidence_mask

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> Tensor:
        evidence, evidence_mask = self.extract_evidence(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        return super().forward(
            state_tokens=evidence,
            state_mask=evidence_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )


class QueryKeyedEvidenceScorer(CoEvidenceSemanticScorer):
    """S1-A: learned two-slot query-keyed evidence extractor.

    Frozen W28/W34 relation geometry is preserved. Only two shared 128->32
    maps are learned:
      - question_heads
      - state_keys

    Each 32-wide output is reshaped to two heads of rank 16. A question head
    attends over state keys, producing two weighted evidence slots in the
    original d_model space. The unchanged W34 co-evidence scorer then scores
    those slots against the dynamic option schema.
    """

    evidence_slots = 2
    head_rank = 16
    extractor_width = evidence_slots * head_rank

    def __init__(self, d_model: int = 256, d_rel: int = 128, rank: int = 8):
        super().__init__(d_model=d_model, d_rel=d_rel, rank=rank)
        self.question_heads = nn.Linear(self.d_rel, self.extractor_width, bias=False)
        self.state_keys = nn.Linear(self.d_rel, self.extractor_width, bias=False)

        # Small initialization keeps the first learned extractor smooth rather
        # than nearly one-hot while still allowing gradients into both maps.
        nn.init.normal_(self.question_heads.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.state_keys.weight, mean=0.0, std=0.01)

    @property
    def extractor_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (self.question_heads, self.state_keys)
            for p in module.parameters()
        )

    @property
    def extractor_trainable_parameter_count(self) -> int:
        return sum(
            p.numel()
            for module in (self.question_heads, self.state_keys)
            for p in module.parameters()
            if p.requires_grad
        )

    @property
    def candidate_parameter_count(self) -> int:
        return super().candidate_parameter_count + self.extractor_parameter_count

    @property
    def candidate_trainable_parameter_count(self) -> int:
        return (
            super().candidate_trainable_parameter_count
            + self.extractor_trainable_parameter_count
        )

    def freeze_extractor(self) -> None:
        for module in (self.question_heads, self.state_keys):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def freeze_candidate(self) -> None:
        super().freeze_candidate()
        self.freeze_extractor()

    def load_extractor_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected = {
            "question_heads.weight": (self.extractor_width, self.d_rel),
            "state_keys.weight": (self.extractor_width, self.d_rel),
        }
        if set(state_dict) != set(expected):
            raise ValueError("Hira v1 S1 extractor checkpoint keys changed")
        own = self.state_dict()
        with torch.no_grad():
            for key, shape in expected.items():
                value = state_dict[key]
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(f"Hira v1 S1 extractor shape mismatch: {key}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Hira v1 S1 extractor non-finite tensor: {key}")
                own[key].copy_(value.to(device=own[key].device, dtype=own[key].dtype))
        if freeze:
            self.freeze_extractor()

    def extract_evidence(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.d_model:
            raise ValueError("state_tokens must be [B,S,D]")
        if state_mask.shape != state_tokens.shape[:2] or state_mask.dtype != torch.bool:
            raise ValueError("state_mask mismatch")
        if question_tokens.ndim != 3 or question_tokens.shape[-1] != self.d_model:
            raise ValueError("question_tokens must be [B,Q,D]")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("question_mask mismatch")
        if state_tokens.shape[0] != question_tokens.shape[0]:
            raise ValueError("batch mismatch")
        if bool((state_mask.sum(-1) < 1).any()):
            raise ValueError("every state requires at least one content token")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("every question requires at least one content token")

        state_rel = F.normalize(self.projection(state_tokens), dim=-1)
        question_rel = F.normalize(self.projection(question_tokens), dim=-1)

        q_low = self.question_heads(question_rel).view(
            question_rel.shape[0],
            question_rel.shape[1],
            self.evidence_slots,
            self.head_rank,
        )
        s_low = self.state_keys(state_rel).view(
            state_rel.shape[0],
            state_rel.shape[1],
            self.evidence_slots,
            self.head_rank,
        )

        q_weights = question_mask.to(q_low.dtype)[..., None, None]
        query_heads = (q_low * q_weights).sum(1) / q_weights.sum(1).clamp_min(1)
        query_heads = F.normalize(query_heads, dim=-1)
        state_keys = F.normalize(s_low, dim=-1)

        # [B,H,S]
        attention_logits = torch.einsum(
            "bhr,bshr->bhs",
            query_heads,
            state_keys,
        ) / sqrt(float(self.head_rank))
        attention_logits = attention_logits.masked_fill(
            ~state_mask[:, None, :],
            -1e4,
        )
        attention = torch.softmax(attention_logits, dim=-1)

        # Two question-specific evidence slots in the original d_model space.
        evidence = torch.einsum("bhs,bsd->bhd", attention, state_tokens)
        evidence_mask = torch.ones(
            evidence.shape[:2],
            dtype=torch.bool,
            device=evidence.device,
        )
        return evidence, evidence_mask, attention

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> Tensor:
        evidence, evidence_mask, _ = self.extract_evidence(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        return super().forward(
            state_tokens=evidence,
            state_mask=evidence_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )


__all__ = [
    "ParameterFreeQueryEvidenceScorer",
    "QueryKeyedEvidenceScorer",
]
