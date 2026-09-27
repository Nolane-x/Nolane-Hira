from __future__ import annotations

from hashlib import sha256
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .semantic import TextBatch, TextSemanticEncoder

M3_ALIGNMENT_RANK = 16
M3_ALIGNMENT_PARAMETER_COUNT = 8192


class MultilingualAlignmentAdapter(nn.Module):
    """Shared identity-initialized low-rank residual alignment map."""

    def __init__(self, d_model: int = 256, rank: int = M3_ALIGNMENT_RANK):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        if self.d_model != 256:
            raise ValueError("M3 alignment adapter requires d_model=256")
        if self.rank != M3_ALIGNMENT_RANK:
            raise ValueError("M3 R1 freezes rank=16")

        self.down = nn.Linear(self.d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.d_model, bias=False)

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

        if self.candidate_parameter_count != M3_ALIGNMENT_PARAMETER_COUNT:
            raise RuntimeError("M3 alignment parameter count changed")

    @property
    def candidate_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.d_model:
            raise ValueError("M3 alignment input dimension changed")
        normalized = F.layer_norm(values, (self.d_model,))
        residual = self.up(F.gelu(self.down(normalized)))
        return values + residual

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def load_candidate_state_dict(
        self,
        state: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected = {"down.weight", "up.weight"}
        if set(state) != expected:
            raise ValueError("M3 alignment checkpoint keys changed")
        self.load_state_dict(state, strict=True)
        if freeze:
            self.freeze()


class AlignedSemanticEncoder(TextSemanticEncoder):
    """Wrap a frozen semantic encoder with one shared EN/VI alignment adapter."""

    def __init__(
        self,
        base: TextSemanticEncoder,
        adapter: MultilingualAlignmentAdapter | None = None,
        *,
        alignment_identity: str = "m3-r1-untrained",
    ):
        super().__init__()
        if int(base.d_model) != 256:
            raise ValueError("M3 aligned semantic encoder requires base d_model=256")
        self.base = base
        self.adapter = adapter or MultilingualAlignmentAdapter()
        self.d_model = int(base.d_model)
        self.alignment_identity = str(alignment_identity)

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    @property
    def encoder_hash(self) -> str:
        identity = (
            f"{self.base.encoder_hash}:m3-align-r{self.adapter.rank}:"
            f"{self.alignment_identity}"
        )
        return sha256(identity.encode()).hexdigest()

    @property
    def tokenizer_hash(self) -> str:
        return self.base.tokenizer_hash

    @property
    def trainable_parameter_count(self) -> int:
        return self.adapter.trainable_parameter_count

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.adapter.train(mode)
        return self

    def encode_base_texts(self, texts: Sequence[str]) -> TextBatch:
        with torch.no_grad():
            return self.base.encode_texts(texts)

    def encode_texts(self, texts: Sequence[str]) -> TextBatch:
        base = self.base.encode_texts(texts)
        tokens = self.adapter(base.token_embeddings)
        pooled = self.adapter(base.pooled_embeddings)
        return TextBatch(
            token_embeddings=tokens,
            pooled_embeddings=pooled,
            attention_mask=base.attention_mask,
            token_ids=base.token_ids,
            special_token_mask=base.special_token_mask,
        )


def alignment_identity_error(
    encoder: AlignedSemanticEncoder,
    texts: Sequence[str],
) -> float:
    """Maximum exact-init deviation from the frozen base on a text batch."""
    with torch.no_grad():
        base = encoder.base.encode_texts(texts)
        aligned = encoder.encode_texts(texts)
        return max(
            float((base.token_embeddings - aligned.token_embeddings).abs().max()),
            float((base.pooled_embeddings - aligned.pooled_embeddings).abs().max()),
        )


__all__ = [
    "AlignedSemanticEncoder",
    "M3_ALIGNMENT_PARAMETER_COUNT",
    "M3_ALIGNMENT_RANK",
    "MultilingualAlignmentAdapter",
    "alignment_identity_error",
]
