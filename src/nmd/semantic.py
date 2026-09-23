from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import torch
from torch import Tensor, nn

from .contracts import StateMemory


@dataclass(frozen=True)
class TextBatch:
    token_embeddings: Tensor
    pooled_embeddings: Tensor
    attention_mask: Tensor
    token_ids: Tensor | None = None
    special_token_mask: Tensor | None = None


class TextSemanticEncoder(nn.Module):
    d_model: int

    @property
    def encoder_hash(self) -> str:
        raise NotImplementedError

    @property
    def tokenizer_hash(self) -> str:
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str]) -> TextBatch:
        raise NotImplementedError

    def encode_state(self, text: str, *, segment_tokens: int = 32) -> StateMemory:
        batch = self.encode_texts([text])
        tokens = batch.token_embeddings[0]
        mask = batch.attention_mask[0].bool()
        valid = tokens[mask]
        if valid.shape[0] == 0:
            valid = tokens[:1]
        if batch.special_token_mask is None:
            content = valid
        else:
            special = batch.special_token_mask[0].bool()
            content_mask = mask & ~special
            content = tokens[content_mask]
            if content.shape[0] == 0:
                content = valid
        segments = torch.stack(
            [valid[i:i + segment_tokens].mean(0) for i in range(0, valid.shape[0], segment_tokens)]
        )
        return StateMemory(
            model_hash=self.encoder_hash,
            tokenizer_hash=self.tokenizer_hash,
            state_hash=sha256(text.encode("utf-8")).hexdigest(),
            global_embedding=batch.pooled_embeddings[0],
            segment_embeddings=segments,
            token_embeddings=valid,
            content_token_embeddings=content,
        )


class HFAutoSemanticEncoder(TextSemanticEncoder):
    """Lazy HF adapter for A13/A22. Optional dependency keeps CI download-free."""

    def __init__(self, model, tokenizer, *, revision: str, max_length: int = 512):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.revision = revision
        self.max_length = int(max_length)
        self.d_model = int(model.config.hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        revision: str,
        max_length: int = 512,
        local_files_only: bool = False,
    ) -> "HFAutoSemanticEncoder":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install 'nolane-hira[hf]'") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        model = AutoModel.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        return cls(model, tokenizer, revision=revision, max_length=max_length)

    @property
    def encoder_hash(self) -> str:
        identity = f"{self.model.config._name_or_path}@{self.revision}:{self.d_model}"
        return sha256(identity.encode()).hexdigest()

    @property
    def tokenizer_hash(self) -> str:
        identity = f"{getattr(self.tokenizer, 'name_or_path', 'tokenizer')}@{self.revision}"
        return sha256(identity.encode()).hexdigest()

    def encode_texts(self, texts: Sequence[str]) -> TextBatch:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask", None)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.model(**encoded, return_dict=True)
        tokens = out.last_hidden_state
        mask = encoded["attention_mask"].bool()
        weights = mask.to(tokens.dtype)[..., None]
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
        special_mask = (
            None if special is None else special.to(device=device).bool()
        )
        return TextBatch(
            tokens,
            pooled,
            mask,
            token_ids=encoded.get("input_ids"),
            special_token_mask=special_mask,
        )


class TrainableSemanticEncoder(TextSemanticEncoder):
    """Self-contained trainable encoder for end-to-end tests/experiments, not A13."""

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        max_length: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.vocab_size = vocab_size
        self.state_encode_calls = 0

    @property
    def encoder_hash(self) -> str:
        return sha256(
            f"trainable:{self.vocab_size}:{self.d_model}:{len(self.body.layers)}".encode()
        ).hexdigest()

    @property
    def tokenizer_hash(self) -> str:
        return sha256(f"hash-tokenizer:{self.vocab_size}".encode()).hexdigest()

    def _tokenize(self, text: str) -> list[int]:
        ids = [1]
        for token in text.lower().split():
            h = int.from_bytes(sha256(token.encode()).digest()[:8], "little")
            ids.append(2 + h % (self.vocab_size - 2))
        return ids[: self.max_length]

    def encode_texts(self, texts: Sequence[str]) -> TextBatch:
        rows = [self._tokenize(t) for t in texts]
        width = max(map(len, rows))
        device = self.embedding.weight.device
        ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row, device=device)
            mask[i, :len(row)] = True
        pos = torch.arange(width, device=device)[None, :].expand_as(ids)
        x = self.embedding(ids) + self.position(pos)
        x = self.body(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        weights = mask.to(x.dtype)[..., None]
        pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1)
        special = ids.eq(1) & mask
        return TextBatch(
            x,
            pooled,
            mask,
            token_ids=ids,
            special_token_mask=special,
        )

    def encode_state(self, text: str, *, segment_tokens: int = 32) -> StateMemory:
        self.state_encode_calls += 1
        return super().encode_state(text, segment_tokens=segment_tokens)
