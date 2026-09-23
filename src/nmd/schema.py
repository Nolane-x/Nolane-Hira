from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable

import torch
import torch.nn.functional as F

from .contracts import CompiledSchema, LogicalOption, Primitive
from .semantic import TextSemanticEncoder


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SchemaCompileReceipt:
    schema_hash: str
    encoder_hash: str
    cache_hit: bool
    option_count: int
    prototype_count: int


class SchemaCompiler:
    """Compile semantic schemas without letting option IDs carry semantic meaning."""

    def __init__(self, encoder: TextSemanticEncoder):
        self.encoder = encoder
        self._cache: dict[tuple[str, bool], CompiledSchema] = {}

    def _payload(self, primitive: Primitive, question_text: str, options: tuple[LogicalOption, ...]):
        return {
            "primitive": primitive,
            "question_text": question_text,
            "options": [{
                "option_id": o.option_id,
                "criterion_text": o.criterion_text,
                "aliases": list(o.aliases),
                "exemplars": list(o.exemplars),
                "counterexamples": list(o.counterexamples),
                "value": o.value,
            } for o in options],
        }

    def schema_hash(self, primitive: Primitive, question_text: str, options: tuple[LogicalOption, ...]):
        return _canonical_hash({
            "encoder_hash": self.encoder.encoder_hash,
            "payload": self._payload(primitive, question_text, options),
        })

    def compile(
        self, *, primitive: Primitive, question_text: str,
        options: Iterable[LogicalOption], use_cache: bool = True,
        include_token_artifacts: bool = False,
    ) -> tuple[CompiledSchema, SchemaCompileReceipt]:
        opts = tuple(options)
        if len(opts) < 2:
            raise ValueError("a decision schema needs at least two logical options")
        if len({o.option_id for o in opts}) != len(opts):
            raise ValueError("option_id values must be unique")

        # Cached tensors are inference artifacts. Reusing an autograd graph across
        # training steps is unsafe and would also make encoder updates stale.
        if self.encoder.training:
            use_cache = False

        key = self.schema_hash(primitive, question_text, opts)
        cache_key = (key, bool(include_token_artifacts))
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached, SchemaCompileReceipt(
                key, self.encoder.encoder_hash, True, len(opts), len(opts)
            )

        q = self.encoder.encode_texts([question_text]).pooled_embeddings[0]
        logical = []
        prototype_count = 0
        for option in opts:
            # IDs are intentionally absent here: they are routing keys, not semantics.
            texts = [option.criterion_text, *option.aliases, *option.exemplars]
            texts = [t for t in texts if t and t.strip()]
            if not texts:
                raise ValueError(f"option {option.option_id!r} has no semantic description")
            proto = self.encoder.encode_texts(texts).pooled_embeddings
            prototype_count += proto.shape[0]
            emb = F.normalize(proto, dim=-1).mean(0)
            logical.append(F.normalize(emb, dim=-1))

        option_token_embeddings = None
        option_token_mask = None
        if include_token_artifacts:
            criterion_batch = self.encoder.encode_texts(
                [option.criterion_text for option in opts]
            )
            option_token_embeddings = criterion_batch.token_embeddings
            option_token_mask = criterion_batch.attention_mask.bool()

        compiled = CompiledSchema(
            schema_hash=key,
            encoder_hash=self.encoder.encoder_hash,
            primitive=primitive,
            question_text=question_text,
            options=opts,
            question_embedding=q,
            option_embeddings=torch.stack(logical),
            option_token_embeddings=option_token_embeddings,
            option_token_mask=option_token_mask,
        )
        if use_cache:
            self._cache[cache_key] = compiled
        return compiled, SchemaCompileReceipt(
            key, self.encoder.encoder_hash, False, len(opts), prototype_count
        )

    def clear(self):
        self._cache.clear()
