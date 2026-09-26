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
        question_token_embeddings = None
        question_token_mask = None
        question_content_token_mask = None
        option_token_ids = None
        option_content_token_mask = None
        option_view_token_embeddings = None
        option_view_token_mask = None
        option_view_mask = None
        if include_token_artifacts:
            token_batch = self.encoder.encode_texts(
                [question_text, *[option.criterion_text for option in opts]]
            )
            question_token_embeddings = token_batch.token_embeddings[0]
            question_token_mask = token_batch.attention_mask[0].bool()
            option_token_embeddings = token_batch.token_embeddings[1:]
            option_token_mask = token_batch.attention_mask[1:].bool()
            if token_batch.token_ids is not None:
                option_token_ids = token_batch.token_ids[1:]
            if token_batch.special_token_mask is None:
                question_content_token_mask = question_token_mask
                option_content_token_mask = option_token_mask
            else:
                special = token_batch.special_token_mask.bool()
                question_content_token_mask = (
                    question_token_mask & ~special[0]
                )
                option_content_token_mask = (
                    option_token_mask & ~special[1:]
                )

            # Positive multi-view artifacts are additive and backward-compatible:
            # criterion_text remains the legacy single-view artifact above, while
            # criterion/aliases/exemplars are compiled independently here.
            view_texts: list[str] = []
            view_counts: list[int] = []
            for option in opts:
                views = [
                    text
                    for text in (
                        option.criterion_text,
                        *option.aliases,
                        *option.exemplars,
                    )
                    if text and text.strip()
                ]
                if not views:
                    raise ValueError(
                        f"option {option.option_id!r} has no positive semantic view"
                    )
                view_counts.append(len(views))
                view_texts.extend(views)

            view_batch = self.encoder.encode_texts(view_texts)
            view_tokens = view_batch.token_embeddings
            view_attention = view_batch.attention_mask.bool()
            if view_batch.special_token_mask is None:
                view_content = view_attention
            else:
                view_content = (
                    view_attention & ~view_batch.special_token_mask.bool()
                )

            max_views = max(view_counts)
            token_width = view_tokens.shape[1]
            d_model = view_tokens.shape[2]
            option_view_token_embeddings = view_tokens.new_zeros(
                len(opts),
                max_views,
                token_width,
                d_model,
            )
            option_view_token_mask = torch.zeros(
                len(opts),
                max_views,
                token_width,
                dtype=torch.bool,
                device=view_tokens.device,
            )
            option_view_mask = torch.zeros(
                len(opts),
                max_views,
                dtype=torch.bool,
                device=view_tokens.device,
            )

            offset = 0
            for option_index, count in enumerate(view_counts):
                option_view_token_embeddings[
                    option_index, :count
                ] = view_tokens[offset : offset + count]
                option_view_token_mask[
                    option_index, :count
                ] = view_content[offset : offset + count]
                option_view_mask[option_index, :count] = True
                offset += count

            if offset != len(view_texts):
                raise RuntimeError("multi-view schema packing mismatch")

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
            question_token_embeddings=question_token_embeddings,
            question_token_mask=question_token_mask,
            question_content_token_mask=question_content_token_mask,
            option_token_ids=option_token_ids,
            option_content_token_mask=option_content_token_mask,
            option_view_token_embeddings=option_view_token_embeddings,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )
        if use_cache:
            self._cache[cache_key] = compiled
        return compiled, SchemaCompileReceipt(
            key, self.encoder.encoder_hash, False, len(opts), prototype_count
        )

    def clear(self):
        self._cache.clear()
