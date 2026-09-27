from __future__ import annotations
from collections import OrderedDict
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


DEFAULT_SCHEMA_CACHE_MAX_ENTRIES = 16
DEFAULT_SCHEMA_CACHE_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SchemaCompileReceipt:
    schema_hash: str
    encoder_hash: str
    cache_hit: bool
    option_count: int
    prototype_count: int
    cache_stored: bool = False
    cache_entry_bytes: int = 0
    cache_entries: int = 0
    cache_bytes: int = 0
    cache_evictions: int = 0


def _compiled_schema_tensor_bytes(schema: CompiledSchema) -> int:
    total = 0
    for value in vars(schema).values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
    return total


class SchemaCompiler:
    """Compile semantic schemas without letting option IDs carry semantic meaning."""

    def __init__(
        self,
        encoder: TextSemanticEncoder,
        *,
        max_cache_entries: int = DEFAULT_SCHEMA_CACHE_MAX_ENTRIES,
        max_cache_bytes: int = DEFAULT_SCHEMA_CACHE_MAX_BYTES,
    ):
        if max_cache_entries < 0:
            raise ValueError("max_cache_entries must be >= 0")
        if max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be >= 0")
        self.encoder = encoder
        self.max_cache_entries = int(max_cache_entries)
        self.max_cache_bytes = int(max_cache_bytes)
        self._cache: OrderedDict[
            tuple[str, bool],
            tuple[CompiledSchema, int],
        ] = OrderedDict()
        self._cache_bytes = 0
        self._cache_evictions = 0

    def cache_info(self) -> dict[str, int]:
        return {
            "entries": len(self._cache),
            "bytes": self._cache_bytes,
            "max_entries": self.max_cache_entries,
            "max_bytes": self.max_cache_bytes,
            "evictions": self._cache_evictions,
        }

    def configure_cache(
        self,
        *,
        max_entries: int,
        max_bytes: int,
        clear: bool = True,
    ) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        if not clear and (
            len(self._cache) > max_entries
            or self._cache_bytes > max_bytes
        ):
            raise ValueError(
                "new cache limits are below current residency; clear is required"
            )
        self.max_cache_entries = int(max_entries)
        self.max_cache_bytes = int(max_bytes)
        if clear:
            self.clear()

    def _store_cache(
        self,
        cache_key: tuple[str, bool],
        compiled: CompiledSchema,
    ) -> tuple[bool, int]:
        entry_bytes = _compiled_schema_tensor_bytes(compiled)
        if (
            self.max_cache_entries == 0
            or self.max_cache_bytes == 0
            or entry_bytes > self.max_cache_bytes
        ):
            return False, entry_bytes

        existing = self._cache.pop(cache_key, None)
        if existing is not None:
            self._cache_bytes -= existing[1]

        while self._cache and (
            len(self._cache) >= self.max_cache_entries
            or self._cache_bytes + entry_bytes > self.max_cache_bytes
        ):
            _old_key, (_old_schema, old_bytes) = self._cache.popitem(last=False)
            self._cache_bytes -= old_bytes
            self._cache_evictions += 1

        self._cache[cache_key] = (compiled, entry_bytes)
        self._cache_bytes += entry_bytes
        return True, entry_bytes

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
            cached, entry_bytes = self._cache[cache_key]
            self._cache.move_to_end(cache_key)
            return cached, SchemaCompileReceipt(
                key,
                self.encoder.encoder_hash,
                True,
                len(opts),
                len(opts),
                cache_stored=True,
                cache_entry_bytes=entry_bytes,
                cache_entries=len(self._cache),
                cache_bytes=self._cache_bytes,
                cache_evictions=self._cache_evictions,
            )

        q = self.encoder.encode_texts([question_text]).pooled_embeddings[0]

        # Compile every positive semantic view in one encoder batch instead of
        # issuing one forward pass per logical option. The mathematical option
        # aggregation contract is unchanged: normalize every prototype, average
        # within each option, then normalize the mean.
        view_texts: list[str] = []
        view_counts: list[int] = []
        for option in opts:
            # IDs are intentionally absent here: they are routing keys, not semantics.
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
                    f"option {option.option_id!r} has no semantic description"
                )
            view_counts.append(len(views))
            view_texts.extend(views)

        prototype_count = len(view_texts)
        view_batch = (
            self.encoder.encode_texts(view_texts)
            if include_token_artifacts
            else None
        )
        prototype_pooled = (
            view_batch.pooled_embeddings
            if view_batch is not None
            else self.encoder.encode_texts(view_texts).pooled_embeddings
        )

        logical = []
        offset = 0
        for count in view_counts:
            proto = prototype_pooled[offset : offset + count]
            emb = F.normalize(proto, dim=-1).mean(0)
            logical.append(F.normalize(emb, dim=-1))
            offset += count
        if offset != prototype_count:
            raise RuntimeError("semantic prototype packing mismatch")

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
            # criterion/aliases/exemplars reuse the already batched semantic view
            # forward pass instead of encoding the same texts a second time.
            if view_batch is None:
                raise RuntimeError("token-artifact compile missing semantic view batch")
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
        cache_stored = False
        entry_bytes = _compiled_schema_tensor_bytes(compiled)
        if use_cache:
            cache_stored, entry_bytes = self._store_cache(cache_key, compiled)
        return compiled, SchemaCompileReceipt(
            key,
            self.encoder.encoder_hash,
            False,
            len(opts),
            prototype_count,
            cache_stored=cache_stored,
            cache_entry_bytes=entry_bytes,
            cache_entries=len(self._cache),
            cache_bytes=self._cache_bytes,
            cache_evictions=self._cache_evictions,
        )

    def clear(self):
        self._cache.clear()
        self._cache_bytes = 0
        self._cache_evictions = 0
