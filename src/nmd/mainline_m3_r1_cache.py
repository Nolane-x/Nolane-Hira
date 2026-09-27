from __future__ import annotations

from hashlib import sha256
from typing import Sequence

import torch
from torch import Tensor

from .contracts import CompiledSchema, StateMemory
from .mainline_m3_alignment import MultilingualAlignmentAdapter
from .mainline_m3_authority import M3PairedCase
from .schema import SchemaCompiler
from .semantic import TextSemanticEncoder

M3_R1_CACHE_SCHEMA = "hira-v0-mainline-m3-r1-base-cache-v1"


def _clone_tensor(value: Tensor | None) -> Tensor | None:
    if value is None:
        return None
    return value.detach().cpu().clone()


def _clone_memory(memory: StateMemory) -> StateMemory:
    return StateMemory(
        model_hash=memory.model_hash,
        tokenizer_hash=memory.tokenizer_hash,
        state_hash=memory.state_hash,
        global_embedding=_clone_tensor(memory.global_embedding),
        segment_embeddings=_clone_tensor(memory.segment_embeddings),
        token_embeddings=_clone_tensor(memory.token_embeddings),
        content_token_embeddings=_clone_tensor(memory.content_token_embeddings),
    )


def _clone_schema(schema: CompiledSchema) -> CompiledSchema:
    return CompiledSchema(
        schema_hash=schema.schema_hash,
        encoder_hash=schema.encoder_hash,
        primitive=schema.primitive,
        question_text=schema.question_text,
        options=schema.options,
        question_embedding=_clone_tensor(schema.question_embedding),
        option_embeddings=_clone_tensor(schema.option_embeddings),
        option_token_embeddings=_clone_tensor(schema.option_token_embeddings),
        option_token_mask=_clone_tensor(schema.option_token_mask),
        question_token_embeddings=_clone_tensor(schema.question_token_embeddings),
        question_token_mask=_clone_tensor(schema.question_token_mask),
        question_content_token_mask=_clone_tensor(schema.question_content_token_mask),
        option_token_ids=_clone_tensor(schema.option_token_ids),
        option_content_token_mask=_clone_tensor(schema.option_content_token_mask),
        option_view_token_embeddings=_clone_tensor(schema.option_view_token_embeddings),
        option_view_token_mask=_clone_tensor(schema.option_view_token_mask),
        option_view_mask=_clone_tensor(schema.option_view_mask),
    )


@torch.no_grad()
def compile_m3_r1_base_cache(
    encoder: TextSemanticEncoder,
    rows: Sequence[M3PairedCase],
    *,
    partition: str,
) -> dict[str, object]:
    if not rows:
        raise ValueError("M3-R1 cache requires paired rows")
    if any(row.partition != partition for row in rows):
        raise ValueError("M3-R1 cache partition mismatch")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("M3-R1 base cache requires frozen semantic encoder")

    encoder.eval()
    compiler = SchemaCompiler(encoder)
    pairs: list[dict[str, object]] = []

    for row in rows:
        record = {
            "pair_id": row.pair_id,
            "domain_id": row.domain_id,
            "partition": row.partition,
            "primitive": row.primitive,
            "gold_index": row.gold_index,
        }
        for language in ("en", "vi"):
            state_text = getattr(row, f"{language}_state_text")
            question_text = getattr(row, f"{language}_question_text")
            options = getattr(row, f"{language}_options")

            memory = encoder.encode_state(state_text)
            schema, _ = compiler.compile(
                primitive=row.primitive,
                question_text=question_text,
                options=options,
                use_cache=False,
                include_token_artifacts=True,
            )
            record[language] = {
                "state_text": state_text,
                "question_text": question_text,
                "memory": _clone_memory(memory),
                "schema": _clone_schema(schema),
            }
        pairs.append(record)

    cache = {
        "metadata": {
            "schema_version": M3_R1_CACHE_SCHEMA,
            "partition": partition,
            "pair_count": len(pairs),
            "language_case_count": 2 * len(pairs),
            "encoder_hash": encoder.encoder_hash,
            "tokenizer_hash": encoder.tokenizer_hash,
            "base_encoder_frozen": True,
            "gradient_updates_used": False,
            "state_encode_count": 2 * len(pairs),
            "state_encodes_per_language_case": 1.0,
        },
        "pairs": pairs,
    }
    validate_m3_r1_base_cache(cache, expected_partition=partition)
    return cache


def validate_m3_r1_base_cache(
    cache: dict[str, object],
    *,
    expected_partition: str | None = None,
) -> None:
    metadata = cache.get("metadata")
    pairs = cache.get("pairs")
    if not isinstance(metadata, dict) or not isinstance(pairs, list):
        raise ValueError("M3-R1 cache requires metadata and pairs")
    if metadata.get("schema_version") != M3_R1_CACHE_SCHEMA:
        raise ValueError("unexpected M3-R1 cache schema")
    if expected_partition is not None and metadata.get("partition") != expected_partition:
        raise ValueError("M3-R1 cache partition mismatch")
    if int(metadata.get("pair_count", -1)) != len(pairs):
        raise ValueError("M3-R1 cache pair count mismatch")
    if int(metadata.get("language_case_count", -1)) != 2 * len(pairs):
        raise ValueError("M3-R1 cache language case count mismatch")
    if metadata.get("base_encoder_frozen") is not True:
        raise ValueError("M3-R1 cache base encoder must be frozen")
    if metadata.get("gradient_updates_used") is not False:
        raise ValueError("M3-R1 base cache cannot contain trained encoder outputs")
    if int(metadata.get("state_encode_count", -1)) != 2 * len(pairs):
        raise ValueError("M3-R1 cache state encode count changed")
    if float(metadata.get("state_encodes_per_language_case", -1.0)) != 1.0:
        raise ValueError("M3-R1 cache state-once contract changed")

    seen = set()
    for pair in pairs:
        pair_id = str(pair.get("pair_id", ""))
        if not pair_id or pair_id in seen:
            raise ValueError("M3-R1 cache duplicate pair id")
        seen.add(pair_id)
        gold = int(pair["gold_index"])

        option_ids = []
        option_values = []
        for language in ("en", "vi"):
            payload = pair.get(language)
            if not isinstance(payload, dict):
                raise ValueError("M3-R1 cache missing language payload")
            memory = payload.get("memory")
            schema = payload.get("schema")
            if not isinstance(memory, StateMemory) or not isinstance(schema, CompiledSchema):
                raise ValueError("M3-R1 cache contains invalid compiled artifacts")
            if memory.model_hash != metadata.get("encoder_hash"):
                raise ValueError("M3-R1 cached state encoder identity changed")
            if schema.encoder_hash != metadata.get("encoder_hash"):
                raise ValueError("M3-R1 cached schema encoder identity changed")
            if memory.tokenizer_hash != metadata.get("tokenizer_hash"):
                raise ValueError("M3-R1 cached tokenizer identity changed")
            if not 0 <= gold < len(schema.options):
                raise ValueError("M3-R1 cached gold index invalid")
            if memory.content_token_embeddings is None:
                raise ValueError("M3-R1 cache requires state content tokens")
            if schema.option_view_token_embeddings is None:
                raise ValueError("M3-R1 cache requires option view tokens")
            if schema.option_view_token_mask is None or schema.option_view_mask is None:
                raise ValueError("M3-R1 cache option view masks missing")
            option_ids.append(tuple(option.option_id for option in schema.options))
            option_values.append(tuple(option.value for option in schema.options))

        if option_ids[0] != option_ids[1]:
            raise ValueError("M3-R1 cached EN/VI option IDs changed")
        if option_values[0] != option_values[1]:
            raise ValueError("M3-R1 cached EN/VI typed values changed")


def _aligned_hash(base_hash: str, alignment_identity: str) -> str:
    return sha256(
        f"{base_hash}:m3-align-r16:{alignment_identity}".encode()
    ).hexdigest()


def align_state_memory(
    memory: StateMemory,
    adapter: MultilingualAlignmentAdapter,
    *,
    alignment_identity: str,
) -> StateMemory:
    model_hash = _aligned_hash(memory.model_hash, alignment_identity)
    return StateMemory(
        model_hash=model_hash,
        tokenizer_hash=memory.tokenizer_hash,
        state_hash=memory.state_hash,
        global_embedding=adapter(memory.global_embedding),
        segment_embeddings=adapter(memory.segment_embeddings),
        token_embeddings=(
            None if memory.token_embeddings is None else adapter(memory.token_embeddings)
        ),
        content_token_embeddings=(
            None
            if memory.content_token_embeddings is None
            else adapter(memory.content_token_embeddings)
        ),
    )


def align_compiled_schema(
    schema: CompiledSchema,
    adapter: MultilingualAlignmentAdapter,
    *,
    alignment_identity: str,
) -> CompiledSchema:
    encoder_hash = _aligned_hash(schema.encoder_hash, alignment_identity)
    schema_hash = sha256(
        f"{schema.schema_hash}:{encoder_hash}".encode()
    ).hexdigest()

    def aligned(value: Tensor | None) -> Tensor | None:
        return None if value is None else adapter(value)

    return CompiledSchema(
        schema_hash=schema_hash,
        encoder_hash=encoder_hash,
        primitive=schema.primitive,
        question_text=schema.question_text,
        options=schema.options,
        question_embedding=adapter(schema.question_embedding),
        option_embeddings=adapter(schema.option_embeddings),
        option_token_embeddings=aligned(schema.option_token_embeddings),
        option_token_mask=schema.option_token_mask,
        question_token_embeddings=aligned(schema.question_token_embeddings),
        question_token_mask=schema.question_token_mask,
        question_content_token_mask=schema.question_content_token_mask,
        option_token_ids=schema.option_token_ids,
        option_content_token_mask=schema.option_content_token_mask,
        option_view_token_embeddings=aligned(schema.option_view_token_embeddings),
        option_view_token_mask=schema.option_view_token_mask,
        option_view_mask=schema.option_view_mask,
    )


def english_anchor_loss(
    base_memory: StateMemory,
    base_schema: CompiledSchema,
    aligned_memory: StateMemory,
    aligned_schema: CompiledSchema,
) -> Tensor:
    losses = [
        torch.mean((aligned_memory.global_embedding - base_memory.global_embedding) ** 2),
        torch.mean(
            (
                aligned_memory.content_token_embeddings
                - base_memory.content_token_embeddings
            )
            ** 2
        ),
        torch.mean(
            (aligned_schema.question_embedding - base_schema.question_embedding) ** 2
        ),
        torch.mean(
            (aligned_schema.option_embeddings - base_schema.option_embeddings) ** 2
        ),
        torch.mean(
            (
                aligned_schema.option_view_token_embeddings
                - base_schema.option_view_token_embeddings
            )
            ** 2
        ),
    ]
    return torch.stack(losses).mean()


def paired_alignment_loss(
    en_memory: StateMemory,
    en_schema: CompiledSchema,
    vi_memory: StateMemory,
    vi_schema: CompiledSchema,
) -> Tensor:
    def cosine_loss(left: Tensor, right: Tensor) -> Tensor:
        left_flat = left.reshape(-1, left.shape[-1])
        right_flat = right.reshape(-1, right.shape[-1])
        if left_flat.shape != right_flat.shape:
            raise ValueError("M3-R1 paired semantic tensor shape changed")
        return (1.0 - torch.nn.functional.cosine_similarity(
            left_flat,
            right_flat,
            dim=-1,
        )).mean()

    return torch.stack(
        [
            cosine_loss(en_memory.global_embedding, vi_memory.global_embedding),
            cosine_loss(en_schema.question_embedding, vi_schema.question_embedding),
            cosine_loss(en_schema.option_embeddings, vi_schema.option_embeddings),
        ]
    ).mean()


__all__ = [
    "M3_R1_CACHE_SCHEMA",
    "align_compiled_schema",
    "align_state_memory",
    "compile_m3_r1_base_cache",
    "english_anchor_loss",
    "paired_alignment_loss",
    "validate_m3_r1_base_cache",
]
