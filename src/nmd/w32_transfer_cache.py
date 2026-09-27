from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .runtime import NolaneHira
from .w32_transfer_authority import (
    FACTOR_IDS,
    PARTITION_DOMAINS,
    QUESTION_TEXT,
    W32TransferCase,
    factor_options,
)

CACHE_SCHEMA = "r8-w32-interaction-transfer-cache-v1"


def _schema_payload(schema, receipt) -> dict[str, object]:
    required = (
        schema.option_view_token_embeddings,
        schema.option_view_token_mask,
        schema.option_view_mask,
    )
    if any(value is None for value in required):
        raise RuntimeError("W32 requires multi-view schema token artifacts")
    return {
        "schema_hash": receipt.schema_hash,
        "option_view_tokens": schema.option_view_token_embeddings.detach().cpu().to(torch.float16),
        "option_view_token_mask": schema.option_view_token_mask.detach().cpu().bool(),
        "option_view_mask": schema.option_view_mask.detach().cpu().bool(),
        "option_ids": tuple(option.option_id for option in schema.options),
        "option_values": tuple(float(option.value) for option in schema.options),
    }


@torch.inference_mode()
def compile_w32_cache(
    model: NolaneHira,
    rows: Sequence[W32TransferCase],
    *,
    partition: str,
) -> dict[str, object]:
    if partition not in {"train", "dev", "confirm"}:
        raise ValueError("W32 cache partition must be train/dev/confirm")
    allowed = set(PARTITION_DOMAINS[partition])
    if any(row.partition != partition or row.domain_id not in allowed for row in rows):
        raise ValueError("W32 rows do not match requested partition")

    model.eval()
    schemas: dict[str, dict[str, object]] = {}
    for domain in PARTITION_DOMAINS[partition]:
        pack: dict[str, object] = {}
        for factor in FACTOR_IDS:
            schema, receipt = model.compile_schema(
                primitive="choice",
                question_text=QUESTION_TEXT["choice"][factor],
                options=factor_options(domain, factor),
                use_cache=True,
                include_token_artifacts=True,
            )
            pack[factor] = _schema_payload(schema, receipt)
        schemas[domain] = pack

    cases: list[dict[str, object]] = []
    for row in rows:
        memory = model.compile_state(row.state_text)
        tokens = memory.content_token_embeddings
        if tokens is None or tokens.ndim != 2 or tokens.shape[0] < 1:
            raise RuntimeError("W32 state content-token cache is unavailable")
        cases.append(
            {
                "case_id": row.case_id,
                "domain_id": row.domain_id,
                "partition": row.partition,
                "style_id": row.style_id,
                "severity": int(row.severity),
                "variant": int(row.variant),
                "factor_vector": tuple(int(x) for x in row.factor_vector),
                "evidence_vector": tuple(int(x) for x in row.evidence_vector),
                "state_text": row.state_text,
                "state_tokens": tokens.detach().cpu().to(torch.float16),
            }
        )

    cache = {
        "metadata": {
            "schema_version": CACHE_SCHEMA,
            "partition": partition,
            "domains": list(PARTITION_DOMAINS[partition]),
            "case_count": len(cases),
            "training_performed": False,
            "selection_performed": False,
            "a13_state_encodes": len(cases),
            "factor_ids": list(FACTOR_IDS),
            "state_adapter_rank": 8,
            "schema_adapter_rank": 8,
            "interaction_rank": 8,
            "candidate_trainable_parameter_count": 6144,
        },
        "schemas": schemas,
        "cases": cases,
    }
    validate_w32_cache(cache, expected_partition=partition)
    return cache


def _validate_schema_payload(payload: dict[str, object]) -> None:
    tokens = payload.get("option_view_tokens")
    token_mask = payload.get("option_view_token_mask")
    view_mask = payload.get("option_view_mask")
    if not isinstance(tokens, Tensor) or tokens.ndim != 4 or tokens.shape[0] != 2:
        raise ValueError("W32 option-view tokens must be [2,V,T,D]")
    if not isinstance(token_mask, Tensor) or token_mask.shape != tokens.shape[:3]:
        raise ValueError("W32 option-view token mask mismatch")
    if not isinstance(view_mask, Tensor) or view_mask.shape != tokens.shape[:2]:
        raise ValueError("W32 option-view mask mismatch")
    if tuple(payload.get("option_values", ())) != (0.0, 1.0):
        raise ValueError("W32 factor option values changed")
    if len(tuple(payload.get("option_ids", ()))) != 2:
        raise ValueError("W32 factor option identity changed")


def validate_w32_cache(
    cache: dict[str, object],
    *,
    expected_partition: str | None = None,
) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W32 cache must be a dict")
    metadata = cache.get("metadata")
    schemas = cache.get("schemas")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(schemas, dict) or not isinstance(cases, list):
        raise ValueError("W32 cache requires metadata/schemas/cases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W32 cache schema")
    partition = str(metadata.get("partition", ""))
    if partition not in {"train", "dev", "confirm"}:
        raise ValueError("invalid W32 cache partition")
    if expected_partition is not None and partition != expected_partition:
        raise ValueError("W32 cache partition mismatch")
    domains = list(PARTITION_DOMAINS[partition])
    if metadata.get("domains") != domains:
        raise ValueError("W32 cache domains changed")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W32 cache case count mismatch")
    if int(metadata.get("state_adapter_rank", -1)) != 8:
        raise ValueError("W32 state adapter rank changed")
    if int(metadata.get("schema_adapter_rank", -1)) != 8:
        raise ValueError("W32 schema adapter rank changed")
    if int(metadata.get("interaction_rank", -1)) != 8:
        raise ValueError("W32 interaction rank changed")
    if int(metadata.get("candidate_trainable_parameter_count", -1)) != 6144:
        raise ValueError("W32 candidate parameter contract changed")
    if set(schemas) != set(domains):
        raise ValueError("W32 schema domains changed")

    for pack in schemas.values():
        if set(pack) != set(FACTOR_IDS):
            raise ValueError("W32 factor schema set changed")
        for factor in FACTOR_IDS:
            _validate_schema_payload(pack[factor])

    seen = set()
    counts = {domain: [0, 0, 0, 0] for domain in domains}
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("W32 invalid/duplicate case")
        seen.add(case_id)
        domain = str(case.get("domain_id", ""))
        if domain not in counts:
            raise ValueError("W32 invalid domain")
        if case.get("partition") != partition:
            raise ValueError("W32 case partition changed")
        severity = int(case.get("severity", -1))
        if severity not in {0, 1, 2, 3}:
            raise ValueError("W32 invalid severity")
        expected = {
            0: (0, 0, 0),
            1: (1, 0, 0),
            2: (1, 1, 0),
            3: (1, 1, 1),
        }[severity]
        if tuple(case.get("factor_vector", ())) != expected:
            raise ValueError("W32 factor vector changed")
        tokens = case.get("state_tokens")
        if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[-1] != 256:
            raise ValueError("W32 state token shape changed")
        if tokens.shape[0] < 1 or not bool(torch.isfinite(tokens).all()):
            raise ValueError("W32 state tokens invalid")
        counts[domain][severity] += 1

    for domain in domains:
        if counts[domain] != [24, 24, 24, 24]:
            raise ValueError("W32 per-domain severity balance changed")


__all__ = ["CACHE_SCHEMA", "compile_w32_cache", "validate_w32_cache"]
