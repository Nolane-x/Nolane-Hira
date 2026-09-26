from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .compositional_projection_authority import (
    FACTOR_DEFINITIONS,
    FACTOR_IDS,
    FACTOR_OPTIONS,
    FACTOR_QUESTIONS,
    PARTITION_DOMAINS,
    REFERENCE_HYPOTHESES,
    VIEW_IDS,
    ProjectionCompositionCase,
)
from .contracts import LogicalOption
from .runtime import NolaneHira

CACHE_SCHEMA = "r8-w28-compositional-projection-cache-v1"


def _content(batch, index: int) -> tuple[Tensor, Tensor]:
    mask = batch.attention_mask[index].bool()
    content_mask = mask
    if batch.special_token_mask is not None:
        content_mask = mask & ~batch.special_token_mask[index].bool()
    if batch.token_ids is None:
        raise RuntimeError("W28 requires token IDs")
    tokens = batch.token_embeddings[index][content_mask]
    ids = batch.token_ids[index][content_mask].long()
    if tokens.shape[0] < 1:
        raise RuntimeError("W28 severity query requires content token")
    return tokens, ids


def _view_options(options, definitions, view_index: int):
    return tuple(
        LogicalOption(
            option_id=option.option_id,
            criterion_text=views[view_index],
            value=option.value,
        )
        for option, views in zip(options, definitions)
    )


def _schema_payload(schema, receipt):
    required = (
        schema.option_token_embeddings,
        schema.option_token_ids,
        schema.option_content_token_mask,
    )
    if any(value is None for value in required):
        raise RuntimeError("W28 schema token artifacts incomplete")
    return {
        "schema_hash": receipt.schema_hash,
        "option_tokens": schema.option_token_embeddings.detach().cpu().to(torch.float16),
        "option_token_ids": schema.option_token_ids.detach().cpu().long(),
        "option_content_mask": schema.option_content_token_mask.detach().cpu().bool(),
        "option_ids": tuple(option.option_id for option in schema.options),
        "option_texts": tuple(option.criterion_text for option in schema.options),
    }


def _compile_views(model: NolaneHira, *, factor_id: str):
    views = {}
    for view_index, view_id in enumerate(VIEW_IDS):
        schema, receipt = model.compile_schema(
            primitive="choice",
            question_text=FACTOR_QUESTIONS[factor_id],
            options=_view_options(
                FACTOR_OPTIONS[factor_id],
                FACTOR_DEFINITIONS[factor_id],
                view_index,
            ),
            use_cache=True,
            include_token_artifacts=True,
        )
        views[view_id] = _schema_payload(schema, receipt)
    ids = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
    if not (ids[0] == ids[1] == ids[2]) or len(ids[0]) != 2:
        raise RuntimeError("W28 factor multiview option identity changed")
    return views


@torch.inference_mode()
def compile_w28_cache(
    model: NolaneHira,
    rows: Sequence[ProjectionCompositionCase],
    *,
    partition: str,
) -> dict:
    if partition not in {"train", "dev", "confirm"}:
        raise ValueError("W28 HIRA cache only supports train/dev/confirm")
    allowed_domains = set(PARTITION_DOMAINS[partition])
    if any(row.domain_id not in allowed_domains or row.partition != partition for row in rows):
        raise ValueError("W28 rows do not match requested partition")

    model.eval()
    schemas = {}
    for domain_id in PARTITION_DOMAINS[partition]:
        pack = {
            "reference_hypotheses": {
                key: tuple(value) for key, value in REFERENCE_HYPOTHESES.items()
            },
        }
        for factor_id in FACTOR_IDS:
            pack[f"factor_{factor_id.lower()}"] = _compile_views(model, factor_id=factor_id)
        schemas[domain_id] = pack

    cases = []
    for row in rows:
        batch = model.encoder.encode_texts([row.severity_field])
        tokens, ids = _content(batch, 0)
        cases.append(
            {
                "case_id": row.case_id,
                "domain_id": row.domain_id,
                "partition": row.partition,
                "severity": int(row.severity),
                "variant": int(row.variant),
                "factor_vector": tuple(int(x) for x in row.factor_vector),
                "evidence_vector": tuple(int(x) for x in row.evidence_vector),
                "severity_field": row.severity_field,
                "representations": {
                    "severity_tokens": tokens.detach().cpu().to(torch.float16),
                    "severity_token_ids": ids.detach().cpu().long(),
                },
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
            "trainable_parameter_count": 0,
            "logical_state_compiles_per_case": 1,
            "a13_query_invocations_per_case": 1,
            "encoded_query_sequences_per_case": 1,
            "isolated_fields": ["severity"],
            "factor_ids": list(FACTOR_IDS),
            "factor_decoder": {"000": 0, "100": 1, "110": 2, "111": 3},
            "primary_reference_f2": "U_AND_C",
        },
        "schemas": schemas,
        "cases": cases,
    }
    validate_w28_cache(cache, expected_partition=partition)
    return cache


def _validate_views(views: dict) -> None:
    if set(views) != set(VIEW_IDS):
        raise ValueError("W28 requires D0/D1/D2")
    ids = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
    if not (ids[0] == ids[1] == ids[2]) or len(ids[0]) != 2:
        raise ValueError("W28 option identity changed")
    for view in views.values():
        tokens = view.get("option_tokens")
        mask = view.get("option_content_mask")
        token_ids = view.get("option_token_ids")
        if not isinstance(tokens, Tensor) or tokens.ndim != 3 or tokens.shape[0] != 2:
            raise ValueError("W28 invalid option tokens")
        if not isinstance(mask, Tensor) or mask.shape != tokens.shape[:2]:
            raise ValueError("W28 invalid option mask")
        if not isinstance(token_ids, Tensor) or token_ids.shape != mask.shape:
            raise ValueError("W28 invalid option token IDs")


def validate_w28_cache(cache: dict, *, expected_partition: str | None = None) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W28 cache must be dict")
    metadata = cache.get("metadata")
    schemas = cache.get("schemas")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(schemas, dict) or not isinstance(cases, list):
        raise ValueError("W28 cache requires metadata/schemas/cases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W28 cache schema")
    partition = str(metadata.get("partition", ""))
    if partition not in {"train", "dev", "confirm"}:
        raise ValueError("W28 invalid HIRA partition")
    if expected_partition is not None and partition != expected_partition:
        raise ValueError("W28 partition mismatch")
    domains = list(PARTITION_DOMAINS[partition])
    if metadata.get("domains") != domains:
        raise ValueError("W28 domain list changed")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W28 case count mismatch")
    if metadata.get("training_performed") is not False:
        raise ValueError("W28 cache cannot train")
    if metadata.get("selection_performed") is not False:
        raise ValueError("W28 cache cannot select")
    if int(metadata.get("trainable_parameter_count", -1)) != 0:
        raise ValueError("W28 cache trainable count changed")
    if int(metadata.get("logical_state_compiles_per_case", -1)) != 1:
        raise ValueError("W28 logical state accounting changed")
    if int(metadata.get("a13_query_invocations_per_case", -1)) != 1:
        raise ValueError("W28 A13 invocation accounting changed")
    if int(metadata.get("encoded_query_sequences_per_case", -1)) != 1:
        raise ValueError("W28 sequence accounting changed")
    if metadata.get("isolated_fields") != ["severity"]:
        raise ValueError("W28 isolated fields changed")
    if tuple(metadata.get("factor_ids", ())) != FACTOR_IDS:
        raise ValueError("W28 factor IDs changed")
    if metadata.get("factor_decoder") != {"000": 0, "100": 1, "110": 2, "111": 3}:
        raise ValueError("W28 factor decoder changed")
    if metadata.get("primary_reference_f2") != "U_AND_C":
        raise ValueError("W28 primary F2 authority changed")

    if set(schemas) != set(domains):
        raise ValueError("W28 schema domains changed")
    for pack in schemas.values():
        for factor_id in FACTOR_IDS:
            _validate_views(pack[f"factor_{factor_id.lower()}"])
        if pack["reference_hypotheses"] != {
            key: tuple(value) for key, value in REFERENCE_HYPOTHESES.items()
        }:
            raise ValueError("W28 reference hypotheses changed")

    seen = set()
    per_domain = {domain: 0 for domain in domains}
    severity_counts = {domain: [0, 0, 0, 0] for domain in domains}
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("W28 invalid/duplicate case")
        seen.add(case_id)
        domain = str(case.get("domain_id", ""))
        if domain not in per_domain:
            raise ValueError("W28 invalid domain")
        if case.get("partition") != partition:
            raise ValueError("W28 case partition changed")
        severity = int(case.get("severity", -1))
        if severity not in {0, 1, 2, 3}:
            raise ValueError("W28 invalid severity")
        expected_factor = {
            0: (0, 0, 0),
            1: (1, 0, 0),
            2: (1, 1, 0),
            3: (1, 1, 1),
        }[severity]
        expected_evidence = {
            0: (0, 0, 0, 0),
            1: (1, 0, 1, 0),
            2: (1, 1, 0, 1),
            3: (1, 1, 1, 1),
        }[severity]
        if tuple(case.get("factor_vector", ())) != expected_factor:
            raise ValueError("W28 factor vector changed")
        if tuple(case.get("evidence_vector", ())) != expected_evidence:
            raise ValueError("W28 evidence vector changed")
        reps = case.get("representations")
        if not isinstance(reps, dict):
            raise ValueError("W28 missing representations")
        tokens = reps.get("severity_tokens")
        ids = reps.get("severity_token_ids")
        if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[-1] != 256:
            raise ValueError("W28 invalid severity tokens")
        if not isinstance(ids, Tensor) or ids.ndim != 1 or ids.shape[0] != tokens.shape[0]:
            raise ValueError("W28 invalid severity token IDs")
        per_domain[domain] += 1
        severity_counts[domain][severity] += 1

    for domain in domains:
        if per_domain[domain] != 96:
            raise ValueError("W28 requires 96 cases/domain")
        if severity_counts[domain] != [24, 24, 24, 24]:
            raise ValueError("W28 severity balance changed")


def save_w28_cache(cache: dict, path: str | Path) -> Path:
    validate_w28_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w28_cache(path: str | Path, *, expected_partition: str | None = None) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w28_cache(cache, expected_partition=expected_partition)
    return cache


__all__ = [
    "CACHE_SCHEMA",
    "compile_w28_cache",
    "load_w28_cache",
    "save_w28_cache",
    "validate_w28_cache",
]
