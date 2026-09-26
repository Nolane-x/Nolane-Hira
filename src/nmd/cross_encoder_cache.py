from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .contracts import LogicalOption
from .cross_encoder_authority import (
    ABSTRACT_CONFIDENCE_DEFINITIONS,
    ABSTRACT_CONFIDENCE_OPTIONS,
    ABSTRACT_CONFIDENCE_QUESTION,
    ABSTRACT_SEVERITY_DEFINITIONS,
    ABSTRACT_SEVERITY_OPTIONS,
    ABSTRACT_SEVERITY_QUESTION,
    CONFIDENCE_PROTOTYPE_QUESTION,
    DOMAINS_ORDER,
    PROTOTYPES_PER_CLASS,
    CrossEncoderAuthorityCase,
    SEVERITY_PROTOTYPE_QUESTION,
    VIEW_IDS,
    confidence_prototype_options,
    severity_prototype_options,
)
from .runtime import NolaneHira

CACHE_SCHEMA = "r8-w23-cross-encoder-cache-v1"


def _content(batch, index: int) -> tuple[Tensor, Tensor]:
    mask = batch.attention_mask[index].bool()
    content_mask = mask
    if batch.special_token_mask is not None:
        content_mask = mask & ~batch.special_token_mask[index].bool()
    if batch.token_ids is None:
        raise RuntimeError("W23 requires token IDs")
    tokens = batch.token_embeddings[index][content_mask]
    ids = batch.token_ids[index][content_mask].long()
    if tokens.shape[0] < 1:
        raise RuntimeError("W23 isolated field requires content token")
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
        raise RuntimeError("W23 schema token artifacts incomplete")
    return {
        "schema_hash": receipt.schema_hash,
        "option_tokens": schema.option_token_embeddings.detach().cpu().to(torch.float16),
        "option_token_ids": schema.option_token_ids.detach().cpu().long(),
        "option_content_mask": schema.option_content_token_mask.detach().cpu().bool(),
        "option_ids": tuple(option.option_id for option in schema.options),
        "option_texts": tuple(option.criterion_text for option in schema.options),
    }


def _compile_views(model: NolaneHira, *, question_text: str, options, definitions):
    views = {}
    for view_index, view_id in enumerate(VIEW_IDS):
        schema, receipt = model.compile_schema(
            primitive="choice",
            question_text=question_text,
            options=_view_options(options, definitions, view_index),
            use_cache=True,
            include_token_artifacts=True,
        )
        views[view_id] = _schema_payload(schema, receipt)
    identities = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
    if not (identities[0] == identities[1] == identities[2]):
        raise RuntimeError("W23 abstract view option identity changed")
    return views


def _compile_prototypes(model: NolaneHira, *, question_text: str, options):
    schema, receipt = model.compile_schema(
        primitive="choice",
        question_text=question_text,
        options=options,
        use_cache=True,
        include_token_artifacts=True,
    )
    return _schema_payload(schema, receipt)


@torch.inference_mode()
def compile_w23_cache(
    model: NolaneHira,
    rows: Sequence[CrossEncoderAuthorityCase],
) -> dict:
    model.eval()

    # Schema-side evidence is shared by domain and does not add query-state invocations.
    schemas = {}
    for domain_id in DOMAINS_ORDER:
        severity_options = severity_prototype_options(domain_id)
        confidence_options = confidence_prototype_options(domain_id)
        schemas[domain_id] = {
            "abstract_severity": _compile_views(
                model,
                question_text=ABSTRACT_SEVERITY_QUESTION,
                options=ABSTRACT_SEVERITY_OPTIONS,
                definitions=ABSTRACT_SEVERITY_DEFINITIONS,
            ),
            "abstract_confidence": _compile_views(
                model,
                question_text=ABSTRACT_CONFIDENCE_QUESTION,
                options=ABSTRACT_CONFIDENCE_OPTIONS,
                definitions=ABSTRACT_CONFIDENCE_DEFINITIONS,
            ),
            "severity_prototypes": _compile_prototypes(
                model,
                question_text=SEVERITY_PROTOTYPE_QUESTION,
                options=severity_options,
            ),
            "confidence_prototypes": _compile_prototypes(
                model,
                question_text=CONFIDENCE_PROTOTYPE_QUESTION,
                options=confidence_options,
            ),
            "severity_prototype_class_index": tuple(
                class_index
                for class_index in range(4)
                for _ in range(PROTOTYPES_PER_CLASS)
            ),
            "confidence_prototype_class_index": tuple(
                class_index
                for class_index in range(3)
                for _ in range(PROTOTYPES_PER_CLASS)
            ),
        }

    cases = []
    for row in rows:
        # Frozen query accounting: one A13 invocation, exactly two isolated sequences.
        batch = model.encoder.encode_texts([row.severity_field, row.confidence_field])
        severity_tokens, severity_ids = _content(batch, 0)
        confidence_tokens, confidence_ids = _content(batch, 1)
        cases.append(
            {
                "case_id": row.case_id,
                "domain_id": row.domain_id,
                "severity": int(row.severity),
                "confidence_index": int(row.confidence_index),
                "variant": int(row.variant),
                "fields": {
                    "severity": row.severity_field,
                    "confidence": row.confidence_field,
                },
                "representations": {
                    "severity_tokens": severity_tokens.detach().cpu().to(torch.float16),
                    "severity_token_ids": severity_ids.detach().cpu().long(),
                    "confidence_tokens": confidence_tokens.detach().cpu().to(torch.float16),
                    "confidence_token_ids": confidence_ids.detach().cpu().long(),
                },
            }
        )

    cache = {
        "metadata": {
            "schema_version": CACHE_SCHEMA,
            "case_count": len(cases),
            "training_performed": False,
            "selection_performed": False,
            "trainable_parameter_count": 0,
            "logical_state_compiles_per_case": 1,
            "a13_query_invocations_per_case": 1,
            "encoded_query_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
            "prototypes_per_class": PROTOTYPES_PER_CLASS,
            "prototype_schema_scope": "shared-per-domain",
        },
        "schemas": schemas,
        "cases": cases,
    }
    validate_w23_cache(cache)
    return cache


def _validate_views(views: dict, width: int) -> None:
    if set(views) != set(VIEW_IDS):
        raise ValueError("W23 requires D0/D1/D2")
    ids = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
    if not (ids[0] == ids[1] == ids[2]) or len(ids[0]) != width:
        raise ValueError("W23 abstract option identity changed")


def _validate_prototype_schema(schema: dict, width: int) -> None:
    required = {
        "schema_hash",
        "option_tokens",
        "option_token_ids",
        "option_content_mask",
        "option_ids",
        "option_texts",
    }
    if set(schema) != required:
        raise ValueError("W23 prototype schema payload changed")
    if len(tuple(schema["option_ids"])) != width:
        raise ValueError("W23 prototype count changed")
    if len(tuple(schema["option_texts"])) != width:
        raise ValueError("W23 prototype text count changed")


def validate_w23_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W23 cache must be dict")
    metadata = cache.get("metadata")
    schemas = cache.get("schemas")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(schemas, dict) or not isinstance(cases, list):
        raise ValueError("W23 cache requires metadata/schemas/cases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W23 cache schema")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W23 case count mismatch")
    if metadata.get("training_performed") is not False:
        raise ValueError("W23 training forbidden")
    if metadata.get("selection_performed") is not False:
        raise ValueError("W23 selection forbidden")
    if int(metadata.get("trainable_parameter_count", -1)) != 0:
        raise ValueError("W23 must be zero-parameter")
    if int(metadata.get("logical_state_compiles_per_case", -1)) != 1:
        raise ValueError("W23 logical state accounting changed")
    if int(metadata.get("a13_query_invocations_per_case", -1)) != 1:
        raise ValueError("W23 query invocation accounting changed")
    if int(metadata.get("encoded_query_sequences_per_case", -1)) != 2:
        raise ValueError("W23 query sequence accounting changed")
    if metadata.get("isolated_fields") != ["severity", "confidence"]:
        raise ValueError("W23 isolated fields changed")
    if int(metadata.get("prototypes_per_class", -1)) != 3:
        raise ValueError("W23 requires exactly three prototypes/class")
    if metadata.get("prototype_schema_scope") != "shared-per-domain":
        raise ValueError("W23 prototype schema scope changed")

    if set(schemas) != set(DOMAINS_ORDER):
        raise ValueError("W23 schema domains changed")
    for domain_id, pack in schemas.items():
        _validate_views(pack["abstract_severity"], 4)
        _validate_views(pack["abstract_confidence"], 3)
        _validate_prototype_schema(pack["severity_prototypes"], 12)
        _validate_prototype_schema(pack["confidence_prototypes"], 9)
        if tuple(pack["severity_prototype_class_index"]) != tuple(
            i for i in range(4) for _ in range(3)
        ):
            raise ValueError("W23 severity prototype membership changed")
        if tuple(pack["confidence_prototype_class_index"]) != tuple(
            i for i in range(3) for _ in range(3)
        ):
            raise ValueError("W23 confidence prototype membership changed")

    seen = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("W23 invalid/duplicate case")
        seen.add(case_id)
        if case.get("domain_id") not in set(DOMAINS_ORDER):
            raise ValueError("W23 invalid domain")
        if int(case.get("severity", -1)) not in {0, 1, 2, 3}:
            raise ValueError("W23 invalid severity")
        if int(case.get("confidence_index", -1)) not in {0, 1, 2}:
            raise ValueError("W23 invalid confidence")
        reps = case.get("representations")
        if not isinstance(reps, dict):
            raise ValueError("W23 missing representations")
        for name in ("severity", "confidence"):
            tokens = reps.get(f"{name}_tokens")
            ids = reps.get(f"{name}_token_ids")
            if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[-1] != 256:
                raise ValueError(f"W23 invalid {name} tokens")
            if not isinstance(ids, Tensor) or ids.ndim != 1 or ids.shape[0] != tokens.shape[0]:
                raise ValueError(f"W23 invalid {name} token IDs")


def save_w23_cache(cache: dict, path: str | Path) -> Path:
    validate_w23_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w23_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w23_cache(cache)
    return cache


__all__ = [
    "CACHE_SCHEMA",
    "compile_w23_cache",
    "load_w23_cache",
    "save_w23_cache",
    "validate_w23_cache",
]
