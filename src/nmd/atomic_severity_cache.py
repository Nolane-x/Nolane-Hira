from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .atomic_severity_authority import (
    CONFIDENCE_DEFINITIONS,
    CONFIDENCE_OPTIONS,
    CONFIDENCE_PROTOTYPE_QUESTION,
    CONFIDENCE_QUESTION,
    DIRECT_SEVERITY_DEFINITIONS,
    DIRECT_SEVERITY_OPTIONS,
    DIRECT_SEVERITY_QUESTION,
    DOMAINS_ORDER,
    FACTOR_DEFINITIONS,
    FACTOR_IDS,
    FACTOR_OPTIONS,
    FACTOR_QUESTIONS,
    FACTOR_REFERENCE_HYPOTHESES,
    PROTOTYPES_PER_CLASS,
    SEVERITY_PROTOTYPE_QUESTION,
    VIEW_IDS,
    AtomicSeverityAuthorityCase,
    confidence_prototype_options,
    severity_prototype_options,
)
from .contracts import LogicalOption
from .runtime import NolaneHira

CACHE_SCHEMA = "r8-w24-atomic-severity-cache-v1"


def _content(batch, index: int) -> tuple[Tensor, Tensor]:
    mask = batch.attention_mask[index].bool()
    content_mask = mask
    if batch.special_token_mask is not None:
        content_mask = mask & ~batch.special_token_mask[index].bool()
    if batch.token_ids is None:
        raise RuntimeError("W24 requires token IDs")
    tokens = batch.token_embeddings[index][content_mask]
    ids = batch.token_ids[index][content_mask].long()
    if tokens.shape[0] < 1:
        raise RuntimeError("W24 isolated field requires content token")
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
        raise RuntimeError("W24 schema token artifacts incomplete")
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
        raise RuntimeError("W24 multiview option identity changed")
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
def compile_w24_cache(
    model: NolaneHira,
    rows: Sequence[AtomicSeverityAuthorityCase],
) -> dict:
    model.eval()

    schemas = {}
    for domain_id in DOMAINS_ORDER:
        pack = {
            "direct_severity": _compile_views(
                model,
                question_text=DIRECT_SEVERITY_QUESTION,
                options=DIRECT_SEVERITY_OPTIONS,
                definitions=DIRECT_SEVERITY_DEFINITIONS,
            ),
            "confidence": _compile_views(
                model,
                question_text=CONFIDENCE_QUESTION,
                options=CONFIDENCE_OPTIONS,
                definitions=CONFIDENCE_DEFINITIONS,
            ),
            "severity_prototypes": _compile_prototypes(
                model,
                question_text=SEVERITY_PROTOTYPE_QUESTION,
                options=severity_prototype_options(domain_id),
            ),
            "confidence_prototypes": _compile_prototypes(
                model,
                question_text=CONFIDENCE_PROTOTYPE_QUESTION,
                options=confidence_prototype_options(domain_id),
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
            "factor_reference_hypotheses": {
                factor_id: tuple(FACTOR_REFERENCE_HYPOTHESES[factor_id])
                for factor_id in FACTOR_IDS
            },
        }
        for factor_id in FACTOR_IDS:
            pack[f"factor_{factor_id.lower()}"] = _compile_views(
                model,
                question_text=FACTOR_QUESTIONS[factor_id],
                options=FACTOR_OPTIONS[factor_id],
                definitions=FACTOR_DEFINITIONS[factor_id],
            )
        schemas[domain_id] = pack

    cases = []
    for row in rows:
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
                "factor_vector": tuple(int(x) for x in row.factor_vector),
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
            "factor_ids": list(FACTOR_IDS),
            "factor_decoder": {
                "000": 0,
                "100": 1,
                "110": 2,
                "111": 3,
            },
        },
        "schemas": schemas,
        "cases": cases,
    }
    validate_w24_cache(cache)
    return cache


def _validate_views(views: dict, width: int) -> None:
    if set(views) != set(VIEW_IDS):
        raise ValueError("W24 requires D0/D1/D2")
    ids = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
    if not (ids[0] == ids[1] == ids[2]) or len(ids[0]) != width:
        raise ValueError("W24 multiview option identity changed")


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
        raise ValueError("W24 prototype schema payload changed")
    if len(tuple(schema["option_ids"])) != width:
        raise ValueError("W24 prototype count changed")
    if len(tuple(schema["option_texts"])) != width:
        raise ValueError("W24 prototype text count changed")


def validate_w24_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W24 cache must be dict")
    metadata = cache.get("metadata")
    schemas = cache.get("schemas")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(schemas, dict) or not isinstance(cases, list):
        raise ValueError("W24 cache requires metadata/schemas/cases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W24 cache schema")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W24 case count mismatch")
    if metadata.get("training_performed") is not False:
        raise ValueError("W24 training forbidden")
    if metadata.get("selection_performed") is not False:
        raise ValueError("W24 selection forbidden")
    if int(metadata.get("trainable_parameter_count", -1)) != 0:
        raise ValueError("W24 must be zero-parameter")
    if int(metadata.get("logical_state_compiles_per_case", -1)) != 1:
        raise ValueError("W24 logical state accounting changed")
    if int(metadata.get("a13_query_invocations_per_case", -1)) != 1:
        raise ValueError("W24 query invocation accounting changed")
    if int(metadata.get("encoded_query_sequences_per_case", -1)) != 2:
        raise ValueError("W24 query sequence accounting changed")
    if metadata.get("isolated_fields") != ["severity", "confidence"]:
        raise ValueError("W24 isolated fields changed")
    if int(metadata.get("prototypes_per_class", -1)) != 3:
        raise ValueError("W24 requires exactly three prototypes/class")
    if metadata.get("prototype_schema_scope") != "shared-per-domain":
        raise ValueError("W24 prototype schema scope changed")
    if tuple(metadata.get("factor_ids", ())) != FACTOR_IDS:
        raise ValueError("W24 factor identity changed")
    if metadata.get("factor_decoder") != {"000": 0, "100": 1, "110": 2, "111": 3}:
        raise ValueError("W24 factor decoder changed")

    if set(schemas) != set(DOMAINS_ORDER):
        raise ValueError("W24 schema domains changed")
    for pack in schemas.values():
        _validate_views(pack["direct_severity"], 4)
        _validate_views(pack["confidence"], 3)
        for factor_id in FACTOR_IDS:
            _validate_views(pack[f"factor_{factor_id.lower()}"], 2)
            if tuple(pack["factor_reference_hypotheses"][factor_id]) != tuple(
                FACTOR_REFERENCE_HYPOTHESES[factor_id]
            ):
                raise ValueError("W24 factor reference hypothesis changed")
        _validate_prototype_schema(pack["severity_prototypes"], 12)
        _validate_prototype_schema(pack["confidence_prototypes"], 9)
        if tuple(pack["severity_prototype_class_index"]) != tuple(
            i for i in range(4) for _ in range(3)
        ):
            raise ValueError("W24 severity prototype membership changed")
        if tuple(pack["confidence_prototype_class_index"]) != tuple(
            i for i in range(3) for _ in range(3)
        ):
            raise ValueError("W24 confidence prototype membership changed")

    seen = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("W24 invalid/duplicate case")
        seen.add(case_id)
        if case.get("domain_id") not in set(DOMAINS_ORDER):
            raise ValueError("W24 invalid domain")
        severity = int(case.get("severity", -1))
        if severity not in {0, 1, 2, 3}:
            raise ValueError("W24 invalid severity")
        if int(case.get("confidence_index", -1)) not in {0, 1, 2}:
            raise ValueError("W24 invalid confidence")
        if tuple(case.get("factor_vector", ())) != {
            0: (0, 0, 0),
            1: (1, 0, 0),
            2: (1, 1, 0),
            3: (1, 1, 1),
        }[severity]:
            raise ValueError("W24 factor vector changed")
        reps = case.get("representations")
        if not isinstance(reps, dict):
            raise ValueError("W24 missing representations")
        for name in ("severity", "confidence"):
            tokens = reps.get(f"{name}_tokens")
            ids = reps.get(f"{name}_token_ids")
            if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[-1] != 256:
                raise ValueError(f"W24 invalid {name} tokens")
            if not isinstance(ids, Tensor) or ids.ndim != 1 or ids.shape[0] != tokens.shape[0]:
                raise ValueError(f"W24 invalid {name} token IDs")


def save_w24_cache(cache: dict, path: str | Path) -> Path:
    validate_w24_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w24_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w24_cache(cache)
    return cache


__all__ = [
    "CACHE_SCHEMA",
    "compile_w24_cache",
    "load_w24_cache",
    "save_w24_cache",
    "validate_w24_cache",
]
