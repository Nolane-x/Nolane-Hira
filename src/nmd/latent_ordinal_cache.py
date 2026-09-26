from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .contracts import LogicalOption
from .latent_ordinal_authority import (
    CONFIDENCE_THRESHOLD_DEFINITIONS,
    CONFIDENCE_THRESHOLD_OPTIONS,
    FLAT_CONFIDENCE_DEFINITIONS,
    FLAT_CONFIDENCE_OPTIONS,
    FLAT_SEVERITY_DEFINITIONS,
    FLAT_SEVERITY_OPTIONS,
    LatentOrdinalAuthorityCase,
    SEVERITY_THRESHOLD_DEFINITIONS,
    SEVERITY_THRESHOLD_OPTIONS,
    VIEW_IDS,
)
from .runtime import NolaneHira

CACHE_SCHEMA = "r8-w19-latent-ordinal-cache-v1"


def _content(batch, index: int) -> tuple[Tensor, Tensor]:
    mask = batch.attention_mask[index].bool()
    content_mask = mask
    if batch.special_token_mask is not None:
        content_mask = mask & ~batch.special_token_mask[index].bool()
    if batch.token_ids is None:
        raise RuntimeError("W19 requires token IDs")
    tokens = batch.token_embeddings[index][content_mask]
    ids = batch.token_ids[index][content_mask].long()
    if tokens.shape[0] < 1:
        raise RuntimeError("W19 isolated field requires content token")
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
        required = (
            schema.option_token_embeddings,
            schema.option_token_ids,
            schema.option_content_token_mask,
        )
        if any(value is None for value in required):
            raise RuntimeError("W19 schema token artifacts incomplete")
        views[view_id] = {
            "schema_hash": receipt.schema_hash,
            "option_tokens": schema.option_token_embeddings.detach().cpu().to(torch.float16),
            "option_token_ids": schema.option_token_ids.detach().cpu().long(),
            "option_content_mask": schema.option_content_token_mask.detach().cpu().bool(),
            "option_ids": tuple(option.option_id for option in schema.options),
            "option_texts": tuple(option.criterion_text for option in schema.options),
        }
    identities = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
    if not (identities[0] == identities[1] == identities[2]):
        raise RuntimeError("W19 view option identity changed")
    return views


@torch.inference_mode()
def compile_w19_cache(
    model: NolaneHira,
    rows: Sequence[LatentOrdinalAuthorityCase],
) -> dict:
    model.eval()
    cases = []
    for row in rows:
        # Frozen accounting: one state encoder invocation, exactly two isolated sequences.
        batch = model.encoder.encode_texts([row.severity_field, row.confidence_field])
        severity_tokens, severity_ids = _content(batch, 0)
        confidence_tokens, confidence_ids = _content(batch, 1)

        flat_severity = _compile_views(
            model,
            question_text="Which impact band matches the isolated disruption evidence?",
            options=FLAT_SEVERITY_OPTIONS,
            definitions=FLAT_SEVERITY_DEFINITIONS,
        )
        flat_confidence = _compile_views(
            model,
            question_text="Which certainty band matches the isolated evidence support?",
            options=FLAT_CONFIDENCE_OPTIONS,
            definitions=FLAT_CONFIDENCE_DEFINITIONS,
        )

        severity_thresholds = []
        for index, (options, definitions) in enumerate(
            zip(SEVERITY_THRESHOLD_OPTIONS, SEVERITY_THRESHOLD_DEFINITIONS), start=1
        ):
            severity_thresholds.append(
                _compile_views(
                    model,
                    question_text=f"Does the isolated disruption evidence satisfy cumulative impact threshold {index}?",
                    options=options,
                    definitions=definitions,
                )
            )

        confidence_thresholds = []
        for index, (options, definitions) in enumerate(
            zip(CONFIDENCE_THRESHOLD_OPTIONS, CONFIDENCE_THRESHOLD_DEFINITIONS), start=1
        ):
            confidence_thresholds.append(
                _compile_views(
                    model,
                    question_text=f"Does the isolated evidence support satisfy cumulative certainty threshold {index}?",
                    options=options,
                    definitions=definitions,
                )
            )

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
                "schemas": {
                    "flat_severity": flat_severity,
                    "flat_confidence": flat_confidence,
                    "severity_thresholds": severity_thresholds,
                    "confidence_thresholds": confidence_thresholds,
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
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
        },
        "cases": cases,
    }
    validate_w19_cache(cache)
    return cache


def _validate_views(views: dict, width: int) -> None:
    if set(views) != set(VIEW_IDS):
        raise ValueError("W19 requires D0/D1/D2")
    ids = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
    if not (ids[0] == ids[1] == ids[2]) or len(ids[0]) != width:
        raise ValueError("W19 option identity changed")


def validate_w19_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W19 cache must be dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("W19 cache requires metadata/cases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W19 cache schema")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W19 case count mismatch")
    if metadata.get("training_performed") is not False:
        raise ValueError("W19 training forbidden")
    if metadata.get("selection_performed") is not False:
        raise ValueError("W19 selection forbidden")
    if int(metadata.get("trainable_parameter_count", -1)) != 0:
        raise ValueError("W19 must be zero-parameter")
    if int(metadata.get("logical_state_compiles_per_case", -1)) != 1:
        raise ValueError("W19 logical state accounting changed")
    if int(metadata.get("a13_invocations_per_case", -1)) != 1:
        raise ValueError("W19 A13 accounting changed")
    if int(metadata.get("encoded_sequences_per_case", -1)) != 2:
        raise ValueError("W19 sequence accounting changed")
    if metadata.get("isolated_fields") != ["severity", "confidence"]:
        raise ValueError("W19 isolated fields changed")

    seen = set()
    domains = {"CV", "CW", "CX", "CY"}
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("W19 invalid/duplicate case")
        seen.add(case_id)
        if case.get("domain_id") not in domains:
            raise ValueError("W19 invalid domain")
        if int(case.get("severity", -1)) not in {0, 1, 2, 3}:
            raise ValueError("W19 invalid severity")
        if int(case.get("confidence_index", -1)) not in {0, 1, 2}:
            raise ValueError("W19 invalid confidence")

        reps = case.get("representations")
        if not isinstance(reps, dict):
            raise ValueError("W19 missing representations")
        for name in ("severity", "confidence"):
            tokens = reps.get(f"{name}_tokens")
            ids = reps.get(f"{name}_token_ids")
            if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[-1] != 256:
                raise ValueError(f"W19 invalid {name} tokens")
            if not isinstance(ids, Tensor) or ids.ndim != 1 or ids.shape[0] != tokens.shape[0]:
                raise ValueError(f"W19 invalid {name} token IDs")

        schemas = case.get("schemas")
        if not isinstance(schemas, dict):
            raise ValueError("W19 missing schemas")
        _validate_views(schemas["flat_severity"], 4)
        _validate_views(schemas["flat_confidence"], 3)
        if len(schemas["severity_thresholds"]) != 3:
            raise ValueError("W19 requires three severity thresholds")
        if len(schemas["confidence_thresholds"]) != 2:
            raise ValueError("W19 requires two confidence thresholds")
        for views in schemas["severity_thresholds"]:
            _validate_views(views, 2)
        for views in schemas["confidence_thresholds"]:
            _validate_views(views, 2)


def save_w19_cache(cache: dict, path: str | Path) -> Path:
    validate_w19_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w19_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w19_cache(cache)
    return cache


__all__ = [
    "CACHE_SCHEMA",
    "compile_w19_cache",
    "load_w19_cache",
    "save_w19_cache",
    "validate_w19_cache",
]
