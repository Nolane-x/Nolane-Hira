from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .contracts import LogicalOption
from .pairwise_latent_authority import (
    CONFIDENCE_PAIR_DEFINITIONS,
    CONFIDENCE_PAIR_OPTIONS,
    CONFIDENCE_PAIR_QUESTIONS,
    FLAT_CONFIDENCE_DEFINITIONS,
    FLAT_CONFIDENCE_OPTIONS,
    FLAT_CONFIDENCE_QUESTION,
    FLAT_SEVERITY_DEFINITIONS,
    FLAT_SEVERITY_OPTIONS,
    FLAT_SEVERITY_QUESTION,
    PairwiseLatentAuthorityCase,
    SEVERITY_PAIR_DEFINITIONS,
    SEVERITY_PAIR_OPTIONS,
    SEVERITY_PAIR_QUESTIONS,
    VIEW_IDS,
)
from .runtime import NolaneHira

CACHE_SCHEMA = "r8-w20-pairwise-latent-cache-v1"


def _content(batch, index: int) -> tuple[Tensor, Tensor]:
    mask = batch.attention_mask[index].bool()
    content_mask = mask
    if batch.special_token_mask is not None:
        content_mask = mask & ~batch.special_token_mask[index].bool()
    if batch.token_ids is None:
        raise RuntimeError("W20 requires token IDs")
    tokens = batch.token_embeddings[index][content_mask]
    ids = batch.token_ids[index][content_mask].long()
    if tokens.shape[0] < 1:
        raise RuntimeError("W20 isolated field requires content token")
    return tokens, ids


def _view_options(options, definitions, view_index: int, *, reverse: bool = False):
    values = [
        LogicalOption(
            option_id=option.option_id,
            criterion_text=views[view_index],
            value=option.value,
        )
        for option, views in zip(options, definitions)
    ]
    if reverse:
        values.reverse()
    return tuple(values)


def _compile_views(
    model: NolaneHira,
    *,
    question_text: str,
    options,
    definitions,
    reverse: bool = False,
):
    views = {}
    for view_index, view_id in enumerate(VIEW_IDS):
        schema, receipt = model.compile_schema(
            primitive="choice",
            question_text=question_text,
            options=_view_options(options, definitions, view_index, reverse=reverse),
            use_cache=True,
            include_token_artifacts=True,
        )
        required = (
            schema.option_token_embeddings,
            schema.option_token_ids,
            schema.option_content_token_mask,
        )
        if any(value is None for value in required):
            raise RuntimeError("W20 schema token artifacts incomplete")
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
        raise RuntimeError("W20 view option identity changed")
    return views


def _compile_pair_views(model, *, question_text, options, definitions):
    canonical = _compile_views(
        model,
        question_text=question_text,
        options=options,
        definitions=definitions,
        reverse=False,
    )
    swapped = _compile_views(
        model,
        question_text=question_text,
        options=options,
        definitions=definitions,
        reverse=True,
    )
    canonical_ids = tuple(canonical[VIEW_IDS[0]]["option_ids"])
    swapped_ids = tuple(swapped[VIEW_IDS[0]]["option_ids"])
    if len(canonical_ids) != 2 or swapped_ids != tuple(reversed(canonical_ids)):
        raise RuntimeError("W20 pair swap identity changed")
    return {"canonical": canonical, "swapped": swapped}


@torch.inference_mode()
def compile_w20_cache(
    model: NolaneHira,
    rows: Sequence[PairwiseLatentAuthorityCase],
) -> dict:
    model.eval()
    cases = []
    for row in rows:
        # Frozen accounting: one state encoder invocation with exactly two isolated sequences.
        batch = model.encoder.encode_texts([row.severity_field, row.confidence_field])
        severity_tokens, severity_ids = _content(batch, 0)
        confidence_tokens, confidence_ids = _content(batch, 1)

        flat_severity = _compile_views(
            model,
            question_text=FLAT_SEVERITY_QUESTION,
            options=FLAT_SEVERITY_OPTIONS,
            definitions=FLAT_SEVERITY_DEFINITIONS,
        )
        flat_confidence = _compile_views(
            model,
            question_text=FLAT_CONFIDENCE_QUESTION,
            options=FLAT_CONFIDENCE_OPTIONS,
            definitions=FLAT_CONFIDENCE_DEFINITIONS,
        )

        severity_pairs = [
            _compile_pair_views(
                model,
                question_text=question_text,
                options=options,
                definitions=definitions,
            )
            for options, definitions, question_text in zip(
                SEVERITY_PAIR_OPTIONS,
                SEVERITY_PAIR_DEFINITIONS,
                SEVERITY_PAIR_QUESTIONS,
            )
        ]
        confidence_pairs = [
            _compile_pair_views(
                model,
                question_text=question_text,
                options=options,
                definitions=definitions,
            )
            for options, definitions, question_text in zip(
                CONFIDENCE_PAIR_OPTIONS,
                CONFIDENCE_PAIR_DEFINITIONS,
                CONFIDENCE_PAIR_QUESTIONS,
            )
        ]

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
                    "severity_pairs": severity_pairs,
                    "confidence_pairs": confidence_pairs,
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
    validate_w20_cache(cache)
    return cache


def _validate_views(views: dict, width: int) -> None:
    if set(views) != set(VIEW_IDS):
        raise ValueError("W20 requires D0/D1/D2")
    ids = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
    if not (ids[0] == ids[1] == ids[2]) or len(ids[0]) != width:
        raise ValueError("W20 option identity changed")


def _validate_pair(pair: dict) -> None:
    if set(pair) != {"canonical", "swapped"}:
        raise ValueError("W20 pair requires canonical/swapped")
    _validate_views(pair["canonical"], 2)
    _validate_views(pair["swapped"], 2)
    canonical_ids = tuple(pair["canonical"][VIEW_IDS[0]]["option_ids"])
    swapped_ids = tuple(pair["swapped"][VIEW_IDS[0]]["option_ids"])
    if swapped_ids != tuple(reversed(canonical_ids)):
        raise ValueError("W20 swapped pair IDs invalid")


def validate_w20_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W20 cache must be dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("W20 cache requires metadata/cases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W20 cache schema")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W20 case count mismatch")
    if metadata.get("training_performed") is not False:
        raise ValueError("W20 training forbidden")
    if metadata.get("selection_performed") is not False:
        raise ValueError("W20 selection forbidden")
    if int(metadata.get("trainable_parameter_count", -1)) != 0:
        raise ValueError("W20 must be zero-parameter")
    if int(metadata.get("logical_state_compiles_per_case", -1)) != 1:
        raise ValueError("W20 logical state accounting changed")
    if int(metadata.get("a13_invocations_per_case", -1)) != 1:
        raise ValueError("W20 A13 accounting changed")
    if int(metadata.get("encoded_sequences_per_case", -1)) != 2:
        raise ValueError("W20 sequence accounting changed")
    if metadata.get("isolated_fields") != ["severity", "confidence"]:
        raise ValueError("W20 isolated fields changed")

    seen = set()
    domains = {"CZ", "DA", "DB", "DC"}
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("W20 invalid/duplicate case")
        seen.add(case_id)
        if case.get("domain_id") not in domains:
            raise ValueError("W20 invalid domain")
        if int(case.get("severity", -1)) not in {0, 1, 2, 3}:
            raise ValueError("W20 invalid severity")
        if int(case.get("confidence_index", -1)) not in {0, 1, 2}:
            raise ValueError("W20 invalid confidence")

        reps = case.get("representations")
        if not isinstance(reps, dict):
            raise ValueError("W20 missing representations")
        for name in ("severity", "confidence"):
            tokens = reps.get(f"{name}_tokens")
            ids = reps.get(f"{name}_token_ids")
            if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[-1] != 256:
                raise ValueError(f"W20 invalid {name} tokens")
            if not isinstance(ids, Tensor) or ids.ndim != 1 or ids.shape[0] != tokens.shape[0]:
                raise ValueError(f"W20 invalid {name} token IDs")

        schemas = case.get("schemas")
        if not isinstance(schemas, dict):
            raise ValueError("W20 missing schemas")
        _validate_views(schemas["flat_severity"], 4)
        _validate_views(schemas["flat_confidence"], 3)
        if len(schemas["severity_pairs"]) != 3:
            raise ValueError("W20 requires three severity pairs")
        if len(schemas["confidence_pairs"]) != 2:
            raise ValueError("W20 requires two confidence pairs")
        for pair in schemas["severity_pairs"]:
            _validate_pair(pair)
        for pair in schemas["confidence_pairs"]:
            _validate_pair(pair)


def save_w20_cache(cache: dict, path: str | Path) -> Path:
    validate_w20_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w20_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w20_cache(cache)
    return cache


__all__ = [
    "CACHE_SCHEMA",
    "compile_w20_cache",
    "load_w20_cache",
    "save_w20_cache",
    "validate_w20_cache",
]
