from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .contracts import LogicalOption
from .latent_composition_authority import (
    CONFIDENCE_KEYS,
    CONFIDENCE_LATENT_DEFINITIONS,
    CONFIDENCE_LATENT_OPTIONS,
    LatentCompositionAuthorityCase,
    SEVERITY_LATENT_DEFINITIONS,
    SEVERITY_LATENT_OPTIONS,
)
from .runtime import NolaneHira, PRIMITIVE_TO_ID

CACHE_SCHEMA = "r8-w18-latent-composition-cache-v1"
VIEW_IDS = ("D0", "D1", "D2")


def _content(batch, index: int) -> tuple[Tensor, Tensor]:
    mask = batch.attention_mask[index].bool()
    if batch.special_token_mask is None:
        content_mask = mask
    else:
        content_mask = mask & ~batch.special_token_mask[index].bool()
    if batch.token_ids is None:
        raise RuntimeError("W18 requires token IDs")
    tokens = batch.token_embeddings[index][content_mask]
    ids = batch.token_ids[index][content_mask].long()
    if tokens.shape[0] < 1:
        raise RuntimeError("W18 isolated field requires content token")
    return tokens, ids


def _view_options(decision, definitions, view_index: int):
    return tuple(
        LogicalOption(
            option_id=option.option_id,
            criterion_text=views[view_index],
            value=option.value,
        )
        for option, views in zip(decision.options, definitions)
    )


def _latent_view_options(options, definitions, view_index: int):
    return tuple(
        LogicalOption(
            option_id=option.option_id,
            criterion_text=views[view_index],
            value=option.value,
        )
        for option, views in zip(options, definitions)
    )


def _compile_views(model: NolaneHira, *, primitive: str, question_text: str, option_views):
    views = {}
    for view_index, view_id in enumerate(VIEW_IDS):
        schema, receipt = model.compile_schema(
            primitive=primitive,
            question_text=question_text,
            options=option_views(view_index),
            use_cache=True,
            include_token_artifacts=True,
        )
        required = (
            schema.option_token_embeddings,
            schema.option_token_ids,
            schema.option_content_token_mask,
        )
        if any(value is None for value in required):
            raise RuntimeError("W18 schema token artifacts incomplete")
        views[view_id] = {
            "schema_hash": receipt.schema_hash,
            "option_embeddings": schema.option_embeddings.detach().cpu().to(torch.float16),
            "option_tokens": schema.option_token_embeddings.detach().cpu().to(torch.float16),
            "option_token_ids": schema.option_token_ids.detach().cpu().long(),
            "option_content_mask": schema.option_content_token_mask.detach().cpu().bool(),
            "option_ids": tuple(option.option_id for option in schema.options),
            "option_texts": tuple(option.criterion_text for option in schema.options),
        }
    ids = [tuple(views[view]["option_ids"]) for view in VIEW_IDS]
    if not (ids[0] == ids[1] == ids[2]):
        raise RuntimeError("W18 view option identity changed")
    return views


@torch.inference_mode()
def compile_w18_cache(
    model: NolaneHira,
    rows: Sequence[LatentCompositionAuthorityCase],
) -> dict:
    model.eval()
    cases: list[dict] = []

    for row in rows:
        # Exactly one A13 invocation per logical case, carrying three isolated sequences.
        batch = model.encoder.encode_texts(
            [row.intent_field, row.severity_field, row.confidence_field]
        )
        field_tokens = {}
        field_ids = {}
        for index, name in enumerate(("intent", "severity", "confidence")):
            tokens, ids = _content(batch, index)
            field_tokens[name] = tokens
            field_ids[name] = ids

        decisions = []
        for decision, definitions in zip(row.typed.decisions, row.option_definitions):
            views = _compile_views(
                model,
                primitive=decision.primitive,
                question_text=decision.question_text,
                option_views=lambda view_index, d=decision, defs=definitions: _view_options(
                    d, defs, view_index
                ),
            )
            score_support = [
                option.value for option in decision.options if option.value is not None
            ]
            decisions.append(
                {
                    "question_id": decision.question_id,
                    "primitive": decision.primitive,
                    "qtype": PRIMITIVE_TO_ID[decision.primitive],
                    "gold_index": int(decision.gold_index),
                    "gold_probabilities": torch.tensor(
                        decision.gold_probabilities, dtype=torch.float32
                    ),
                    "gold_score": (
                        None if decision.gold_score is None else float(decision.gold_score)
                    ),
                    "score_support": (
                        None
                        if not score_support
                        else torch.tensor(score_support, dtype=torch.float32)
                    ),
                    "views": views,
                }
            )

        severity_views = _compile_views(
            model,
            primitive="choice",
            question_text="Which latent severity category matches the isolated impact evidence?",
            option_views=lambda view_index: _latent_view_options(
                SEVERITY_LATENT_OPTIONS, SEVERITY_LATENT_DEFINITIONS, view_index
            ),
        )
        confidence_views = _compile_views(
            model,
            primitive="choice",
            question_text="Which latent confidence category matches the isolated evidence-quality statement?",
            option_views=lambda view_index: _latent_view_options(
                CONFIDENCE_LATENT_OPTIONS, CONFIDENCE_LATENT_DEFINITIONS, view_index
            ),
        )

        cases.append(
            {
                "case_id": row.typed.case_id,
                "domain_id": row.domain_id,
                "diagnosis_k": int(row.diagnosis_k),
                "severity": int(row.severity),
                "confidence": row.confidence,
                "confidence_index": CONFIDENCE_KEYS.index(row.confidence),
                "fields": {
                    "intent": row.intent_field,
                    "severity": row.severity_field,
                    "confidence": row.confidence_field,
                },
                "representations": {
                    "intent_tokens": field_tokens["intent"].detach().cpu().to(torch.float16),
                    "severity_tokens": field_tokens["severity"].detach().cpu().to(torch.float16),
                    "confidence_tokens": field_tokens["confidence"].detach().cpu().to(torch.float16),
                    "intent_token_ids": field_ids["intent"].detach().cpu().long(),
                    "severity_token_ids": field_ids["severity"].detach().cpu().long(),
                    "confidence_token_ids": field_ids["confidence"].detach().cpu().long(),
                },
                "latent_views": {
                    "severity": severity_views,
                    "confidence": confidence_views,
                },
                "decisions": decisions,
            }
        )

    cache = {
        "metadata": {
            "schema_version": CACHE_SCHEMA,
            "case_count": len(cases),
            "decision_count": sum(len(case["decisions"]) for case in cases),
            "training_performed": False,
            "selection_performed": False,
            "trainable_parameter_count": 0,
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 3,
            "isolated_fields": ["intent", "severity", "confidence"],
        },
        "cases": cases,
    }
    validate_w18_cache(cache)
    return cache


def validate_w18_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W18 cache must be dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("W18 cache requires metadata/cases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W18 cache schema")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W18 case count mismatch")
    if int(metadata.get("decision_count", -1)) != 5 * len(cases):
        raise ValueError("W18 decision count mismatch")
    if metadata.get("training_performed") is not False:
        raise ValueError("W18 training forbidden")
    if metadata.get("selection_performed") is not False:
        raise ValueError("W18 selection forbidden")
    if int(metadata.get("trainable_parameter_count", -1)) != 0:
        raise ValueError("W18 must be zero-parameter")
    if int(metadata.get("logical_state_compiles_per_case", -1)) != 1:
        raise ValueError("W18 logical state accounting changed")
    if int(metadata.get("a13_invocations_per_case", -1)) != 1:
        raise ValueError("W18 A13 invocation accounting changed")
    if int(metadata.get("encoded_sequences_per_case", -1)) != 3:
        raise ValueError("W18 sequence accounting changed")
    if metadata.get("isolated_fields") != ["intent", "severity", "confidence"]:
        raise ValueError("W18 isolated field set changed")

    seen = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("W18 invalid/duplicate case")
        seen.add(case_id)
        if case.get("domain_id") not in {"CR", "CS", "CT", "CU"}:
            raise ValueError("W18 invalid domain")
        if int(case.get("diagnosis_k", -1)) not in {4, 8, 16}:
            raise ValueError("W18 invalid diagnosis K")
        if int(case.get("severity", -1)) not in {0, 1, 2, 3}:
            raise ValueError("W18 invalid severity")
        if int(case.get("confidence_index", -1)) not in {0, 1, 2}:
            raise ValueError("W18 invalid confidence")

        reps = case.get("representations")
        if not isinstance(reps, dict):
            raise ValueError("W18 missing isolated representations")
        for name in ("intent", "severity", "confidence"):
            tokens = reps.get(f"{name}_tokens")
            ids = reps.get(f"{name}_token_ids")
            if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[-1] != 256:
                raise ValueError(f"W18 invalid {name} tokens")
            if not isinstance(ids, Tensor) or ids.ndim != 1 or ids.shape[0] != tokens.shape[0]:
                raise ValueError(f"W18 invalid {name} token IDs")

        latent = case.get("latent_views")
        if not isinstance(latent, dict) or set(latent) != {"severity", "confidence"}:
            raise ValueError("W18 latent schema set changed")
        for name, width in (("severity", 4), ("confidence", 3)):
            views = latent[name]
            if set(views) != set(VIEW_IDS):
                raise ValueError("W18 latent views changed")
            ids = [tuple(views[v]["option_ids"]) for v in VIEW_IDS]
            if not (ids[0] == ids[1] == ids[2]) or len(ids[0]) != width:
                raise ValueError("W18 latent option identity changed")

        decisions = case.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 5:
            raise ValueError("W18 requires five typed decisions")
        expected_q = {"diagnosis", "response", "needs_review", "risk", "urgency"}
        if {str(row["question_id"]) for row in decisions} != expected_q:
            raise ValueError("W18 typed question set changed")
        for decision in decisions:
            if set(decision["views"]) != set(VIEW_IDS):
                raise ValueError("W18 requires D0/D1/D2")
            ids = [tuple(decision["views"][v]["option_ids"]) for v in VIEW_IDS]
            if not (ids[0] == ids[1] == ids[2]):
                raise ValueError("W18 typed paraphrase option identity changed")


def save_w18_cache(cache: dict, path: str | Path) -> Path:
    validate_w18_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w18_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w18_cache(cache)
    return cache


__all__ = [
    "CACHE_SCHEMA",
    "VIEW_IDS",
    "compile_w18_cache",
    "load_w18_cache",
    "save_w18_cache",
    "validate_w18_cache",
]
