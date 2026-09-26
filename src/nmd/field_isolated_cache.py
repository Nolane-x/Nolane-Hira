from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .contracts import LogicalOption
from .field_isolated_authority import FieldIsolatedAuthorityCase
from .runtime import NolaneHira, PRIMITIVE_TO_ID

CACHE_SCHEMA = "r8-w17-field-isolated-cache-v1"


def _content(batch, index: int) -> tuple[Tensor, Tensor]:
    mask = batch.attention_mask[index].bool()
    if batch.special_token_mask is None:
        content_mask = mask
    else:
        content_mask = mask & ~batch.special_token_mask[index].bool()
    if batch.token_ids is None:
        raise RuntimeError("W17 requires token IDs")
    tokens = batch.token_embeddings[index][content_mask]
    ids = batch.token_ids[index][content_mask].long()
    if tokens.shape[0] < 1:
        raise RuntimeError("W17 field requires content token")
    return tokens, ids


def _segments(batch, index: int, segment_tokens: int = 32) -> Tensor:
    mask = batch.attention_mask[index].bool()
    valid = batch.token_embeddings[index][mask]
    if valid.shape[0] < 1:
        raise RuntimeError("W17 full state requires valid token")
    return torch.stack(
        [
            valid[start : start + segment_tokens].mean(0)
            for start in range(0, valid.shape[0], segment_tokens)
        ]
    )


def _view_options(decision, definitions, view_index: int):
    rows = []
    for option, views in zip(decision.options, definitions):
        rows.append(
            LogicalOption(
                option_id=option.option_id,
                criterion_text=views[view_index],
                value=option.value,
            )
        )
    return tuple(rows)


@torch.inference_mode()
def compile_w17_cache(
    model: NolaneHira,
    rows: Sequence[FieldIsolatedAuthorityCase],
    *,
    expected_split_prefix: str | None = None,
) -> dict:
    model.eval()
    cases: list[dict] = []

    for row in rows:
        if expected_split_prefix is not None and not row.split.startswith(
            expected_split_prefix
        ):
            raise ValueError("W17 split mismatch")

        full_single = model.encoder.encode_texts([row.full_state_text])
        full_triplicate = model.encoder.encode_texts([row.full_state_text] * 3)
        isolated = model.encoder.encode_texts(
            [row.intent_field, row.severity_field, row.confidence_field]
        )

        single_tokens, single_ids = _content(full_single, 0)
        trip_tokens = []
        trip_ids = []
        for idx in range(3):
            t, i = _content(full_triplicate, idx)
            trip_tokens.append(t)
            trip_ids.append(i)
        if any(not torch.equal(ids.cpu(), single_ids.cpu()) for ids in trip_ids):
            raise RuntimeError("W17 triplicate token identity changed")
        if any(tokens.shape != single_tokens.shape for tokens in trip_tokens):
            raise RuntimeError("W17 triplicate token shape changed")

        max_trip_copy_diff = max(
            float((trip_tokens[0].float() - tokens.float()).abs().max())
            for tokens in trip_tokens[1:]
        )
        max_single_trip_diff = float(
            (single_tokens.float() - trip_tokens[0].float()).abs().max()
        )

        field_tokens = {}
        field_ids = {}
        for idx, name in enumerate(("intent", "severity", "confidence")):
            t, i = _content(isolated, idx)
            field_tokens[name] = t
            field_ids[name] = i

        decisions = []
        for decision, definitions in zip(
            row.typed.decisions,
            row.option_definitions,
        ):
            views = {}
            for view_index, view_id in enumerate(("D0", "D1", "D2")):
                schema, receipt = model.compile_schema(
                    primitive=decision.primitive,
                    question_text=decision.question_text,
                    options=_view_options(decision, definitions, view_index),
                    use_cache=True,
                    include_token_artifacts=True,
                )
                required = (
                    schema.option_token_embeddings,
                    schema.option_token_ids,
                    schema.option_content_token_mask,
                )
                if any(value is None for value in required):
                    raise RuntimeError("W17 schema token artifacts incomplete")
                views[view_id] = {
                    "schema_hash": receipt.schema_hash,
                    "question_embedding": (
                        schema.question_embedding.detach().cpu().to(torch.float16)
                    ),
                    "option_embeddings": (
                        schema.option_embeddings.detach().cpu().to(torch.float16)
                    ),
                    "question_tokens": (
                        schema.question_token_embeddings.detach().cpu().to(torch.float16)
                        if schema.question_token_embeddings is not None
                        else None
                    ),
                    "question_content_mask": (
                        schema.question_content_token_mask.detach().cpu().bool()
                        if schema.question_content_token_mask is not None
                        else None
                    ),
                    "option_tokens": (
                        schema.option_token_embeddings.detach().cpu().to(torch.float16)
                    ),
                    "option_token_ids": schema.option_token_ids.detach().cpu().long(),
                    "option_content_mask": (
                        schema.option_content_token_mask.detach().cpu().bool()
                    ),
                    "option_ids": tuple(option.option_id for option in schema.options),
                    "option_texts": tuple(option.criterion_text for option in schema.options),
                }

            score_support = [
                option.value
                for option in decision.options
                if option.value is not None
            ]
            decisions.append(
                {
                    "question_id": decision.question_id,
                    "primitive": decision.primitive,
                    "qtype": PRIMITIVE_TO_ID[decision.primitive],
                    "gold_index": int(decision.gold_index),
                    "gold_probabilities": decision.gold_probabilities.detach().cpu().float(),
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

        cases.append(
            {
                "case_id": row.typed.case_id,
                "domain_id": row.domain_id,
                "split": row.split,
                "diagnosis_k": int(row.diagnosis_k),
                "severity": int(row.severity),
                "confidence": row.confidence,
                "fields": {
                    "intent": row.intent_field,
                    "severity": row.severity_field,
                    "confidence": row.confidence_field,
                },
                "full_state_text": row.full_state_text,
                "representations": {
                    "FULL_SINGLE": {
                        "tokens": single_tokens.detach().cpu().to(torch.float16),
                        "token_ids": single_ids.detach().cpu().long(),
                        "segments": _segments(full_single, 0).detach().cpu().to(torch.float16),
                    },
                    "FULL_TRIPLICATE": {
                        "tokens": trip_tokens[0].detach().cpu().to(torch.float16),
                        "token_ids": trip_ids[0].detach().cpu().long(),
                        "max_copy_embedding_diff": max_trip_copy_diff,
                        "max_single_embedding_diff": max_single_trip_diff,
                    },
                    "FIELD_ISOLATED": {
                        "intent_tokens": field_tokens["intent"].detach().cpu().to(torch.float16),
                        "severity_tokens": field_tokens["severity"].detach().cpu().to(torch.float16),
                        "confidence_tokens": field_tokens["confidence"].detach().cpu().to(torch.float16),
                        "intent_token_ids": field_ids["intent"].detach().cpu().long(),
                        "severity_token_ids": field_ids["severity"].detach().cpu().long(),
                        "confidence_token_ids": field_ids["confidence"].detach().cpu().long(),
                    },
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
            "logical_state_compiles_per_candidate_case": {
                "FULL_SINGLE": 1,
                "FULL_TRIPLICATE": 1,
                "FIELD_ISOLATED": 1,
            },
            "a13_invocations_per_candidate_case": {
                "FULL_SINGLE": 1,
                "FULL_TRIPLICATE": 1,
                "FIELD_ISOLATED": 1,
            },
            "encoded_sequences_per_candidate_case": {
                "FULL_SINGLE": 1,
                "FULL_TRIPLICATE": 3,
                "FIELD_ISOLATED": 3,
            },
        },
        "cases": cases,
    }
    validate_w17_cache(cache)
    return cache


def validate_w17_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W17 cache must be dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("W17 cache requires metadata/cases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W17 cache schema")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W17 case count mismatch")
    if metadata.get("training_performed") is not False:
        raise ValueError("W17 must not train")
    if metadata.get("logical_state_compiles_per_candidate_case") != {
        "FULL_SINGLE": 1,
        "FULL_TRIPLICATE": 1,
        "FIELD_ISOLATED": 1,
    }:
        raise ValueError("W17 logical state accounting changed")
    if metadata.get("a13_invocations_per_candidate_case") != {
        "FULL_SINGLE": 1,
        "FULL_TRIPLICATE": 1,
        "FIELD_ISOLATED": 1,
    }:
        raise ValueError("W17 A13 invocation accounting changed")
    if metadata.get("encoded_sequences_per_candidate_case") != {
        "FULL_SINGLE": 1,
        "FULL_TRIPLICATE": 3,
        "FIELD_ISOLATED": 3,
    }:
        raise ValueError("W17 encoded sequence accounting changed")

    seen = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("W17 duplicate/invalid case")
        seen.add(case_id)
        reps = case.get("representations")
        if not isinstance(reps, dict) or set(reps) != {
            "FULL_SINGLE",
            "FULL_TRIPLICATE",
            "FIELD_ISOLATED",
        }:
            raise ValueError("W17 representation set changed")
        single = reps["FULL_SINGLE"]
        trip = reps["FULL_TRIPLICATE"]
        if not torch.equal(single["token_ids"], trip["token_ids"]):
            raise ValueError("W17 full/triplicate token identity changed")
        if float(trip["max_copy_embedding_diff"]) > 1e-5:
            raise ValueError("W17 identical triplicate copies diverged")
        if len(case.get("decisions", [])) != 5:
            raise ValueError("W17 requires five typed decisions")
        for decision in case["decisions"]:
            if set(decision["views"]) != {"D0", "D1", "D2"}:
                raise ValueError("W17 requires D0/D1/D2")
            ids = [
                tuple(decision["views"][view]["option_ids"])
                for view in ("D0", "D1", "D2")
            ]
            if not (ids[0] == ids[1] == ids[2]):
                raise ValueError("W17 paraphrase option identity changed")


def save_w17_cache(cache: dict, path: str | Path) -> Path:
    validate_w17_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w17_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w17_cache(cache)
    return cache


__all__ = [
    "CACHE_SCHEMA",
    "compile_w17_cache",
    "load_w17_cache",
    "save_w17_cache",
    "validate_w17_cache",
]
