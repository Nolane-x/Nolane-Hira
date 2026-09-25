from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .anchor_preserving_residual_authority import (
    AnchorPreservingAuthorityCase,
    K_VALUES,
    PARAPHRASE_VIEWS,
)
from .contracts import LogicalOption
from .runtime import NolaneHira, PRIMITIVE_TO_ID


W15_CACHE_SCHEMA_VERSION = "r8-w15-anchor-preserving-cache-v1"


def _sha_lines(values: Sequence[str]) -> str:
    return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _schema_options(
    decision,
    definitions: tuple[tuple[str, str, str], ...],
    view_index: int,
) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=option.option_id,
            criterion_text=paraphrases[view_index],
            value=option.value,
        )
        for option, paraphrases in zip(decision.options, definitions)
    )


@torch.inference_mode()
def compile_w15_cache(
    model: NolaneHira,
    cases: Sequence[AnchorPreservingAuthorityCase],
    *,
    expected_split: str,
) -> dict:
    model.eval()
    before = model.state_encode_calls
    rows: list[dict] = []
    domain_counts: Counter[str] = Counter()
    k_counts: Counter[int] = Counter()

    for authority in cases:
        if authority.split != expected_split:
            raise ValueError("W15 authority split mismatch")
        typed = authority.typed
        memory = model.compile_state(
            typed.state_text,
            segment_tokens=32,
        )
        if memory.content_token_embeddings is None:
            raise RuntimeError("W15 cache requires state content tokens")

        cached_decisions: list[dict] = []
        for decision, definition_rows in zip(
            typed.decisions,
            authority.option_definitions,
        ):
            if len(definition_rows) != len(decision.options):
                raise RuntimeError("W15 option definition count mismatch")

            views: dict[str, dict] = {}
            for view_index, view_id in enumerate(PARAPHRASE_VIEWS):
                options = _schema_options(
                    decision,
                    definition_rows,
                    view_index,
                )
                schema, receipt = model.compile_schema(
                    primitive=decision.primitive,
                    question_text=decision.question_text,
                    options=options,
                    use_cache=False,
                    include_token_artifacts=True,
                )
                required = (
                    schema.question_token_embeddings,
                    schema.question_content_token_mask,
                    schema.option_token_embeddings,
                    schema.option_token_ids,
                    schema.option_content_token_mask,
                )
                if any(value is None for value in required):
                    raise RuntimeError("W15 schema token artifacts incomplete")
                views[view_id] = {
                    "schema_hash": receipt.schema_hash,
                    "question_embedding": (
                        schema.question_embedding.detach().cpu().to(torch.float16)
                    ),
                    "option_embeddings": (
                        schema.option_embeddings.detach().cpu().to(torch.float16)
                    ),
                    "question_tokens": (
                        schema.question_token_embeddings.detach().cpu().to(
                            torch.float16
                        )
                    ),
                    "question_content_mask": (
                        schema.question_content_token_mask.detach().cpu().bool()
                    ),
                    "option_tokens": (
                        schema.option_token_embeddings.detach().cpu().to(
                            torch.float16
                        )
                    ),
                    "option_token_ids": (
                        schema.option_token_ids.detach().cpu().long()
                    ),
                    "option_content_mask": (
                        schema.option_content_token_mask.detach().cpu().bool()
                    ),
                    "option_ids": tuple(option.option_id for option in options),
                    "option_texts": tuple(
                        option.criterion_text for option in options
                    ),
                }

            raw_values = [option.value for option in decision.options]
            if decision.primitive == "score":
                if any(value is None for value in raw_values):
                    raise RuntimeError("W15 score options require numeric values")
                score_support = torch.tensor(
                    raw_values,
                    dtype=torch.float32,
                )
            else:
                score_support = torch.empty(0, dtype=torch.float32)

            cached_decisions.append(
                {
                    "question_id": decision.question_id,
                    "primitive": decision.primitive,
                    "qtype": int(PRIMITIVE_TO_ID[decision.primitive]),
                    "gold_index": int(decision.gold_index),
                    "gold_probabilities": torch.tensor(
                        decision.gold_probabilities,
                        dtype=torch.float32,
                    ),
                    "gold_score": (
                        None
                        if decision.gold_score is None
                        else float(decision.gold_score)
                    ),
                    "score_support": score_support,
                    "views": views,
                }
            )

        rows.append(
            {
                "case_id": typed.case_id,
                "workflow": typed.workflow,
                "split": authority.split,
                "domain_id": authority.domain_id,
                "diagnosis_k": int(authority.diagnosis_k),
                "severity": int(authority.severity),
                "confidence": authority.confidence,
                "state_segments": (
                    memory.segment_embeddings.detach().cpu().to(torch.float16)
                ),
                "state_content_tokens": (
                    memory.content_token_embeddings.detach().cpu().to(
                        torch.float16
                    )
                ),
                "decisions": cached_decisions,
            }
        )
        domain_counts[authority.domain_id] += 1
        k_counts[int(authority.diagnosis_k)] += 1

    state_calls = model.state_encode_calls - before
    if state_calls != len(cases):
        raise RuntimeError("W15 violated one-state-encode-per-case")

    case_ids = [authority.typed.case_id for authority in cases]
    schema_hashes = [
        decision["views"][view]["schema_hash"]
        for row in rows
        for decision in row["decisions"]
        for view in PARAPHRASE_VIEWS
    ]
    cache = {
        "metadata": {
            "schema_version": W15_CACHE_SCHEMA_VERSION,
            "split": expected_split,
            "case_count": len(rows),
            "decision_count": 5 * len(rows),
            "schema_view_count": 15 * len(rows),
            "case_id_sha256": _sha_lines(case_ids),
            "schema_hash_sha256": _sha_lines(schema_hashes),
            "state_encode_calls": state_calls,
            "state_encode_calls_per_case": state_calls / max(1, len(rows)),
            "domain_case_counts": dict(sorted(domain_counts.items())),
            "diagnosis_k_counts": {
                str(k): int(k_counts[k]) for k in K_VALUES
            },
            "confirm_exposed": expected_split.startswith("confirm-"),
            "training_performed": False,
        },
        "cases": rows,
    }
    validate_w15_cache(cache, expected_split=expected_split)
    return cache


def validate_w15_cache(
    cache: dict,
    *,
    expected_split: str | None = None,
) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W15 cache must be a dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("W15 cache requires metadata/cases")
    if metadata.get("schema_version") != W15_CACHE_SCHEMA_VERSION:
        raise ValueError("unexpected W15 cache schema")
    split = metadata.get("split")
    if expected_split is not None and split != expected_split:
        raise ValueError("W15 cache split mismatch")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("W15 case count mismatch")
    if int(metadata.get("decision_count", -1)) != 5 * len(cases):
        raise ValueError("W15 decision count mismatch")
    if int(metadata.get("schema_view_count", -1)) != 15 * len(cases):
        raise ValueError("W15 schema view count mismatch")
    if float(metadata.get("state_encode_calls_per_case", -1.0)) != 1.0:
        raise ValueError("W15 must encode state exactly once per case")

    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("invalid/duplicate W15 case id")
        seen.add(case_id)
        if case.get("split") != split:
            raise ValueError("W15 case/cache split mismatch")
        if int(case.get("diagnosis_k", -1)) not in K_VALUES:
            raise ValueError("invalid W15 diagnosis K")

        segments = case.get("state_segments")
        state_tokens = case.get("state_content_tokens")
        if (
            not isinstance(segments, Tensor)
            or segments.ndim != 2
            or segments.shape[-1] != 256
        ):
            raise ValueError("W15 state_segments must be [S,256]")
        if (
            not isinstance(state_tokens, Tensor)
            or state_tokens.ndim != 2
            or state_tokens.shape[-1] != 256
            or state_tokens.shape[0] < 1
        ):
            raise ValueError("W15 state content tokens invalid")

        decisions = case.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 5:
            raise ValueError("W15 requires five decisions/case")
        if [row.get("question_id") for row in decisions] != [
            "diagnosis",
            "response",
            "needs_review",
            "risk",
            "urgency",
        ]:
            raise ValueError("W15 decision order changed")

        for decision in decisions:
            primitive = decision.get("primitive")
            if primitive not in PRIMITIVE_TO_ID:
                raise ValueError("invalid W15 primitive")
            if int(decision.get("qtype", -1)) != PRIMITIVE_TO_ID[primitive]:
                raise ValueError("W15 primitive/qtype mismatch")

            views = decision.get("views")
            if not isinstance(views, dict) or set(views) != set(PARAPHRASE_VIEWS):
                raise ValueError("W15 decision requires D0/D1/D2")
            option_ids = None
            option_count = None
            for view_id in PARAPHRASE_VIEWS:
                view = views[view_id]
                options = view.get("option_embeddings")
                option_tokens = view.get("option_tokens")
                token_ids = view.get("option_token_ids")
                token_mask = view.get("option_content_mask")
                question = view.get("question_embedding")
                question_tokens = view.get("question_tokens")
                question_mask = view.get("question_content_mask")
                ids = tuple(view.get("option_ids", ()))
                texts = tuple(view.get("option_texts", ()))

                if (
                    not isinstance(question, Tensor)
                    or question.shape != (256,)
                ):
                    raise ValueError("W15 question embedding mismatch")
                if (
                    not isinstance(question_tokens, Tensor)
                    or question_tokens.ndim != 2
                    or question_tokens.shape[-1] != 256
                ):
                    raise ValueError("W15 question tokens mismatch")
                if (
                    not isinstance(question_mask, Tensor)
                    or question_mask.shape != question_tokens.shape[:1]
                    or question_mask.dtype != torch.bool
                    or int(question_mask.sum()) < 1
                ):
                    raise ValueError("W15 question mask mismatch")
                if (
                    not isinstance(options, Tensor)
                    or options.ndim != 2
                    or options.shape[-1] != 256
                    or options.shape[0] < 2
                ):
                    raise ValueError("W15 option embeddings mismatch")
                if (
                    not isinstance(option_tokens, Tensor)
                    or option_tokens.ndim != 3
                    or option_tokens.shape[0] != options.shape[0]
                    or option_tokens.shape[-1] != 256
                ):
                    raise ValueError("W15 option tokens mismatch")
                if (
                    not isinstance(token_ids, Tensor)
                    or token_ids.shape != option_tokens.shape[:2]
                    or token_ids.dtype != torch.long
                ):
                    raise ValueError("W15 option token IDs mismatch")
                if (
                    not isinstance(token_mask, Tensor)
                    or token_mask.shape != option_tokens.shape[:2]
                    or token_mask.dtype != torch.bool
                    or (token_mask.sum(-1) < 1).any()
                ):
                    raise ValueError("W15 option mask mismatch")
                if len(ids) != options.shape[0] or len(texts) != options.shape[0]:
                    raise ValueError("W15 option identity/text mismatch")
                if len(set(ids)) != len(ids):
                    raise ValueError("W15 option IDs must be unique")
                if option_ids is None:
                    option_ids = ids
                    option_count = options.shape[0]
                elif ids != option_ids or options.shape[0] != option_count:
                    raise ValueError("W15 paraphrase option identity changed")

            gold_probs = decision.get("gold_probabilities")
            if (
                not isinstance(gold_probs, Tensor)
                or gold_probs.shape != (option_count,)
                or not torch.isfinite(gold_probs).all()
                or abs(float(gold_probs.sum()) - 1.0) > 1e-6
            ):
                raise ValueError("W15 gold probability contract failed")
            gold = int(decision.get("gold_index", -1))
            if not 0 <= gold < int(option_count):
                raise ValueError("W15 gold index invalid")
            if int(gold_probs.argmax()) != gold:
                raise ValueError("W15 hard/soft gold mismatch")

            support = decision.get("score_support")
            if primitive == "score":
                if (
                    not isinstance(support, Tensor)
                    or support.shape != (option_count,)
                ):
                    raise ValueError("W15 score support mismatch")
                if decision.get("gold_score") is None:
                    raise ValueError("W15 score gold missing")


def save_w15_cache(cache: dict, path: str | Path) -> Path:
    validate_w15_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w15_cache(
    path: str | Path,
    *,
    expected_split: str | None = None,
) -> dict:
    cache = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    validate_w15_cache(cache, expected_split=expected_split)
    return cache


__all__ = [
    "W15_CACHE_SCHEMA_VERSION",
    "compile_w15_cache",
    "load_w15_cache",
    "save_w15_cache",
    "validate_w15_cache",
]
