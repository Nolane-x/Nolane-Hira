from __future__ import annotations

from math import log
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .mainline import (
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    HiraV0Mainline,
)
from .mainline_m1_authority import M1AuthorityCase
from .mainline_reliability import confidence_diagnostics
from .runtime import PRIMITIVE_TO_ID
from .semantic_core import W28_T0_CHECKPOINT_SHA256

M1_CACHE_SCHEMA = "hira-v0-mainline-m1-frozen-cache-v1"

OOD_FEATURE_NAMES = (
    "state_question_cosine",
    "state_option_max_cosine",
    "state_option_mean_cosine",
    "state_option_std_cosine",
    "max_probability",
    "normalized_entropy",
    "top_margin",
    "normalized_log_k",
)
SEMANTIC_OOD_FEATURE_COUNT = 4
CONFIDENCE_OOD_FEATURE_COUNT = 3


def _cosine_vector(left: Tensor, right: Tensor) -> Tensor:
    if left.ndim != 1 or right.ndim != 2 or left.shape[0] != right.shape[-1]:
        raise ValueError("M1 cosine feature shape mismatch")
    a = F.normalize(left.float(), dim=-1)
    b = F.normalize(right.float(), dim=-1)
    return torch.mv(b, a)


def _semantic_ood_features(memory, schema, probabilities: Tensor) -> Tensor:
    state = memory.global_embedding.detach().float().cpu()
    question = schema.question_embedding.detach().float().cpu()
    options = schema.option_embeddings.detach().float().cpu()
    if state.ndim != 1 or question.ndim != 1 or options.ndim != 2:
        raise ValueError("M1 frozen semantic feature tensors have invalid rank")
    if options.shape[0] < 2:
        raise ValueError("M1 OOD features require K>=2")

    state_question = float(
        F.cosine_similarity(
            state.unsqueeze(0),
            question.unsqueeze(0),
            dim=-1,
        )[0]
    )
    state_option = _cosine_vector(state, options)
    diagnostics = confidence_diagnostics(probabilities.detach().cpu())
    normalized_log_k = log(float(options.shape[0])) / log(255.0)

    values = torch.tensor(
        [
            state_question,
            float(state_option.max()),
            float(state_option.mean()),
            float(state_option.std(unbiased=False)),
            diagnostics.max_probability,
            diagnostics.normalized_entropy,
            diagnostics.top_margin,
            normalized_log_k,
        ],
        dtype=torch.float32,
    )
    if values.numel() != len(OOD_FEATURE_NAMES):
        raise RuntimeError("M1 OOD feature count changed")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("M1 OOD feature vector contains non-finite values")
    return values


def _validate_model(model: HiraV0Mainline) -> None:
    manifest = model.manifest
    if manifest.t0_checkpoint_sha256 != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1 cache requires exact W28 T0 provenance")
    if (
        manifest.transfer_checkpoint_sha256
        != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
    ):
        raise RuntimeError("M1 cache requires exact W34 provisional core")
    if manifest.transfer_core != "provisional":
        raise RuntimeError("M1 cache transfer maturity changed")
    if manifest.production_ready:
        raise RuntimeError("M1 cache cannot use production-ready claim")
    if model.runtime.reliability_calibrator is not None:
        raise RuntimeError("M1 cache must be compiled before calibration")
    if any(parameter.requires_grad for parameter in model.runtime.parameters()):
        raise RuntimeError("M1 cache requires a completely frozen decision core")


@torch.no_grad()
def compile_m1_frozen_cache(
    model: HiraV0Mainline,
    rows: Sequence[M1AuthorityCase],
    *,
    partition: str,
) -> dict[str, object]:
    _validate_model(model)
    if not rows:
        raise ValueError("M1 cache requires at least one authority case")
    if any(row.partition != partition for row in rows):
        raise ValueError("M1 authority rows do not match requested partition")

    before = model.runtime.state_encode_calls
    cases: list[dict[str, object]] = []

    for row in rows:
        session = model.open_session(row.state_text)
        schema, _ = model.runtime.compile_schema(
            primitive=row.primitive,
            question_text=row.question_text,
            options=row.options,
            use_cache=True,
            include_token_artifacts=True,
        )
        decision = session.decide(
            primitive=row.primitive,
            question_text=row.question_text,
            options=row.options,
            use_schema_cache=True,
        )
        if session.query_count != 1:
            raise RuntimeError("M1 cache must execute one decision per state")
        if int(decision.hira.candidate_budget.item()) != len(row.options):
            raise RuntimeError("M1 cache requires full-K decision execution")
        if not bool(decision.hira.selected_mask.all()):
            raise RuntimeError("M1 cache selected mask changed")
        if float(decision.hira.relation_delta.abs().max()) != 0.0:
            raise RuntimeError("M1 cache relation refinement changed")

        logits = decision.logits.detach().float().cpu().clone()
        probabilities = decision.probabilities.detach().float().cpu().clone()
        if logits.is_inference() or probabilities.is_inference():
            logits = logits.clone()
            probabilities = probabilities.clone()

        gold = (
            None
            if row.gold_probabilities is None
            else torch.tensor(row.gold_probabilities, dtype=torch.float32)
        )
        cases.append(
            {
                "case_id": row.case_id,
                "domain_id": row.domain_id,
                "partition": row.partition,
                "primitive": row.primitive,
                "primitive_id": PRIMITIVE_TO_ID[row.primitive],
                "is_ood": bool(row.is_ood),
                "ood_kind": row.ood_kind,
                "confidence_band": row.confidence_band,
                "gold_index": row.gold_index,
                "gold_probabilities": gold,
                "logits": logits,
                "probabilities": probabilities,
                "ood_features": _semantic_ood_features(
                    session.memory,
                    schema,
                    probabilities,
                ),
                "option_values": tuple(
                    None if option.value is None else float(option.value)
                    for option in row.options
                ),
                "option_ids": tuple(option.option_id for option in row.options),
            }
        )

    state_encode_delta = model.runtime.state_encode_calls - before
    if state_encode_delta != len(rows):
        raise RuntimeError("M1 frozen cache violated state-once case encoding")

    cache = {
        "metadata": {
            "schema_version": M1_CACHE_SCHEMA,
            "partition": partition,
            "case_count": len(cases),
            "state_encode_count": state_encode_delta,
            "state_encodes_per_case": 1.0,
            "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
            "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
            "transfer_core_maturity": model.manifest.transfer_core,
            "production_ready": model.manifest.production_ready,
            "decision_core_frozen": True,
            "calibration_applied": False,
            "ood_feature_names": list(OOD_FEATURE_NAMES),
            "semantic_ood_feature_count": SEMANTIC_OOD_FEATURE_COUNT,
            "confidence_ood_feature_count": CONFIDENCE_OOD_FEATURE_COUNT,
        },
        "cases": cases,
    }
    validate_m1_frozen_cache(cache, expected_partition=partition)
    return cache


def validate_m1_frozen_cache(
    cache: dict[str, object],
    *,
    expected_partition: str | None = None,
) -> None:
    if not isinstance(cache, dict):
        raise ValueError("M1 cache must be a dict")
    metadata = cache.get("metadata")
    cases = cache.get("cases")
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise ValueError("M1 cache requires metadata and cases")
    if metadata.get("schema_version") != M1_CACHE_SCHEMA:
        raise ValueError("unexpected M1 cache schema")
    if expected_partition is not None and metadata.get("partition") != expected_partition:
        raise ValueError("M1 cache partition mismatch")
    if int(metadata.get("case_count", -1)) != len(cases):
        raise ValueError("M1 cache case count mismatch")
    if float(metadata.get("state_encodes_per_case", -1.0)) != 1.0:
        raise ValueError("M1 cache state-once contract changed")
    if metadata.get("decision_core_frozen") is not True:
        raise ValueError("M1 cache decision core must be frozen")
    if metadata.get("calibration_applied") is not False:
        raise ValueError("M1 cache must contain pre-calibration logits")
    if metadata.get("production_ready") is not False:
        raise ValueError("M1 cache cannot claim production readiness")
    if metadata.get("t0_checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise ValueError("M1 cache T0 provenance changed")
    if (
        metadata.get("transfer_checkpoint_sha256")
        != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
    ):
        raise ValueError("M1 cache transfer provenance changed")
    if tuple(metadata.get("ood_feature_names", ())) != OOD_FEATURE_NAMES:
        raise ValueError("M1 OOD feature contract changed")
    if int(metadata.get("semantic_ood_feature_count", -1)) < 4:
        raise ValueError("M1 OOD authority cannot be confidence-only")

    seen = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("M1 cache invalid or duplicate case id")
        seen.add(case_id)

        logits = case.get("logits")
        probabilities = case.get("probabilities")
        features = case.get("ood_features")
        if (
            not isinstance(logits, Tensor)
            or not isinstance(probabilities, Tensor)
            or logits.ndim != 1
            or probabilities.shape != logits.shape
            or logits.numel() < 2
        ):
            raise ValueError("M1 cached decision tensor shape changed")
        if not isinstance(features, Tensor) or features.shape != (len(OOD_FEATURE_NAMES),):
            raise ValueError("M1 OOD feature tensor shape changed")
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(features).all()):
            raise ValueError("M1 cache contains non-finite values")
        if abs(float(probabilities.sum()) - 1.0) > 1e-6:
            raise ValueError("M1 cached probability mass changed")

        is_ood = bool(case.get("is_ood"))
        gold = case.get("gold_probabilities")
        gold_index = case.get("gold_index")
        if is_ood:
            if gold is not None or gold_index is not None:
                raise ValueError("M1 OOD rows must not carry fake task gold")
        else:
            if (
                not isinstance(gold, Tensor)
                or gold.shape != logits.shape
                or gold_index is None
            ):
                raise ValueError("M1 ID row missing calibration target")
            if abs(float(gold.sum()) - 1.0) > 1e-6:
                raise ValueError("M1 calibration target mass changed")


__all__ = [
    "CONFIDENCE_OOD_FEATURE_COUNT",
    "M1_CACHE_SCHEMA",
    "OOD_FEATURE_NAMES",
    "SEMANTIC_OOD_FEATURE_COUNT",
    "compile_m1_frozen_cache",
    "validate_m1_frozen_cache",
]
