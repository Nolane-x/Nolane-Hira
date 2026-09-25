from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .conjunctive_authority import ConjunctiveAuthorityCase
from .conjunctive_cache import (
    one_field_negative_indices,
    semantic_signature_sha256,
)
from .runtime import NolaneHira
from .typed_competitive_cache import (
    compile_w6b_cache,
    validate_w6b_cache,
)


W7B_CACHE_SCHEMA_VERSION = "r8-w7b-freeform-attribution-cache-v1"


@torch.inference_mode()
def compile_w7b_cache(
    model: NolaneHira,
    cases: Sequence[ConjunctiveAuthorityCase],
    *,
    expected_split: str,
) -> dict:
    """Compile only production free-form artifacts for W7b.

    State compilation stays exactly once/case via the frozen W6b path.
    Unlike W7, W7b does not encode factor phrases because no W7b scorer
    consumes a factor branch. Synthetic signatures are retained only as
    labels for controlled one-field pair attribution.
    """
    base = compile_w6b_cache(
        model,
        cases,
        expected_split=expected_split,
    )
    if len(base["cases"]) != len(cases):
        raise RuntimeError("W7b base cache/case count mismatch")

    domain_counts: Counter[str] = Counter()
    for cached, authority in zip(base["cases"], cases):
        if cached["case_id"] != authority.typed.case_id:
            raise RuntimeError("W7b base cache order mismatch")
        if authority.split != expected_split:
            raise ValueError("W7b authority split mismatch")

        diagnosis = cached["decisions"][0]
        diagnosis["one_field_negative_indices"] = (
            one_field_negative_indices(authority)
        )
        diagnosis["diagnosis_signatures"] = tuple(
            tuple(row) for row in authority.diagnosis_signatures
        )
        cached["domain_id"] = authority.domain_id
        domain_counts[authority.domain_id] += 1

    metadata = base["metadata"]
    metadata["w7b_schema_version"] = W7B_CACHE_SCHEMA_VERSION
    metadata["semantic_signature_sha256"] = semantic_signature_sha256(cases)
    metadata["domain_case_counts"] = dict(sorted(domain_counts.items()))
    metadata["confirm_exposed"] = expected_split.startswith("confirm-")
    metadata["factor_encoder_batches"] = 0
    metadata["factor_text_count"] = 0

    validate_w7b_cache(base, expected_split=expected_split)
    return base


def validate_w7b_cache(
    cache: dict,
    *,
    expected_split: str | None = None,
) -> None:
    validate_w6b_cache(cache, expected_split=expected_split)
    metadata = cache["metadata"]
    cases = cache["cases"]

    if metadata.get("w7b_schema_version") != W7B_CACHE_SCHEMA_VERSION:
        raise ValueError("unexpected W7b cache schema")
    if float(metadata.get("state_encode_calls_per_case", -1.0)) != 1.0:
        raise ValueError("W7b state-once contract failed")
    if int(metadata.get("factor_encoder_batches", -1)) != 0:
        raise ValueError("W7b must not encode factor phrases")
    if int(metadata.get("factor_text_count", -1)) != 0:
        raise ValueError("W7b factor text count must be zero")

    for case in cases:
        if not str(case.get("domain_id", "")).strip():
            raise ValueError("W7b cached domain id missing")
        diagnosis = case["decisions"][0]
        k = int(case["diagnosis_k"])
        signatures = diagnosis.get("diagnosis_signatures")
        labels = diagnosis.get("one_field_negative_indices")
        if (
            not isinstance(signatures, tuple)
            or len(signatures) != k
            or any(len(row) != 4 for row in signatures)
        ):
            raise ValueError("W7b cached signatures mismatch")
        if not isinstance(labels, dict):
            raise ValueError("W7b one-field labels missing")
        gold = signatures[int(diagnosis["gold_index"])]
        expected = sum(
            sum(a != b for a, b in zip(gold, row)) == 1
            for index, row in enumerate(signatures)
            if index != int(diagnosis["gold_index"])
        )
        observed = sum(
            len(labels.get(role, labels.get(str(role), [])))
            for role in range(4)
        )
        if observed != expected:
            raise ValueError("W7b one-field label accounting mismatch")


def save_w7b_cache(cache: dict, path: str | Path) -> Path:
    validate_w7b_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w7b_cache(
    path: str | Path,
    *,
    expected_split: str | None = None,
) -> dict:
    cache = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    validate_w7b_cache(cache, expected_split=expected_split)
    return cache
