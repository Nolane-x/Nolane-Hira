from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .conjunctive_authority import (
    ConjunctiveAuthorityCase,
    signature_distance,
)
from .runtime import NolaneHira
from .typed_competitive_cache import (
    compile_w6b_cache,
    validate_w6b_cache,
)


W7_CACHE_SCHEMA_VERSION = "r8-w7-conjunctive-cache-v1"


def _sha256_rows(rows: Sequence[str]) -> str:
    return sha256("\n".join(rows).encode("utf-8")).hexdigest()


def factor_identity_sha256(
    cases: Sequence[ConjunctiveAuthorityCase],
) -> str:
    rows: list[str] = []
    for case in sorted(cases, key=lambda row: row.typed.case_id):
        diagnosis = case.typed.decisions[0]
        if len(diagnosis.options) != len(case.diagnosis_factors):
            raise ValueError("W7 factor/option count mismatch")
        for option, factors in zip(
            diagnosis.options,
            case.diagnosis_factors,
        ):
            rows.append(
                "\x1e".join(
                    (
                        case.typed.case_id,
                        option.option_id,
                        option.criterion_text,
                        "\x1d".join(factors),
                    )
                )
            )
    return _sha256_rows(rows)


def semantic_signature_sha256(
    cases: Sequence[ConjunctiveAuthorityCase],
) -> str:
    rows: list[str] = []
    for case in sorted(cases, key=lambda row: row.typed.case_id):
        rows.append(
            "\x1e".join(
                (
                    case.typed.case_id,
                    case.domain_id,
                    str(case.diagnosis_k),
                    "\x1d".join(
                        "\x1f".join(signature)
                        for signature in case.diagnosis_signatures
                    ),
                )
            )
        )
    return _sha256_rows(rows)


def one_field_negative_indices(
    case: ConjunctiveAuthorityCase,
) -> dict[int, list[int]]:
    diagnosis = case.typed.decisions[0]
    gold_index = int(diagnosis.gold_index)
    gold = case.diagnosis_signatures[gold_index]
    result = {index: [] for index in range(4)}
    for index, signature in enumerate(case.diagnosis_signatures):
        if index == gold_index:
            continue
        changed = [
            role
            for role, (left, right) in enumerate(zip(gold, signature))
            if left != right
        ]
        if len(changed) == 1:
            result[changed[0]].append(index)
    if sum(len(rows) for rows in result.values()) < 1:
        raise ValueError("W7 requires one-field diagnosis negatives")
    return result


def factor_target_mask(
    case: ConjunctiveAuthorityCase,
) -> Tensor:
    diagnosis = case.typed.decisions[0]
    gold = case.diagnosis_signatures[int(diagnosis.gold_index)]
    rows = [
        [candidate[role] == gold[role] for role in range(4)]
        for candidate in case.diagnosis_signatures
    ]
    target = torch.tensor(rows, dtype=torch.bool)
    if target.shape != (int(case.diagnosis_k), 4):
        raise ValueError("W7 factor-target shape mismatch")
    if not bool(target[int(diagnosis.gold_index)].all()):
        raise ValueError("W7 gold factor target must be all-positive")
    return target


def _content_mask(batch) -> Tensor:
    mask = batch.attention_mask.bool()
    if batch.special_token_mask is None:
        return mask
    return mask & ~batch.special_token_mask.bool()


@torch.inference_mode()
def compile_w7_cache(
    model: NolaneHira,
    cases: Sequence[ConjunctiveAuthorityCase],
    *,
    expected_split: str,
) -> dict:
    """Compile production typed artifacts plus explicit factor-token artifacts.

    State compilation is delegated to the frozen W6b cache path and therefore
    remains exactly once/case. Factor phrases are schema-side encoder work and
    are counted separately; they never trigger a second state compilation.
    """
    base = compile_w6b_cache(
        model,
        cases,
        expected_split=expected_split,
    )
    if len(base["cases"]) != len(cases):
        raise RuntimeError("W7 base cache/case count mismatch")

    factor_batches = 0
    factor_text_count = 0
    domain_counts: Counter[str] = Counter()

    for cached, authority in zip(base["cases"], cases):
        if cached["case_id"] != authority.typed.case_id:
            raise RuntimeError("W7 base cache order mismatch")
        if authority.split != expected_split:
            raise ValueError("W7 authority split mismatch")

        diagnosis = cached["decisions"][0]
        factors = authority.diagnosis_factors
        if len(factors) != int(authority.diagnosis_k):
            raise RuntimeError("W7 factor cardinality mismatch")
        if any(len(row) != 4 for row in factors):
            raise RuntimeError("W7 requires four factors per diagnosis option")

        factor_texts = [
            phrase
            for option_factors in factors
            for phrase in option_factors
        ]
        batch = model.encoder.encode_texts(factor_texts)
        factor_batches += 1
        factor_text_count += len(factor_texts)

        tokens = batch.token_embeddings
        if tokens.ndim != 3 or tokens.shape[-1] != 256:
            raise RuntimeError("W7 factor encoder token shape mismatch")
        mask = _content_mask(batch)
        if mask.shape != tokens.shape[:2]:
            raise RuntimeError("W7 factor content-mask shape mismatch")
        if (mask.sum(-1) < 1).any():
            raise RuntimeError("W7 factor phrase has no content token")

        k = int(authority.diagnosis_k)
        width = int(tokens.shape[1])
        diagnosis["factor_tokens"] = (
            tokens.reshape(k, 4, width, 256)
            .detach().cpu().to(torch.float16)
        )
        diagnosis["factor_token_mask"] = (
            mask.reshape(k, 4, width)
            .detach().cpu().bool()
        )
        diagnosis["factor_present_mask"] = torch.ones(
            (k, 4),
            dtype=torch.bool,
        )
        diagnosis["factor_target_mask"] = factor_target_mask(authority)
        diagnosis["one_field_negative_indices"] = (
            one_field_negative_indices(authority)
        )
        diagnosis["diagnosis_signatures"] = tuple(
            tuple(row)
            for row in authority.diagnosis_signatures
        )
        diagnosis["diagnosis_factors"] = tuple(
            tuple(row)
            for row in authority.diagnosis_factors
        )

        cached["domain_id"] = authority.domain_id
        cached["factor_role_count"] = 4
        domain_counts[authority.domain_id] += 1

    metadata = base["metadata"]
    metadata["w7_schema_version"] = W7_CACHE_SCHEMA_VERSION
    metadata["factor_identity_sha256"] = factor_identity_sha256(cases)
    metadata["semantic_signature_sha256"] = semantic_signature_sha256(cases)
    metadata["factor_encoder_batches"] = factor_batches
    metadata["factor_text_count"] = factor_text_count
    metadata["factor_texts_per_case"] = (
        factor_text_count / max(1, len(cases))
    )
    metadata["domain_case_counts"] = dict(sorted(domain_counts.items()))
    metadata["confirm_exposed"] = expected_split.startswith("confirm-")

    validate_w7_cache(base, expected_split=expected_split)
    return base


def validate_w7_cache(
    cache: dict,
    *,
    expected_split: str | None = None,
) -> None:
    validate_w6b_cache(cache, expected_split=expected_split)
    metadata = cache["metadata"]
    cases = cache["cases"]
    if metadata.get("w7_schema_version") != W7_CACHE_SCHEMA_VERSION:
        raise ValueError("unexpected W7 cache schema")
    if int(metadata.get("factor_encoder_batches", -1)) != len(cases):
        raise ValueError("W7 factor encoder batch accounting mismatch")
    if float(metadata.get("state_encode_calls_per_case", -1.0)) != 1.0:
        raise ValueError("W7 state-once contract failed")

    for case in cases:
        if not str(case.get("domain_id", "")).strip():
            raise ValueError("W7 cached domain id missing")
        if int(case.get("factor_role_count", -1)) != 4:
            raise ValueError("W7 factor role count mismatch")
        diagnosis = case["decisions"][0]
        k = int(case["diagnosis_k"])

        factor_tokens = diagnosis.get("factor_tokens")
        factor_mask = diagnosis.get("factor_token_mask")
        factor_present = diagnosis.get("factor_present_mask")
        factor_targets = diagnosis.get("factor_target_mask")
        signatures = diagnosis.get("diagnosis_signatures")
        phrases = diagnosis.get("diagnosis_factors")
        labels = diagnosis.get("one_field_negative_indices")

        if (
            not isinstance(factor_tokens, Tensor)
            or factor_tokens.ndim != 4
            or factor_tokens.shape[0] != k
            or factor_tokens.shape[1] != 4
            or factor_tokens.shape[-1] != 256
        ):
            raise ValueError("W7 factor_tokens must be [K,4,T,256]")
        if (
            not isinstance(factor_mask, Tensor)
            or factor_mask.shape != factor_tokens.shape[:3]
            or factor_mask.dtype != torch.bool
            or (factor_mask.sum(-1) < 1).any()
        ):
            raise ValueError("W7 factor_token_mask mismatch")
        if (
            not isinstance(factor_present, Tensor)
            or factor_present.shape != (k, 4)
            or factor_present.dtype != torch.bool
            or not bool(factor_present.all())
        ):
            raise ValueError("W7 factor_present_mask mismatch")
        if (
            not isinstance(factor_targets, Tensor)
            or factor_targets.shape != (k, 4)
            or factor_targets.dtype != torch.bool
        ):
            raise ValueError("W7 factor_target_mask mismatch")
        gold_index = int(diagnosis["gold_index"])
        if not bool(factor_targets[gold_index].all()):
            raise ValueError("W7 gold factor target mismatch")
        if (
            not isinstance(signatures, tuple)
            or len(signatures) != k
            or any(len(row) != 4 for row in signatures)
        ):
            raise ValueError("W7 cached signatures mismatch")
        if (
            not isinstance(phrases, tuple)
            or len(phrases) != k
            or any(len(row) != 4 for row in phrases)
        ):
            raise ValueError("W7 cached factor phrases mismatch")
        if not isinstance(labels, dict):
            raise ValueError("W7 one-field labels missing")
        negative_total = sum(
            len(labels.get(role, labels.get(str(role), [])))
            for role in range(4)
        )
        expected_one_field = sum(
            signature_distance(
                signatures[gold_index],
                signature,
            ) == 1
            for index, signature in enumerate(signatures)
            if index != gold_index
        )
        if negative_total != expected_one_field:
            raise ValueError("W7 one-field label accounting mismatch")


def save_w7_cache(cache: dict, path: str | Path) -> Path:
    validate_w7_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w7_cache(
    path: str | Path,
    *,
    expected_split: str | None = None,
) -> dict:
    cache = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    validate_w7_cache(cache, expected_split=expected_split)
    return cache
