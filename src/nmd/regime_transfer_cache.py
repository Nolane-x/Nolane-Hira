from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .regime_transfer_authority import RegimeTransferView
from .runtime import NolaneHira

CACHE_SCHEMA = "r8-w16-regime-transfer-cache-v1"
VALID_DOMAINS = {"CG", "CH", "CI", "CJ"}
VALID_VIEWS = {"D0", "D1", "D2"}
VALID_K = {4, 8, 16}


def _hash_ids(values: Sequence[str]) -> str:
    return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _content_row(batch, index: int) -> tuple[Tensor, Tensor]:
    mask = batch.attention_mask[index].bool()
    if batch.special_token_mask is None:
        content_mask = mask
    else:
        content_mask = mask & ~batch.special_token_mask[index].bool()
    tokens = batch.token_embeddings[index][content_mask]
    if batch.token_ids is None:
        raise RuntimeError("W16 requires encoder token IDs")
    ids = batch.token_ids[index][content_mask].long()
    if tokens.shape[0] < 1:
        raise RuntimeError("W16 state requires at least one content token")
    return tokens, ids


@torch.inference_mode()
def compile_w16_cache(
    model: NolaneHira,
    rows: Sequence[RegimeTransferView],
) -> dict:
    model.eval()
    by_base: dict[str, list[RegimeTransferView]] = defaultdict(list)
    for row in rows:
        by_base[row.base_id].append(row)

    bases: list[dict] = []
    state_encoder_batches = 0
    encoded_state_texts = 0

    for base_id in sorted(by_base):
        base_rows = by_base[base_id]
        domains = {row.domain_id for row in base_rows}
        intents = {row.intent_id for row in base_rows}
        bare_texts = {row.bare_state_text for row in base_rows}
        decorated_texts = {row.decorated_state_text for row in base_rows}
        questions = {row.question_text for row in base_rows}
        severity = {row.severity for row in base_rows}
        confidence = {row.confidence for row in base_rows}
        if not all(
            len(values) == 1
            for values in (
                domains,
                intents,
                bare_texts,
                decorated_texts,
                questions,
                severity,
                confidence,
            )
        ):
            raise ValueError("W16 base identity mismatch")

        bare = next(iter(bare_texts))
        decorated = next(iter(decorated_texts))
        state_batch = model.encoder.encode_texts([bare, decorated])
        state_encoder_batches += 1
        encoded_state_texts += 2

        r0_tokens, r0_ids = _content_row(state_batch, 0)
        r1_tokens, r1_ids = _content_row(state_batch, 1)
        prefix_n = int(r0_ids.numel())
        if r1_ids.numel() <= prefix_n:
            raise RuntimeError("W16 decorated state must add content tokens")
        if not torch.equal(r1_ids[:prefix_n].cpu(), r0_ids.cpu()):
            raise RuntimeError("W16 decorated state prefix token IDs changed")
        r2_tokens = r1_tokens[:prefix_n]
        r2_ids = r1_ids[:prefix_n]

        views: list[dict] = []
        for row in sorted(
            base_rows,
            key=lambda item: (item.diagnosis_k, item.view_id),
        ):
            schema, receipt = model.compile_schema(
                primitive="choice",
                question_text=row.question_text,
                options=row.logical_options(),
                use_cache=False,
                include_token_artifacts=True,
            )
            required = (
                schema.option_token_embeddings,
                schema.option_token_ids,
                schema.option_content_token_mask,
            )
            if any(value is None for value in required):
                raise RuntimeError("W16 schema token artifacts incomplete")
            if tuple(option.option_id for option in schema.options) != row.option_ids:
                raise RuntimeError("W16 compiled option identity changed")

            views.append(
                {
                    "case_id": row.case_id,
                    "view_id": row.view_id,
                    "diagnosis_k": int(row.diagnosis_k),
                    "gold_index": int(row.gold_index),
                    "option_ids": row.option_ids,
                    "option_texts": row.option_texts,
                    "schema_hash": receipt.schema_hash,
                    "option_tokens": (
                        schema.option_token_embeddings.detach().cpu().to(torch.float16)
                    ),
                    "option_token_ids": (
                        schema.option_token_ids.detach().cpu().long()
                    ),
                    "option_content_mask": (
                        schema.option_content_token_mask.detach().cpu().bool()
                    ),
                }
            )

        bases.append(
            {
                "base_id": base_id,
                "domain_id": next(iter(domains)),
                "intent_id": next(iter(intents)),
                "bare_state_text": bare,
                "decorated_state_text": decorated,
                "severity": int(next(iter(severity))),
                "confidence": str(next(iter(confidence))),
                "state_renderings": {
                    "R0": {
                        "content_tokens": r0_tokens.detach().cpu().to(torch.float16),
                        "content_token_ids": r0_ids.detach().cpu().long(),
                        "active_token_count": prefix_n,
                    },
                    "R1": {
                        "content_tokens": r1_tokens.detach().cpu().to(torch.float16),
                        "content_token_ids": r1_ids.detach().cpu().long(),
                        "active_token_count": int(r1_ids.numel()),
                    },
                    "R2": {
                        "content_tokens": r2_tokens.detach().cpu().to(torch.float16),
                        "content_token_ids": r2_ids.detach().cpu().long(),
                        "active_token_count": prefix_n,
                    },
                },
                "prefix_token_identity": True,
                "views": views,
            }
        )

    cache = {
        "metadata": {
            "schema_version": CACHE_SCHEMA,
            "base_count": len(bases),
            "view_count": sum(len(base["views"]) for base in bases),
            "state_encoder_batches": state_encoder_batches,
            "encoded_state_text_count": encoded_state_texts,
            "state_encoder_batches_per_base": state_encoder_batches / max(1, len(bases)),
            "encoded_state_texts_per_base": encoded_state_texts / max(1, len(bases)),
            "r2_additional_encoder_calls": 0,
            "prefix_token_identity_rate": (
                sum(bool(base["prefix_token_identity"]) for base in bases)
                / max(1, len(bases))
            ),
            "base_id_sha256": _hash_ids([str(base["base_id"]) for base in bases]),
            "case_id_sha256": _hash_ids(
                [
                    str(view["case_id"])
                    for base in bases
                    for view in base["views"]
                ]
            ),
        },
        "bases": bases,
    }
    validate_w16_cache(cache)
    return cache


def validate_w16_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W16 cache must be dict")
    metadata = cache.get("metadata")
    bases = cache.get("bases")
    if not isinstance(metadata, dict) or not isinstance(bases, list):
        raise ValueError("W16 cache requires metadata/bases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W16 cache schema")
    if int(metadata.get("base_count", -1)) != len(bases):
        raise ValueError("W16 base count mismatch")
    if float(metadata.get("encoded_state_texts_per_base", -1)) != 2.0:
        raise ValueError("W16 requires exactly R0+R1 state encodes/base")
    if int(metadata.get("r2_additional_encoder_calls", -1)) != 0:
        raise ValueError("W16 R2 must derive from R1 without re-encoding")
    if float(metadata.get("prefix_token_identity_rate", -1)) != 1.0:
        raise ValueError("W16 prefix token identity failed")

    seen_bases: set[str] = set()
    seen_cases: set[str] = set()
    view_count = 0

    for base in bases:
        base_id = str(base.get("base_id", ""))
        if not base_id or base_id in seen_bases:
            raise ValueError("invalid/duplicate W16 base")
        seen_bases.add(base_id)
        if base.get("domain_id") not in VALID_DOMAINS:
            raise ValueError("invalid W16 domain")
        renderings = base.get("state_renderings")
        if not isinstance(renderings, dict) or set(renderings) != {"R0", "R1", "R2"}:
            raise ValueError("W16 rendering set changed")

        for rendering in ("R0", "R1", "R2"):
            row = renderings[rendering]
            tokens = row.get("content_tokens")
            ids = row.get("content_token_ids")
            if (
                not isinstance(tokens, Tensor)
                or tokens.ndim != 2
                or tokens.shape[-1] != 256
            ):
                raise ValueError("W16 state content tokens invalid")
            if (
                not isinstance(ids, Tensor)
                or ids.ndim != 1
                or ids.shape[0] != tokens.shape[0]
                or ids.dtype != torch.long
            ):
                raise ValueError("W16 state token IDs invalid")
            if int(row.get("active_token_count", -1)) != tokens.shape[0]:
                raise ValueError("W16 active token count mismatch")

        r0 = renderings["R0"]
        r1 = renderings["R1"]
        r2 = renderings["R2"]
        if r1["content_tokens"].shape[0] <= r0["content_tokens"].shape[0]:
            raise ValueError("W16 R1 must contain suffix tokens")
        if not torch.equal(r0["content_token_ids"], r2["content_token_ids"]):
            raise ValueError("W16 R2 prefix token IDs differ from R0")
        if not torch.equal(
            r1["content_token_ids"][: r0["content_token_ids"].shape[0]],
            r0["content_token_ids"],
        ):
            raise ValueError("W16 R1 prefix token IDs differ from R0")
        if r2["content_tokens"].shape[0] != r0["content_tokens"].shape[0]:
            raise ValueError("W16 R2 prefix width changed")

        views = base.get("views")
        if not isinstance(views, list) or len(views) != 9:
            raise ValueError("W16 each base requires exactly 9 views")

        identities: dict[int, tuple[str, ...]] = {}
        golds: dict[int, int] = {}
        view_sets: dict[int, set[str]] = defaultdict(set)
        for view in views:
            view_count += 1
            case_id = str(view.get("case_id", ""))
            if not case_id or case_id in seen_cases:
                raise ValueError("invalid/duplicate W16 case")
            seen_cases.add(case_id)
            k = int(view.get("diagnosis_k", -1))
            view_id = str(view.get("view_id", ""))
            if k not in VALID_K or view_id not in VALID_VIEWS:
                raise ValueError("invalid W16 view/K")
            view_sets[k].add(view_id)
            option_ids = tuple(view.get("option_ids", ()))
            gold = int(view.get("gold_index", -1))
            otokens = view.get("option_tokens")
            omask = view.get("option_content_mask")
            oids = view.get("option_token_ids")
            if len(option_ids) != k or len(set(option_ids)) != k or not 0 <= gold < k:
                raise ValueError("W16 option identity invalid")
            if (
                not isinstance(otokens, Tensor)
                or otokens.ndim != 3
                or otokens.shape[0] != k
                or otokens.shape[-1] != 256
            ):
                raise ValueError("W16 option tokens invalid")
            if (
                not isinstance(omask, Tensor)
                or omask.shape != otokens.shape[:2]
                or omask.dtype != torch.bool
            ):
                raise ValueError("W16 option mask invalid")
            if (
                not isinstance(oids, Tensor)
                or oids.shape != omask.shape
                or oids.dtype != torch.long
            ):
                raise ValueError("W16 option token IDs invalid")
            if k not in identities:
                identities[k] = option_ids
                golds[k] = gold
            elif identities[k] != option_ids or golds[k] != gold:
                raise ValueError("W16 paraphrase identity changed")

        if any(view_sets[k] != VALID_VIEWS for k in VALID_K):
            raise ValueError("W16 requires D0/D1/D2 at every K")
        ids4, ids8, ids16 = identities[4], identities[8], identities[16]
        if not (set(ids4) < set(ids8) < set(ids16)):
            raise ValueError("W16 nested membership failed")
        if [x for x in ids8 if x in ids4] != list(ids4):
            raise ValueError("W16 K4 order changed at K8")
        if [x for x in ids16 if x in ids8] != list(ids8):
            raise ValueError("W16 K8 order changed at K16")

    if int(metadata.get("view_count", -1)) != view_count:
        raise ValueError("W16 view count mismatch")


def save_w16_cache(cache: dict, path: str | Path) -> Path:
    validate_w16_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w16_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w16_cache(cache)
    return cache


__all__ = [
    "CACHE_SCHEMA",
    "compile_w16_cache",
    "load_w16_cache",
    "save_w16_cache",
    "validate_w16_cache",
]
