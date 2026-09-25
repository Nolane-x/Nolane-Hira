from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .anchor_residual_authority import AnchorResidualView
from .runtime import NolaneHira

CACHE_SCHEMA = "r8-w12-anchor-residual-cache-v1"


def _hash_ids(values: Sequence[str]) -> str:
    return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


@torch.inference_mode()
def compile_w12_cache(
    model: NolaneHira,
    rows: Sequence[AnchorResidualView],
) -> dict:
    model.eval()
    by_base: dict[str, list[AnchorResidualView]] = defaultdict(list)
    for row in rows:
        by_base[row.base_id].append(row)

    before = model.state_encode_calls
    bases: list[dict] = []

    for base_id in sorted(by_base):
        base_rows = by_base[base_id]
        state_texts = {row.state_text for row in base_rows}
        domains = {row.domain_id for row in base_rows}
        intents = {row.intent_id for row in base_rows}
        questions = {row.question_text for row in base_rows}
        if not (
            len(state_texts) == len(domains) == len(intents) == len(questions) == 1
        ):
            raise ValueError("W12 base identity mismatch")

        state_text = next(iter(state_texts))
        memory = model.compile_state(state_text, segment_tokens=32)
        if memory.content_token_embeddings is None:
            raise RuntimeError("W12 requires state content tokens")

        views: list[dict] = []
        for row in sorted(base_rows, key=lambda x: (x.diagnosis_k, x.view_id)):
            schema, receipt = model.compile_schema(
                primitive="choice",
                question_text=row.question_text,
                options=row.logical_options(),
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
            if any(x is None for x in required):
                raise RuntimeError("W12 schema token artifacts incomplete")
            if tuple(option.option_id for option in schema.options) != row.option_ids:
                raise RuntimeError("W12 compiled option identity changed")
            views.append(
                {
                    "case_id": row.case_id,
                    "view_id": row.view_id,
                    "diagnosis_k": int(row.diagnosis_k),
                    "gold_index": int(row.gold_index),
                    "option_ids": row.option_ids,
                    "option_texts": row.option_texts,
                    "schema_hash": receipt.schema_hash,
                    "question_embedding": schema.question_embedding.detach().cpu().to(torch.float16),
                    "option_embeddings": schema.option_embeddings.detach().cpu().to(torch.float16),
                    "question_tokens": schema.question_token_embeddings.detach().cpu().to(torch.float16),
                    "question_content_mask": schema.question_content_token_mask.detach().cpu().bool(),
                    "option_tokens": schema.option_token_embeddings.detach().cpu().to(torch.float16),
                    "option_token_ids": schema.option_token_ids.detach().cpu().long(),
                    "option_content_mask": schema.option_content_token_mask.detach().cpu().bool(),
                }
            )
        bases.append(
            {
                "base_id": base_id,
                "domain_id": next(iter(domains)),
                "intent_id": next(iter(intents)),
                "state_text": state_text,
                "state_segments": memory.segment_embeddings.detach().cpu().to(torch.float16),
                "state_content_tokens": memory.content_token_embeddings.detach().cpu().to(torch.float16),
                "views": views,
            }
        )

    state_calls = model.state_encode_calls - before
    if state_calls != len(bases):
        raise RuntimeError("W12 violated one-state-encode-per-base contract")

    cache = {
        "metadata": {
            "schema_version": CACHE_SCHEMA,
            "base_count": len(bases),
            "view_count": sum(len(base["views"]) for base in bases),
            "state_encode_calls": state_calls,
            "state_encode_calls_per_base": state_calls / max(1, len(bases)),
            "base_id_sha256": _hash_ids([str(base["base_id"]) for base in bases]),
            "case_id_sha256": _hash_ids([
                str(view["case_id"])
                for base in bases
                for view in base["views"]
            ]),
        },
        "bases": bases,
    }
    validate_w12_cache(cache)
    return cache


def validate_w12_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W12 cache must be dict")
    metadata = cache.get("metadata")
    bases = cache.get("bases")
    if not isinstance(metadata, dict) or not isinstance(bases, list):
        raise ValueError("W12 cache requires metadata/bases")
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected W12 cache schema")
    if int(metadata.get("base_count", -1)) != len(bases):
        raise ValueError("W12 base count mismatch")
    if float(metadata.get("state_encode_calls_per_base", -1)) != 1.0:
        raise ValueError("W12 state must encode once/base")

    seen_bases: set[str] = set()
    seen_cases: set[str] = set()
    view_count = 0

    for base in bases:
        base_id = str(base.get("base_id", ""))
        if not base_id or base_id in seen_bases:
            raise ValueError("invalid/duplicate W12 base")
        seen_bases.add(base_id)
        if base.get("domain_id") not in {"BN", "BO", "BP", "BQ"}:
            raise ValueError("invalid W12 domain")
        state_tokens = base.get("state_content_tokens")
        segments = base.get("state_segments")
        if not isinstance(state_tokens, Tensor) or state_tokens.ndim != 2 or state_tokens.shape[-1] != 256:
            raise ValueError("W12 state content tokens invalid")
        if not isinstance(segments, Tensor) or segments.ndim != 2 or segments.shape[-1] != 256:
            raise ValueError("W12 state segments invalid")

        views = base.get("views")
        if not isinstance(views, list) or len(views) != 6:
            raise ValueError("W12 each base requires exactly 6 views")

        identities: dict[int, tuple[str, ...]] = {}
        golds: dict[int, int] = {}
        for view in views:
            view_count += 1
            case_id = str(view.get("case_id", ""))
            if not case_id or case_id in seen_cases:
                raise ValueError("invalid/duplicate W12 case")
            seen_cases.add(case_id)
            k = int(view.get("diagnosis_k", -1))
            view_id = view.get("view_id")
            if k not in {4, 8, 16} or view_id not in {"definition", "label"}:
                raise ValueError("invalid W12 view/K")

            option_ids = tuple(view.get("option_ids", ()))
            gold = int(view.get("gold_index", -1))
            otokens = view.get("option_tokens")
            omask = view.get("option_content_mask")
            oids = view.get("option_token_ids")
            qtokens = view.get("question_tokens")
            qmask = view.get("question_content_mask")
            if len(option_ids) != k or len(set(option_ids)) != k or not 0 <= gold < k:
                raise ValueError("W12 option identity invalid")
            if not isinstance(otokens, Tensor) or otokens.ndim != 3 or otokens.shape[0] != k or otokens.shape[-1] != 256:
                raise ValueError("W12 option tokens invalid")
            if not isinstance(omask, Tensor) or omask.shape != otokens.shape[:2] or omask.dtype != torch.bool:
                raise ValueError("W12 option mask invalid")
            if (omask.sum(-1) < 1).any():
                raise ValueError("W12 option requires content token")
            if not isinstance(oids, Tensor) or oids.shape != omask.shape or oids.dtype != torch.long:
                raise ValueError("W12 option token ids invalid")
            if not isinstance(qtokens, Tensor) or qtokens.ndim != 2 or qtokens.shape[-1] != 256:
                raise ValueError("W12 question tokens invalid")
            if not isinstance(qmask, Tensor) or qmask.shape != qtokens.shape[:1] or qmask.dtype != torch.bool:
                raise ValueError("W12 question mask invalid")

            if k not in identities:
                identities[k] = option_ids
                golds[k] = gold
            elif option_ids != identities[k] or gold != golds[k]:
                raise ValueError("W12 paired label/definition identity changed")

        ids4, ids8, ids16 = identities[4], identities[8], identities[16]
        if not (set(ids4) < set(ids8) < set(ids16)):
            raise ValueError("W12 nested membership failed")
        if [x for x in ids8 if x in ids4] != list(ids4):
            raise ValueError("W12 K4 order changed at K8")
        if [x for x in ids16 if x in ids8] != list(ids8):
            raise ValueError("W12 K8 order changed at K16")

    if int(metadata.get("view_count", -1)) != view_count:
        raise ValueError("W12 view count mismatch")


def save_w12_cache(cache: dict, path: str | Path) -> Path:
    validate_w12_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w12_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w12_cache(cache)
    return cache
