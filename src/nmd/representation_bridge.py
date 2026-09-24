from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .hira import HIRACore
from .representation_bridge_authority import (
    DOMAINS,
    ROLE_KEYS,
    VIEW_IDS,
    RepresentationBridgeView,
    canonical_tagged_text,
    factorized_role_value_phrases,
)
from .runtime import NolaneHira, PRIMITIVE_TO_ID
from .second_order_localization import (
    _content_from_text_batch,
    _isolated_score,
    _rank_metrics,
    isolated_value_probe,
    role_value_phrase_probe,
)


PRIOR_W6G_P0_ANCHOR = 0.60
PRIOR_W6H_P2_ANCHOR = 0.95
DOMAIN_SHIFT_MATERIAL_DELTA = 0.10


def _semantic_view_sha256(
    views: Sequence[RepresentationBridgeView],
) -> str:
    rows = []
    for view in sorted(views, key=lambda row: row.typed.case_id):
        rows.append(
            "\x1e".join(
                (
                    view.typed.case_id,
                    view.base_id,
                    view.domain_id,
                    view.view_id,
                    "\x1f".join(view.gold_signature),
                    "\x1d".join(
                        "\x1f".join(signature)
                        for signature in view.option_signatures
                    ),
                )
            )
        )
    return sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _case_id_sha256(
    views: Sequence[RepresentationBridgeView],
) -> str:
    payload = "\n".join(
        sorted(view.typed.case_id for view in views)
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _content_mask(batch) -> Tensor:
    mask = batch.attention_mask.bool()
    if batch.special_token_mask is None:
        return mask
    return mask & ~batch.special_token_mask.bool()


def _rendered_option_artifacts(
    batch,
    *,
    start: int,
    count: int,
) -> dict[str, Tensor]:
    stop = start + count
    pooled = F.normalize(
        batch.pooled_embeddings[start:stop],
        dim=-1,
    )
    token_ids = batch.token_ids
    if token_ids is None:
        width = batch.token_embeddings.shape[1]
        ids = torch.arange(
            width,
            dtype=torch.long,
            device=batch.token_embeddings.device,
        ).unsqueeze(0).expand(count, -1)
    else:
        ids = token_ids[start:stop].long()
    return {
        "option_embeddings": pooled.detach().cpu().to(torch.float16),
        "option_tokens": (
            batch.token_embeddings[start:stop].detach().cpu().to(torch.float16)
        ),
        "option_token_ids": ids.detach().cpu().long(),
        "option_content_mask": (
            _content_mask(batch)[start:stop].detach().cpu().bool()
        ),
    }


def _factorized_artifacts(
    batch,
    *,
    start: int,
    option_count: int,
) -> dict[str, Tensor]:
    count = option_count * 4
    stop = start + count
    tokens = batch.token_embeddings[start:stop]
    mask = _content_mask(batch)[start:stop]
    width = tokens.shape[1]
    dim = tokens.shape[2]
    return {
        "tokens": (
            tokens.reshape(option_count, 4, width, dim)
            .detach().cpu().to(torch.float16)
        ),
        "mask": (
            mask.reshape(option_count, 4, width)
            .detach().cpu().bool()
        ),
    }


def validate_w6i_cache(cache: dict) -> None:
    metadata = cache.get("metadata", {})
    if metadata.get("schema_version") != "r8-w6i-representation-cache-v1":
        raise ValueError("unexpected W6i cache schema")
    bases = cache.get("bases", [])
    views = cache.get("views", [])
    if int(metadata.get("base_count", -1)) != len(bases):
        raise ValueError("W6i base count mismatch")
    if int(metadata.get("view_count", -1)) != len(views):
        raise ValueError("W6i view count mismatch")
    if int(metadata.get("state_encode_calls", -1)) != len(bases):
        raise ValueError("W6i state-once contract failed")
    if int(metadata.get("representation_encoder_batches", -1)) != len(bases):
        raise ValueError("W6i representation batch accounting mismatch")

    by_base: dict[str, list[dict]] = defaultdict(list)
    for view in views:
        by_base[str(view["base_id"])].append(view)
    if set(by_base) != {str(base["base_id"]) for base in bases}:
        raise ValueError("W6i base/view identity mismatch")

    for base in bases:
        rows = by_base[str(base["base_id"])]
        if {row["view_id"] for row in rows} != set(VIEW_IDS):
            raise ValueError("W6i base view set mismatch")
        if set(base["field_probes"]) != set(ROLE_KEYS):
            raise ValueError("W6i field probe role mismatch")
        core = next(row for row in rows if row["view_id"] == "core-k8")
        master = next(row for row in rows if row["view_id"] == "master-k64")
        if int(core["diagnosis_k"]) != 8 or int(master["diagnosis_k"]) != 64:
            raise ValueError("W6i K8/K64 cardinality mismatch")
        if not set(core["option_ids"]).issubset(set(master["option_ids"])):
            raise ValueError("W6i K8 must be nested inside K64")
        for row in (core, master):
            if "canonical" not in row or "factorized" not in row:
                raise ValueError("W6i counterfactual representation missing")


def compile_w6i_cache(
    model: NolaneHira,
    views: Sequence[RepresentationBridgeView],
) -> dict:
    model.eval()
    grouped: dict[str, list[RepresentationBridgeView]] = defaultdict(list)
    for view in views:
        grouped[view.base_id].append(view)

    before = model.state_encode_calls
    bases: list[dict] = []
    rows: list[dict] = []
    representation_batches = 0
    representation_text_count = 0

    for base_id in sorted(grouped):
        base_views = grouped[base_id]
        if {view.view_id for view in base_views} != set(VIEW_IDS):
            raise ValueError("W6i base view set mismatch")
        reference = base_views[0]
        if len({view.typed.state_text for view in base_views}) != 1:
            raise ValueError("W6i base state drift")
        if len({view.gold_signature for view in base_views}) != 1:
            raise ValueError("W6i base gold drift")

        memory = model.compile_state(
            reference.typed.state_text,
            segment_tokens=32,
        )
        if memory.content_token_embeddings is None:
            raise RuntimeError("W6i requires state content token embeddings")

        spec = DOMAINS[reference.domain_id]
        lookup = {view.view_id: view for view in base_views}
        core_view = lookup["core-k8"]
        master_view = lookup["master-k64"]

        probe_texts: list[str] = []
        for role_index, role in enumerate(ROLE_KEYS):
            gold_value = reference.gold_signature[role_index]
            negative_value = (
                reference.one_field_signatures[role_index][role_index]
            )
            role_text = spec.roles[role_index]
            swapped_role = spec.roles[(role_index + 1) % 4]
            probe_texts.extend(
                [
                    gold_value,
                    negative_value,
                    f"{role_text} {gold_value}",
                    f"{role_text} {negative_value}",
                    f"{swapped_role} {gold_value}",
                ]
            )

        canonical_core = [
            canonical_tagged_text(signature)
            for signature in core_view.option_signatures
        ]
        canonical_master = [
            canonical_tagged_text(signature)
            for signature in master_view.option_signatures
        ]
        factorized_core = [
            phrase
            for signature in core_view.option_signatures
            for phrase in factorized_role_value_phrases(spec, signature)
        ]
        factorized_master = [
            phrase
            for signature in master_view.option_signatures
            for phrase in factorized_role_value_phrases(spec, signature)
        ]
        representation_texts = [
            *probe_texts,
            *canonical_core,
            *canonical_master,
            *factorized_core,
            *factorized_master,
        ]
        rep_batch = model.encoder.encode_texts(representation_texts)
        representation_batches += 1
        representation_text_count += len(representation_texts)

        field_probes = {}
        cursor = 0
        for role in ROLE_KEYS:
            field_probes[role] = {
                "gold": _content_from_text_batch(rep_batch, cursor),
                "negative": _content_from_text_batch(rep_batch, cursor + 1),
                "gold_phrase": _content_from_text_batch(rep_batch, cursor + 2),
                "negative_phrase": _content_from_text_batch(rep_batch, cursor + 3),
                "swapped_role_gold_phrase": _content_from_text_batch(
                    rep_batch, cursor + 4
                ),
            }
            cursor += 5

        core_canonical = _rendered_option_artifacts(
            rep_batch,
            start=cursor,
            count=8,
        )
        cursor += 8
        master_canonical = _rendered_option_artifacts(
            rep_batch,
            start=cursor,
            count=64,
        )
        cursor += 64
        core_factorized = _factorized_artifacts(
            rep_batch,
            start=cursor,
            option_count=8,
        )
        cursor += 8 * 4
        master_factorized = _factorized_artifacts(
            rep_batch,
            start=cursor,
            option_count=64,
        )
        cursor += 64 * 4
        if cursor != len(representation_texts):
            raise RuntimeError("W6i representation cursor mismatch")

        bases.append(
            {
                "base_id": base_id,
                "domain_id": reference.domain_id,
                "template_id": reference.template_id,
                "severity": int(reference.severity),
                "confidence": reference.confidence,
                "gold_signature": tuple(reference.gold_signature),
                "one_field_signatures": tuple(
                    reference.one_field_signatures
                ),
                "roles": tuple(spec.roles),
                "state_segments": (
                    memory.segment_embeddings.detach().cpu().to(torch.float16)
                ),
                "state_content_tokens": (
                    memory.content_token_embeddings.detach().cpu().to(
                        torch.float16
                    )
                ),
                "field_probes": field_probes,
            }
        )

        counterfactuals = {
            "core-k8": {
                "canonical": core_canonical,
                "factorized": core_factorized,
            },
            "master-k64": {
                "canonical": master_canonical,
                "factorized": master_factorized,
            },
        }

        for view in sorted(
            base_views,
            key=lambda item: VIEW_IDS.index(item.view_id),
        ):
            decision = view.typed.decisions[0]
            schema, receipt = model.compile_schema(
                primitive=decision.primitive,
                question_text=decision.question_text,
                options=decision.options,
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
                raise RuntimeError("W6i production schema artifacts incomplete")

            row = {
                "case_id": view.typed.case_id,
                "base_id": base_id,
                "domain_id": view.domain_id,
                "template_id": view.template_id,
                "view_id": view.view_id,
                "diagnosis_k": int(view.diagnosis_k),
                "severity": int(view.severity),
                "confidence": view.confidence,
                "target_role": view.target_role,
                "target_role_index": view.target_role_index,
                "gold_signature": tuple(view.gold_signature),
                "target_negative_signature": (
                    None
                    if view.target_negative_signature is None
                    else tuple(view.target_negative_signature)
                ),
                "one_field_signatures": tuple(view.one_field_signatures),
                "option_ids": tuple(
                    option.option_id for option in decision.options
                ),
                "option_signatures": tuple(view.option_signatures),
                "gold_option_id": view.gold_option_id,
                "target_negative_option_id": view.target_negative_option_id,
                "roles": tuple(spec.roles),
                "decisions": [
                    {
                        "question_id": decision.question_id,
                        "primitive": decision.primitive,
                        "schema_hash": receipt.schema_hash,
                        "question_embedding": (
                            schema.question_embedding.detach().cpu().to(
                                torch.float16
                            )
                        ),
                        "option_embeddings": (
                            schema.option_embeddings.detach().cpu().to(
                                torch.float16
                            )
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
                        "gold_index": int(decision.gold_index),
                    }
                ],
            }
            if view.view_id in counterfactuals:
                row.update(counterfactuals[view.view_id])
            rows.append(row)

    state_calls = model.state_encode_calls - before
    if state_calls != len(grouped):
        raise RuntimeError("W6i state encode contract failed")

    cache = {
        "metadata": {
            "schema_version": "r8-w6i-representation-cache-v1",
            "base_count": len(bases),
            "view_count": len(rows),
            "case_id_sha256": _case_id_sha256(views),
            "semantic_view_sha256": _semantic_view_sha256(views),
            "domain_semantic_view_sha256": {
                domain: _semantic_view_sha256(
                    [
                        view
                        for view in views
                        if view.domain_id == domain
                    ]
                )
                for domain in ("AA", "AB", "AC")
            },
            "state_encode_calls": state_calls,
            "state_encodes_per_base": (
                state_calls / max(1, len(bases))
            ),
            "representation_encoder_batches": representation_batches,
            "representation_text_count": representation_text_count,
            "representation_texts_per_base": (
                representation_text_count / max(1, len(bases))
            ),
            "domain_base_counts": dict(
                Counter(base["domain_id"] for base in bases)
            ),
            "view_counts": dict(
                Counter(row["view_id"] for row in rows)
            ),
        },
        "bases": bases,
        "views": rows,
    }
    validate_w6i_cache(cache)
    return cache


def save_w6i_cache(cache: dict, path: str | Path) -> Path:
    validate_w6i_cache(cache)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, target)
    return target


def load_w6i_cache(path: str | Path) -> dict:
    cache = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    validate_w6i_cache(cache)
    return cache


def _logical_index(view: dict, option_id: str) -> int:
    try:
        return tuple(view["option_ids"]).index(option_id)
    except ValueError as exc:
        raise ValueError("W6i logical option identity missing") from exc


def _production_tensors(base: dict, view: dict) -> dict[str, Tensor]:
    decision = view["decisions"][0]
    state_tokens = base["state_content_tokens"].float().unsqueeze(0)
    return {
        "state_segments": base["state_segments"].float().unsqueeze(0),
        "state_tokens": state_tokens,
        "state_mask": torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
        ),
        "question": decision["question_embedding"].float().unsqueeze(0),
        "question_tokens": decision["question_tokens"].float().unsqueeze(0),
        "question_mask": (
            decision["question_content_mask"].bool().unsqueeze(0)
        ),
        "options": decision["option_embeddings"].float().unsqueeze(0),
        "option_tokens": decision["option_tokens"].float().unsqueeze(0),
        "option_token_ids": (
            decision["option_token_ids"].long().unsqueeze(0)
        ),
        "option_mask": (
            decision["option_content_mask"].bool().unsqueeze(0)
        ),
        "qtype": torch.tensor(
            [PRIMITIVE_TO_ID[decision["primitive"]]],
            dtype=torch.long,
        ),
    }


def _canonical_tensors(base: dict, view: dict) -> dict[str, Tensor]:
    tensors = _production_tensors(base, view)
    canonical = view["canonical"]
    tensors["options"] = (
        canonical["option_embeddings"].float().unsqueeze(0)
    )
    tensors["option_tokens"] = (
        canonical["option_tokens"].float().unsqueeze(0)
    )
    tensors["option_token_ids"] = (
        canonical["option_token_ids"].long().unsqueeze(0)
    )
    tensors["option_mask"] = (
        canonical["option_content_mask"].bool().unsqueeze(0)
    )
    return tensors


@torch.inference_mode()
def _forward_rank(
    hira: HIRACore,
    scorer,
    base: dict,
    view: dict,
    *,
    representation: str,
) -> dict[str, object]:
    if representation == "production":
        tensors = _production_tensors(base, view)
    elif representation == "canonical":
        tensors = _canonical_tensors(base, view)
    else:
        raise ValueError("unsupported W6i forward representation")

    coarse = scorer(
        state_tokens=tensors["state_tokens"],
        state_mask=tensors["state_mask"],
        question_tokens=tensors["question_tokens"],
        question_mask=tensors["question_mask"],
        option_tokens=tensors["option_tokens"],
        option_token_ids=tensors["option_token_ids"],
        option_mask=tensors["option_mask"],
    )
    out = hira(
        tensors["question"],
        tensors["state_segments"],
        tensors["options"],
        tensors["qtype"],
        coarse_override=coarse,
        forced_budget=int(view["diagnosis_k"]),
        adaptive_budget=False,
    )
    gold_index = _logical_index(view, view["gold_option_id"])
    probabilities = torch.softmax(out.logits[0], dim=-1)
    return {
        "coarse": _rank_metrics(coarse[0], gold_index),
        "final": _rank_metrics(out.logits[0], gold_index),
        "probability_mass_error": abs(
            float(probabilities.sum()) - 1.0
        ),
        "coarse_logits": coarse[0].detach().cpu(),
        "final_logits": out.logits[0].detach().cpu(),
    }


@torch.inference_mode()
def _factorized_rank(
    scorer,
    base: dict,
    view: dict,
    *,
    operator: str,
) -> dict[str, object]:
    artifact = view["factorized"]
    state = base["state_content_tokens"].float()
    tokens = artifact["tokens"].float()
    masks = artifact["mask"].bool()
    rows = []
    per_field = []
    for option_index in range(tokens.shape[0]):
        field_scores = []
        for role_index in range(4):
            candidate = tokens[option_index, role_index][
                masks[option_index, role_index]
            ]
            score = _isolated_score(scorer, candidate, state)
            field_scores.append(score)
        stacked = torch.stack(field_scores)
        per_field.append(stacked.detach().cpu())
        if operator == "mean":
            rows.append(stacked.mean())
        elif operator == "min":
            rows.append(stacked.min())
        else:
            raise ValueError("W6i factorized operator must be mean or min")
    logits = torch.stack(rows)
    scale = scorer.scale().to(logits.device, logits.dtype)
    logits = logits * scale
    gold_index = _logical_index(view, view["gold_option_id"])
    return {
        "rank": _rank_metrics(logits, gold_index),
        "per_field_scores": torch.stack(per_field),
    }


def _pair_margin_from_logits(
    logits: Tensor,
    view: dict,
    target_option_id: str,
) -> float:
    gold_index = _logical_index(view, view["gold_option_id"])
    target_index = _logical_index(view, target_option_id)
    return float(logits[gold_index] - logits[target_index])


def _role_swap_probe(
    scorer,
    probe: dict,
    state_tokens: Tensor,
) -> dict[str, object]:
    state = state_tokens.float()
    correct = _isolated_score(
        scorer,
        probe["gold_phrase"]["tokens"].float(),
        state,
    )
    swapped = _isolated_score(
        scorer,
        probe["swapped_role_gold_phrase"]["tokens"].float(),
        state,
    )
    margin = float((correct - swapped).detach())
    return {
        "rejected": margin > 0.0,
        "margin": margin,
        "correct_score": float(correct.detach()),
        "swapped_score": float(swapped.detach()),
    }


@torch.inference_mode()
def diagnose_w6i_base(
    hira: HIRACore,
    scorer,
    base: dict,
    views: dict[str, dict],
) -> dict[str, object]:
    if set(views) != set(VIEW_IDS):
        raise ValueError("W6i base evaluator view mismatch")

    production = {
        view_id: _forward_rank(
            hira,
            scorer,
            base,
            view,
            representation="production",
        )
        for view_id, view in views.items()
    }
    canonical = {
        view_id: _forward_rank(
            hira,
            scorer,
            base,
            views[view_id],
            representation="canonical",
        )
        for view_id in ("core-k8", "master-k64")
    }
    factorized_mean = {
        view_id: _factorized_rank(
            scorer,
            base,
            views[view_id],
            operator="mean",
        )
        for view_id in ("core-k8", "master-k64")
    }
    factorized_min = {
        view_id: _factorized_rank(
            scorer,
            base,
            views[view_id],
            operator="min",
        )
        for view_id in ("core-k8", "master-k64")
    }

    role_rows = {}
    for role_index, role in enumerate(ROLE_KEYS):
        pair_view = views[f"pair-{role}"]
        target_id = pair_view["target_negative_option_id"]
        if target_id is None:
            raise ValueError("W6i pair target missing")

        p0 = isolated_value_probe(
            scorer,
            base["field_probes"][role],
            base["state_content_tokens"],
        )
        p1 = role_value_phrase_probe(
            scorer,
            base["field_probes"][role],
            base["state_content_tokens"],
        )
        role_swap = _role_swap_probe(
            scorer,
            base["field_probes"][role],
            base["state_content_tokens"],
        )
        pair_logits = production[f"pair-{role}"]["coarse_logits"]
        core_logits = production["core-k8"]["coarse_logits"]
        master_logits = production["master-k64"]["coarse_logits"]
        p2_margin = _pair_margin_from_logits(
            pair_logits,
            pair_view,
            target_id,
        )
        p3_k8_margin = _pair_margin_from_logits(
            core_logits,
            views["core-k8"],
            target_id,
        )
        p3_k64_margin = _pair_margin_from_logits(
            master_logits,
            views["master-k64"],
            target_id,
        )
        role_rows[role] = {
            "target_option_id": target_id,
            "p0_isolated_value": p0,
            "p1_role_value_phrase": p1,
            "role_swap_rejection": role_swap,
            "p2_structured_k2_margin": p2_margin,
            "p2_structured_k2_gold_wins": p2_margin > 0.0,
            "p3_k8_pair_margin": p3_k8_margin,
            "p3_k8_pair_gold_wins": p3_k8_margin > 0.0,
            "p3_k64_pair_margin": p3_k64_margin,
            "p3_k64_pair_gold_wins": p3_k64_margin > 0.0,
        }

    return {
        "base_id": base["base_id"],
        "domain_id": base["domain_id"],
        "roles": role_rows,
        "production": {
            view_id: {
                "coarse": production[view_id]["coarse"],
                "final": production[view_id]["final"],
                "probability_mass_error": production[view_id][
                    "probability_mass_error"
                ],
            }
            for view_id in VIEW_IDS
        },
        "canonical": {
            view_id: {
                "coarse": canonical[view_id]["coarse"],
                "final": canonical[view_id]["final"],
                "probability_mass_error": canonical[view_id][
                    "probability_mass_error"
                ],
            }
            for view_id in ("core-k8", "master-k64")
        },
        "factorized_mean": {
            view_id: factorized_mean[view_id]["rank"]
            for view_id in ("core-k8", "master-k64")
        },
        "factorized_min": {
            view_id: factorized_min[view_id]["rank"]
            for view_id in ("core-k8", "master-k64")
        },
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _accuracy(values: list[bool]) -> float:
    return _mean([1.0 if value else 0.0 for value in values])


def aggregate_w6i_records(
    records: list[dict[str, object]],
) -> dict[str, object]:
    if not records:
        raise ValueError("W6i aggregation requires records")

    role_rows = [
        role_row
        for record in records
        for role_row in record["roles"].values()
    ]
    p0 = [
        bool(row["p0_isolated_value"]["gold_wins"])
        for row in role_rows
    ]
    p1 = [
        bool(row["p1_role_value_phrase"]["gold_wins"])
        for row in role_rows
    ]
    p2 = [
        bool(row["p2_structured_k2_gold_wins"])
        for row in role_rows
    ]
    role_swap = [
        bool(row["role_swap_rejection"]["rejected"])
        for row in role_rows
    ]
    p3_k64 = [
        bool(row["p3_k64_pair_gold_wins"])
        for row in role_rows
    ]
    p0_wrong_p2_right = [
        (not left) and right
        for left, right in zip(p0, p2)
    ]
    p0_right_p2_wrong = [
        left and (not right)
        for left, right in zip(p0, p2)
    ]

    def rank_block(path: str, view_id: str, stage: str | None = None):
        values = []
        for record in records:
            row = record[path][view_id]
            if stage is not None:
                row = row[stage]
            values.append(row)
        return {
            "top1": _accuracy([bool(row["top1"]) for row in values]),
            "top5": _accuracy([bool(row["top5"]) for row in values]),
            "mrr": _mean(
                [float(row["reciprocal_rank"]) for row in values]
            ),
            "margin": _mean([float(row["margin"]) for row in values]),
        }

    production_k8 = rank_block("production", "core-k8", "final")
    production_k64 = rank_block(
        "production", "master-k64", "final"
    )
    canonical_k8 = rank_block("canonical", "core-k8", "final")
    canonical_k64 = rank_block(
        "canonical", "master-k64", "final"
    )
    factorized_mean_k8 = rank_block(
        "factorized_mean", "core-k8"
    )
    factorized_mean_k64 = rank_block(
        "factorized_mean", "master-k64"
    )
    factorized_min_k8 = rank_block(
        "factorized_min", "core-k8"
    )
    factorized_min_k64 = rank_block(
        "factorized_min", "master-k64"
    )

    output = {
        "base_count": len(records),
        "role_pair_count": len(role_rows),
        "p0_accuracy": _accuracy(p0),
        "p1_accuracy": _accuracy(p1),
        "p2_accuracy": _accuracy(p2),
        "role_swap_rejection_accuracy": _accuracy(role_swap),
        "role_swap_margin": _mean(
            [
                float(row["role_swap_rejection"]["margin"])
                for row in role_rows
            ]
        ),
        "p3_k64_fixed_pair_accuracy": _accuracy(p3_k64),
        "p0_wrong_p2_right_rate": _accuracy(p0_wrong_p2_right),
        "p0_right_p2_wrong_rate": _accuracy(p0_right_p2_wrong),
        "p0_margin": _mean(
            [
                float(row["p0_isolated_value"]["margin"])
                for row in role_rows
            ]
        ),
        "p1_margin": _mean(
            [
                float(row["p1_role_value_phrase"]["margin"])
                for row in role_rows
            ]
        ),
        "p2_margin": _mean(
            [
                float(row["p2_structured_k2_margin"])
                for row in role_rows
            ]
        ),
        "production_k8": production_k8,
        "production_k64": production_k64,
        "canonical_k8": canonical_k8,
        "canonical_k64": canonical_k64,
        "factorized_mean_k8": factorized_mean_k8,
        "factorized_mean_k64": factorized_mean_k64,
        "factorized_min_k8": factorized_min_k8,
        "factorized_min_k64": factorized_min_k64,
    }
    output["canonical_k64_gain"] = (
        canonical_k64["top1"] - production_k64["top1"]
    )
    output["factorized_mean_k64_gain"] = (
        factorized_mean_k64["top1"] - production_k64["top1"]
    )
    output["factorized_min_k64_gain"] = (
        factorized_min_k64["top1"] - production_k64["top1"]
    )
    return output


def representation_classification(
    metrics: dict[str, object],
) -> dict[str, object]:
    p0 = float(metrics["p0_accuracy"])
    p2 = float(metrics["p2_accuracy"])
    mismatch = float(metrics["p0_wrong_p2_right_rate"])
    production_k64 = float(metrics["production_k64"]["top1"])
    fixed_pair = float(metrics["p3_k64_fixed_pair_accuracy"])
    gains = {
        "canonical": float(metrics["canonical_k64_gain"]),
        "factorized_mean": float(
            metrics["factorized_mean_k64_gain"]
        ),
        "factorized_min": float(
            metrics["factorized_min_k64_gain"]
        ),
    }
    best_gain = max(gains.values())

    proxy_mismatch = (
        p0 < 0.80
        and p2 >= 0.90
        and mismatch >= 0.15
    )
    free_form_limit = (
        p2 >= 0.90
        and production_k64 < 0.55
        and fixed_pair >= 0.85
        and best_gain >= 0.10
    )
    global_without_gain = (
        p2 >= 0.90
        and production_k64 < 0.55
        and best_gain < 0.10
    )
    domain_shift = (
        abs(p0 - p2) < 0.08
        and abs(p0 - PRIOR_W6G_P0_ANCHOR)
        >= DOMAIN_SHIFT_MATERIAL_DELTA
        and abs(p2 - PRIOR_W6H_P2_ANCHOR)
        >= DOMAIN_SHIFT_MATERIAL_DELTA
    )

    active = [
        label
        for label, enabled in (
            ("STRUCTURED_PAIR_PROXY_MISMATCH", proxy_mismatch),
            ("FREE_FORM_CONJUNCTION_LIMIT", free_form_limit),
            (
                "GLOBAL_RANKING_WITHOUT_INTERFACE_GAIN",
                global_without_gain,
            ),
            ("DOMAIN_DIFFICULTY_SHIFT", domain_shift),
        )
        if enabled
    ]
    if len(active) > 1:
        classification = "MIXED_REPRESENTATION_FAILURE"
    elif len(active) == 1:
        classification = active[0]
    else:
        classification = "REPRESENTATION_BRIDGE_UNRESOLVED"

    return {
        "classification": classification,
        "active_rules": active,
        "p0_accuracy": p0,
        "p2_accuracy": p2,
        "p0_wrong_p2_right_rate": mismatch,
        "production_k64_top1": production_k64,
        "fixed_pair_k64_accuracy": fixed_pair,
        "representation_k64_gains": gains,
    }


def checkpoint_stability(
    per_domain: dict[str, dict[str, object]],
) -> dict[str, object]:
    labels = [
        str(row["classification"]["classification"])
        for row in per_domain.values()
    ]
    counts = Counter(
        label
        for label in labels
        if label != "REPRESENTATION_BRIDGE_UNRESOLVED"
    )
    if not counts:
        return {
            "stable": False,
            "classification": "REPRESENTATION_BRIDGE_UNRESOLVED",
            "domains": [],
        }
    label, count = counts.most_common(1)[0]
    if count < 2:
        return {
            "stable": False,
            "classification": "REPRESENTATION_BRIDGE_UNRESOLVED",
            "domains": [],
        }
    conflicting = [
        other
        for other, other_count in counts.items()
        if other != label and other_count >= 2
    ]
    if conflicting:
        return {
            "stable": False,
            "classification": "MIXED_REPRESENTATION_FAILURE",
            "domains": [],
        }
    domains = sorted(
        domain
        for domain, row in per_domain.items()
        if row["classification"]["classification"] == label
    )
    return {
        "stable": True,
        "classification": label,
        "domains": domains,
    }


def diagnostic_stability(
    results: dict[str, dict[str, object]],
) -> dict[str, object]:
    required = {
        "w6e-joint-primary",
        "w6h-projection-retune",
        "w6h-semantic-adapter",
    }
    if set(results) != required:
        raise ValueError("W6i checkpoint set mismatch")

    checkpoint = {
        name: checkpoint_stability(row["per_domain"])
        for name, row in results.items()
    }
    primary = checkpoint["w6e-joint-primary"]
    trained = [
        checkpoint["w6h-projection-retune"],
        checkpoint["w6h-semantic-adapter"],
    ]

    if not primary["stable"]:
        outcome = "REPRESENTATION_BRIDGE_UNRESOLVED"
        stable_class = None
    else:
        matches = [
            row
            for row in trained
            if row["stable"]
            and row["classification"] == primary["classification"]
        ]
        opposites = [
            row
            for row in trained
            if row["stable"]
            and row["classification"] != primary["classification"]
        ]
        if opposites:
            outcome = "MIXED_REPRESENTATION_FAILURE"
            stable_class = None
        elif matches:
            outcome = "STABLE_REPRESENTATION_BRIDGE"
            stable_class = primary["classification"]
        else:
            outcome = "REPRESENTATION_BRIDGE_UNRESOLVED"
            stable_class = None

    return {
        "outcome": outcome,
        "stable_classification": stable_class,
        "checkpoint_stability": checkpoint,
        "rescue_lane_authorized": False,
    }
