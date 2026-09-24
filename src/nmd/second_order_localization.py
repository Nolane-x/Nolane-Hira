from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .competitive import (
    CompetitiveCoarseScorer,
    MIN_COVERAGE_WEIGHT,
    SALIENCE_THRESHOLD,
    candidate_relative_idf,
)
from .hira import HIRACore
from .runtime import NolaneHira, PRIMITIVE_TO_ID
from .second_order_localization_authority import (
    DOMAINS,
    ROLE_KEYS,
    VIEW_IDS,
    SecondOrderDiagnosticView,
)


def _case_id_sha256(views: Sequence[SecondOrderDiagnosticView]) -> str:
    payload = "\n".join(
        sorted(view.typed.case_id for view in views)
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _semantic_view_sha256(
    views: Sequence[SecondOrderDiagnosticView],
) -> str:
    rows = []
    for view in sorted(views, key=lambda item: item.typed.case_id):
        decision = view.typed.decisions[0]
        rows.append(
            {
                "case_id": view.typed.case_id,
                "base_id": view.base_id,
                "domain_id": view.domain_id,
                "view_id": view.view_id,
                "state_text": view.typed.state_text,
                "question_text": decision.question_text,
                "gold_index": int(decision.gold_index),
                "options": [
                    {
                        "option_id": option.option_id,
                        "criterion_text": option.criterion_text,
                    }
                    for option in decision.options
                ],
            }
        )
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _content_from_text_batch(batch, row: int) -> dict[str, Tensor]:
    mask = batch.attention_mask[row].bool()
    special = (
        torch.zeros_like(mask)
        if batch.special_token_mask is None
        else batch.special_token_mask[row].bool()
    )
    content_mask = mask & ~special
    if int(content_mask.sum()) == 0:
        content_mask = mask
    embeddings = batch.token_embeddings[row][content_mask]
    token_ids = (
        torch.arange(
            embeddings.shape[0],
            device=embeddings.device,
            dtype=torch.long,
        )
        if batch.token_ids is None
        else batch.token_ids[row][content_mask].long()
    )
    return {
        "tokens": embeddings.detach().cpu().to(torch.float16),
        "token_ids": token_ids.detach().cpu().long(),
        "mask": torch.ones(embeddings.shape[0], dtype=torch.bool),
    }


def validate_w6g_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W6g cache must be a dict")
    metadata = cache.get("metadata")
    bases = cache.get("bases")
    views = cache.get("views")
    if (
        not isinstance(metadata, dict)
        or not isinstance(bases, list)
        or not isinstance(views, list)
    ):
        raise ValueError("W6g cache requires metadata/bases/views")
    if metadata.get("schema_version") != "r8-w6g-second-order-cache-v1":
        raise ValueError("unexpected W6g cache schema")
    if metadata.get("split") != "diagnostic":
        raise ValueError("W6g cache split must be diagnostic")
    if int(metadata.get("base_count", -1)) != len(bases):
        raise ValueError("W6g base count mismatch")
    if int(metadata.get("view_count", -1)) != len(views):
        raise ValueError("W6g view count mismatch")
    if float(metadata.get("state_encode_calls_per_base", -1.0)) != 1.0:
        raise ValueError("W6g must encode state exactly once per base")

    base_ids: set[str] = set()
    base_lookup: dict[str, dict] = {}
    for base in bases:
        base_id = str(base.get("base_id", ""))
        if not base_id or base_id in base_ids:
            raise ValueError("invalid or duplicate W6g base id")
        base_ids.add(base_id)
        base_lookup[base_id] = base
        if base.get("domain_id") not in {"Q", "R", "S"}:
            raise ValueError("invalid W6g base domain")
        state_segments = base.get("state_segments")
        state_tokens = base.get("state_content_tokens")
        if (
            not isinstance(state_segments, Tensor)
            or state_segments.ndim != 2
            or state_segments.shape[-1] != 256
        ):
            raise ValueError("W6g state_segments must be [S,256]")
        if (
            not isinstance(state_tokens, Tensor)
            or state_tokens.ndim != 2
            or state_tokens.shape[-1] != 256
            or state_tokens.shape[0] < 1
        ):
            raise ValueError("W6g state_content_tokens must be [T,256]")
        probes = base.get("field_probes")
        if not isinstance(probes, dict) or set(probes) != set(ROLE_KEYS):
            raise ValueError("W6g field probes must cover four roles")
        for role, probe in probes.items():
            if role not in ROLE_KEYS or not isinstance(probe, dict):
                raise ValueError("invalid W6g field probe")
            for key in (
                "gold",
                "negative",
                "role",
                "gold_phrase",
                "negative_phrase",
            ):
                item = probe.get(key)
                if not isinstance(item, dict):
                    raise ValueError("W6g field probe item missing")
                tokens = item.get("tokens")
                ids = item.get("token_ids")
                mask = item.get("mask")
                if (
                    not isinstance(tokens, Tensor)
                    or tokens.ndim != 2
                    or tokens.shape[-1] != 256
                    or tokens.shape[0] < 1
                ):
                    raise ValueError("W6g field probe tokens invalid")
                if (
                    not isinstance(ids, Tensor)
                    or ids.ndim != 1
                    or ids.shape[0] != tokens.shape[0]
                    or ids.dtype != torch.long
                ):
                    raise ValueError("W6g field probe IDs invalid")
                if (
                    not isinstance(mask, Tensor)
                    or mask.shape != ids.shape
                    or mask.dtype != torch.bool
                    or not bool(mask.all())
                ):
                    raise ValueError("W6g field probe mask invalid")

    seen_cases: set[str] = set()
    views_by_base: dict[str, set[str]] = defaultdict(set)
    for case in views:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen_cases:
            raise ValueError("invalid or duplicate W6g case id")
        seen_cases.add(case_id)
        base_id = str(case.get("base_id", ""))
        if base_id not in base_lookup:
            raise ValueError("W6g view references unknown base")
        view_id = str(case.get("view_id", ""))
        if view_id not in VIEW_IDS or view_id in views_by_base[base_id]:
            raise ValueError("invalid or duplicate W6g view id")
        views_by_base[base_id].add(view_id)

        k = int(case.get("diagnosis_k", -1))
        expected_k = (
            2 if view_id.startswith("pair-")
            else 8 if view_id == "core-k8"
            else 64
        )
        if k != expected_k:
            raise ValueError("W6g view K mismatch")
        decisions = case.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 1:
            raise ValueError("W6g requires diagnosis-only views")
        decision = decisions[0]
        if (
            decision.get("question_id") != "diagnosis"
            or decision.get("primitive") != "choice"
        ):
            raise ValueError("W6g cached decision must be diagnosis/choice")

        question = decision.get("question_embedding")
        options = decision.get("option_embeddings")
        question_tokens = decision.get("question_tokens")
        question_mask = decision.get("question_content_mask")
        option_tokens = decision.get("option_tokens")
        option_ids = decision.get("option_token_ids")
        option_mask = decision.get("option_content_mask")
        gold_probs = decision.get("gold_probabilities")
        if not isinstance(question, Tensor) or question.shape != (256,):
            raise ValueError("W6g question embedding invalid")
        if not isinstance(options, Tensor) or options.shape != (k, 256):
            raise ValueError("W6g option embeddings invalid")
        if (
            not isinstance(question_tokens, Tensor)
            or question_tokens.ndim != 2
            or question_tokens.shape[-1] != 256
        ):
            raise ValueError("W6g question tokens invalid")
        if (
            not isinstance(question_mask, Tensor)
            or question_mask.shape != question_tokens.shape[:1]
            or question_mask.dtype != torch.bool
            or int(question_mask.sum()) < 1
        ):
            raise ValueError("W6g question mask invalid")
        if (
            not isinstance(option_tokens, Tensor)
            or option_tokens.ndim != 3
            or option_tokens.shape[0] != k
            or option_tokens.shape[-1] != 256
        ):
            raise ValueError("W6g option tokens invalid")
        if (
            not isinstance(option_ids, Tensor)
            or option_ids.shape != option_tokens.shape[:2]
            or option_ids.dtype != torch.long
        ):
            raise ValueError("W6g option token IDs invalid")
        if (
            not isinstance(option_mask, Tensor)
            or option_mask.shape != option_tokens.shape[:2]
            or option_mask.dtype != torch.bool
            or (option_mask.sum(-1) < 1).any()
        ):
            raise ValueError("W6g option masks invalid")
        if (
            not isinstance(gold_probs, Tensor)
            or gold_probs.shape != (k,)
            or not torch.isfinite(gold_probs).all()
            or abs(float(gold_probs.sum()) - 1.0) > 1e-6
        ):
            raise ValueError("W6g gold probability contract failed")
        gold = int(decision.get("gold_index", -1))
        if not 0 <= gold < k or int(gold_probs.argmax()) != gold:
            raise ValueError("W6g hard/soft gold mismatch")

        logical_ids = case.get("option_ids")
        signatures = case.get("option_signatures")
        distances = case.get("option_distances")
        changed = case.get("option_changed_roles")
        if (
            not isinstance(logical_ids, tuple)
            or len(logical_ids) != k
            or len(set(logical_ids)) != k
        ):
            raise ValueError("W6g logical option IDs invalid")
        if not isinstance(signatures, tuple) or len(signatures) != k:
            raise ValueError("W6g signatures invalid")
        if not isinstance(distances, tuple) or len(distances) != k:
            raise ValueError("W6g distances invalid")
        if not isinstance(changed, tuple) or len(changed) != k:
            raise ValueError("W6g changed-role metadata invalid")
        if int(distances[gold]) != 0:
            raise ValueError("W6g gold distance must be zero")
        if logical_ids[gold] != case.get("gold_option_id"):
            raise ValueError("W6g gold logical ID mismatch")

    if any(view_ids != set(VIEW_IDS) for view_ids in views_by_base.values()):
        raise ValueError("W6g every base must expose all ten views")


@torch.inference_mode()
def compile_w6g_cache(
    model: NolaneHira,
    views: Sequence[SecondOrderDiagnosticView],
) -> dict:
    model.eval()
    grouped: dict[str, list[SecondOrderDiagnosticView]] = defaultdict(list)
    for view in views:
        if view.split != "diagnostic":
            raise ValueError("W6g view split mismatch")
        grouped[view.base_id].append(view)

    before = model.state_encode_calls
    bases: list[dict] = []
    rows: list[dict] = []

    for base_id in sorted(grouped):
        base_views = grouped[base_id]
        if {view.view_id for view in base_views} != set(VIEW_IDS):
            raise ValueError("W6g base view set mismatch")
        reference = base_views[0]
        if len({view.typed.state_text for view in base_views}) != 1:
            raise ValueError("W6g base state text drift")
        if len({view.gold_signature for view in base_views}) != 1:
            raise ValueError("W6g base gold drift")

        memory = model.compile_state(
            reference.typed.state_text,
            segment_tokens=32,
        )
        if memory.content_token_embeddings is None:
            raise RuntimeError("W6g requires state content token embeddings")

        spec = DOMAINS[reference.domain_id]
        probe_texts: list[str] = []
        for index, role in enumerate(ROLE_KEYS):
            gold_value = reference.gold_signature[index]
            negative_value = reference.one_field_signatures[index][index]
            role_text = spec.roles[index]
            probe_texts.extend(
                [
                    gold_value,
                    negative_value,
                    role_text,
                    f"{role_text} {gold_value}",
                    f"{role_text} {negative_value}",
                ]
            )
        probe_batch = model.encoder.encode_texts(probe_texts)
        field_probes = {}
        cursor = 0
        for role in ROLE_KEYS:
            field_probes[role] = {
                "gold": _content_from_text_batch(probe_batch, cursor),
                "negative": _content_from_text_batch(probe_batch, cursor + 1),
                "role": _content_from_text_batch(probe_batch, cursor + 2),
                "gold_phrase": _content_from_text_batch(
                    probe_batch, cursor + 3
                ),
                "negative_phrase": _content_from_text_batch(
                    probe_batch, cursor + 4
                ),
            }
            cursor += 5

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
                "one_field_option_ids": tuple(
                    reference.one_field_option_ids
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

        for view in sorted(base_views, key=lambda item: VIEW_IDS.index(item.view_id)):
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
                raise RuntimeError("W6g schema token artifacts incomplete")

            rows.append(
                {
                    "case_id": view.typed.case_id,
                    "split": "diagnostic",
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
                    "one_field_signatures": tuple(
                        view.one_field_signatures
                    ),
                    "one_field_option_ids": tuple(
                        view.one_field_option_ids
                    ),
                    "option_ids": tuple(
                        option.option_id for option in decision.options
                    ),
                    "option_signatures": tuple(view.option_signatures),
                    "option_distances": tuple(
                        int(value) for value in view.option_distances
                    ),
                    "option_changed_roles": tuple(
                        tuple(value) for value in view.option_changed_roles
                    ),
                    "gold_option_id": view.gold_option_id,
                    "target_negative_option_id": (
                        view.target_negative_option_id
                    ),
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
                            "gold_probabilities": torch.tensor(
                                decision.gold_probabilities,
                                dtype=torch.float32,
                            ),
                        }
                    ],
                }
            )

    state_calls = model.state_encode_calls - before
    if state_calls != len(grouped):
        raise RuntimeError("W6g base-centric state encode contract failed")

    cache = {
        "metadata": {
            "schema_version": "r8-w6g-second-order-cache-v1",
            "split": "diagnostic",
            "base_count": len(bases),
            "view_count": len(rows),
            "case_id_sha256": _case_id_sha256(views),
            "semantic_view_sha256": _semantic_view_sha256(views),
            "domain_semantic_view_sha256": {
                domain: _semantic_view_sha256(
                    [view for view in views if view.domain_id == domain]
                )
                for domain in ("Q", "R", "S")
            },
            "state_encode_calls": state_calls,
            "state_encode_calls_per_base": (
                state_calls / max(1, len(bases))
            ),
            "state_encode_calls_per_view": (
                state_calls / max(1, len(rows))
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
    validate_w6g_cache(cache)
    return cache


def save_w6g_cache(cache: dict, path: str | Path) -> Path:
    validate_w6g_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w6g_cache(path: str | Path) -> dict:
    cache = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    validate_w6g_cache(cache)
    return cache


def _full_tensors(base: dict, view: dict) -> dict[str, Tensor]:
    decision = view["decisions"][0]
    state_tokens = base["state_content_tokens"].float().unsqueeze(0)
    return {
        "state_segments": base["state_segments"].float().unsqueeze(0),
        "state_tokens": state_tokens,
        "state_mask": torch.ones(
            1, state_tokens.shape[1], dtype=torch.bool
        ),
        "question": decision["question_embedding"].float().unsqueeze(0),
        "question_tokens": decision["question_tokens"].float().unsqueeze(0),
        "question_mask": decision["question_content_mask"].bool().unsqueeze(0),
        "options": decision["option_embeddings"].float().unsqueeze(0),
        "option_tokens": decision["option_tokens"].float().unsqueeze(0),
        "option_token_ids": decision["option_token_ids"].long().unsqueeze(0),
        "option_mask": decision["option_content_mask"].bool().unsqueeze(0),
        "qtype": torch.tensor(
            [PRIMITIVE_TO_ID[decision["primitive"]]],
            dtype=torch.long,
        ),
    }


def _logical_index(view: dict, option_id: str) -> int:
    ids = tuple(view["option_ids"])
    try:
        return ids.index(option_id)
    except ValueError as exc:
        raise ValueError(f"W6g option {option_id} missing from view") from exc


def _rank_metrics(logits: Tensor, gold_index: int) -> dict[str, object]:
    row = logits.detach().cpu().to(torch.float64).flatten()
    values = [float(value) for value in row.tolist()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("W6g rank metrics require finite logits")
    order = sorted(
        range(len(values)),
        key=lambda index: (-values[index], index),
    )
    rank = order.index(gold_index) + 1
    predicted = order[0]
    top5 = set(order[: min(5, len(order))])
    negatives = [index for index in order if index != gold_index]
    best_negative = negatives[0]
    return {
        "rank": rank,
        "top1": predicted == gold_index,
        "top5": gold_index in top5,
        "reciprocal_rank": 1.0 / rank,
        "gold_logit": values[gold_index],
        "margin": values[gold_index] - values[best_negative],
        "predicted_index": predicted,
        "best_negative_index": best_negative,
    }


def _salience_for_view(view: dict) -> Tensor:
    decision = view["decisions"][0]
    ids = decision["option_token_ids"].long().unsqueeze(0)
    mask = decision["option_content_mask"].bool().unsqueeze(0)
    return candidate_relative_idf(ids, mask)[0]


def _aligned_reference_salience(
    current_view: dict,
    reference_view: dict,
    current_index: int,
    reference_index: int,
    reference_salience: Tensor,
) -> Tensor:
    current_decision = current_view["decisions"][0]
    reference_decision = reference_view["decisions"][0]
    current_ids = current_decision["option_token_ids"][current_index]
    current_mask = current_decision["option_content_mask"][current_index].bool()
    reference_ids = reference_decision["option_token_ids"][reference_index]
    reference_mask = reference_decision["option_content_mask"][reference_index].bool()

    current_valid = current_ids[current_mask].long()
    reference_valid = reference_ids[reference_mask].long()
    if not torch.equal(current_valid, reference_valid):
        raise ValueError("W6g reference option token identity drift")

    values = reference_salience[reference_index][reference_mask].float()
    aligned = torch.zeros(current_mask.shape[0], dtype=torch.float32)
    aligned[current_mask] = values
    return aligned


def _pair_pathway_logits(
    scorer: CompetitiveCoarseScorer,
    *,
    tensors: dict[str, Tensor],
    pair_indices: tuple[int, int],
    common_salience: Tensor,
    aggregate_salience: Tensor,
) -> Tensor:
    context = torch.cat(
        [tensors["state_tokens"], tensors["question_tokens"]],
        dim=1,
    )
    context_mask = torch.cat(
        [tensors["state_mask"], tensors["question_mask"]],
        dim=1,
    )
    pair = tensors["option_tokens"][:, list(pair_indices)]
    pair_mask = tensors["option_mask"][:, list(pair_indices)]

    projected_context = scorer._project(context)
    projected_pair = scorer._project(pair)
    similarity = torch.einsum(
        "bktd,bcd->bktc",
        projected_pair,
        projected_context,
    )
    similarity = similarity.masked_fill(
        ~context_mask[:, None, None, :],
        -1e4,
    )

    common_weights = (
        common_salience.unsqueeze(0).to(similarity.dtype)
        * pair_mask.to(similarity.dtype)
    )
    common_denom = common_weights.sum(
        dim=2, keepdim=True
    ).clamp_min(1e-8)
    common = (
        similarity * common_weights[..., None]
    ).sum(dim=2) / common_denom

    adjusted = similarity - common[:, :, None, :]
    adjusted = adjusted.masked_fill(
        ~context_mask[:, None, None, :],
        -1e4,
    )
    coverage = adjusted.max(dim=-1).values

    aggregate_weights = (
        aggregate_salience.unsqueeze(0).to(similarity.dtype)
        * pair_mask.to(similarity.dtype)
    )
    salient_mask = (
        pair_mask
        & (aggregate_salience.unsqueeze(0) >= SALIENCE_THRESHOLD)
    )
    empty = salient_mask.sum(-1) == 0
    if empty.any():
        salient_mask = salient_mask.clone()
        salient_mask[empty] = pair_mask[empty]

    weighted_mean = (
        (coverage * aggregate_weights).sum(-1)
        / aggregate_weights.sum(-1).clamp_min(1e-8)
    )
    min_coverage = coverage.masked_fill(
        ~salient_mask,
        1e4,
    ).min(dim=-1).values
    raw = weighted_mean + MIN_COVERAGE_WEIGHT * min_coverage
    logits = raw * scorer.scale().to(raw.device, raw.dtype)
    if not torch.isfinite(logits).all():
        raise ValueError("W6g pathway probe produced non-finite logits")
    return logits[0]



def _pair_margin(logits: Tensor, gold_position: int = 0) -> float:
    if logits.numel() != 2:
        raise ValueError("W6g pair margin requires two logits")
    other = 1 - gold_position
    return float(logits[gold_position] - logits[other])


def _isolated_score(
    scorer: CompetitiveCoarseScorer,
    candidate: Tensor,
    state: Tensor,
) -> Tensor:
    p_candidate = scorer._project(candidate.unsqueeze(0))[0]
    p_state = scorer._project(state.unsqueeze(0))[0]
    similarity = torch.einsum("td,sd->ts", p_candidate, p_state)
    coverage = similarity.max(dim=-1).values
    return (
        coverage.mean()
        + MIN_COVERAGE_WEIGHT * coverage.min()
    )


@torch.inference_mode()
def _counterfactual_probe(
    scorer: CompetitiveCoarseScorer,
    *,
    gold: Tensor,
    negative: Tensor,
) -> dict[str, object]:
    state = gold
    gold_score = _isolated_score(scorer, gold, state)
    negative_score = _isolated_score(scorer, negative, state)
    margin = float((gold_score - negative_score).detach())
    return {
        "gold_wins": margin > 0.0,
        "margin": margin,
        "gold_score": float(gold_score.detach()),
        "negative_score": float(negative_score.detach()),
    }


def isolated_value_probe(
    scorer: CompetitiveCoarseScorer,
    probe: dict,
) -> dict[str, object]:
    return _counterfactual_probe(
        scorer,
        gold=probe["gold"]["tokens"].float(),
        negative=probe["negative"]["tokens"].float(),
    )


def role_value_phrase_probe(
    scorer: CompetitiveCoarseScorer,
    probe: dict,
) -> dict[str, object]:
    return _counterfactual_probe(
        scorer,
        gold=probe["gold_phrase"]["tokens"].float(),
        negative=probe["negative_phrase"]["tokens"].float(),
    )


def _wrong_winner_anatomy(
    view: dict,
    metrics: dict[str, object],
    gold_index: int,
) -> dict[str, object] | None:
    predicted = int(metrics["predicted_index"])
    if predicted == gold_index:
        return None
    return {
        "option_id": tuple(view["option_ids"])[predicted],
        "distance": int(tuple(view["option_distances"])[predicted]),
        "changed_roles": tuple(
            tuple(view["option_changed_roles"])[predicted]
        ),
    }


@torch.inference_mode()
def diagnose_w6g_view(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    base: dict,
    view: dict,
    core_view: dict,
) -> dict[str, object]:
    tensors = _full_tensors(base, view)
    native_logits = scorer(
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
        coarse_override=native_logits,
        forced_budget=tensors["options"].shape[1],
        adaptive_budget=False,
    )
    gold_index = _logical_index(view, view["gold_option_id"])
    native_rank = _rank_metrics(native_logits[0], gold_index)
    final_rank = _rank_metrics(out.logits[0], gold_index)

    current_salience = _salience_for_view(view)
    reference_salience = _salience_for_view(core_view)

    roles = {}
    for role_index, role in enumerate(ROLE_KEYS):
        target_id = tuple(view["one_field_option_ids"])[role_index]
        if target_id not in tuple(view["option_ids"]):
            continue
        if target_id not in tuple(core_view["option_ids"]):
            raise ValueError("W6g core reference missing one-field target")

        current_target = _logical_index(view, target_id)
        core_gold = _logical_index(core_view, core_view["gold_option_id"])
        core_target = _logical_index(core_view, target_id)
        pair_indices = (gold_index, current_target)

        native_pair_salience = torch.stack(
            [
                current_salience[gold_index].float(),
                current_salience[current_target].float(),
            ]
        )
        ref_gold = _aligned_reference_salience(
            view,
            core_view,
            gold_index,
            core_gold,
            reference_salience,
        )
        ref_target = _aligned_reference_salience(
            view,
            core_view,
            current_target,
            core_target,
            reference_salience,
        )
        reference_pair_salience = torch.stack(
            [ref_gold, ref_target]
        )

        native_pair_margin = float(
            native_logits[0, gold_index]
            - native_logits[0, current_target]
        )
        aggregation_ref = _pair_pathway_logits(
            scorer,
            tensors=tensors,
            pair_indices=pair_indices,
            common_salience=native_pair_salience,
            aggregate_salience=reference_pair_salience,
        )
        common_ref = _pair_pathway_logits(
            scorer,
            tensors=tensors,
            pair_indices=pair_indices,
            common_salience=reference_pair_salience,
            aggregate_salience=native_pair_salience,
        )
        full_ref = _pair_pathway_logits(
            scorer,
            tensors=tensors,
            pair_indices=pair_indices,
            common_salience=reference_pair_salience,
            aggregate_salience=reference_pair_salience,
        )

        roles[role] = {
            "target_option_id": target_id,
            "native_margin": native_pair_margin,
            "native_gold_wins": native_pair_margin > 0.0,
            "aggregation_idf_reference_margin": _pair_margin(
                aggregation_ref
            ),
            "common_mode_idf_reference_margin": _pair_margin(
                common_ref
            ),
            "full_idf_reference_margin": _pair_margin(full_ref),
            "isolated_value": isolated_value_probe(
                scorer,
                base["field_probes"][role],
            ),
            "role_value_phrase": role_value_phrase_probe(
                scorer,
                base["field_probes"][role],
            ),
        }

    probabilities = torch.softmax(out.logits[0], dim=-1)
    return {
        "case_id": view["case_id"],
        "base_id": view["base_id"],
        "domain_id": view["domain_id"],
        "view_id": view["view_id"],
        "diagnosis_k": int(view["diagnosis_k"]),
        "native": native_rank,
        "final": final_rank,
        "native_wrong_winner": _wrong_winner_anatomy(
            view, native_rank, gold_index
        ),
        "final_wrong_winner": _wrong_winner_anatomy(
            view, final_rank, gold_index
        ),
        "role_pairs": roles,
        "probability_mass_error": abs(
            float(probabilities.sum()) - 1.0
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def aggregate_role_context(
    records: Sequence[dict],
    role: str,
    view_id: str,
) -> dict[str, object]:
    rows = [
        row["role_pairs"][role]
        for row in records
        if row["view_id"] == view_id and role in row["role_pairs"]
    ]
    if not rows:
        raise ValueError("W6g role/context aggregation is empty")

    def margins(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    native = margins("native_margin")
    aggregation = margins("aggregation_idf_reference_margin")
    common = margins("common_mode_idf_reference_margin")
    full = margins("full_idf_reference_margin")
    isolated = [
        float(row["isolated_value"]["margin"])
        for row in rows
    ]
    phrases = [
        float(row["role_value_phrase"]["margin"])
        for row in rows
    ]
    return {
        "n": len(rows),
        "native_pair_accuracy": sum(value > 0 for value in native) / len(native),
        "native_margin_mean": _mean(native),
        "native_margin_median": _median(native),
        "aggregation_idf_reference_accuracy": (
            sum(value > 0 for value in aggregation) / len(aggregation)
        ),
        "aggregation_idf_reference_margin_mean": _mean(aggregation),
        "common_mode_idf_reference_accuracy": (
            sum(value > 0 for value in common) / len(common)
        ),
        "common_mode_idf_reference_margin_mean": _mean(common),
        "full_idf_reference_accuracy": (
            sum(value > 0 for value in full) / len(full)
        ),
        "full_idf_reference_margin_mean": _mean(full),
        "isolated_value_accuracy": (
            sum(value > 0 for value in isolated) / len(isolated)
        ),
        "isolated_value_margin_mean": _mean(isolated),
        "role_value_phrase_accuracy": (
            sum(value > 0 for value in phrases) / len(phrases)
        ),
        "role_value_phrase_margin_mean": _mean(phrases),
        "value_to_phrase_accuracy_delta": (
            sum(value > 0 for value in phrases) / len(phrases)
            - sum(value > 0 for value in isolated) / len(isolated)
        ),
        "phrase_to_full_accuracy_delta": (
            sum(value > 0 for value in native) / len(native)
            - sum(value > 0 for value in phrases) / len(phrases)
        ),
        "value_to_phrase_margin_delta": (
            _mean(phrases) - _mean(isolated)
        ),
        "phrase_to_full_margin_delta": (
            _mean(native) - _mean(phrases)
        ),
    }


def pair_context_trajectory(
    records: Sequence[dict],
    role: str,
) -> dict[str, object]:
    contexts = (
        f"pair-{role}",
        "core-k8",
        "far64",
        f"dense-{role}64",
    )
    by_base: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in records:
        if row["view_id"] in contexts and role in row["role_pairs"]:
            by_base[str(row["base_id"])][str(row["view_id"])] = row

    complete = {
        base_id: views
        for base_id, views in by_base.items()
        if set(views) == set(contexts)
    }
    if not complete:
        raise ValueError("W6g pair trajectory has no complete bases")

    deltas = {
        "pair_to_core": [],
        "core_to_far": [],
        "far_to_dense": [],
    }
    first_loss = Counter()
    for views in complete.values():
        margins = {
            context: float(
                views[context]["role_pairs"][role]["native_margin"]
            )
            for context in contexts
        }
        deltas["pair_to_core"].append(
            margins["core-k8"] - margins[f"pair-{role}"]
        )
        deltas["core_to_far"].append(
            margins["far64"] - margins["core-k8"]
        )
        deltas["far_to_dense"].append(
            margins[f"dense-{role}64"] - margins["far64"]
        )

        label = "never"
        for context in contexts:
            if margins[context] <= 0.0:
                label = context
                break
        first_loss[label] += 1

    return {
        "complete_base_count": len(complete),
        "mean_margin_delta_pair_to_core": _mean(
            deltas["pair_to_core"]
        ),
        "mean_margin_delta_core_to_far": _mean(
            deltas["core_to_far"]
        ),
        "mean_margin_delta_far_to_dense": _mean(
            deltas["far_to_dense"]
        ),
        "first_pair_loss_context": dict(first_loss),
    }


def aggregate_fullset_rank(
    records: Sequence[dict],
    view_id: str,
) -> dict[str, object]:
    rows = [row for row in records if row["view_id"] == view_id]
    if not rows:
        raise ValueError("W6g full-set rank aggregation is empty")
    return {
        "n": len(rows),
        "native_top1": sum(bool(row["native"]["top1"]) for row in rows) / len(rows),
        "native_top5": sum(bool(row["native"]["top5"]) for row in rows) / len(rows),
        "native_mrr": _mean([
            float(row["native"]["reciprocal_rank"]) for row in rows
        ]),
        "native_margin_mean": _mean([
            float(row["native"]["margin"]) for row in rows
        ]),
        "final_top1": sum(bool(row["final"]["top1"]) for row in rows) / len(rows),
        "final_top5": sum(bool(row["final"]["top5"]) for row in rows) / len(rows),
        "final_mrr": _mean([
            float(row["final"]["reciprocal_rank"]) for row in rows
        ]),
        "final_margin_mean": _mean([
            float(row["final"]["margin"]) for row in rows
        ]),
        "probability_mass_max_error": max(
            float(row["probability_mass_error"]) for row in rows
        ),
        "native_wrong_winner_distance_counts": dict(
            Counter(
                int(row["native_wrong_winner"]["distance"])
                for row in rows
                if row["native_wrong_winner"] is not None
            )
        ),
        "final_wrong_winner_distance_counts": dict(
            Counter(
                int(row["final_wrong_winner"]["distance"])
                for row in rows
                if row["final_wrong_winner"] is not None
            )
        ),
        "native_wrong_winner_role_counts": dict(
            Counter(
                role
                for row in rows
                if row["native_wrong_winner"] is not None
                for role in row["native_wrong_winner"]["changed_roles"]
            )
        ),
        "final_wrong_winner_role_counts": dict(
            Counter(
                role
                for row in rows
                if row["final_wrong_winner"] is not None
                for role in row["final_wrong_winner"]["changed_roles"]
            )
        ),
    }


def stable_role_target(
    *,
    scorer_only: dict[str, str],
    joint_primary: dict[str, str],
    joint_replica: dict[str, str],
) -> dict[str, object]:
    labels = {
        label
        for label in joint_primary.values()
        if label != "NO_SECOND_ORDER_LOCALIZATION"
    }
    winners = []
    for label in sorted(labels):
        agreeing_domains = [
            domain
            for domain in sorted(joint_primary)
            if (
                joint_primary.get(domain) == label
                and joint_replica.get(domain) == label
            )
        ]
        if len(agreeing_domains) < 2:
            continue
        contradicted = any(
            scorer_only.get(domain)
            not in {label, "NO_SECOND_ORDER_LOCALIZATION"}
            for domain in agreeing_domains
        )
        if not contradicted:
            winners.append(
                {
                    "classification": label,
                    "domains": agreeing_domains,
                }
            )
    return {
        "stable": len(winners) == 1,
        "targets": winners,
    }


def role_localization_classification(
    *,
    pair: dict[str, object],
    far: dict[str, object],
    dense: dict[str, object],
) -> dict[str, object]:
    pair_acc = float(pair["native_pair_accuracy"])
    far_acc = float(far["native_pair_accuracy"])
    dense_acc = float(dense["native_pair_accuracy"])
    isolated_acc = float(pair["isolated_value_accuracy"])

    aggregation_acc_recovery = (
        float(dense["aggregation_idf_reference_accuracy"]) - dense_acc
    )
    common_acc_recovery = (
        float(dense["common_mode_idf_reference_accuracy"]) - dense_acc
    )
    aggregation_margin_recovery = (
        float(dense["aggregation_idf_reference_margin_mean"])
        - float(dense["native_margin_mean"])
    )
    common_margin_recovery = (
        float(dense["common_mode_idf_reference_margin_mean"])
        - float(dense["native_margin_mean"])
    )

    aggregation_recovers = (
        aggregation_acc_recovery >= 0.07
        or aggregation_margin_recovery >= 0.15
    )
    common_recovers = (
        common_acc_recovery >= 0.07
        or common_margin_recovery >= 0.15
    )
    density_sensitive = (
        pair_acc >= 0.90
        and far_acc >= pair_acc - 0.05
        and dense_acc <= far_acc - 0.10
    )

    if pair_acc < 0.85 and isolated_acc < 0.85:
        label = "FIELD_SEMANTIC_COLLAPSE"
    elif pair_acc < 0.85 and isolated_acc >= 0.90:
        label = "ROLE_BINDING_COLLAPSE"
    elif density_sensitive and aggregation_recovers and (
        common_acc_recovery < aggregation_acc_recovery / 2
        and common_margin_recovery < aggregation_margin_recovery / 2
    ):
        label = "IDF_AGGREGATION_INTERFERENCE"
    elif density_sensitive and common_recovers and (
        aggregation_acc_recovery < common_acc_recovery / 2
        and aggregation_margin_recovery < common_margin_recovery / 2
    ):
        label = "IDF_COMMON_MODE_INTERFERENCE"
    elif density_sensitive and aggregation_recovers and common_recovers:
        label = "MIXED_IDF_PATHWAY_INTERFERENCE"
    elif density_sensitive and not aggregation_recovers and not common_recovers:
        label = "DENSITY_NEAR_NEIGHBOR_LIMIT"
    else:
        label = "NO_SECOND_ORDER_LOCALIZATION"

    return {
        "classification": label,
        "pair_accuracy": pair_acc,
        "far_accuracy": far_acc,
        "dense_accuracy": dense_acc,
        "isolated_value_accuracy": isolated_acc,
        "density_sensitive": density_sensitive,
        "aggregation_accuracy_recovery": aggregation_acc_recovery,
        "common_mode_accuracy_recovery": common_acc_recovery,
        "aggregation_margin_recovery": aggregation_margin_recovery,
        "common_mode_margin_recovery": common_margin_recovery,
    }
