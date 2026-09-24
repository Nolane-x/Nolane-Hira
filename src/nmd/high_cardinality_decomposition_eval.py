from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .high_cardinality_decomposition import architecture_classification
from .high_cardinality_decomposition_authority import (
    DecompositionView,
    DOMAINS,
    VIEW_IDS,
)
from .hira import HIRACore
from .runtime import NolaneHira, PRIMITIVE_TO_ID


def _case_id_sha256(views: Sequence[DecompositionView]) -> str:
    payload = "\n".join(
        sorted(view.typed.case_id for view in views)
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _semantic_sha256(views: Sequence[DecompositionView]) -> str:
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


def validate_w6j_cache(cache: dict) -> None:
    if not isinstance(cache, dict):
        raise ValueError("W6j cache must be a dict")
    metadata = cache.get("metadata")
    bases = cache.get("bases")
    if not isinstance(metadata, dict) or not isinstance(bases, list):
        raise ValueError("W6j cache requires metadata and bases")
    if metadata.get("schema_version") != "r8-w6j-decomposition-cache-v1":
        raise ValueError("unexpected W6j cache schema")
    if int(metadata.get("base_count", -1)) != len(bases):
        raise ValueError("W6j base count mismatch")
    if float(metadata.get("state_encodes_per_base", -1.0)) != 1.0:
        raise ValueError("W6j must encode state exactly once per base")

    seen: set[str] = set()
    for base in bases:
        base_id = str(base.get("base_id", ""))
        if not base_id or base_id in seen:
            raise ValueError("invalid or duplicate W6j base id")
        seen.add(base_id)
        if base.get("domain_id") not in {"AD", "AE", "AF"}:
            raise ValueError("invalid W6j domain")

        segments = base.get("state_segments")
        state_tokens = base.get("state_content_tokens")
        if (
            not isinstance(segments, Tensor)
            or segments.ndim != 2
            or segments.shape[-1] != 256
        ):
            raise ValueError("W6j state_segments must be [S,256]")
        if (
            not isinstance(state_tokens, Tensor)
            or state_tokens.ndim != 2
            or state_tokens.shape[-1] != 256
            or state_tokens.shape[0] < 1
        ):
            raise ValueError("W6j state_content_tokens must be [T,256]")

        question = base.get("question_embedding")
        question_tokens = base.get("question_tokens")
        question_mask = base.get("question_content_mask")
        options = base.get("master_option_embeddings")
        option_tokens = base.get("master_option_tokens")
        option_ids = base.get("master_option_token_ids")
        option_mask = base.get("master_option_content_mask")
        signatures = base.get("master_option_signatures")
        distances = base.get("master_option_distances")
        stable_ids = base.get("master_option_ids")

        if not isinstance(question, Tensor) or question.shape != (256,):
            raise ValueError("W6j question embedding must be [256]")
        if (
            not isinstance(question_tokens, Tensor)
            or question_tokens.ndim != 2
            or question_tokens.shape[-1] != 256
        ):
            raise ValueError("W6j question tokens must be [T,256]")
        if (
            not isinstance(question_mask, Tensor)
            or question_mask.shape != question_tokens.shape[:1]
            or question_mask.dtype != torch.bool
        ):
            raise ValueError("W6j question content mask mismatch")
        if not isinstance(options, Tensor) or options.shape != (64, 256):
            raise ValueError("W6j master option embeddings must be [64,256]")
        if (
            not isinstance(option_tokens, Tensor)
            or option_tokens.ndim != 3
            or option_tokens.shape[0] != 64
            or option_tokens.shape[-1] != 256
        ):
            raise ValueError("W6j master option tokens must be [64,T,256]")
        if (
            not isinstance(option_ids, Tensor)
            or option_ids.shape != option_tokens.shape[:2]
            or option_ids.dtype != torch.long
        ):
            raise ValueError("W6j option token IDs mismatch")
        if (
            not isinstance(option_mask, Tensor)
            or option_mask.shape != option_tokens.shape[:2]
            or option_mask.dtype != torch.bool
            or (option_mask.sum(-1) < 1).any()
        ):
            raise ValueError("W6j option content mask mismatch")
        if not isinstance(signatures, tuple) or len(signatures) != 64:
            raise ValueError("W6j master signature metadata mismatch")
        if not isinstance(distances, tuple) or len(distances) != 64:
            raise ValueError("W6j master distance metadata mismatch")
        if not isinstance(stable_ids, tuple) or len(stable_ids) != 64:
            raise ValueError("W6j master option identity mismatch")

        gold = int(base.get("gold_master_index", -1))
        if not 0 <= gold < 64 or int(distances[gold]) != 0:
            raise ValueError("W6j gold master index mismatch")
        if Counter(int(x) for x in distances) != {
            0: 1,
            1: 12,
            2: 20,
            3: 15,
            4: 16,
        }:
            raise ValueError("W6j master distance histogram mismatch")

        view_indices = base.get("view_indices")
        view_gold = base.get("view_gold_positions")
        if not isinstance(view_indices, dict) or not isinstance(view_gold, dict):
            raise ValueError("W6j nested view metadata missing")
        if set(view_indices) != {"8", "16", "32", "64"}:
            raise ValueError("W6j nested view index keys mismatch")
        if set(view_gold) != set(view_indices):
            raise ValueError("W6j nested gold keys mismatch")
        previous: set[int] = set()
        for k in (8, 16, 32, 64):
            indices = tuple(int(x) for x in view_indices[str(k)])
            if len(indices) != k or len(set(indices)) != k:
                raise ValueError("W6j nested view index size mismatch")
            if not all(0 <= index < 64 for index in indices):
                raise ValueError("W6j nested view index out of range")
            if gold not in indices:
                raise ValueError("W6j nested view lost gold")
            if int(view_gold[str(k)]) != indices.index(gold):
                raise ValueError("W6j nested gold position mismatch")
            current = set(indices)
            if previous and not previous.issubset(current):
                raise ValueError("W6j nested candidate identity mismatch")
            previous = current


@torch.inference_mode()
def compile_w6j_cache(
    model: NolaneHira,
    views: Sequence[DecompositionView],
) -> dict:
    """Compile one state and one master K64 schema per fresh W6j base."""
    model.eval()
    grouped: dict[str, list[DecompositionView]] = defaultdict(list)
    for view in views:
        grouped[view.base_id].append(view)

    before = model.state_encode_calls
    bases: list[dict] = []

    for base_id in sorted(grouped):
        rows = grouped[base_id]
        if {row.view_id for row in rows} != set(VIEW_IDS):
            raise ValueError("W6j base must expose K8/K16/K32/K64")
        if len({row.typed.state_text for row in rows}) != 1:
            raise ValueError("W6j base state drift")
        if len({row.gold_signature for row in rows}) != 1:
            raise ValueError("W6j base gold drift")

        lookup = {row.view_id: row for row in rows}
        master = lookup["k64"]
        decision = master.typed.decisions[0]
        if decision.question_id != "diagnosis" or decision.primitive != "choice":
            raise ValueError("W6j master decision must be diagnosis/choice")

        memory = model.compile_state(master.typed.state_text, segment_tokens=32)
        if memory.content_token_embeddings is None:
            raise RuntimeError("W6j requires state content token embeddings")

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
            raise RuntimeError("W6j master schema token artifacts incomplete")

        master_ids = tuple(option.option_id for option in decision.options)
        id_to_master = {
            option_id: index
            for index, option_id in enumerate(master_ids)
        }
        if len(id_to_master) != 64:
            raise RuntimeError("W6j master option IDs must be unique")

        view_indices: dict[str, tuple[int, ...]] = {}
        view_gold_positions: dict[str, int] = {}
        for k in (8, 16, 32, 64):
            view = lookup[f"k{k}"]
            view_decision = view.typed.decisions[0]
            if view_decision.question_text != decision.question_text:
                raise RuntimeError("W6j question text drift across views")
            indices = tuple(
                id_to_master[option.option_id]
                for option in view_decision.options
            )
            for option, master_index in zip(view_decision.options, indices):
                master_option = decision.options[master_index]
                if option.criterion_text != master_option.criterion_text:
                    raise RuntimeError("W6j candidate text drift across views")
            view_indices[str(k)] = indices
            gold_master = id_to_master[view.gold_option_id]
            if gold_master != int(decision.gold_index):
                raise RuntimeError("W6j gold identity drift across views")
            view_gold_positions[str(k)] = indices.index(gold_master)

        bases.append(
            {
                "base_id": base_id,
                "domain_id": master.domain_id,
                "template_id": master.template_id,
                "severity": int(master.severity),
                "confidence": master.confidence,
                "roles": tuple(DOMAINS[master.domain_id].roles),
                "gold_signature": tuple(master.gold_signature),
                "state_segments": (
                    memory.segment_embeddings.detach().cpu().to(torch.float16)
                ),
                "state_content_tokens": (
                    memory.content_token_embeddings.detach().cpu().to(torch.float16)
                ),
                "question_embedding": (
                    schema.question_embedding.detach().cpu().to(torch.float16)
                ),
                "question_tokens": (
                    schema.question_token_embeddings.detach().cpu().to(torch.float16)
                ),
                "question_content_mask": (
                    schema.question_content_token_mask.detach().cpu().bool()
                ),
                "master_schema_hash": receipt.schema_hash,
                "master_option_embeddings": (
                    schema.option_embeddings.detach().cpu().to(torch.float16)
                ),
                "master_option_tokens": (
                    schema.option_token_embeddings.detach().cpu().to(torch.float16)
                ),
                "master_option_token_ids": (
                    schema.option_token_ids.detach().cpu().long()
                ),
                "master_option_content_mask": (
                    schema.option_content_token_mask.detach().cpu().bool()
                ),
                "master_option_ids": master_ids,
                "master_option_signatures": tuple(master.option_signatures),
                "master_option_distances": tuple(
                    int(x) for x in master.option_distances
                ),
                "gold_master_index": int(decision.gold_index),
                "view_indices": view_indices,
                "view_gold_positions": view_gold_positions,
            }
        )

    state_calls = model.state_encode_calls - before
    if state_calls != len(bases):
        raise RuntimeError("W6j violated one-state-encode-per-base contract")

    cache = {
        "metadata": {
            "schema_version": "r8-w6j-decomposition-cache-v1",
            "scope": "fresh architectural diagnostic; no training",
            "base_count": len(bases),
            "view_count": len(views),
            "case_id_sha256": _case_id_sha256(views),
            "semantic_view_sha256": _semantic_sha256(views),
            "state_encode_calls": state_calls,
            "state_encodes_per_base": state_calls / max(1, len(bases)),
            "domain_base_counts": dict(
                Counter(base["domain_id"] for base in bases)
            ),
        },
        "bases": bases,
    }
    validate_w6j_cache(cache)
    return cache


def save_w6j_cache(cache: dict, path: str | Path) -> Path:
    validate_w6j_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_w6j_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_w6j_cache(cache)
    return cache


def _rank_metrics(logits: Tensor, gold_index: int) -> dict[str, object]:
    row = logits.detach().cpu().to(torch.float64).flatten()
    values = [float(value) for value in row.tolist()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("W6j rank metrics require finite logits")
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    rank = order.index(gold_index) + 1
    negatives = [index for index in order if index != gold_index]
    best_negative = negatives[0]
    return {
        "rank": rank,
        "top1": order[0] == gold_index,
        "top5": gold_index in set(order[: min(5, len(order))]),
        "reciprocal_rank": 1.0 / rank,
        "margin": values[gold_index] - values[best_negative],
        "gold_logit": values[gold_index],
        "predicted_index": order[0],
        "best_negative_index": best_negative,
    }


def _subset_tensors(base: dict, indices: tuple[int, ...]) -> dict[str, Tensor]:
    index = torch.tensor(indices, dtype=torch.long)
    state_tokens = base["state_content_tokens"].float().unsqueeze(0)
    return {
        "state_segments": base["state_segments"].float().unsqueeze(0),
        "state_tokens": state_tokens,
        "state_mask": torch.ones(
            1, state_tokens.shape[1], dtype=torch.bool
        ),
        "question": base["question_embedding"].float().unsqueeze(0),
        "question_tokens": base["question_tokens"].float().unsqueeze(0),
        "question_mask": base["question_content_mask"].bool().unsqueeze(0),
        "options": base["master_option_embeddings"].float()[index].unsqueeze(0),
        "option_tokens": (
            base["master_option_tokens"].float()[index].unsqueeze(0)
        ),
        "option_token_ids": (
            base["master_option_token_ids"].long()[index].unsqueeze(0)
        ),
        "option_mask": (
            base["master_option_content_mask"].bool()[index].unsqueeze(0)
        ),
        "qtype": torch.tensor(
            [PRIMITIVE_TO_ID["choice"]],
            dtype=torch.long,
        ),
    }


@torch.inference_mode()
def _score_subset(
    hira: HIRACore,
    scorer: nn.Module,
    base: dict,
    indices: tuple[int, ...],
) -> dict[str, object]:
    tensors = _subset_tensors(base, indices)
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
        forced_budget=len(indices),
        adaptive_budget=False,
    )
    gold_master = int(base["gold_master_index"])
    gold_position = indices.index(gold_master)
    coarse_metrics = _rank_metrics(out.coarse_logits[0], gold_position)
    final_metrics = _rank_metrics(out.logits[0], gold_position)
    predicted_master = indices[int(final_metrics["predicted_index"])]
    coarse_predicted_master = indices[int(coarse_metrics["predicted_index"])]
    return {
        "indices": indices,
        "gold_position": gold_position,
        "coarse": coarse_metrics,
        "final": final_metrics,
        "predicted_master_index": predicted_master,
        "coarse_predicted_master_index": coarse_predicted_master,
        "probability_mass_error": abs(float(out.probabilities[0].sum()) - 1.0),
    }


@torch.inference_mode()
def _score_all_gold_negative_pairs(
    hira: HIRACore,
    scorer: nn.Module,
    base: dict,
) -> list[dict[str, object]]:
    """Evaluate the 63 independent gold-vs-negative K2 contexts in one batch.

    Each batch row remains a distinct two-candidate scorer context, so
    candidate-relative IDF/common-mode are computed independently per pair.
    This is a compute optimization only; it does not change the diagnostic.
    """
    gold = int(base["gold_master_index"])
    negatives = [index for index in range(64) if index != gold]
    pair_indices = torch.tensor(
        [sorted((gold, negative)) for negative in negatives],
        dtype=torch.long,
    )
    batch = pair_indices.shape[0]

    state_segments = base["state_segments"].float().unsqueeze(0).expand(
        batch, -1, -1
    )
    state_tokens = base["state_content_tokens"].float().unsqueeze(0).expand(
        batch, -1, -1
    )
    state_mask = torch.ones(
        batch,
        state_tokens.shape[1],
        dtype=torch.bool,
    )
    question = base["question_embedding"].float().unsqueeze(0).expand(
        batch, -1
    )
    question_tokens = base["question_tokens"].float().unsqueeze(0).expand(
        batch, -1, -1
    )
    question_mask = base["question_content_mask"].bool().unsqueeze(0).expand(
        batch, -1
    )

    master_options = base["master_option_embeddings"].float()
    master_option_tokens = base["master_option_tokens"].float()
    master_option_ids = base["master_option_token_ids"].long()
    master_option_mask = base["master_option_content_mask"].bool()

    options = master_options[pair_indices]
    option_tokens = master_option_tokens[pair_indices]
    option_token_ids = master_option_ids[pair_indices]
    option_mask = master_option_mask[pair_indices]
    qtype = torch.full(
        (batch,),
        PRIMITIVE_TO_ID["choice"],
        dtype=torch.long,
    )

    coarse = scorer(
        state_tokens=state_tokens,
        state_mask=state_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_tokens=option_tokens,
        option_token_ids=option_token_ids,
        option_mask=option_mask,
    )
    out = hira(
        question,
        state_segments,
        options,
        qtype,
        coarse_override=coarse,
        forced_budget=2,
        adaptive_budget=False,
    )

    records: list[dict[str, object]] = []
    for row_index, negative in enumerate(negatives):
        indices = tuple(int(x) for x in pair_indices[row_index].tolist())
        gold_position = indices.index(gold)
        coarse_metrics = _rank_metrics(
            out.coarse_logits[row_index],
            gold_position,
        )
        final_metrics = _rank_metrics(
            out.logits[row_index],
            gold_position,
        )
        records.append(
            {
                "negative_master_index": negative,
                "distance": int(
                    base["master_option_distances"][negative]
                ),
                "coarse_gold_win": bool(coarse_metrics["top1"]),
                "final_gold_win": bool(final_metrics["top1"]),
                "coarse_margin": float(coarse_metrics["margin"]),
                "final_margin": float(final_metrics["margin"]),
                "probability_mass_error": abs(
                    float(out.probabilities[row_index].sum()) - 1.0
                ),
            }
        )
    return records


@torch.inference_mode()
def _oracle_relation_probe(
    hira: HIRACore,
    base: dict,
    indices: tuple[int, ...],
) -> dict[str, object]:
    tensors = _subset_tensors(base, indices)
    distances = [
        int(base["master_option_distances"][index])
        for index in indices
    ]
    oracle = torch.tensor(
        [[4.0 - float(distance) for distance in distances]],
        dtype=tensors["options"].dtype,
    )
    out = hira(
        tensors["question"],
        tensors["state_segments"],
        tensors["options"],
        tensors["qtype"],
        coarse_override=oracle,
        forced_budget=len(indices),
        adaptive_budget=False,
    )
    gold_position = indices.index(int(base["gold_master_index"]))
    return {
        "coarse": _rank_metrics(out.coarse_logits[0], gold_position),
        "final": _rank_metrics(out.logits[0], gold_position),
        "probability_mass_error": abs(float(out.probabilities[0].sum()) - 1.0),
    }


@torch.inference_mode()
def diagnose_w6j_base(
    hira: HIRACore,
    scorer: nn.Module,
    base: dict,
) -> dict[str, object]:
    hira.eval()
    scorer.eval()

    views = {
        str(k): _score_subset(
            hira,
            scorer,
            base,
            tuple(int(x) for x in base["view_indices"][str(k)]),
        )
        for k in (8, 16, 32, 64)
    }

    gold = int(base["gold_master_index"])
    pairs = _score_all_gold_negative_pairs(
        hira,
        scorer,
        base,
    )

    master_indices = tuple(
        int(x) for x in base["view_indices"]["64"]
    )
    oracle = _oracle_relation_probe(hira, base, master_indices)

    return {
        "base_id": base["base_id"],
        "domain_id": base["domain_id"],
        "views": views,
        "pairs": pairs,
        "oracle": oracle,
    }


def _mean_bool(values: list[bool]) -> float:
    return sum(int(value) for value in values) / len(values) if values else 0.0


def aggregate_w6j_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("W6j aggregation requires records")

    k64_rows = [row["views"]["64"] for row in records]  # type: ignore[index]
    coarse_correct = [bool(row["coarse"]["top1"]) for row in k64_rows]  # type: ignore[index]
    final_correct = [bool(row["final"]["top1"]) for row in k64_rows]  # type: ignore[index]
    rescue = [
        (not coarse) and final
        for coarse, final in zip(coarse_correct, final_correct)
    ]
    damage = [
        coarse and (not final)
        for coarse, final in zip(coarse_correct, final_correct)
    ]

    global_error_indices = [
        index
        for index, value in enumerate(final_correct)
        if not value
    ]

    error_has_pair_loss: list[bool] = []
    winner_reversal: list[bool] = []
    winner_beats_k2_coarse: list[bool] = []
    all_pair_win_flags: list[bool] = []
    pair_loss_counts: list[int] = []
    pair_loss_distance_counts = Counter()

    for index, record in enumerate(records):
        pairs = record["pairs"]  # type: ignore[index]
        pair_lookup = {
            int(pair["negative_master_index"]): pair
            for pair in pairs
        }
        losses = [
            pair for pair in pairs
            if not bool(pair["final_gold_win"])
        ]
        pair_loss_counts.append(len(losses))
        all_pair_win = len(losses) == 0
        all_pair_win_flags.append(all_pair_win)
        for pair in losses:
            pair_loss_distance_counts[int(pair["distance"])] += 1

        if index not in global_error_indices:
            continue

        error_has_pair_loss.append(bool(losses))
        winner = int(
            record["views"]["64"]["predicted_master_index"]  # type: ignore[index]
        )
        winner_pair = pair_lookup[winner]
        winner_reversal.append(bool(winner_pair["final_gold_win"]))
        winner_beats_k2_coarse.append(
            not bool(winner_pair["coarse_gold_win"])
        )

    all_pair_win_indices = [
        index
        for index, value in enumerate(all_pair_win_flags)
        if value
    ]
    all_pair_fail_count = sum(
        not final_correct[index]
        for index in all_pair_win_indices
    )
    all_pair_fail_rate = (
        all_pair_fail_count / len(all_pair_win_indices)
        if all_pair_win_indices
        else 0.0
    )

    coarse_correct_count = sum(coarse_correct)
    relation_damage_given_correct = (
        sum(damage) / coarse_correct_count
        if coarse_correct_count
        else 0.0
    )

    oracle_final = [
        bool(record["oracle"]["final"]["top1"])  # type: ignore[index]
        for record in records
    ]

    metrics: dict[str, object] = {
        "base_count": len(records),
        "k64_coarse_top1": _mean_bool(coarse_correct),
        "k64_final_top1": _mean_bool(final_correct),
        "relation_rescue_rate": _mean_bool(rescue),
        "relation_damage_rate": _mean_bool(damage),
        "relation_rescue_top1_gain": (
            _mean_bool(final_correct) - _mean_bool(coarse_correct)
        ),
        "coarse_correct_case_rate": _mean_bool(coarse_correct),
        "relation_damage_given_coarse_correct_rate": (
            relation_damage_given_correct
        ),
        "global_error_count": len(global_error_indices),
        "global_error_has_pair_loss_rate": (
            _mean_bool(error_has_pair_loss)
            if global_error_indices
            else 0.0
        ),
        "winner_reversal_rate": (
            _mean_bool(winner_reversal)
            if global_error_indices
            else 0.0
        ),
        "winner_k2_final_to_k64_final_sign_reversal_rate": (
            _mean_bool(winner_reversal)
            if global_error_indices
            else 0.0
        ),
        "global_error_winner_beats_gold_k2_coarse_rate": (
            _mean_bool(winner_beats_k2_coarse)
            if global_error_indices
            else 0.0
        ),
        "all_pair_win_case_rate": _mean_bool(all_pair_win_flags),
        "all_pair_win_case_count": len(all_pair_win_indices),
        "all_pair_win_but_k64_fail_rate": all_pair_fail_rate,
        "mean_pair_loss_count": (
            sum(pair_loss_counts) / len(pair_loss_counts)
        ),
        "pair_loss_count_histogram": dict(Counter(pair_loss_counts)),
        "pair_loss_distance_histogram": dict(pair_loss_distance_counts),
        "oracle_final_top1": _mean_bool(oracle_final),
        "oracle_relation_damage_rate": 1.0 - _mean_bool(oracle_final),
    }

    for k in (8, 16, 32, 64):
        rows = [record["views"][str(k)] for record in records]  # type: ignore[index]
        metrics[f"k{k}_coarse_top1"] = _mean_bool(
            [bool(row["coarse"]["top1"]) for row in rows]  # type: ignore[index]
        )
        metrics[f"k{k}_final_top1"] = _mean_bool(
            [bool(row["final"]["top1"]) for row in rows]  # type: ignore[index]
        )

    classification = architecture_classification(
        {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    )
    return {
        "metrics": metrics,
        "classification": classification,
    }


def aggregate_w6j_checkpoint(
    records: list[dict[str, object]],
) -> dict[str, object]:
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_domain[str(record["domain_id"])].append(record)
    if set(by_domain) != {"AD", "AE", "AF"}:
        raise ValueError("W6j checkpoint aggregation requires AD/AE/AF")
    return {
        "per_domain": {
            domain: aggregate_w6j_records(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": aggregate_w6j_records(records),
    }
