from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .competitive import (
    CompetitiveCoarseScorer,
    MIN_COVERAGE_WEIGHT,
    SALIENCE_THRESHOLD,
    candidate_relative_idf,
)
from .hira import HIRACore
from .interface_decomposition_cache import validate_w11_cache
from .runtime import PRIMITIVE_TO_ID


STAGES = ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7")
ADJACENT = tuple(zip(STAGES[:-1], STAGES[1:]))
DOMAINS = ("BJ", "BK", "BL", "BM")
K_VALUES = (4, 8, 16)
VIEWS = ("definition", "label")


def _rank(logits: Tensor, gold: int) -> tuple[int, float, bool]:
    x = logits.detach().cpu().float()
    gold_value = float(x[gold])
    better = int((x > gold_value).sum())
    tied_before = sum(1 for i in range(gold) if float(x[i]) == gold_value)
    rank = better + tied_before + 1
    return rank, 1.0 / rank, rank <= min(5, x.numel())


def _margin(logits: Tensor, gold: int) -> float:
    x = logits.detach().cpu().float()
    others = torch.cat([x[:gold], x[gold + 1 :]])
    return float(x[gold]) - float(others.max())


def _order(logits: Tensor) -> tuple[int, ...]:
    values = [float(x) for x in logits.detach().cpu().float()]
    return tuple(sorted(range(len(values)), key=lambda i: (-values[i], i)))


def _project(x: Tensor, projection: Tensor) -> Tensor:
    return F.normalize(F.linear(x.float(), projection.float()), dim=-1)


def _option_to_context(
    projected_options: Tensor,
    option_mask: Tensor,
    projected_context: Tensor,
    context_mask: Tensor | None = None,
) -> Tensor:
    similarity = torch.einsum("ktd,cd->ktc", projected_options, projected_context)
    if context_mask is not None:
        similarity = similarity.masked_fill(~context_mask[None, None, :], -1e4)
    coverage = similarity.max(dim=-1).values
    coverage = coverage.masked_fill(~option_mask, 0.0)
    return coverage


def _q0_symmetric(
    state_tokens: Tensor,
    option_tokens: Tensor,
    option_mask: Tensor,
    projection: Tensor,
) -> Tensor:
    state = _project(state_tokens, projection)
    options = _project(option_tokens, projection)
    similarity = torch.einsum("ktd,sd->kts", options, state)

    option_best = similarity.max(dim=-1).values
    option_best = option_best.masked_fill(~option_mask, 0.0)
    option_mean = option_best.sum(-1) / option_mask.sum(-1).clamp_min(1)

    masked = similarity.masked_fill(~option_mask[:, :, None], -1e4)
    state_best = masked.max(dim=1).values
    state_mean = state_best.mean(-1)
    return 0.5 * (option_mean + state_mean)


def _manual_q1_q5(
    *,
    state_tokens: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
    option_tokens: Tensor,
    option_token_ids: Tensor,
    option_mask: Tensor,
    projection: Tensor,
) -> dict[str, Tensor]:
    state = _project(state_tokens, projection)
    question = _project(question_tokens, projection)
    options = _project(option_tokens, projection)

    state_mask = torch.ones(state.shape[0], dtype=torch.bool)
    context = torch.cat([state, question], dim=0)
    context_mask = torch.cat([state_mask, question_mask.bool()], dim=0)

    q1_coverage = _option_to_context(options, option_mask, state)
    q1 = q1_coverage.sum(-1) / option_mask.sum(-1).clamp_min(1)

    q2_coverage = _option_to_context(options, option_mask, context, context_mask)
    q2 = q2_coverage.sum(-1) / option_mask.sum(-1).clamp_min(1)

    salience = candidate_relative_idf(
        option_token_ids.long().unsqueeze(0),
        option_mask.bool().unsqueeze(0),
    )[0].to(q2_coverage.dtype)
    weights = salience * option_mask.to(salience.dtype)
    denom = weights.sum(-1).clamp_min(1e-8)
    q3 = (q2_coverage * weights).sum(-1) / denom

    similarity = torch.einsum("ktd,cd->ktc", options, context)
    similarity = similarity.masked_fill(~context_mask[None, None, :], -1e4)
    common = (similarity * weights[..., None]).sum(dim=1) / denom[:, None]
    adjusted = similarity - common[:, None, :]
    adjusted = adjusted.masked_fill(~context_mask[None, None, :], -1e4)
    q4_coverage = adjusted.max(dim=-1).values
    q4 = (q4_coverage * weights).sum(-1) / denom

    salient_mask = option_mask.bool() & (salience >= SALIENCE_THRESHOLD)
    empty = salient_mask.sum(-1) == 0
    if empty.any():
        salient_mask = salient_mask.clone()
        salient_mask[empty] = option_mask.bool()[empty]
    min_coverage = q4_coverage.masked_fill(~salient_mask, 1e4).min(dim=-1).values
    q5 = q4 + MIN_COVERAGE_WEIGHT * min_coverage

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}


@torch.inference_mode()
def _q6_actual(
    scorer: CompetitiveCoarseScorer,
    base: Mapping[str, object],
    view: Mapping[str, object],
) -> Tensor:
    state = base["state_content_tokens"].float().unsqueeze(0)
    question = view["question_tokens"].float().unsqueeze(0)
    options = view["option_tokens"].float().unsqueeze(0)
    option_ids = view["option_token_ids"].long().unsqueeze(0)
    option_mask = view["option_content_mask"].bool().unsqueeze(0)
    question_mask = view["question_content_mask"].bool().unsqueeze(0)
    return scorer(
        state_tokens=state,
        state_mask=torch.ones(1, state.shape[1], dtype=torch.bool),
        question_tokens=question,
        question_mask=question_mask,
        option_tokens=options,
        option_token_ids=option_ids,
        option_mask=option_mask,
    )[0]


@torch.inference_mode()
def _q7_final(
    hira: HIRACore,
    base: Mapping[str, object],
    view: Mapping[str, object],
    coarse: Tensor,
):
    question = view["question_embedding"].float().unsqueeze(0)
    state_segments = base["state_segments"].float().unsqueeze(0)
    options = view["option_embeddings"].float().unsqueeze(0)
    qtype = torch.tensor([PRIMITIVE_TO_ID["choice"]], dtype=torch.long)
    out = hira(
        question,
        state_segments,
        options,
        qtype,
        coarse_override=coarse.unsqueeze(0),
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )
    return out


def _record(
    *,
    base: Mapping[str, object],
    view: Mapping[str, object],
    stage: str,
    logits: Tensor,
) -> dict[str, object]:
    gold = int(view["gold_index"])
    rank, mrr, top5 = _rank(logits, gold)
    return {
        "base_id": str(base["base_id"]),
        "case_id": str(view["case_id"]),
        "domain_id": str(base["domain_id"]),
        "view_id": str(view["view_id"]),
        "k": int(view["diagnosis_k"]),
        "stage": stage,
        "correct": rank == 1,
        "rank": rank,
        "mrr": mrr,
        "top5": top5,
        "margin": _margin(logits, gold),
        "order": _order(logits),
    }


def _cell(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("W11 summary cell cannot be empty")
    n = len(rows)
    return {
        "n": n,
        "top1": sum(bool(x["correct"]) for x in rows) / n,
        "top5": sum(bool(x["top5"]) for x in rows) / n,
        "mrr": sum(float(x["mrr"]) for x in rows) / n,
        "mean_margin": sum(float(x["margin"]) for x in rows) / n,
    }


def summarize_stage(records: list[dict[str, object]]) -> dict[str, object]:
    per_domain: dict[str, object] = {}
    for domain in DOMAINS:
        domain_rows = [x for x in records if x["domain_id"] == domain]
        views: dict[str, object] = {}
        for view_id in VIEWS:
            views[view_id] = {}
            for k in K_VALUES:
                rows = [
                    x for x in domain_rows
                    if x["view_id"] == view_id and int(x["k"]) == k
                ]
                views[view_id][str(k)] = _cell(rows)
        per_domain[domain] = {"views": views}

    pooled: dict[str, object] = {}
    for view_id in VIEWS:
        pooled[view_id] = {}
        for k in K_VALUES:
            rows = [
                x for x in records
                if x["view_id"] == view_id and int(x["k"]) == k
            ]
            pooled[view_id][str(k)] = _cell(rows)
    return {"per_domain": per_domain, "pooled": {"views": pooled}}


def _index(records: list[dict[str, object]]) -> dict[tuple[str, str, int], dict[str, object]]:
    out: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in records:
        key = (str(row["base_id"]), str(row["view_id"]), int(row["k"]))
        if key in out:
            raise ValueError("duplicate W11 stage identity")
        out[key] = row
    return out


def _transition(
    source: list[dict[str, object]],
    target: list[dict[str, object]],
) -> dict[str, object]:
    a = _index(source)
    b = _index(target)
    if set(a) != set(b):
        raise ValueError("W11 adjacent stages lost paired identity")

    def summarize(keys: list[tuple[str, str, int]]) -> dict[str, float | int]:
        n = len(keys)
        ctw = sum(bool(a[k]["correct"]) and not bool(b[k]["correct"]) for k in keys)
        wtc = sum(not bool(a[k]["correct"]) and bool(b[k]["correct"]) for k in keys)
        return {
            "n": n,
            "correct_to_wrong_count": ctw,
            "correct_to_wrong_rate": ctw / n,
            "wrong_to_correct_count": wtc,
            "wrong_to_correct_rate": wtc / n,
            "mean_rank_delta": sum(int(b[k]["rank"]) - int(a[k]["rank"]) for k in keys) / n,
            "mean_margin_delta": sum(float(b[k]["margin"]) - float(a[k]["margin"]) for k in keys) / n,
        }

    by_cell: dict[str, object] = {}
    for view_id in VIEWS:
        by_cell[view_id] = {}
        for k in K_VALUES:
            keys = [key for key in a if key[1] == view_id and key[2] == k]
            by_cell[view_id][str(k)] = summarize(keys)

    gate_keys = [
        key for key in a
        if key[1] == "definition" and key[2] in {4, 16}
    ]
    gate = summarize(gate_keys)
    return {
        "cells": by_cell,
        "gate_n": gate["n"],
        "gate_correct_to_wrong_count": gate["correct_to_wrong_count"],
        "gate_correct_to_wrong_rate": gate["correct_to_wrong_rate"],
        "gate_wrong_to_correct_count": gate["wrong_to_correct_count"],
        "gate_wrong_to_correct_rate": gate["wrong_to_correct_rate"],
        "gate_mean_rank_delta": gate["mean_rank_delta"],
        "gate_mean_margin_delta": gate["mean_margin_delta"],
    }


def _first_event_histograms(
    stage_records: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    indexed = {stage: _index(rows) for stage, rows in stage_records.items()}
    keys = set(indexed["Q0"])
    if any(set(rows) != keys for rows in indexed.values()):
        raise ValueError("W11 stage identity mismatch for first-event analysis")

    result: dict[str, object] = {}
    for k_value in K_VALUES:
        loss = Counter()
        rescue = Counter()
        eligible_loss = 0
        eligible_rescue = 0
        for key in sorted(keys):
            if key[1] != "definition" or key[2] != k_value:
                continue
            q0 = indexed["Q0"][key]
            q7 = indexed["Q7"][key]

            if bool(q0["correct"]) and not bool(q7["correct"]):
                eligible_loss += 1
                current = True
                found = None
                for source, target in ADJACENT:
                    a = bool(indexed[source][key]["correct"])
                    b = bool(indexed[target][key]["correct"])
                    if a and not b:
                        found = f"{source}->{target}"
                        break
                    current = b
                loss[found or "NO_ADJACENT_FIRST_LOSS"] += 1

            if not bool(q0["correct"]) and bool(q7["correct"]):
                eligible_rescue += 1
                found = None
                for source, target in ADJACENT:
                    a = bool(indexed[source][key]["correct"])
                    b = bool(indexed[target][key]["correct"])
                    if not a and b:
                        found = f"{source}->{target}"
                        break
                rescue[found or "NO_ADJACENT_FIRST_RESCUE"] += 1

        result[str(k_value)] = {
            "q0_correct_q7_wrong": eligible_loss,
            "first_loss_histogram": dict(loss),
            "q0_wrong_q7_correct": eligible_rescue,
            "first_rescue_histogram": dict(rescue),
        }
    return result


def _equivalence(
    q5: list[dict[str, object]],
    q6: list[dict[str, object]],
) -> dict[str, object]:
    a = _index(q5)
    b = _index(q6)
    if set(a) != set(b):
        raise ValueError("W11 Q5/Q6 identity mismatch")
    out: dict[str, object] = {}
    for k in K_VALUES:
        keys = [key for key in a if key[1] == "definition" and key[2] == k]
        top1_equal = sum(a[key]["order"][0] == b[key]["order"][0] for key in keys)
        full_equal = sum(a[key]["order"] == b[key]["order"] for key in keys)
        out[str(k)] = {
            "n": len(keys),
            "top1_identity_rate": top1_equal / len(keys),
            "full_rank_order_identity_rate": full_equal / len(keys),
        }
    return out


@torch.inference_mode()
def evaluate_interface_stages(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w11_cache(cache)
    hira.eval()
    scorer.eval()

    projection = scorer.projection.weight.detach().float()
    records: dict[str, list[dict[str, object]]] = {stage: [] for stage in STAGES}
    max_mass_error = 0.0

    for base in cache["bases"]:
        state_tokens = base["state_content_tokens"].float()
        for view in base["views"]:
            option_tokens = view["option_tokens"].float()
            option_ids = view["option_token_ids"].long()
            option_mask = view["option_content_mask"].bool()
            question_tokens = view["question_tokens"].float()
            question_mask = view["question_content_mask"].bool()

            logits: dict[str, Tensor] = {}
            logits["Q0"] = _q0_symmetric(
                state_tokens,
                option_tokens,
                option_mask,
                projection,
            )
            logits.update(
                _manual_q1_q5(
                    state_tokens=state_tokens,
                    question_tokens=question_tokens,
                    question_mask=question_mask,
                    option_tokens=option_tokens,
                    option_token_ids=option_ids,
                    option_mask=option_mask,
                    projection=projection,
                )
            )
            logits["Q6"] = _q6_actual(scorer, base, view)
            out = _q7_final(hira, base, view, logits["Q6"])
            logits["Q7"] = out.logits[0].detach().cpu().float()

            p = out.probabilities[0].detach().cpu().to(torch.float64)
            max_mass_error = max(max_mass_error, abs(float(p.sum()) - 1.0))

            for stage in STAGES:
                records[stage].append(
                    _record(
                        base=base,
                        view=view,
                        stage=stage,
                        logits=logits[stage],
                    )
                )

    transitions: dict[str, object] = {}
    for source, target in ADJACENT:
        transitions[f"{source}->{target}"] = _transition(
            records[source],
            records[target],
        )

    per_domain_transitions: dict[str, dict[str, object]] = {}
    per_domain_first: dict[str, object] = {}
    for domain in DOMAINS:
        stage_subset = {
            stage: [x for x in rows if x["domain_id"] == domain]
            for stage, rows in records.items()
        }
        per_domain_transitions[domain] = {
            f"{source}->{target}": _transition(
                stage_subset[source],
                stage_subset[target],
            )
            for source, target in ADJACENT
        }
        per_domain_first[domain] = _first_event_histograms(stage_subset)

    return {
        "stages": {
            stage: summarize_stage(rows)
            for stage, rows in records.items()
        },
        "transitions": transitions,
        "per_domain_transitions": per_domain_transitions,
        "first_events": per_domain_first,
        "q5_q6_equivalence": _equivalence(records["Q5"], records["Q6"]),
        "probability_mass_max_error": max_mass_error,
        "state_encodes_per_base": float(cache["metadata"]["state_encode_calls_per_base"]),
        "cached_base_count": int(cache["metadata"]["base_count"]),
        "cached_view_count": int(cache["metadata"]["view_count"]),
    }


def make_reference_records(
    cache: dict,
    score_lookup: dict[str, list[float]],
) -> list[dict[str, object]]:
    validate_w11_cache(cache)
    rows: list[dict[str, object]] = []
    for base in cache["bases"]:
        for view in base["views"]:
            case_id = str(view["case_id"])
            scores = score_lookup.get(case_id)
            if scores is None:
                raise ValueError(f"missing W11 reference scores for {case_id}")
            logits = torch.tensor(scores, dtype=torch.float32)
            if logits.numel() != int(view["diagnosis_k"]):
                raise ValueError("W11 reference score width mismatch")
            rows.append(
                _record(
                    base=base,
                    view=view,
                    stage="R0",
                    logits=logits,
                )
            )
    return rows


def summarize_reference(records: list[dict[str, object]]) -> dict[str, object]:
    return summarize_stage(records)
