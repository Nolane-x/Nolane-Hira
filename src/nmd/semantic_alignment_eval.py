from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .hira import HIRACore
from .runtime import PRIMITIVE_TO_ID
from .semantic_alignment_bridge import semantic_alignment_scores
from .semantic_alignment_cache import validate_w9_cache


ALIGNMENT_TEMPERATURE = 0.07


@torch.inference_mode()
def _cached_forward(
    hira: HIRACore,
    scorer: nn.Module,
    base: Mapping[str, object],
    view: Mapping[str, object],
):
    hira.eval()
    scorer.eval()

    state_segments = base["state_segments"].float().unsqueeze(0)
    state_tokens = base["state_content_tokens"].float().unsqueeze(0)
    question = view["question_embedding"].float().unsqueeze(0)
    options = view["option_embeddings"].float().unsqueeze(0)
    question_tokens = view["question_tokens"].float().unsqueeze(0)
    question_mask = view["question_content_mask"].bool().unsqueeze(0)
    option_tokens = view["option_tokens"].float().unsqueeze(0)
    option_ids = view["option_token_ids"].long().unsqueeze(0)
    option_mask = view["option_content_mask"].bool().unsqueeze(0)

    coarse = scorer(
        state_tokens=state_tokens,
        state_mask=torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
        ),
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_tokens=option_tokens,
        option_token_ids=option_ids,
        option_mask=option_mask,
    )
    qtype = torch.tensor(
        [PRIMITIVE_TO_ID["choice"]],
        dtype=torch.long,
    )
    return hira(
        question,
        state_segments,
        options,
        qtype,
        coarse_override=coarse,
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )


def _rank_metrics(logits: Tensor, gold: int) -> tuple[int, float, bool]:
    values = logits.detach().cpu()
    gold_value = float(values[gold])
    better = int((values > gold_value).sum())
    tied_before = sum(
        1
        for index in range(gold)
        if float(values[index]) == gold_value
    )
    rank = better + tied_before + 1
    return rank, 1.0 / rank, rank <= min(5, values.numel())


def _gold_margin(logits: Tensor, gold: int) -> float:
    values = logits.detach().cpu()
    gold_value = float(values[gold])
    others = torch.cat([values[:gold], values[gold + 1 :]])
    return gold_value - float(others.max())


def _definition_k16_view(base: Mapping[str, object]) -> Mapping[str, object]:
    matches = [
        view
        for view in base["views"]
        if view["view_id"] == "definition"
        and int(view["diagnosis_k"]) == 16
    ]
    if len(matches) != 1:
        raise ValueError("W9 base must contain exactly one definition K16 view")
    return matches[0]


@torch.inference_mode()
def alignment_metrics(
    scorer: nn.Module,
    cache: dict,
) -> dict[str, object]:
    validate_w9_cache(cache)
    scorer.eval()
    records: list[dict[str, object]] = []

    for base in cache["bases"]:
        view = _definition_k16_view(base)
        state = base["state_content_tokens"].float().unsqueeze(0)
        definitions = view["option_tokens"].float().unsqueeze(0)
        definition_mask = view["option_content_mask"].bool().unsqueeze(0)
        state_mask = torch.ones(
            1,
            state.shape[1],
            dtype=torch.bool,
        )
        scores = semantic_alignment_scores(
            scorer,
            state_tokens=state,
            state_mask=state_mask,
            definition_tokens=definitions,
            definition_mask=definition_mask,
        )[0]
        gold = int(view["gold_index"])
        rank, mrr, _ = _rank_metrics(scores, gold)
        logits = (scores / ALIGNMENT_TEMPERATURE).unsqueeze(0)
        target = torch.tensor([gold], dtype=torch.long)
        ce = float(F.cross_entropy(logits, target))
        positive = float(scores[gold])
        negative = torch.cat([scores[:gold], scores[gold + 1 :]])
        hardest = float(negative.max())
        records.append(
            {
                "domain_id": base["domain_id"],
                "correct": rank == 1,
                "mrr": mrr,
                "ce": ce,
                "positive_score": positive,
                "hardest_negative_score": hardest,
                "margin": positive - hardest,
            }
        )

    return _summarize_alignment(records)


def _summarize_alignment(
    records: list[dict[str, object]],
) -> dict[str, object]:
    if not records:
        raise ValueError("W9 alignment metrics require records")

    def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
        n = len(rows)
        return {
            "n": float(n),
            "top1": sum(bool(row["correct"]) for row in rows) / n,
            "mrr": sum(float(row["mrr"]) for row in rows) / n,
            "ce": sum(float(row["ce"]) for row in rows) / n,
            "mean_positive_score": (
                sum(float(row["positive_score"]) for row in rows) / n
            ),
            "mean_hardest_negative_score": (
                sum(float(row["hardest_negative_score"]) for row in rows) / n
            ),
            "mean_margin": sum(float(row["margin"]) for row in rows) / n,
        }

    per_domain = {}
    for domain in sorted({str(row["domain_id"]) for row in records}):
        rows = [row for row in records if row["domain_id"] == domain]
        per_domain[domain] = summarize(rows)

    return {
        "per_domain": per_domain,
        "pooled": summarize(records),
    }


@torch.inference_mode()
def evaluate_w9_checkpoint(
    hira: HIRACore,
    scorer: nn.Module,
    cache: dict,
) -> dict[str, object]:
    validate_w9_cache(cache)
    hira.eval()
    scorer.eval()

    records: list[dict[str, object]] = []
    max_mass_error = 0.0

    for base in cache["bases"]:
        for view in base["views"]:
            out = _cached_forward(hira, scorer, base, view)
            coarse = out.coarse_logits[0].detach().cpu()
            final = out.logits[0].detach().cpu()
            p = out.probabilities[0].detach().cpu().to(torch.float64)
            gold = int(view["gold_index"])
            coarse_rank, coarse_mrr, coarse_top5 = _rank_metrics(coarse, gold)
            final_rank, final_mrr, final_top5 = _rank_metrics(final, gold)
            max_mass_error = max(
                max_mass_error,
                abs(float(p.sum()) - 1.0),
            )
            records.append(
                {
                    "base_id": base["base_id"],
                    "domain_id": base["domain_id"],
                    "view_id": view["view_id"],
                    "k": int(view["diagnosis_k"]),
                    "coarse_correct": coarse_rank == 1,
                    "final_correct": final_rank == 1,
                    "coarse_mrr": coarse_mrr,
                    "final_mrr": final_mrr,
                    "coarse_top5": coarse_top5,
                    "final_top5": final_top5,
                    "coarse_margin": _gold_margin(coarse, gold),
                    "final_margin": _gold_margin(final, gold),
                    "relation_rescue": coarse_rank != 1 and final_rank == 1,
                    "relation_damage": coarse_rank == 1 and final_rank != 1,
                }
            )

    domains = sorted({str(row["domain_id"]) for row in records})
    per_domain = {
        domain: _summarize_domain(
            [row for row in records if row["domain_id"] == domain]
        )
        for domain in domains
    }

    return {
        "per_domain": per_domain,
        "pooled": _summarize_domain(records),
        "alignment": alignment_metrics(scorer, cache),
        "probability_mass_max_error": max_mass_error,
        "state_encodes_per_base": 1.0,
        "cached_base_count": int(cache["metadata"]["base_count"]),
        "cached_view_count": int(cache["metadata"]["view_count"]),
    }


def _summarize_domain(
    records: list[dict[str, object]],
) -> dict[str, object]:
    if not records:
        raise ValueError("W9 summary requires records")

    by_view_k: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    by_base: dict[str, dict[tuple[str, int], dict[str, object]]] = defaultdict(dict)
    for row in records:
        key = (str(row["view_id"]), int(row["k"]))
        by_view_k[key].append(row)
        by_base[str(row["base_id"])][key] = row

    views: dict[str, dict[str, object]] = {}
    for view_id in ("label", "definition"):
        views[view_id] = {}
        for k in (4, 8, 16):
            rows = by_view_k[(view_id, k)]
            if not rows:
                raise ValueError("W9 missing view/K cell")
            n = len(rows)
            views[view_id][str(k)] = {
                "n": n,
                "coarse_top1": sum(bool(x["coarse_correct"]) for x in rows) / n,
                "final_top1": sum(bool(x["final_correct"]) for x in rows) / n,
                "coarse_top5": sum(bool(x["coarse_top5"]) for x in rows) / n,
                "final_top5": sum(bool(x["final_top5"]) for x in rows) / n,
                "coarse_mrr": sum(float(x["coarse_mrr"]) for x in rows) / n,
                "final_mrr": sum(float(x["final_mrr"]) for x in rows) / n,
                "mean_coarse_margin": (
                    sum(float(x["coarse_margin"]) for x in rows) / n
                ),
                "mean_final_margin": (
                    sum(float(x["final_margin"]) for x in rows) / n
                ),
                "relation_rescue_rate": (
                    sum(bool(x["relation_rescue"]) for x in rows) / n
                ),
                "relation_damage_rate": (
                    sum(bool(x["relation_damage"]) for x in rows) / n
                ),
            }

    label_wrong_definition_right = 0
    k4_right_k16_wrong = 0
    base_count = len(by_base)
    for rows in by_base.values():
        label4 = rows[("label", 4)]
        definition4 = rows[("definition", 4)]
        definition16 = rows[("definition", 16)]
        label_wrong_definition_right += int(
            not bool(label4["final_correct"])
            and bool(definition4["final_correct"])
        )
        k4_right_k16_wrong += int(
            bool(definition4["final_correct"])
            and not bool(definition16["final_correct"])
        )

    return {
        "base_count": base_count,
        "views": views,
        "transitions": {
            "label_wrong_definition_right_k4": {
                "count": label_wrong_definition_right,
                "rate": label_wrong_definition_right / base_count,
            },
            "definition_k4_right_k16_wrong": {
                "count": k4_right_k16_wrong,
                "rate": k4_right_k16_wrong / base_count,
            },
        },
    }
