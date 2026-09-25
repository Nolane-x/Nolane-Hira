from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch
from torch import Tensor

from .competitive import CompetitiveCoarseScorer
from .hira import HIRACore
from .runtime import PRIMITIVE_TO_ID
from .semantic_transfer import transfer_classification
from .semantic_transfer_cache import validate_w8_cache


@torch.inference_mode()
def _cached_forward(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
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
            1, state_tokens.shape[1], dtype=torch.bool
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
    out = hira(
        question,
        state_segments,
        options,
        qtype,
        coarse_override=coarse,
        forced_budget=options.shape[1],
        adaptive_budget=False,
    )
    return out


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


@torch.inference_mode()
def evaluate_w8_checkpoint(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w8_cache(cache)
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
                    "gold_index": gold,
                    "coarse_correct": coarse_rank == 1,
                    "final_correct": final_rank == 1,
                    "coarse_rank": coarse_rank,
                    "final_rank": final_rank,
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

    per_domain: dict[str, dict[str, object]] = {}
    for domain in ("AU", "AV", "AW", "AX"):
        rows = [row for row in records if row["domain_id"] == domain]
        per_domain[domain] = _summarize_domain(rows)

    pooled = _summarize_domain(records)

    return {
        "per_domain": per_domain,
        "pooled": pooled,
        "probability_mass_max_error": max_mass_error,
        "state_encodes_per_base": 1.0,
        "cached_base_count": int(cache["metadata"]["base_count"]),
        "cached_view_count": int(cache["metadata"]["view_count"]),
    }


def _summarize_domain(
    records: list[dict[str, object]],
) -> dict[str, object]:
    if not records:
        raise ValueError("W8 summary requires records")

    by_view_k: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    by_base: dict[str, dict[tuple[str, int], dict[str, object]]] = defaultdict(dict)
    for row in records:
        key = (str(row["view_id"]), int(row["k"]))
        by_view_k[key].append(row)
        by_base[str(row["base_id"])][key] = row

    views: dict[str, dict[str, object]] = {}
    for view_id in ("V0", "V1", "V2", "V3"):
        views[view_id] = {}
        for k in (4, 8, 16):
            rows = by_view_k[(view_id, k)]
            if not rows:
                raise ValueError("W8 missing view/K cell")
            n = len(rows)
            final_errors = [row for row in rows if not bool(row["final_correct"])]
            coarse_wrong_final_error = (
                sum(not bool(row["coarse_correct"]) for row in final_errors)
                / len(final_errors)
                if final_errors
                else 0.0
            )
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
                "coarse_wrong_given_final_error_rate": (
                    coarse_wrong_final_error
                ),
            }

    transitions = {}
    base_count = len(by_base)
    for source, target, name in (
        ("V0", "V1", "v0_wrong_v1_right"),
        ("V1", "V2", "v1_wrong_v2_right"),
        ("V1", "V3", "v1_wrong_v3_right"),
        ("V2", "V0", "v2_right_v0_wrong"),
    ):
        count = 0
        for rows in by_base.values():
            left = rows[(source, 4)]
            right = rows[(target, 4)]
            if name == "v2_right_v0_wrong":
                active = bool(left["final_correct"]) and not bool(right["final_correct"])
            else:
                active = not bool(left["final_correct"]) and bool(right["final_correct"])
            count += int(active)
        transitions[name] = {
            "count": count,
            "rate": count / base_count,
            "stage": "K4_FINAL",
        }

    for view_id in ("V0", "V1", "V2", "V3"):
        right_to_wrong = 0
        wrong_to_wrong = 0
        for rows in by_base.values():
            k4 = rows[(view_id, 4)]
            k16 = rows[(view_id, 16)]
            right_to_wrong += int(
                bool(k4["final_correct"]) and not bool(k16["final_correct"])
            )
            wrong_to_wrong += int(
                not bool(k4["final_correct"]) and not bool(k16["final_correct"])
            )
        transitions[f"{view_id.lower()}_k4_right_k16_wrong"] = {
            "count": right_to_wrong,
            "rate": right_to_wrong / base_count,
        }
        transitions[f"{view_id.lower()}_k4_wrong_k16_wrong"] = {
            "count": wrong_to_wrong,
            "rate": wrong_to_wrong / base_count,
        }

    v1_k4 = float(views["V1"]["4"]["final_top1"])
    v3_k4 = float(views["V3"]["4"]["final_top1"])
    natural_view = "V1" if v1_k4 >= v3_k4 else "V3"
    classifier_metrics = {
        "v0_k4_top1": float(views["V0"]["4"]["final_top1"]),
        "v1_k4_top1": v1_k4,
        "v2_k4_top1": float(views["V2"]["4"]["final_top1"]),
        "v3_k4_top1": v3_k4,
        "v1_k16_top1": float(views["V1"]["16"]["final_top1"]),
        "v3_k16_top1": float(views["V3"]["16"]["final_top1"]),
        "k16_coarse_wrong_given_final_error_rate": float(
            views[natural_view]["16"][
                "coarse_wrong_given_final_error_rate"
            ]
        ),
    }
    classification = transfer_classification(classifier_metrics)

    return {
        "base_count": base_count,
        "views": views,
        "transitions": transitions,
        "classifier_metrics": classifier_metrics,
        "classification": classification,
    }
