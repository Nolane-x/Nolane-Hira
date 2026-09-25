from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .competitive import CompetitiveCoarseScorer
from .hira import HIRACore
from .representation_ceiling_cache import validate_w10_cache
from .runtime import PRIMITIVE_TO_ID


def _rank(logits: Tensor, gold: int) -> tuple[int, float, bool]:
    x = logits.detach().cpu()
    gold_value = float(x[gold])
    better = int((x > gold_value).sum())
    tied_before = sum(1 for i in range(gold) if float(x[i]) == gold_value)
    rank = better + tied_before + 1
    return rank, 1.0 / rank, rank <= min(5, x.numel())


def _margin(logits: Tensor, gold: int) -> float:
    x = logits.detach().cpu()
    others = torch.cat([x[:gold], x[gold + 1 :]])
    return float(x[gold]) - float(others.max())


def _raw_mean_scores(state_tokens: Tensor, option_tokens: Tensor, option_mask: Tensor) -> Tensor:
    state = F.normalize(state_tokens.float().mean(0), dim=-1)
    weights = option_mask.to(option_tokens.dtype)[..., None]
    pooled = (option_tokens.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
    pooled = F.normalize(pooled, dim=-1)
    return torch.einsum("d,kd->k", state, pooled)


def _symmetric_maxsim(
    state_tokens: Tensor,
    option_tokens: Tensor,
    option_mask: Tensor,
    projection: Tensor | None = None,
) -> Tensor:
    s = state_tokens.float()
    o = option_tokens.float()
    if projection is not None:
        s = F.linear(s, projection.float())
        o = F.linear(o, projection.float())
    s = F.normalize(s, dim=-1)
    o = F.normalize(o, dim=-1)

    sim = torch.einsum("sd,ktd->kst", s, o)
    valid = option_mask.bool()

    # option/schema -> state coverage: max over state tokens for each option token
    schema_best = sim.max(dim=1).values
    schema_best = schema_best.masked_fill(~valid, 0.0)
    schema_mean = schema_best.sum(-1) / valid.sum(-1).clamp_min(1)

    # state -> option coverage: max over valid option tokens for each state token
    masked = sim.masked_fill(~valid[:, None, :], -1e4)
    state_best = masked.max(dim=-1).values
    state_mean = state_best.mean(-1)
    return 0.5 * (schema_mean + state_mean)


@torch.inference_mode()
def _production_forward(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    base: Mapping[str, object],
    view: Mapping[str, object],
):
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
        state_mask=torch.ones(1, state_tokens.shape[1], dtype=torch.bool),
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_tokens=option_tokens,
        option_token_ids=option_ids,
        option_mask=option_mask,
    )
    qtype = torch.tensor([PRIMITIVE_TO_ID["choice"]], dtype=torch.long)
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


def _record(
    *,
    base: Mapping[str, object],
    view: Mapping[str, object],
    operator: str,
    logits: Tensor,
    stage: str,
) -> dict[str, object]:
    gold = int(view["gold_index"])
    rank, mrr, top5 = _rank(logits, gold)
    return {
        "base_id": str(base["base_id"]),
        "domain_id": str(base["domain_id"]),
        "view_id": str(view["view_id"]),
        "k": int(view["diagnosis_k"]),
        "operator": operator,
        "stage": stage,
        "correct": rank == 1,
        "rank": rank,
        "mrr": mrr,
        "top5": top5,
        "margin": _margin(logits, gold),
    }


def summarize_operator(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("W10 operator summary requires records")

    def cell(rows: list[dict[str, object]]) -> dict[str, float | int]:
        n = len(rows)
        return {
            "n": n,
            "top1": sum(bool(x["correct"]) for x in rows) / n,
            "top5": sum(bool(x["top5"]) for x in rows) / n,
            "mrr": sum(float(x["mrr"]) for x in rows) / n,
            "mean_margin": sum(float(x["margin"]) for x in rows) / n,
        }

    by_domain: dict[str, dict[str, object]] = {}
    for domain in ("BF", "BG", "BH", "BI"):
        domain_rows = [x for x in records if x["domain_id"] == domain]
        views: dict[str, dict[str, object]] = {}
        for view_id in ("definition", "label"):
            views[view_id] = {}
            for k in (4, 8, 16):
                rows = [x for x in domain_rows if x["view_id"] == view_id and int(x["k"]) == k]
                if not rows:
                    raise ValueError("W10 missing domain/view/K cell")
                views[view_id][str(k)] = cell(rows)
        by_domain[domain] = {"views": views}

    pooled: dict[str, dict[str, object]] = {}
    for view_id in ("definition", "label"):
        pooled[view_id] = {}
        for k in (4, 8, 16):
            rows = [x for x in records if x["view_id"] == view_id and int(x["k"]) == k]
            pooled[view_id][str(k)] = cell(rows)
    return {"per_domain": by_domain, "pooled": {"views": pooled}}


@torch.inference_mode()
def evaluate_hira_representation_operators(
    hira: HIRACore,
    w6e_scorer: CompetitiveCoarseScorer,
    w9_scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w10_cache(cache)
    hira.eval()
    w6e_scorer.eval()
    w9_scorer.eval()

    records: dict[str, list[dict[str, object]]] = {
        name: [] for name in ("A0", "A1", "P0", "P1", "S0", "S1")
    }
    max_mass_error = 0.0

    p0_weight = w6e_scorer.projection.weight.detach()
    p1_weight = w9_scorer.projection.weight.detach()

    for base in cache["bases"]:
        state_tokens = base["state_content_tokens"].float()
        for view in base["views"]:
            option_tokens = view["option_tokens"].float()
            option_mask = view["option_content_mask"].bool()

            a0 = _raw_mean_scores(state_tokens, option_tokens, option_mask)
            a1 = _symmetric_maxsim(state_tokens, option_tokens, option_mask)
            p0 = _symmetric_maxsim(state_tokens, option_tokens, option_mask, p0_weight)
            p1 = _symmetric_maxsim(state_tokens, option_tokens, option_mask, p1_weight)
            for name, logits in (("A0", a0), ("A1", a1), ("P0", p0), ("P1", p1)):
                records[name].append(
                    _record(base=base, view=view, operator=name, logits=logits, stage="semantic")
                )

            for name, scorer in (("S0", w6e_scorer), ("S1", w9_scorer)):
                out = _production_forward(hira, scorer, base, view)
                final = out.logits[0].detach().cpu()
                records[name].append(
                    _record(base=base, view=view, operator=name, logits=final, stage="final")
                )
                p = out.probabilities[0].detach().cpu().to(torch.float64)
                max_mass_error = max(max_mass_error, abs(float(p.sum()) - 1.0))

    return {
        "operators": {
            name: summarize_operator(rows)
            for name, rows in records.items()
        },
        "probability_mass_max_error": max_mass_error,
        "state_encodes_per_base": 1.0,
        "cached_base_count": int(cache["metadata"]["base_count"]),
        "cached_view_count": int(cache["metadata"]["view_count"]),
    }


def make_reference_records(
    cache: dict,
    score_lookup: dict[str, list[float]],
) -> list[dict[str, object]]:
    validate_w10_cache(cache)
    rows: list[dict[str, object]] = []
    for base in cache["bases"]:
        for view in base["views"]:
            case_id = str(view["case_id"])
            scores = score_lookup.get(case_id)
            if scores is None:
                raise ValueError(f"missing W10 reference scores for {case_id}")
            logits = torch.tensor(scores, dtype=torch.float32)
            if logits.numel() != int(view["diagnosis_k"]):
                raise ValueError("W10 reference score width mismatch")
            rows.append(
                _record(base=base, view=view, operator="R0", logits=logits, stage="reference")
            )
    return rows
