from __future__ import annotations

from collections import Counter
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .regime_transfer_authority import DOMAINS, K_VALUES
from .regime_transfer_cache import validate_w16_cache

RENDERINGS = ("R0", "R1", "R2")
OPERATORS = ("D2S", "S2D", "SYM")

EXTRA_STATE_TOKEN_DILUTION = "EXTRA_STATE_TOKEN_DILUTION"
CONTEXTUAL_STATE_CONTAMINATION = "CONTEXTUAL_STATE_CONTAMINATION"
DECORATION_EFFECT_MIXED = "DECORATION_EFFECT_MIXED"
DECORATION_NOT_PRIMARY = "DECORATION_NOT_PRIMARY"
BASE_SEMANTIC_REGIME_DIFFICULTY = "BASE_SEMANTIC_REGIME_DIFFICULTY"
REGIME_TRANSFER_UNRESOLVED = "REGIME_TRANSFER_UNRESOLVED"

OUTCOME_STABLE = "STABLE_REGIME_TRANSFER_LOCALIZATION"
OUTCOME_MIXED = "MIXED_REGIME_TRANSFER_LOCALIZATION"
OUTCOME_UNRESOLVED = "REGIME_TRANSFER_UNRESOLVED"


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


def directional_scores(
    state_tokens: Tensor,
    option_tokens: Tensor,
    option_mask: Tensor,
    projection: Tensor,
) -> dict[str, Tensor]:
    s = F.normalize(F.linear(state_tokens.float(), projection.float()), dim=-1)
    o = F.normalize(F.linear(option_tokens.float(), projection.float()), dim=-1)
    sim = torch.einsum("sd,ktd->kst", s, o)
    valid = option_mask.bool()

    d2s_best = sim.max(dim=1).values.masked_fill(~valid, 0.0)
    d2s = d2s_best.sum(-1) / valid.sum(-1).clamp_min(1)

    masked = sim.masked_fill(~valid[:, None, :], -1e4)
    s2d = masked.max(dim=-1).values.mean(-1)
    return {"D2S": d2s, "S2D": s2d, "SYM": 0.5 * (d2s + s2d)}


def _record(
    *,
    base: Mapping[str, object],
    k: int,
    rendering: str,
    operator: str,
    logits: Tensor,
    gold: int,
) -> dict[str, object]:
    rank, mrr, top5 = _rank(logits, gold)
    return {
        "base_id": str(base["base_id"]),
        "domain_id": str(base["domain_id"]),
        "k": int(k),
        "rendering": rendering,
        "operator": operator,
        "correct": rank == 1,
        "rank": rank,
        "mrr": mrr,
        "top5": top5,
        "margin": _margin(logits, gold),
    }


def _cell(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("W16 metric cell requires rows")
    n = len(rows)
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / n,
        "top5": sum(bool(row["top5"]) for row in rows) / n,
        "mrr": sum(float(row["mrr"]) for row in rows) / n,
        "mean_margin": sum(float(row["margin"]) for row in rows) / n,
    }


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    per_domain: dict[str, object] = {}
    for domain in DOMAINS:
        per_domain[domain] = {}
        for rendering in RENDERINGS:
            per_domain[domain][rendering] = {}
            for operator in OPERATORS:
                per_domain[domain][rendering][operator] = {}
                for k in K_VALUES:
                    rows = [
                        row
                        for row in records
                        if row["domain_id"] == domain
                        and row["rendering"] == rendering
                        and row["operator"] == operator
                        and int(row["k"]) == k
                    ]
                    per_domain[domain][rendering][operator][str(k)] = _cell(rows)

    pooled: dict[str, object] = {}
    for rendering in RENDERINGS:
        pooled[rendering] = {}
        for operator in OPERATORS:
            pooled[rendering][operator] = {}
            for k in K_VALUES:
                rows = [
                    row
                    for row in records
                    if row["rendering"] == rendering
                    and row["operator"] == operator
                    and int(row["k"]) == k
                ]
                pooled[rendering][operator][str(k)] = _cell(rows)
    return {"per_domain": per_domain, "pooled": pooled}


def _index(records: list[dict[str, object]], rendering: str) -> dict[tuple[str, int], dict]:
    return {
        (str(row["base_id"]), int(row["k"])): row
        for row in records
        if row["rendering"] == rendering and row["operator"] == "SYM"
    }


def _transition(
    a: dict[tuple[str, int], dict],
    b: dict[tuple[str, int], dict],
    keys: list[tuple[str, int]],
) -> dict[str, float | int]:
    if not keys:
        raise ValueError("W16 transition requires keys")
    ctw = sum(bool(a[key]["correct"]) and not bool(b[key]["correct"]) for key in keys)
    wtc = sum(not bool(a[key]["correct"]) and bool(b[key]["correct"]) for key in keys)
    n = len(keys)
    return {
        "n": n,
        "correct_to_wrong_count": ctw,
        "correct_to_wrong_rate": ctw / n,
        "wrong_to_correct_count": wtc,
        "wrong_to_correct_rate": wtc / n,
        "mean_rank_delta": sum(int(b[key]["rank"]) - int(a[key]["rank"]) for key in keys) / n,
        "mean_margin_delta": sum(float(b[key]["margin"]) - float(a[key]["margin"]) for key in keys) / n,
    }


@torch.inference_mode()
def evaluate_regime_transfer(
    scorer,
    cache: dict,
) -> dict[str, object]:
    validate_w16_cache(cache)
    scorer.eval()
    projection = scorer.projection.weight.detach().float()

    records: list[dict[str, object]] = []
    token_counts = {rendering: [] for rendering in RENDERINGS}

    for base in cache["bases"]:
        renderings = base["state_renderings"]
        for rendering in RENDERINGS:
            token_counts[rendering].append(int(renderings[rendering]["active_token_count"]))

        by_k: dict[int, dict[str, dict]] = {k: {} for k in K_VALUES}
        for view in base["views"]:
            by_k[int(view["diagnosis_k"])][str(view["view_id"])] = view

        for k in K_VALUES:
            views = by_k[k]
            if set(views) != {"D0", "D1", "D2"}:
                raise RuntimeError("W16 missing D0/D1/D2")
            option_ids = tuple(views["D0"]["option_ids"])
            gold = int(views["D0"]["gold_index"])
            if any(tuple(views[v]["option_ids"]) != option_ids for v in ("D1", "D2")):
                raise RuntimeError("W16 paraphrase candidate identity changed")
            if any(int(views[v]["gold_index"]) != gold for v in ("D1", "D2")):
                raise RuntimeError("W16 paraphrase gold changed")

            for rendering in RENDERINGS:
                state_tokens = renderings[rendering]["content_tokens"].float()
                per_view: dict[str, list[Tensor]] = {op: [] for op in OPERATORS}
                for view_id in ("D0", "D1", "D2"):
                    view = views[view_id]
                    scores = directional_scores(
                        state_tokens,
                        view["option_tokens"].float(),
                        view["option_content_mask"].bool(),
                        projection,
                    )
                    for op in OPERATORS:
                        per_view[op].append(scores[op])
                for op in OPERATORS:
                    ensemble = torch.stack(per_view[op], dim=0).mean(dim=0)
                    records.append(
                        _record(
                            base=base,
                            k=k,
                            rendering=rendering,
                            operator=op,
                            logits=ensemble,
                            gold=gold,
                        )
                    )

    summary = _summary(records)
    r0 = _index(records, "R0")
    r1 = _index(records, "R1")
    r2 = _index(records, "R2")
    per_domain_transition: dict[str, object] = {}

    for domain in DOMAINS:
        domain_ids = {
            str(base["base_id"])
            for base in cache["bases"]
            if base["domain_id"] == domain
        }
        keys = sorted(key for key in r0 if key[0] in domain_ids)
        per_domain_transition[domain] = {
            "R0_to_R1": _transition(r0, r1, keys),
            "R1_to_R2": _transition(r1, r2, keys),
            "R0_to_R2": _transition(r0, r2, keys),
        }

    directional_attribution: dict[str, object] = {}
    for domain in DOMAINS:
        directional_attribution[domain] = {}
        for k in K_VALUES:
            d = summary["per_domain"][domain]
            r0_d2s = d["R0"]["D2S"][str(k)]
            r1_d2s = d["R1"]["D2S"][str(k)]
            r0_s2d = d["R0"]["S2D"][str(k)]
            r1_s2d = d["R1"]["S2D"][str(k)]
            r0_sym = d["R0"]["SYM"][str(k)]
            r1_sym = d["R1"]["SYM"][str(k)]
            sym_margin_loss = float(r0_sym["mean_margin"]) - float(r1_sym["mean_margin"])
            s2d_margin_loss = float(r0_s2d["mean_margin"]) - float(r1_s2d["mean_margin"])
            directional_attribution[domain][str(k)] = {
                "d2s_top1_delta_r1_minus_r0": float(r1_d2s["top1"]) - float(r0_d2s["top1"]),
                "s2d_top1_delta_r1_minus_r0": float(r1_s2d["top1"]) - float(r0_s2d["top1"]),
                "sym_top1_delta_r1_minus_r0": float(r1_sym["top1"]) - float(r0_sym["top1"]),
                "d2s_margin_delta_r1_minus_r0": float(r1_d2s["mean_margin"]) - float(r0_d2s["mean_margin"]),
                "s2d_margin_delta_r1_minus_r0": float(r1_s2d["mean_margin"]) - float(r0_s2d["mean_margin"]),
                "sym_margin_delta_r1_minus_r0": float(r1_sym["mean_margin"]) - float(r0_sym["mean_margin"]),
                "s2d_margin_loss_to_sym_margin_loss_ratio": (
                    s2d_margin_loss / sym_margin_loss
                    if sym_margin_loss > 1e-9 else 0.0
                ),
            }

    return {
        "metrics": summary,
        "transitions": per_domain_transition,
        "directional_attribution": directional_attribution,
        "token_accounting": {
            rendering: {
                "mean_active_tokens": sum(values) / len(values),
                "min_active_tokens": min(values),
                "max_active_tokens": max(values),
            }
            for rendering, values in token_counts.items()
        },
        "state_encoder_batches_per_base": float(
            cache["metadata"]["state_encoder_batches_per_base"]
        ),
        "encoded_state_texts_per_base": float(
            cache["metadata"]["encoded_state_texts_per_base"]
        ),
        "r2_additional_encoder_calls": int(
            cache["metadata"]["r2_additional_encoder_calls"]
        ),
        "prefix_token_identity_rate": float(
            cache["metadata"]["prefix_token_identity_rate"]
        ),
        "cached_base_count": int(cache["metadata"]["base_count"]),
        "cached_view_count": int(cache["metadata"]["view_count"]),
    }


def reference_summary(cache: dict, score_lookup: Mapping[str, list[float]]) -> dict[str, object]:
    validate_w16_cache(cache)
    records: list[dict[str, object]] = []
    for base in cache["bases"]:
        for view in base["views"]:
            if view["view_id"] != "D0":
                continue
            for rendering in ("R0", "R1"):
                key = f"{rendering}|{view['case_id']}"
                scores = score_lookup.get(key)
                if scores is None:
                    raise ValueError(f"missing W16 reference score: {key}")
                records.append(
                    _record(
                        base=base,
                        k=int(view["diagnosis_k"]),
                        rendering=rendering,
                        operator="SYM",
                        logits=torch.tensor(scores, dtype=torch.float32),
                        gold=int(view["gold_index"]),
                    )
                )
    # Shape compatible with main summary, but only R0/R1 SYM exist.
    per_domain: dict[str, object] = {}
    pooled: dict[str, object] = {}
    for domain in DOMAINS:
        per_domain[domain] = {}
        for rendering in ("R0", "R1"):
            per_domain[domain][rendering] = {}
            for k in K_VALUES:
                rows = [
                    row for row in records
                    if row["domain_id"] == domain
                    and row["rendering"] == rendering
                    and int(row["k"]) == k
                ]
                per_domain[domain][rendering][str(k)] = _cell(rows)
    for rendering in ("R0", "R1"):
        pooled[rendering] = {}
        for k in K_VALUES:
            rows = [
                row for row in records
                if row["rendering"] == rendering and int(row["k"]) == k
            ]
            pooled[rendering][str(k)] = _cell(rows)
    return {"per_domain": per_domain, "pooled": pooled}


def classify_domain(metrics: dict, reference: dict) -> dict[str, object]:
    r0 = metrics["R0"]["SYM"]
    r1 = metrics["R1"]["SYM"]
    r2 = metrics["R2"]["SYM"]
    d2s0 = metrics["R0"]["D2S"]
    d2s1 = metrics["R1"]["D2S"]
    s2d0 = metrics["R0"]["S2D"]
    s2d1 = metrics["R1"]["S2D"]

    adequate = (
        float(reference["R0"]["4"]["top1"]) >= .75
        and float(r0["4"]["top1"]) >= .70
        and float(r0["16"]["top1"]) >= .50
    )
    if not adequate:
        return {
            "classification": BASE_SEMANTIC_REGIME_DIFFICULTY,
            "adequate": False,
        }

    def drop(a, b, k): return float(a[str(k)]["top1"]) - float(b[str(k)]["top1"])
    dilution = (
        drop(r0, r1, 4) >= .12
        and drop(r0, r1, 16) >= .12
        and drop(r2, r1, 4) >= .10
        and drop(r2, r1, 16) >= .10
        and drop(r0, r2, 4) <= .05
        and drop(r0, r2, 16) <= .05
        and drop(d2s0, d2s1, 4) <= .05
        and drop(d2s0, d2s1, 16) <= .05
        and drop(s2d0, s2d1, 4) >= .10
        and drop(s2d0, s2d1, 16) >= .10
    )
    contextual = (
        drop(r0, r1, 4) >= .12
        and drop(r0, r1, 16) >= .12
        and (drop(r2, r1, 4) < .06 or drop(r2, r1, 16) < .06)
        and drop(r0, r2, 4) >= .08
        and drop(r0, r2, 16) >= .08
    )
    mixed = (
        drop(r0, r1, 4) >= .08
        and drop(r0, r1, 16) >= .08
        and not dilution
        and not contextual
    )
    not_primary = (
        abs(float(r1["4"]["top1"]) - float(r0["4"]["top1"])) < .08
        and abs(float(r1["16"]["top1"]) - float(r0["16"]["top1"])) < .08
    )

    if dilution:
        label = EXTRA_STATE_TOKEN_DILUTION
    elif contextual:
        label = CONTEXTUAL_STATE_CONTAMINATION
    elif mixed:
        label = DECORATION_EFFECT_MIXED
    elif not_primary:
        label = DECORATION_NOT_PRIMARY
    else:
        label = REGIME_TRANSFER_UNRESOLVED

    return {
        "classification": label,
        "adequate": True,
        "r0_k4_top1": float(r0["4"]["top1"]),
        "r0_k16_top1": float(r0["16"]["top1"]),
        "r1_k4_top1": float(r1["4"]["top1"]),
        "r1_k16_top1": float(r1["16"]["top1"]),
        "r2_k4_top1": float(r2["4"]["top1"]),
        "r2_k16_top1": float(r2["16"]["top1"]),
        "d2s_k4_drop": drop(d2s0, d2s1, 4),
        "d2s_k16_drop": drop(d2s0, d2s1, 16),
        "s2d_k4_drop": drop(s2d0, s2d1, 4),
        "s2d_k16_drop": drop(s2d0, s2d1, 16),
    }


def cross_domain_outcome(per_domain: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    labels = [str(per_domain[d]["classification"]) for d in DOMAINS]
    counts = Counter(labels)
    excluded = {REGIME_TRANSFER_UNRESOLVED, BASE_SEMANTIC_REGIME_DIFFICULTY}
    stable = [
        label for label, count in counts.items()
        if label not in excluded and count >= 3
    ]
    if stable:
        stable.sort(key=lambda label: (-counts[label], label))
        return {
            "outcome": OUTCOME_STABLE,
            "stable_classification": stable[0],
            "counts": dict(counts),
        }

    nonexcluded = [(label, count) for label, count in counts.items() if label not in excluded]
    if sum(count >= 2 for _, count in nonexcluded) >= 2:
        return {
            "outcome": OUTCOME_MIXED,
            "stable_classification": None,
            "counts": dict(counts),
        }
    return {
        "outcome": OUTCOME_UNRESOLVED,
        "stable_classification": None,
        "counts": dict(counts),
    }


__all__ = [
    "BASE_SEMANTIC_REGIME_DIFFICULTY",
    "CONTEXTUAL_STATE_CONTAMINATION",
    "DECORATION_EFFECT_MIXED",
    "DECORATION_NOT_PRIMARY",
    "EXTRA_STATE_TOKEN_DILUTION",
    "OUTCOME_MIXED",
    "OUTCOME_STABLE",
    "OUTCOME_UNRESOLVED",
    "REGIME_TRANSFER_UNRESOLVED",
    "classify_domain",
    "cross_domain_outcome",
    "directional_scores",
    "evaluate_regime_transfer",
    "reference_summary",
]
