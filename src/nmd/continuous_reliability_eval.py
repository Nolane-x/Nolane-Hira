from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from .competitive import CompetitiveCoarseScorer
from .continuous_reliability import (
    DOMAINS,
    K_VALUES,
    continuous_reliability,
    fixed_tertiles,
    normalized_anchor_margin,
    spearman,
    top_margin_ids,
)
from .continuous_reliability_cache import validate_w14_cache
from .hira import HIRACore
from .interface_decomposition_eval import _q0_symmetric, _q6_actual, _q7_final

STAGES = ("A0", "A1", "A2", "E", "F")


def _rank(logits: Tensor, gold: int) -> tuple[int, float, bool]:
    x = logits.detach().cpu().float()
    gold_value = float(x[gold])
    better = int((x > gold_value).sum())
    tied_before = sum(
        1 for index in range(gold)
        if float(x[index]) == gold_value
    )
    rank = better + tied_before + 1
    return rank, 1.0 / rank, rank <= min(5, x.numel())


def _margin(logits: Tensor, gold: int) -> float:
    x = logits.detach().cpu().float()
    others = torch.cat([x[:gold], x[gold + 1 :]])
    return float(x[gold]) - float(others.max())


def _order(logits: Tensor) -> tuple[int, ...]:
    values = [float(value) for value in logits.detach().cpu().float()]
    return tuple(
        sorted(range(len(values)), key=lambda index: (-values[index], index))
    )


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
        return {
            "n": 0,
            "top1": 0.0,
            "top5": 0.0,
            "mrr": 0.0,
            "mean_margin": 0.0,
        }
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
        for k in K_VALUES:
            rows = [
                row for row in records
                if row["domain_id"] == domain and int(row["k"]) == k
            ]
            per_domain[domain][str(k)] = _cell(rows)

    pooled: dict[str, object] = {}
    for k in K_VALUES:
        pooled[str(k)] = _cell(
            [row for row in records if int(row["k"]) == k]
        )
    return {"per_domain": per_domain, "pooled": pooled}


def _index(
    records: list[dict[str, object]],
) -> dict[tuple[str, int], dict[str, object]]:
    out: dict[tuple[str, int], dict[str, object]] = {}
    for row in records:
        key = (str(row["base_id"]), int(row["k"]))
        if key in out:
            raise ValueError("duplicate W14 stage identity")
        out[key] = row
    return out


def _transition(
    source: list[dict[str, object]],
    target: list[dict[str, object]],
    keys: list[tuple[str, int]],
) -> dict[str, float | int]:
    a = _index(source)
    b = _index(target)
    if any(key not in a or key not in b for key in keys):
        raise ValueError("W14 transition lost paired identity")
    n = len(keys)
    if n == 0:
        return {
            "n": 0,
            "correct_to_wrong_count": 0,
            "correct_to_wrong_rate": 0.0,
            "wrong_to_correct_count": 0,
            "wrong_to_correct_rate": 0.0,
            "mean_rank_delta": 0.0,
            "mean_margin_delta": 0.0,
        }
    ctw = sum(
        bool(a[key]["correct"]) and not bool(b[key]["correct"])
        for key in keys
    )
    wtc = sum(
        not bool(a[key]["correct"]) and bool(b[key]["correct"])
        for key in keys
    )
    return {
        "n": n,
        "correct_to_wrong_count": ctw,
        "correct_to_wrong_rate": ctw / n,
        "wrong_to_correct_count": wtc,
        "wrong_to_correct_rate": wtc / n,
        "mean_rank_delta": sum(
            int(b[key]["rank"]) - int(a[key]["rank"]) for key in keys
        ) / n,
        "mean_margin_delta": sum(
            float(b[key]["margin"]) - float(a[key]["margin"]) for key in keys
        ) / n,
    }


def _chosen_record(
    ensemble: dict[str, object],
    final: dict[str, object],
    use_ensemble: bool,
    stage: str,
) -> dict[str, object]:
    return {**(ensemble if use_ensemble else final), "stage": stage}


def _tertile_cell(
    *,
    keys: list[tuple[str, int]],
    ensemble_index: dict[tuple[str, int], dict[str, object]],
    final_index: dict[tuple[str, int], dict[str, object]],
    meta: dict[tuple[str, int], dict[str, float]],
) -> dict[str, float | int]:
    if not keys:
        return {
            "n": 0,
            "ensemble_top1": 0.0,
            "final_top1": 0.0,
            "ensemble_mrr": 0.0,
            "final_mrr": 0.0,
            "mean_R": 0.0,
            "mean_V": 0.0,
            "mean_S": 0.0,
            "mean_O": 0.0,
        }
    n = len(keys)
    return {
        "n": n,
        "ensemble_top1": sum(
            bool(ensemble_index[key]["correct"]) for key in keys
        ) / n,
        "final_top1": sum(
            bool(final_index[key]["correct"]) for key in keys
        ) / n,
        "ensemble_mrr": sum(
            float(ensemble_index[key]["mrr"]) for key in keys
        ) / n,
        "final_mrr": sum(
            float(final_index[key]["mrr"]) for key in keys
        ) / n,
        "mean_R": sum(float(meta[key]["R"]) for key in keys) / n,
        "mean_V": sum(float(meta[key]["V"]) for key in keys) / n,
        "mean_S": sum(float(meta[key]["S"]) for key in keys) / n,
        "mean_O": sum(float(meta[key]["O"]) for key in keys) / n,
    }


@torch.inference_mode()
def evaluate_continuous_reliability(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w14_cache(cache)
    hira.eval()
    scorer.eval()

    projection = scorer.projection.weight.detach().float()
    records: dict[str, list[dict[str, object]]] = {
        stage: [] for stage in STAGES
    }
    meta: dict[tuple[str, int], dict[str, float]] = {}
    max_mass_error = 0.0

    for base in cache["bases"]:
        state_tokens = base["state_content_tokens"].float()
        by_k: dict[int, dict[str, dict[str, object]]] = {
            k: {} for k in K_VALUES
        }
        for view in base["views"]:
            by_k[int(view["diagnosis_k"])][str(view["view_id"])] = view

        for k in K_VALUES:
            views = by_k[k]
            if set(views) != {"D0", "D1", "D2"}:
                raise RuntimeError("W14 missing paraphrase view")
            d0, d1, d2 = views["D0"], views["D1"], views["D2"]
            if not (
                tuple(d0["option_ids"])
                == tuple(d1["option_ids"])
                == tuple(d2["option_ids"])
            ):
                raise RuntimeError("W14 paraphrase option identity changed")

            anchor_logits: list[Tensor] = []
            for view_id, view in (("D0", d0), ("D1", d1), ("D2", d2)):
                logits = _q0_symmetric(
                    state_tokens,
                    view["option_tokens"].float(),
                    view["option_content_mask"].bool(),
                    projection,
                )
                anchor_logits.append(logits)
                records[f"A{int(view_id[-1])}"].append(
                    _record(
                        base=base,
                        view=view,
                        stage=f"A{int(view_id[-1])}",
                        logits=logits,
                    )
                )

            ensemble = torch.stack(anchor_logits, dim=0).mean(dim=0)
            records["E"].append(
                _record(base=base, view=d0, stage="E", logits=ensemble)
            )

            coarse = _q6_actual(scorer, base, d0)
            out = _q7_final(hira, base, d0, coarse)
            final = out.logits[0].detach().cpu().float()
            records["F"].append(
                _record(base=base, view=d0, stage="F", logits=final)
            )
            probability = out.probabilities[0].detach().cpu().to(torch.float64)
            max_mass_error = max(
                max_mass_error,
                abs(float(probability.sum()) - 1.0),
            )

            orders = [_order(logits) for logits in anchor_logits]
            option_ids = tuple(str(value) for value in d0["option_ids"])
            top1_ids = [option_ids[order[0]] for order in orders]
            reliability = continuous_reliability(top1_ids, orders)
            key = (str(base["base_id"]), k)
            meta[key] = {
                **reliability,
                "a0_normalized_margin": normalized_anchor_margin(
                    [float(value) for value in anchor_logits[0]]
                ),
            }

    ensemble_index = _index(records["E"])
    final_index = _index(records["F"])
    guard_consistency: list[dict[str, object]] = []
    guard_margin: list[dict[str, object]] = []
    guard_nonlow: list[dict[str, object]] = []
    per_domain: dict[str, object] = {}

    for domain in DOMAINS:
        domain_base_ids = {
            str(base["base_id"])
            for base in cache["bases"]
            if base["domain_id"] == domain
        }
        domain_keys = [
            key for key in ensemble_index if key[0] in domain_base_ids
        ]

        tertile_ids: dict[int, dict[str, set[str]]] = {}
        margin_ids: dict[int, set[str]] = {}
        for k in K_VALUES:
            keys = sorted(
                [key for key in domain_keys if key[1] == k],
                key=lambda item: item[0],
            )
            tertile_ids[k] = fixed_tertiles(
                [(key[0], float(meta[key]["R"])) for key in keys]
            )
            margin_ids[k] = top_margin_ids(
                [
                    (key[0], float(meta[key]["a0_normalized_margin"]))
                    for key in keys
                ]
            )

            for key in keys:
                e = ensemble_index[key]
                f = final_index[key]
                high = key[0] in tertile_ids[k]["HIGH"]
                nonlow = key[0] not in tertile_ids[k]["LOW"]
                margin_high = key[0] in margin_ids[k]
                guard_consistency.append(
                    _chosen_record(e, f, high, "G_consistency_high")
                )
                guard_margin.append(
                    _chosen_record(e, f, margin_high, "G_margin_high")
                )
                guard_nonlow.append(
                    _chosen_record(e, f, nonlow, "G_consistency_nonlow")
                )

        tertiles: dict[str, object] = {
            "HIGH": {},
            "MIDDLE": {},
            "LOW": {},
        }
        for bucket in ("HIGH", "MIDDLE", "LOW"):
            pooled_primary: list[tuple[str, int]] = []
            for k in K_VALUES:
                keys = [
                    key for key in domain_keys
                    if key[1] == k and key[0] in tertile_ids[k][bucket]
                ]
                tertiles[bucket][str(k)] = _tertile_cell(
                    keys=keys,
                    ensemble_index=ensemble_index,
                    final_index=final_index,
                    meta=meta,
                )
                if k in (4, 16):
                    pooled_primary.extend(keys)
            tertiles[bucket]["pooled_primary_transition"] = _transition(
                records["E"], records["F"], pooled_primary
            )

        correlations: dict[str, object] = {}
        for k in K_VALUES:
            keys = [key for key in domain_keys if key[1] == k]
            e_correct = [
                1.0 if bool(ensemble_index[key]["correct"]) else 0.0
                for key in keys
            ]
            f_minus_e = [
                (1.0 if bool(final_index[key]["correct"]) else 0.0)
                - (1.0 if bool(ensemble_index[key]["correct"]) else 0.0)
                for key in keys
            ]
            correlations[str(k)] = {
                "spearman_R_ensemble_correct": spearman(
                    [float(meta[key]["R"]) for key in keys], e_correct
                ),
                "spearman_V_ensemble_correct": spearman(
                    [float(meta[key]["V"]) for key in keys], e_correct
                ),
                "spearman_S_ensemble_correct": spearman(
                    [float(meta[key]["S"]) for key in keys], e_correct
                ),
                "spearman_O_ensemble_correct": spearman(
                    [float(meta[key]["O"]) for key in keys], e_correct
                ),
                "spearman_R_final_minus_ensemble": spearman(
                    [float(meta[key]["R"]) for key in keys], f_minus_e
                ),
            }

        per_domain[domain] = {
            "tertiles": tertiles,
            "correlations": correlations,
            "ensemble_to_final": _transition(
                records["E"], records["F"], sorted(domain_keys)
            ),
        }

    guard_summaries = {
        "G_consistency_high": _summary(guard_consistency),
        "G_margin_high": _summary(guard_margin),
        "G_consistency_nonlow": _summary(guard_nonlow),
    }
    for domain in DOMAINS:
        per_domain[domain]["guards"] = {
            name: summary["per_domain"][domain]
            for name, summary in guard_summaries.items()
        }

    return {
        "stages": {
            stage: _summary(rows)
            for stage, rows in records.items()
        },
        "per_domain": per_domain,
        "guards": guard_summaries,
        "reliability_meta": {
            f"{base_id}|{k}": value
            for (base_id, k), value in sorted(meta.items())
        },
        "probability_mass_max_error": max_mass_error,
        "state_encodes_per_base": float(
            cache["metadata"]["state_encode_calls_per_base"]
        ),
        "cached_base_count": int(cache["metadata"]["base_count"]),
        "cached_view_count": int(cache["metadata"]["view_count"]),
    }


def reference_summary(
    cache: dict,
    score_lookup: Mapping[str, list[float]],
) -> dict[str, object]:
    validate_w14_cache(cache)
    rows: list[dict[str, object]] = []
    for base in cache["bases"]:
        for view in base["views"]:
            if view["view_id"] != "D0":
                continue
            scores = score_lookup.get(str(view["case_id"]))
            if scores is None:
                raise ValueError(
                    f"missing W14 reference score for {view['case_id']}"
                )
            logits = torch.tensor(scores, dtype=torch.float32)
            rows.append(
                _record(
                    base=base,
                    view=view,
                    stage="R0",
                    logits=logits,
                )
            )
    return _summary(rows)


__all__ = [
    "STAGES",
    "evaluate_continuous_reliability",
    "reference_summary",
]
