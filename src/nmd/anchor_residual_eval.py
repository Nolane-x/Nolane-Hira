from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch

from .anchor_residual import (
    ConfidenceRow,
    deterministic_quartiles,
    normalized_anchor_margin,
    spearman,
)
from .anchor_residual_authority import DOMAINS, K_VALUES
from .anchor_residual_cache import validate_w12_cache
from .competitive import CompetitiveCoarseScorer
from .hira import HIRACore
from .interface_decomposition_eval import _q0_symmetric, _q6_actual, _q7_final, _record


def _cell(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("W12 summary cell cannot be empty")
    n = len(rows)
    return {
        "n": n,
        "top1": sum(bool(x["correct"]) for x in rows) / n,
        "top5": sum(bool(x["top5"]) for x in rows) / n,
        "mrr": sum(float(x["mrr"]) for x in rows) / n,
        "mean_margin": sum(float(x["margin"]) for x in rows) / n,
    }


def _index(rows: list[dict[str, object]]) -> dict[tuple[str, str, int], dict[str, object]]:
    out = {}
    for row in rows:
        key = (str(row["base_id"]), str(row["view_id"]), int(row["k"]))
        if key in out:
            raise ValueError("duplicate W12 identity")
        out[key] = row
    return out


def _transition(
    anchor: list[dict[str, object]],
    target: list[dict[str, object]],
    keys: list[tuple[str, str, int]],
) -> dict[str, float | int]:
    a = _index(anchor)
    b = _index(target)
    if not keys:
        raise ValueError("W12 transition subset cannot be empty")
    ctw = sum(bool(a[k]["correct"]) and not bool(b[k]["correct"]) for k in keys)
    wtc = sum(not bool(a[k]["correct"]) and bool(b[k]["correct"]) for k in keys)
    return {
        "n": len(keys),
        "correct_to_wrong_count": ctw,
        "correct_to_wrong_rate": ctw / len(keys),
        "wrong_to_correct_count": wtc,
        "wrong_to_correct_rate": wtc / len(keys),
        "mean_rank_delta": sum(int(b[k]["rank"]) - int(a[k]["rank"]) for k in keys) / len(keys),
        "mean_margin_delta": sum(float(b[k]["margin"]) - float(a[k]["margin"]) for k in keys) / len(keys),
    }


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    per_domain: dict[str, object] = {}
    for domain in DOMAINS:
        per_domain[domain] = {}
        for view in ("definition", "label"):
            per_domain[domain][view] = {}
            for k in K_VALUES:
                rows = [
                    x for x in records
                    if x["domain_id"] == domain
                    and x["view_id"] == view
                    and int(x["k"]) == k
                ]
                per_domain[domain][view][str(k)] = _cell(rows)

    pooled: dict[str, object] = {}
    for view in ("definition", "label"):
        pooled[view] = {}
        for k in K_VALUES:
            rows = [x for x in records if x["view_id"] == view and int(x["k"]) == k]
            pooled[view][str(k)] = _cell(rows)
    return {"per_domain": per_domain, "pooled": pooled}


def _guard_record(
    anchor: dict[str, object],
    final: dict[str, object],
    use_anchor: bool,
    stage: str,
) -> dict[str, object]:
    source = anchor if use_anchor else final
    row = dict(source)
    row["stage"] = stage
    return row


@torch.inference_mode()
def evaluate_anchor_residual(
    hira: HIRACore,
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w12_cache(cache)
    hira.eval()
    scorer.eval()
    projection = scorer.projection.weight.detach().float()

    records: dict[str, list[dict[str, object]]] = {"A": [], "C": [], "F": []}
    anchor_conf: dict[tuple[str, str, int], float] = {}
    max_mass_error = 0.0

    for base in cache["bases"]:
        state_tokens = base["state_content_tokens"].float()
        for view in base["views"]:
            option_tokens = view["option_tokens"].float()
            option_mask = view["option_content_mask"].bool()

            a_logits = _q0_symmetric(
                state_tokens,
                option_tokens,
                option_mask,
                projection,
            )
            c_logits = _q6_actual(scorer, base, view)
            out = _q7_final(hira, base, view, c_logits)
            f_logits = out.logits[0].detach().cpu().float()
            p = out.probabilities[0].detach().cpu().to(torch.float64)
            max_mass_error = max(max_mass_error, abs(float(p.sum()) - 1.0))

            a_row = _record(base=base, view=view, stage="A", logits=a_logits)
            c_row = _record(base=base, view=view, stage="C", logits=c_logits)
            f_row = _record(base=base, view=view, stage="F", logits=f_logits)
            records["A"].append(a_row)
            records["C"].append(c_row)
            records["F"].append(f_row)

            if str(view["view_id"]) == "definition":
                key = (str(base["base_id"]), "definition", int(view["diagnosis_k"]))
                anchor_conf[key] = normalized_anchor_margin(a_logits.tolist())

    a_idx = _index(records["A"])
    c_idx = _index(records["C"])
    f_idx = _index(records["F"])

    strata_by_domain_k: dict[tuple[str, int], dict[str, str]] = {}
    for domain in DOMAINS:
        for k in K_VALUES:
            conf_rows = [
                ConfidenceRow(base_id=base_id, margin=margin)
                for (base_id, view_id, row_k), margin in anchor_conf.items()
                if view_id == "definition"
                and row_k == k
                and a_idx[(base_id, view_id, row_k)]["domain_id"] == domain
            ]
            strata_by_domain_k[(domain, k)] = deterministic_quartiles(conf_rows)

    guard_high: list[dict[str, object]] = []
    guard_low: list[dict[str, object]] = []
    per_domain: dict[str, object] = {}

    for domain in DOMAINS:
        domain_strata: dict[str, object] = {}
        domain_keys = [
            key for key, row in a_idx.items()
            if row["domain_id"] == domain and row["view_id"] == "definition"
        ]

        for bucket in ("HIGH", "LOW", "MIDDLE"):
            bucket_metrics: dict[str, object] = {}
            pooled_primary_keys: list[tuple[str, str, int]] = []
            for k in K_VALUES:
                keys = [
                    key for key in domain_keys
                    if key[2] == k
                    and strata_by_domain_k[(domain, k)][key[0]] == bucket
                ]
                if len(keys) != (16 if bucket != "MIDDLE" else 32):
                    raise RuntimeError("W12 confidence stratum count mismatch")
                if k in (4, 16):
                    pooled_primary_keys.extend(keys)
                bucket_metrics[str(k)] = {
                    "anchor_top1": sum(bool(a_idx[x]["correct"]) for x in keys) / len(keys),
                    "coarse_top1": sum(bool(c_idx[x]["correct"]) for x in keys) / len(keys),
                    "final_top1": sum(bool(f_idx[x]["correct"]) for x in keys) / len(keys),
                    "anchor_mrr": sum(float(a_idx[x]["mrr"]) for x in keys) / len(keys),
                    "final_mrr": sum(float(f_idx[x]["mrr"]) for x in keys) / len(keys),
                }
            bucket_metrics["pooled_primary_transition"] = _transition(
                records["A"], records["F"], pooled_primary_keys
            )
            domain_strata[bucket] = bucket_metrics

        for key in domain_keys:
            base_id, _, k = key
            bucket = strata_by_domain_k[(domain, k)][base_id]
            guard_high.append(
                _guard_record(a_idx[key], f_idx[key], bucket == "HIGH", "G_high_anchor")
            )
            guard_low.append(
                _guard_record(a_idx[key], f_idx[key], bucket == "LOW", "G_low_anchor_control")
            )

        margins = [anchor_conf[key] for key in domain_keys if key[2] in (4, 16)]
        anchor_correct = [1.0 if bool(a_idx[key]["correct"]) else 0.0 for key in domain_keys if key[2] in (4, 16)]
        delta_correct = [
            (1.0 if bool(f_idx[key]["correct"]) else 0.0)
            - (1.0 if bool(a_idx[key]["correct"]) else 0.0)
            for key in domain_keys if key[2] in (4, 16)
        ]
        per_domain[domain] = {
            "strata": domain_strata,
            "spearman_margin_anchor_correct": spearman(margins, anchor_correct),
            "spearman_margin_final_minus_anchor_correct": spearman(margins, delta_correct),
            "anchor_to_coarse": _transition(records["A"], records["C"], domain_keys),
            "anchor_to_final": _transition(records["A"], records["F"], domain_keys),
        }

    guard_high_summary = _summary(guard_high)
    guard_low_summary = _summary(guard_low)

    for domain in DOMAINS:
        per_domain[domain]["guards"] = {
            "G_high_anchor": guard_high_summary["per_domain"][domain]["definition"],
            "G_low_anchor_control": guard_low_summary["per_domain"][domain]["definition"],
        }

    return {
        "stages": {name: _summary(rows) for name, rows in records.items()},
        "per_domain": per_domain,
        "guards": {
            "G_high_anchor": guard_high_summary,
            "G_low_anchor_control": guard_low_summary,
        },
        "anchor_confidence": {
            f"{base_id}|{k}": value
            for (base_id, view_id, k), value in anchor_conf.items()
            if view_id == "definition"
        },
        "probability_mass_max_error": max_mass_error,
        "state_encodes_per_base": float(cache["metadata"]["state_encode_calls_per_base"]),
        "cached_base_count": int(cache["metadata"]["base_count"]),
        "cached_view_count": int(cache["metadata"]["view_count"]),
    }


def reference_summary(
    cache: dict,
    score_lookup: Mapping[str, list[float]],
) -> dict[str, object]:
    validate_w12_cache(cache)
    rows: list[dict[str, object]] = []
    for base in cache["bases"]:
        for view in base["views"]:
            scores = score_lookup.get(str(view["case_id"]))
            if scores is None:
                raise ValueError(f"missing W12 reference scores for {view['case_id']}")
            logits = torch.tensor(scores, dtype=torch.float32)
            if logits.numel() != int(view["diagnosis_k"]):
                raise ValueError("W12 reference width mismatch")
            rows.append(_record(base=base, view=view, stage="R0", logits=logits))
    return _summary(rows)
