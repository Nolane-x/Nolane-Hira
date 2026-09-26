from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .competitive import CompetitiveCoarseScorer
from .interface_decomposition_eval import _q0_symmetric
from .latent_ordinal_cache import validate_w19_cache
from .latent_ordinal_authority import VIEW_IDS


@torch.inference_mode()
def _multiview_logits(
    scorer: CompetitiveCoarseScorer,
    state_tokens: Tensor,
    views: Mapping[str, Mapping[str, object]],
) -> Tensor:
    projection = scorer.projection.weight.detach().float()
    logits = []
    for view_id in VIEW_IDS:
        view = views[view_id]
        raw = _q0_symmetric(
            state_tokens.float(),
            view["option_tokens"].float(),
            view["option_content_mask"].bool(),
            projection,
        )
        logits.append(raw)
    return torch.stack(logits, dim=0).mean(dim=0) * scorer.scale().detach().float()


def _record(logits: Tensor, gold: int) -> dict[str, object]:
    rank, mrr, _ = _rank_metrics(logits, int(gold))
    pred = int(logits.argmax())
    return {
        "gold": int(gold),
        "pred": pred,
        "correct": pred == int(gold),
        "rank": int(rank),
        "mrr": float(mrr),
        "margin": float(_margin(logits, int(gold))),
    }


def _soft_ordinal(q: list[float]) -> Tensor:
    if len(q) == 3:
        q1, q2, q3 = q
        p = torch.tensor(
            [
                1.0 - q1,
                q1 * (1.0 - q2),
                q1 * q2 * (1.0 - q3),
                q1 * q2 * q3,
            ],
            dtype=torch.float64,
        )
    elif len(q) == 2:
        q1, q2 = q
        p = torch.tensor(
            [1.0 - q1, q1 * (1.0 - q2), q1 * q2],
            dtype=torch.float64,
        )
    else:
        raise ValueError("W19 ordinal width must be 2 or 3 thresholds")
    total = p.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        raise RuntimeError("W19 invalid ordinal probability mass")
    return p / total


def _threshold_gold(level: int, threshold_index: int) -> int:
    return int(int(level) >= threshold_index + 1)


def _valid_pattern(bits: tuple[int, ...]) -> bool:
    if len(bits) == 3:
        return bits in {(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)}
    if len(bits) == 2:
        return bits in {(0, 0), (1, 0), (1, 1)}
    raise ValueError("W19 unexpected threshold count")


def _categorical_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    wrong = [row for row in rows if not bool(row["correct"])]
    adjacent = sum(abs(int(row["pred"]) - int(row["gold"])) == 1 for row in wrong)
    non_adjacent = sum(abs(int(row["pred"]) - int(row["gold"])) > 1 for row in wrong)
    mae = sum(abs(int(row["pred"]) - int(row["gold"])) for row in rows) / max(1, n)
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / max(1, n),
        "mrr": sum(float(row["mrr"]) for row in rows) / max(1, n),
        "mean_margin": sum(float(row["margin"]) for row in rows) / max(1, n),
        "mae": mae,
        "adjacent_error_rate": adjacent / max(1, n),
        "non_adjacent_error_rate": non_adjacent / max(1, n),
    }


def _threshold_summary(rows_by_threshold: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    return [_categorical_summary(rows) for rows in rows_by_threshold]


def _summarize(raw: list[dict[str, object]]) -> dict[str, object]:
    flat_s = [row["flat_severity"] for row in raw]
    flat_c = [row["flat_confidence"] for row in raw]
    ord_s = [row["ordinal_severity"] for row in raw]
    ord_c = [row["ordinal_confidence"] for row in raw]

    s_thresholds = [
        [row["severity_thresholds"][index] for row in raw]
        for index in range(3)
    ]
    c_thresholds = [
        [row["confidence_thresholds"][index] for row in raw]
        for index in range(2)
    ]

    n = len(raw)
    flat_joint = sum(
        bool(s["correct"]) and bool(c["correct"])
        for s, c in zip(flat_s, flat_c)
    ) / max(1, n)
    ordinal_joint = sum(
        bool(s["correct"]) and bool(c["correct"])
        for s, c in zip(ord_s, ord_c)
    ) / max(1, n)

    flat_wrong_ordinal_right_s = sum(
        (not bool(fs["correct"])) and bool(os["correct"])
        for fs, os in zip(flat_s, ord_s)
    ) / max(1, n)
    flat_right_ordinal_wrong_s = sum(
        bool(fs["correct"]) and (not bool(os["correct"]))
        for fs, os in zip(flat_s, ord_s)
    ) / max(1, n)
    flat_wrong_ordinal_right_c = sum(
        (not bool(fc["correct"])) and bool(oc["correct"])
        for fc, oc in zip(flat_c, ord_c)
    ) / max(1, n)
    flat_right_ordinal_wrong_c = sum(
        bool(fc["correct"]) and (not bool(oc["correct"]))
        for fc, oc in zip(flat_c, ord_c)
    ) / max(1, n)

    flat_s_summary = _categorical_summary(flat_s)
    flat_c_summary = _categorical_summary(flat_c)
    ord_s_summary = _categorical_summary(ord_s)
    ord_c_summary = _categorical_summary(ord_c)

    soft_s_top1 = sum(int(row["soft_severity_pred"]) == int(row["severity_gold"]) for row in raw) / max(1, n)
    soft_c_top1 = sum(int(row["soft_confidence_pred"]) == int(row["confidence_gold"]) for row in raw) / max(1, n)

    return {
        "case_count": n,
        "flat": {
            "severity": flat_s_summary,
            "confidence": flat_c_summary,
            "joint_severity_confidence_top1": flat_joint,
        },
        "ordinal": {
            "severity_thresholds": _threshold_summary(s_thresholds),
            "confidence_thresholds": _threshold_summary(c_thresholds),
            "severity": ord_s_summary,
            "confidence": ord_c_summary,
            "joint_severity_confidence_top1": ordinal_joint,
            "severity_soft_top1": soft_s_top1,
            "confidence_soft_top1": soft_c_top1,
            "severity_monotonicity": sum(bool(row["severity_monotone"]) for row in raw) / max(1, n),
            "confidence_monotonicity": sum(bool(row["confidence_monotone"]) for row in raw) / max(1, n),
            "probability_mass_max_error": max(
                [float(row["probability_mass_error"]) for row in raw] or [0.0]
            ),
        },
        "transitions": {
            "severity_flat_wrong_to_ordinal_right": flat_wrong_ordinal_right_s,
            "severity_flat_right_to_ordinal_wrong": flat_right_ordinal_wrong_s,
            "confidence_flat_wrong_to_ordinal_right": flat_wrong_ordinal_right_c,
            "confidence_flat_right_to_ordinal_wrong": flat_right_ordinal_wrong_c,
        },
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
        },
    }


def _case_from_logits(
    case: dict,
    flat_severity_logits: Tensor,
    flat_confidence_logits: Tensor,
    severity_threshold_logits: list[Tensor],
    confidence_threshold_logits: list[Tensor],
) -> dict[str, object]:
    severity_gold = int(case["severity"])
    confidence_gold = int(case["confidence_index"])
    fs = _record(flat_severity_logits, severity_gold)
    fc = _record(flat_confidence_logits, confidence_gold)

    s_threshold_records = []
    s_bits = []
    s_q = []
    for index, logits in enumerate(severity_threshold_logits):
        gold = _threshold_gold(severity_gold, index)
        rec = _record(logits, gold)
        s_threshold_records.append(rec)
        s_bits.append(int(rec["pred"]))
        s_q.append(float(torch.softmax(logits.float(), dim=0)[1]))

    c_threshold_records = []
    c_bits = []
    c_q = []
    for index, logits in enumerate(confidence_threshold_logits):
        gold = _threshold_gold(confidence_gold, index)
        rec = _record(logits, gold)
        c_threshold_records.append(rec)
        c_bits.append(int(rec["pred"]))
        c_q.append(float(torch.softmax(logits.float(), dim=0)[1]))

    hard_s = sum(s_bits)
    hard_c = sum(c_bits)
    soft_s = _soft_ordinal(s_q)
    soft_c = _soft_ordinal(c_q)
    probability_mass_error = max(
        abs(float(soft_s.sum()) - 1.0),
        abs(float(soft_c.sum()) - 1.0),
    )

    return {
        "case_id": str(case["case_id"]),
        "domain_id": str(case["domain_id"]),
        "severity_gold": severity_gold,
        "confidence_gold": confidence_gold,
        "flat_severity": fs,
        "flat_confidence": fc,
        "severity_thresholds": s_threshold_records,
        "confidence_thresholds": c_threshold_records,
        "ordinal_severity": {
            "gold": severity_gold,
            "pred": hard_s,
            "correct": hard_s == severity_gold,
            "rank": 1 if hard_s == severity_gold else 2,
            "mrr": 1.0 if hard_s == severity_gold else 0.5,
            "margin": 0.0,
        },
        "ordinal_confidence": {
            "gold": confidence_gold,
            "pred": hard_c,
            "correct": hard_c == confidence_gold,
            "rank": 1 if hard_c == confidence_gold else 2,
            "mrr": 1.0 if hard_c == confidence_gold else 0.5,
            "margin": 0.0,
        },
        "soft_severity_pred": int(soft_s.argmax()),
        "soft_confidence_pred": int(soft_c.argmax()),
        "severity_monotone": _valid_pattern(tuple(s_bits)),
        "confidence_monotone": _valid_pattern(tuple(c_bits)),
        "probability_mass_error": probability_mass_error,
    }


@torch.inference_mode()
def evaluate_w19(
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w19_cache(cache)
    scorer.eval()
    raw = []
    for case in cache["cases"]:
        reps = case["representations"]
        schemas = case["schemas"]
        severity_tokens = reps["severity_tokens"].float()
        confidence_tokens = reps["confidence_tokens"].float()
        raw.append(
            _case_from_logits(
                case,
                _multiview_logits(scorer, severity_tokens, schemas["flat_severity"]),
                _multiview_logits(scorer, confidence_tokens, schemas["flat_confidence"]),
                [
                    _multiview_logits(scorer, severity_tokens, views)
                    for views in schemas["severity_thresholds"]
                ],
                [
                    _multiview_logits(scorer, confidence_tokens, views)
                    for views in schemas["confidence_thresholds"]
                ],
            )
        )

    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    return {
        "per_domain": {
            domain: _summarize(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": _summarize(raw),
    }


def reference_summary_from_scores(
    cases: list[dict],
    score_lookup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    raw = []
    for case in cases:
        key = str(case["case_id"])
        lookup = score_lookup.get(key)
        if lookup is None:
            raise ValueError(f"missing W19 reference scores for {key}")
        raw.append(
            _case_from_logits(
                case,
                torch.tensor(lookup["flat_severity"], dtype=torch.float32),
                torch.tensor(lookup["flat_confidence"], dtype=torch.float32),
                [
                    torch.tensor(values, dtype=torch.float32)
                    for values in lookup["severity_thresholds"]
                ],
                [
                    torch.tensor(values, dtype=torch.float32)
                    for values in lookup["confidence_thresholds"]
                ],
            )
        )
    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    return {
        "per_domain": {
            domain: _summarize(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": _summarize(raw),
    }


def _domain_gate(metrics: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    flat = metrics["flat"]
    ordinal = metrics["ordinal"]
    ref_ordinal = reference["ordinal"]

    reference_adequacy = (
        float(ref_ordinal["severity"]["top1"]) >= .90
        and float(ref_ordinal["confidence"]["top1"]) >= .90
        and float(ref_ordinal["severity_monotonicity"]) >= .95
        and float(ref_ordinal["confidence_monotonicity"]) >= .97
    )

    ordinal_adequacy = (
        float(ordinal["severity"]["top1"]) >= .82
        and float(ordinal["confidence"]["top1"]) >= .85
        and float(ordinal["joint_severity_confidence_top1"]) >= .72
        and float(ordinal["severity_monotonicity"]) >= .92
        and float(ordinal["confidence_monotonicity"]) >= .95
    )

    severity_gain = (
        float(ordinal["severity"]["top1"]) >= float(flat["severity"]["top1"]) + .12
        and float(flat["severity"]["mae"]) - float(ordinal["severity"]["mae"]) >= .20
    )
    confidence_preservation = (
        float(ordinal["confidence"]["top1"])
        >= max(.85, float(flat["confidence"]["top1"]) - .03)
    )
    joint_gain = (
        float(ordinal["joint_severity_confidence_top1"])
        >= float(flat["joint_severity_confidence_top1"]) + .10
    )

    flat_adequate = (
        float(flat["severity"]["top1"]) >= .80
        and float(flat["confidence"]["top1"]) >= .85
        and float(flat["joint_severity_confidence_top1"]) >= .70
        and (
            float(ordinal["joint_severity_confidence_top1"])
            - float(flat["joint_severity_confidence_top1"])
        ) < .08
    )

    integrity = (
        float(ordinal["probability_mass_max_error"]) <= 1e-6
        and metrics["representation_accounting"] == {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
        }
    )

    if not reference_adequacy:
        classification = "W19_REFERENCE_INADEQUATE"
    elif not ordinal_adequacy:
        classification = "LATENT_AXIS_EXTRACTION_LIMIT"
    elif severity_gain and confidence_preservation and joint_gain:
        classification = "ORDINAL_LATENT_INTERFACE_LIMIT"
    elif flat_adequate:
        classification = "FLAT_LATENT_INTERFACE_ADEQUATE"
    else:
        classification = "LATENT_AXIS_UNRESOLVED"

    return {
        "integrity": integrity,
        "reference_adequacy": reference_adequacy,
        "hira_ordinal_adequacy": ordinal_adequacy,
        "severity_causal_gain": severity_gain,
        "confidence_preservation_or_gain": confidence_preservation,
        "joint_gain": joint_gain,
        "flat_interface_adequate": flat_adequate,
        "classification": classification,
    }


def classify_w19(
    metrics: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    domains = sorted(metrics["per_domain"])
    per_domain = {
        domain: _domain_gate(
            metrics["per_domain"][domain],
            reference["per_domain"][domain],
        )
        for domain in domains
    }
    counts = Counter(row["classification"] for row in per_domain.values())
    excluded = {"LATENT_AXIS_UNRESOLVED", "W19_REFERENCE_INADEQUATE"}
    stable = None
    for name, count in counts.items():
        if name not in excluded and count >= 3:
            stable = name
            break

    if stable is not None:
        outcome = "STABLE_LATENT_AXIS_LOCALIZATION"
    else:
        incompatible = [
            name for name, count in counts.items()
            if name not in excluded and count >= 2
        ]
        outcome = (
            "MIXED_LATENT_AXIS_LOCALIZATION"
            if len(incompatible) >= 2
            else "LATENT_AXIS_UNRESOLVED"
        )
    return {
        "outcome": outcome,
        "stable_classification": stable,
        "classification_counts": dict(sorted(counts.items())),
        "per_domain": per_domain,
    }


__all__ = [
    "classify_w19",
    "evaluate_w19",
    "reference_summary_from_scores",
]
