from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .competitive import CompetitiveCoarseScorer
from .interface_decomposition_eval import _q0_symmetric
from .pairwise_latent_authority import VIEW_IDS
from .pairwise_latent_cache import validate_w20_cache


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
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W20 requires finite rank logits")
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


def _categorical_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "top1": 0.0,
            "mrr": 0.0,
            "mean_margin": 0.0,
            "mae": 0.0,
        }
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / n,
        "mrr": sum(float(row["mrr"]) for row in rows) / n,
        "mean_margin": sum(float(row["margin"]) for row in rows) / n,
        "mae": sum(abs(int(row["pred"]) - int(row["gold"])) for row in rows) / n,
    }


def _binary_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    base = _categorical_summary(rows)
    n = len(rows)
    if n == 0:
        return {
            **base,
            "balanced_accuracy": 0.0,
            "swap_prediction_identity": 0.0,
            "swap_score_max_abs_diff": float("inf"),
            "probability_mass_max_error": float("inf"),
        }
    recalls = []
    for gold in (0, 1):
        subset = [row for row in rows if int(row["gold"]) == gold]
        if not subset:
            recalls.append(0.0)
        else:
            recalls.append(
                sum(int(row["pred"]) == gold for row in subset) / len(subset)
            )
    return {
        **base,
        "balanced_accuracy": sum(recalls) / 2.0,
        "swap_prediction_identity": (
            sum(bool(row["swap_prediction_identity"]) for row in rows) / n
        ),
        "swap_score_max_abs_diff": max(
            float(row["swap_score_max_abs_diff"]) for row in rows
        ),
        "probability_mass_max_error": max(
            float(row["probability_mass_error"]) for row in rows
        ),
    }


def _pair_record(
    canonical_logits: Tensor,
    swapped_logits: Tensor,
    gold_relative: int,
) -> dict[str, object]:
    if canonical_logits.numel() != 2 or swapped_logits.numel() != 2:
        raise RuntimeError("W20 pair requires width two")
    mapped_swapped = torch.flip(swapped_logits, dims=(0,))
    rec = _record(canonical_logits, gold_relative)
    canonical_prob = torch.softmax(canonical_logits.float(), dim=0)
    swapped_prob = torch.flip(torch.softmax(swapped_logits.float(), dim=0), dims=(0,))
    rec.update(
        {
            "swap_prediction_identity": (
                int(canonical_logits.argmax()) == int(mapped_swapped.argmax())
            ),
            "swap_score_max_abs_diff": float(
                (canonical_logits.float() - mapped_swapped.float()).abs().max()
            ),
            "probability_mass_error": max(
                abs(float(canonical_prob.sum()) - 1.0),
                abs(float(swapped_prob.sum()) - 1.0),
            ),
        }
    )
    return rec


def _pair_bundle(
    state_tokens: Tensor,
    scorer: CompetitiveCoarseScorer,
    pairs: list[dict],
) -> list[dict[str, Tensor]]:
    out = []
    for pair in pairs:
        out.append(
            {
                "canonical": _multiview_logits(
                    scorer, state_tokens, pair["canonical"]
                ),
                "swapped": _multiview_logits(
                    scorer, state_tokens, pair["swapped"]
                ),
            }
        )
    return out


def _global_utilities(pair_logits: list[Tensor], width: int) -> Tensor:
    deltas = [float(logits[1] - logits[0]) for logits in pair_logits]
    utilities = [0.0]
    running = 0.0
    for delta in deltas:
        running += delta
        utilities.append(running)
    if len(utilities) != width:
        raise RuntimeError("W20 global pair reconstruction width changed")
    return torch.tensor(utilities, dtype=torch.float32)


def _case_from_logits(
    case: dict,
    flat_severity_logits: Tensor,
    flat_confidence_logits: Tensor,
    severity_pairs: list[dict[str, Tensor]],
    confidence_pairs: list[dict[str, Tensor]],
) -> dict[str, object]:
    severity_gold = int(case["severity"])
    confidence_gold = int(case["confidence_index"])
    flat_s = _record(flat_severity_logits, severity_gold)
    flat_c = _record(flat_confidence_logits, confidence_gold)

    severity_local = []
    for boundary, pair in enumerate(severity_pairs):
        low = boundary
        high = boundary + 1
        eligible = severity_gold in {low, high}
        pair_rec = None
        flat_local_rec = None
        if eligible:
            relative_gold = severity_gold - low
            pair_rec = _pair_record(
                pair["canonical"], pair["swapped"], relative_gold
            )
            flat_local_rec = _record(
                flat_severity_logits[[low, high]], relative_gold
            )
        severity_local.append(
            {
                "boundary": f"S{low}{high}",
                "eligible": eligible,
                "pairwise": pair_rec,
                "flat_local": flat_local_rec,
            }
        )

    confidence_local = []
    for boundary, pair in enumerate(confidence_pairs):
        low = boundary
        high = boundary + 1
        eligible = confidence_gold in {low, high}
        pair_rec = None
        flat_local_rec = None
        if eligible:
            relative_gold = confidence_gold - low
            pair_rec = _pair_record(
                pair["canonical"], pair["swapped"], relative_gold
            )
            flat_local_rec = _record(
                flat_confidence_logits[[low, high]], relative_gold
            )
        confidence_local.append(
            {
                "boundary": f"C{low}{high}",
                "eligible": eligible,
                "pairwise": pair_rec,
                "flat_local": flat_local_rec,
            }
        )

    severity_utilities = _global_utilities(
        [pair["canonical"] for pair in severity_pairs], 4
    )
    confidence_utilities = _global_utilities(
        [pair["canonical"] for pair in confidence_pairs], 3
    )
    global_s = _record(severity_utilities, severity_gold)
    global_c = _record(confidence_utilities, confidence_gold)
    s_prob = torch.softmax(severity_utilities, dim=0)
    c_prob = torch.softmax(confidence_utilities, dim=0)

    all_pair_probs = []
    all_pair_probs.extend(
        torch.softmax(pair["canonical"].float(), dim=0)
        for pair in severity_pairs + confidence_pairs
    )
    probability_mass_error = max(
        [abs(float(prob.sum()) - 1.0) for prob in all_pair_probs]
        + [
            abs(float(s_prob.sum()) - 1.0),
            abs(float(c_prob.sum()) - 1.0),
        ]
    )

    return {
        "case_id": str(case["case_id"]),
        "domain_id": str(case["domain_id"]),
        "severity_gold": severity_gold,
        "confidence_gold": confidence_gold,
        "flat_severity": flat_s,
        "flat_confidence": flat_c,
        "severity_local": severity_local,
        "confidence_local": confidence_local,
        "global_pairwise_severity": global_s,
        "global_pairwise_confidence": global_c,
        "probability_mass_error": probability_mass_error,
    }


def _local_boundary_summaries(
    raw: list[dict[str, object]],
    field_name: str,
    boundary_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pair_summaries = []
    flat_summaries = []
    local_key = f"{field_name}_local"
    for boundary in range(boundary_count):
        eligible = [
            row[local_key][boundary]
            for row in raw
            if bool(row[local_key][boundary]["eligible"])
        ]
        pair_rows = [entry["pairwise"] for entry in eligible]
        flat_rows = [entry["flat_local"] for entry in eligible]
        pair_summaries.append(_binary_summary(pair_rows))
        flat_summaries.append(_categorical_summary(flat_rows))
    return pair_summaries, flat_summaries


def _weighted_mean_top1(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(float(row["top1"]) for row in rows) / len(rows)


def _summarize(raw: list[dict[str, object]]) -> dict[str, object]:
    n = len(raw)
    flat_s = [row["flat_severity"] for row in raw]
    flat_c = [row["flat_confidence"] for row in raw]
    global_s = [row["global_pairwise_severity"] for row in raw]
    global_c = [row["global_pairwise_confidence"] for row in raw]

    severity_pairs, severity_flat_local = _local_boundary_summaries(
        raw, "severity", 3
    )
    confidence_pairs, confidence_flat_local = _local_boundary_summaries(
        raw, "confidence", 2
    )

    severity_pair_mean = _weighted_mean_top1(severity_pairs)
    confidence_pair_mean = _weighted_mean_top1(confidence_pairs)
    severity_flat_mean = _weighted_mean_top1(severity_flat_local)
    confidence_flat_mean = _weighted_mean_top1(confidence_flat_local)

    flat_joint = (
        sum(
            bool(s["correct"]) and bool(c["correct"])
            for s, c in zip(flat_s, flat_c)
        )
        / max(1, n)
    )
    global_joint = (
        sum(
            bool(s["correct"]) and bool(c["correct"])
            for s, c in zip(global_s, global_c)
        )
        / max(1, n)
    )

    severity_transitions = {
        "flat_wrong_to_pairwise_right": (
            sum(
                (not bool(fs["correct"])) and bool(gs["correct"])
                for fs, gs in zip(flat_s, global_s)
            )
            / max(1, n)
        ),
        "flat_right_to_pairwise_wrong": (
            sum(
                bool(fs["correct"]) and (not bool(gs["correct"]))
                for fs, gs in zip(flat_s, global_s)
            )
            / max(1, n)
        ),
    }
    confidence_transitions = {
        "flat_wrong_to_pairwise_right": (
            sum(
                (not bool(fc["correct"])) and bool(gc["correct"])
                for fc, gc in zip(flat_c, global_c)
            )
            / max(1, n)
        ),
        "flat_right_to_pairwise_wrong": (
            sum(
                bool(fc["correct"]) and (not bool(gc["correct"]))
                for fc, gc in zip(flat_c, global_c)
            )
            / max(1, n)
        ),
    }

    return {
        "case_count": n,
        "flat": {
            "severity": _categorical_summary(flat_s),
            "confidence": _categorical_summary(flat_c),
            "joint_severity_confidence_top1": flat_joint,
            "severity_local_pairs": severity_flat_local,
            "confidence_local_pairs": confidence_flat_local,
            "severity_local_mean_top1": severity_flat_mean,
            "confidence_local_mean_top1": confidence_flat_mean,
        },
        "pairwise": {
            "severity_pairs": severity_pairs,
            "confidence_pairs": confidence_pairs,
            "severity_mean_top1": severity_pair_mean,
            "confidence_mean_top1": confidence_pair_mean,
            "severity_min_top1": min(
                [float(x["top1"]) for x in severity_pairs] or [0.0]
            ),
            "confidence_min_top1": min(
                [float(x["top1"]) for x in confidence_pairs] or [0.0]
            ),
            "probability_mass_max_error": max(
                [float(row["probability_mass_error"]) for row in raw] or [0.0]
            ),
        },
        "global_reconstruction": {
            "severity": _categorical_summary(global_s),
            "confidence": _categorical_summary(global_c),
            "joint_severity_confidence_top1": global_joint,
            "severity_transitions": severity_transitions,
            "confidence_transitions": confidence_transitions,
        },
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
        },
    }


@torch.inference_mode()
def evaluate_w20(
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w20_cache(cache)
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
                _multiview_logits(
                    scorer, severity_tokens, schemas["flat_severity"]
                ),
                _multiview_logits(
                    scorer, confidence_tokens, schemas["flat_confidence"]
                ),
                _pair_bundle(
                    severity_tokens, scorer, schemas["severity_pairs"]
                ),
                _pair_bundle(
                    confidence_tokens, scorer, schemas["confidence_pairs"]
                ),
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
            raise ValueError(f"missing W20 reference scores for {key}")

        severity_pairs = [
            {
                "canonical": torch.tensor(pair["canonical"], dtype=torch.float32),
                "swapped": torch.tensor(pair["swapped"], dtype=torch.float32),
            }
            for pair in lookup["severity_pairs"]
        ]
        confidence_pairs = [
            {
                "canonical": torch.tensor(pair["canonical"], dtype=torch.float32),
                "swapped": torch.tensor(pair["swapped"], dtype=torch.float32),
            }
            for pair in lookup["confidence_pairs"]
        ]
        raw.append(
            _case_from_logits(
                case,
                torch.tensor(lookup["flat_severity"], dtype=torch.float32),
                torch.tensor(lookup["flat_confidence"], dtype=torch.float32),
                severity_pairs,
                confidence_pairs,
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


def _all_boundaries_at_least(rows: list[dict[str, object]], threshold: float) -> bool:
    return all(float(row["top1"]) >= threshold for row in rows)


def _all_swap_identity(rows: list[dict[str, object]], threshold: float = .99) -> bool:
    return all(float(row["swap_prediction_identity"]) >= threshold for row in rows)


def _mean_gain(pairwise: list[dict[str, object]], flat: list[dict[str, object]]) -> float:
    if len(pairwise) != len(flat) or not pairwise:
        raise ValueError("W20 gain rows invalid")
    return sum(
        float(p["top1"]) - float(f["top1"])
        for p, f in zip(pairwise, flat)
    ) / len(pairwise)


def _min_gain(pairwise: list[dict[str, object]], flat: list[dict[str, object]]) -> float:
    if len(pairwise) != len(flat) or not pairwise:
        raise ValueError("W20 gain rows invalid")
    return min(
        float(p["top1"]) - float(f["top1"])
        for p, f in zip(pairwise, flat)
    )


def _domain_gate(
    metrics: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    pair = metrics["pairwise"]
    flat = metrics["flat"]
    ref_pair = reference["pairwise"]

    ref_s = ref_pair["severity_pairs"]
    ref_c = ref_pair["confidence_pairs"]
    reference_adequacy = (
        _all_boundaries_at_least(ref_s, .90)
        and _all_boundaries_at_least(ref_c, .90)
        and _all_swap_identity(ref_s)
        and _all_swap_identity(ref_c)
        and float(ref_pair["probability_mass_max_error"]) <= 1e-6
    )

    s_pairs = pair["severity_pairs"]
    c_pairs = pair["confidence_pairs"]
    hira_pairwise_adequacy = (
        float(pair["severity_mean_top1"]) >= .82
        and _all_boundaries_at_least(s_pairs, .75)
        and float(pair["confidence_mean_top1"]) >= .85
        and _all_boundaries_at_least(c_pairs, .80)
        and _all_swap_identity(s_pairs)
        and _all_swap_identity(c_pairs)
    )

    severity_mean_gain = _mean_gain(s_pairs, flat["severity_local_pairs"])
    confidence_mean_gain = _mean_gain(c_pairs, flat["confidence_local_pairs"])
    severity_min_gain = _min_gain(s_pairs, flat["severity_local_pairs"])
    confidence_min_gain = _min_gain(c_pairs, flat["confidence_local_pairs"])

    severity_causal_gain = (
        severity_mean_gain >= .08 and severity_min_gain >= -.03
    )
    confidence_causal_gain = (
        confidence_mean_gain >= .05 and confidence_min_gain >= -.03
    )

    flat_local_adequate = (
        float(flat["severity_local_mean_top1"]) >= .82
        and float(flat["confidence_local_mean_top1"]) >= .85
        and severity_mean_gain < .05
        and confidence_mean_gain < .05
    )

    integrity = (
        float(pair["probability_mass_max_error"]) <= 1e-6
        and metrics["representation_accounting"] == {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
        }
    )

    if not reference_adequacy:
        classification = "W20_REFERENCE_INADEQUATE"
    elif not hira_pairwise_adequacy:
        classification = "LATENT_BOUNDARY_EXTRACTION_LIMIT"
    elif severity_causal_gain and confidence_causal_gain:
        classification = "LOCAL_PAIRWISE_INTERFACE_LIMIT"
    elif flat_local_adequate:
        classification = "FLAT_LOCAL_INTERFACE_ADEQUATE"
    else:
        classification = "LATENT_BOUNDARY_UNRESOLVED"

    return {
        "integrity": integrity,
        "reference_adequacy": reference_adequacy,
        "hira_pairwise_adequacy": hira_pairwise_adequacy,
        "severity_mean_gain": severity_mean_gain,
        "confidence_mean_gain": confidence_mean_gain,
        "severity_min_gain": severity_min_gain,
        "confidence_min_gain": confidence_min_gain,
        "severity_causal_gain": severity_causal_gain,
        "confidence_causal_gain": confidence_causal_gain,
        "flat_local_interface_adequate": flat_local_adequate,
        "classification": classification,
    }


def classify_w20(
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
    excluded = {
        "LATENT_BOUNDARY_UNRESOLVED",
        "W20_REFERENCE_INADEQUATE",
    }
    stable = None
    for name, count in counts.items():
        if name not in excluded and count >= 3:
            stable = name
            break

    if stable is not None:
        outcome = "STABLE_LATENT_BOUNDARY_LOCALIZATION"
    else:
        incompatible = [
            name
            for name, count in counts.items()
            if name not in excluded and count >= 2
        ]
        outcome = (
            "MIXED_LATENT_BOUNDARY_LOCALIZATION"
            if len(incompatible) >= 2
            else "LATENT_BOUNDARY_UNRESOLVED"
        )
    return {
        "outcome": outcome,
        "stable_classification": stable,
        "classification_counts": dict(sorted(counts.items())),
        "per_domain": per_domain,
    }


__all__ = [
    "classify_w20",
    "evaluate_w20",
    "reference_summary_from_scores",
]
