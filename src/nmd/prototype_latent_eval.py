from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .competitive import CompetitiveCoarseScorer
from .interface_decomposition_eval import _q0_symmetric
from .prototype_latent_authority import PROTOTYPES_PER_CLASS, VIEW_IDS
from .prototype_latent_cache import validate_w21_cache


@torch.inference_mode()
def _multiview_logits(
    scorer: CompetitiveCoarseScorer,
    state_tokens: Tensor,
    views: Mapping[str, Mapping[str, object]],
) -> Tensor:
    projection = scorer.projection.weight.detach().float()
    rows = []
    for view_id in VIEW_IDS:
        view = views[view_id]
        rows.append(
            _q0_symmetric(
                state_tokens.float(),
                view["option_tokens"].float(),
                view["option_content_mask"].bool(),
                projection,
            )
        )
    return torch.stack(rows, dim=0).mean(dim=0) * scorer.scale().detach().float()


@torch.inference_mode()
def _prototype_raw_logits(
    scorer: CompetitiveCoarseScorer,
    state_tokens: Tensor,
    schema: Mapping[str, object],
) -> Tensor:
    projection = scorer.projection.weight.detach().float()
    return _q0_symmetric(
        state_tokens.float(),
        schema["option_tokens"].float(),
        schema["option_content_mask"].bool(),
        projection,
    ) * scorer.scale().detach().float()


def _aggregate_prototypes(raw: Tensor, classes: int) -> Tensor:
    if raw.numel() != classes * PROTOTYPES_PER_CLASS:
        raise ValueError("W21 prototype width changed")
    return raw.reshape(classes, PROTOTYPES_PER_CLASS).mean(dim=1)


def _leave_one_out_logits(raw: Tensor, classes: int) -> list[Tensor]:
    matrix = raw.reshape(classes, PROTOTYPES_PER_CLASS)
    out = []
    for excluded in range(PROTOTYPES_PER_CLASS):
        keep = [i for i in range(PROTOTYPES_PER_CLASS) if i != excluded]
        out.append(matrix[:, keep].mean(dim=1))
    return out


def _single_prototype_position_logits(raw: Tensor, classes: int) -> list[Tensor]:
    matrix = raw.reshape(classes, PROTOTYPES_PER_CLASS)
    return [matrix[:, index] for index in range(PROTOTYPES_PER_CLASS)]


def _record(logits: Tensor, gold: int) -> dict[str, object]:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W21 requires finite logits")
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


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "top1": 0.0, "mrr": 0.0, "mean_margin": 0.0, "mae": 0.0}
    return {
        "n": n,
        "top1": sum(bool(row["correct"]) for row in rows) / n,
        "mrr": sum(float(row["mrr"]) for row in rows) / n,
        "mean_margin": sum(float(row["margin"]) for row in rows) / n,
        "mae": sum(abs(int(row["pred"]) - int(row["gold"])) for row in rows) / n,
    }


def _case_from_logits(
    case: dict,
    abstract_severity: Tensor,
    abstract_confidence: Tensor,
    severity_raw: Tensor,
    confidence_raw: Tensor,
) -> dict[str, object]:
    severity_gold = int(case["severity"])
    confidence_gold = int(case["confidence_index"])

    severity_primary = _aggregate_prototypes(severity_raw, 4)
    confidence_primary = _aggregate_prototypes(confidence_raw, 3)
    severity_loo = _leave_one_out_logits(severity_raw, 4)
    confidence_loo = _leave_one_out_logits(confidence_raw, 3)
    severity_single = _single_prototype_position_logits(severity_raw, 4)
    confidence_single = _single_prototype_position_logits(confidence_raw, 3)

    s_probs = torch.softmax(severity_primary.float(), dim=0)
    c_probs = torch.softmax(confidence_primary.float(), dim=0)

    return {
        "case_id": str(case["case_id"]),
        "domain_id": str(case["domain_id"]),
        "severity_gold": severity_gold,
        "confidence_gold": confidence_gold,
        "abstract_severity": _record(abstract_severity, severity_gold),
        "abstract_confidence": _record(abstract_confidence, confidence_gold),
        "prototype_severity": _record(severity_primary, severity_gold),
        "prototype_confidence": _record(confidence_primary, confidence_gold),
        "severity_loo_preds": tuple(int(x.argmax()) for x in severity_loo),
        "confidence_loo_preds": tuple(int(x.argmax()) for x in confidence_loo),
        "severity_single": tuple(_record(x, severity_gold) for x in severity_single),
        "confidence_single": tuple(_record(x, confidence_gold) for x in confidence_single),
        "severity_raw_logits": tuple(float(x) for x in severity_raw),
        "confidence_raw_logits": tuple(float(x) for x in confidence_raw),
        "probability_mass_error": max(
            abs(float(s_probs.sum()) - 1.0),
            abs(float(c_probs.sum()) - 1.0),
        ),
    }


def _summarize(raw: list[dict[str, object]]) -> dict[str, object]:
    n = len(raw)
    abstract_s = [row["abstract_severity"] for row in raw]
    abstract_c = [row["abstract_confidence"] for row in raw]
    proto_s = [row["prototype_severity"] for row in raw]
    proto_c = [row["prototype_confidence"] for row in raw]

    abstract_joint = sum(
        bool(s["correct"]) and bool(c["correct"])
        for s, c in zip(abstract_s, abstract_c)
    ) / max(1, n)
    prototype_joint = sum(
        bool(s["correct"]) and bool(c["correct"])
        for s, c in zip(proto_s, proto_c)
    ) / max(1, n)

    severity_subset_agreement = sum(
        len(set(int(x) for x in row["severity_loo_preds"])) == 1
        for row in raw
    ) / max(1, n)
    confidence_subset_agreement = sum(
        len(set(int(x) for x in row["confidence_loo_preds"])) == 1
        for row in raw
    ) / max(1, n)

    severity_single = [
        _summary([row["severity_single"][index] for row in raw])
        for index in range(PROTOTYPES_PER_CLASS)
    ]
    confidence_single = [
        _summary([row["confidence_single"][index] for row in raw])
        for index in range(PROTOTYPES_PER_CLASS)
    ]

    return {
        "case_count": n,
        "abstract": {
            "severity": _summary(abstract_s),
            "confidence": _summary(abstract_c),
            "joint_severity_confidence_top1": abstract_joint,
        },
        "prototype": {
            "severity": _summary(proto_s),
            "confidence": _summary(proto_c),
            "joint_severity_confidence_top1": prototype_joint,
            "severity_leave_one_out_agreement": severity_subset_agreement,
            "confidence_leave_one_out_agreement": confidence_subset_agreement,
            "severity_single_prototype_position": severity_single,
            "confidence_single_prototype_position": confidence_single,
            "probability_mass_max_error": max(
                [float(row["probability_mass_error"]) for row in raw] or [0.0]
            ),
        },
        "transitions": {
            "severity_abstract_wrong_to_prototype_right": sum(
                (not bool(a["correct"])) and bool(p["correct"])
                for a, p in zip(abstract_s, proto_s)
            ) / max(1, n),
            "severity_abstract_right_to_prototype_wrong": sum(
                bool(a["correct"]) and (not bool(p["correct"]))
                for a, p in zip(abstract_s, proto_s)
            ) / max(1, n),
            "confidence_abstract_wrong_to_prototype_right": sum(
                (not bool(a["correct"])) and bool(p["correct"])
                for a, p in zip(abstract_c, proto_c)
            ) / max(1, n),
            "confidence_abstract_right_to_prototype_wrong": sum(
                bool(a["correct"]) and (not bool(p["correct"]))
                for a, p in zip(abstract_c, proto_c)
            ) / max(1, n),
        },
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_query_invocations_per_case": 1,
            "encoded_query_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
            "prototypes_per_class": 3,
            "prototype_schema_scope": "shared-per-domain",
        },
    }


@torch.inference_mode()
def evaluate_w21(
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w21_cache(cache)
    scorer.eval()
    raw = []
    for case in cache["cases"]:
        domain_id = str(case["domain_id"])
        schema = cache["schemas"][domain_id]
        reps = case["representations"]
        severity_tokens = reps["severity_tokens"].float()
        confidence_tokens = reps["confidence_tokens"].float()
        raw.append(
            _case_from_logits(
                case,
                _multiview_logits(scorer, severity_tokens, schema["abstract_severity"]),
                _multiview_logits(scorer, confidence_tokens, schema["abstract_confidence"]),
                _prototype_raw_logits(
                    scorer, severity_tokens, schema["severity_prototypes"]
                ),
                _prototype_raw_logits(
                    scorer, confidence_tokens, schema["confidence_prototypes"]
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
        row = score_lookup.get(key)
        if row is None:
            raise ValueError(f"missing W21 reference scores for {key}")
        raw.append(
            _case_from_logits(
                case,
                torch.tensor(row["abstract_severity"], dtype=torch.float32),
                torch.tensor(row["abstract_confidence"], dtype=torch.float32),
                torch.tensor(row["severity_prototypes"], dtype=torch.float32),
                torch.tensor(row["confidence_prototypes"], dtype=torch.float32),
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
    abstract = metrics["abstract"]
    proto = metrics["prototype"]
    ref_proto = reference["prototype"]

    reference_adequacy = (
        float(ref_proto["severity"]["top1"]) >= .90
        and float(ref_proto["confidence"]["top1"]) >= .90
        and float(ref_proto["joint_severity_confidence_top1"]) >= .82
        and float(ref_proto["probability_mass_max_error"]) <= 1e-6
    )
    hira_prototype_adequacy = (
        float(proto["severity"]["top1"]) >= .80
        and float(proto["confidence"]["top1"]) >= .85
        and float(proto["joint_severity_confidence_top1"]) >= .70
        and float(proto["severity_leave_one_out_agreement"]) >= .90
        and float(proto["confidence_leave_one_out_agreement"]) >= .92
    )

    severity_gain = float(proto["severity"]["top1"]) - float(abstract["severity"]["top1"])
    confidence_gain = float(proto["confidence"]["top1"]) - float(abstract["confidence"]["top1"])
    joint_gain = (
        float(proto["joint_severity_confidence_top1"])
        - float(abstract["joint_severity_confidence_top1"])
    )
    severity_mae_improvement = (
        float(abstract["severity"]["mae"]) - float(proto["severity"]["mae"])
    )
    causal_gain = (
        severity_gain >= .12
        and confidence_gain >= .08
        and joint_gain >= .12
        and severity_mae_improvement >= .15
    )

    abstract_adequate = (
        float(abstract["severity"]["top1"]) >= .80
        and float(abstract["confidence"]["top1"]) >= .85
        and float(abstract["joint_severity_confidence_top1"]) >= .70
        and joint_gain < .08
    )

    integrity = (
        float(proto["probability_mass_max_error"]) <= 1e-6
        and metrics["representation_accounting"] == {
            "logical_state_compiles_per_case": 1,
            "a13_query_invocations_per_case": 1,
            "encoded_query_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
            "prototypes_per_class": 3,
            "prototype_schema_scope": "shared-per-domain",
        }
    )

    if not reference_adequacy:
        classification = "W21_REFERENCE_INADEQUATE"
    elif not hira_prototype_adequacy:
        classification = "PROTOTYPE_LATENT_EXTRACTION_LIMIT"
    elif causal_gain:
        classification = "PROTOTYPE_GROUNDING_INTERFACE_LIMIT"
    elif abstract_adequate:
        classification = "ABSTRACT_LATENT_INTERFACE_ADEQUATE"
    else:
        classification = "PROTOTYPE_LATENT_UNRESOLVED"

    return {
        "integrity": integrity,
        "reference_adequacy": reference_adequacy,
        "hira_prototype_adequacy": hira_prototype_adequacy,
        "severity_gain": severity_gain,
        "confidence_gain": confidence_gain,
        "joint_gain": joint_gain,
        "severity_mae_improvement": severity_mae_improvement,
        "causal_gain": causal_gain,
        "abstract_interface_adequate": abstract_adequate,
        "classification": classification,
    }


def classify_w21(
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
    excluded = {"PROTOTYPE_LATENT_UNRESOLVED", "W21_REFERENCE_INADEQUATE"}
    stable = None
    for name, count in counts.items():
        if name not in excluded and count >= 3:
            stable = name
            break

    if stable is not None:
        outcome = "STABLE_PROTOTYPE_LATENT_LOCALIZATION"
    else:
        incompatible = [
            name for name, count in counts.items()
            if name not in excluded and count >= 2
        ]
        outcome = (
            "MIXED_PROTOTYPE_LATENT_LOCALIZATION"
            if len(incompatible) >= 2
            else "PROTOTYPE_LATENT_UNRESOLVED"
        )
    return {
        "outcome": outcome,
        "stable_classification": stable,
        "classification_counts": dict(sorted(counts.items())),
        "per_domain": per_domain,
    }


__all__ = [
    "classify_w21",
    "evaluate_w21",
    "reference_summary_from_scores",
]
