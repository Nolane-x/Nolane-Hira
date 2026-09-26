from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .competitive import CompetitiveCoarseScorer
from .cross_encoder_authority import PROTOTYPES_PER_CLASS, VIEW_IDS
from .cross_encoder_cache import validate_w23_cache
from .interface_decomposition_eval import _q0_symmetric

CROSS_ENCODER_ORDER = ("deberta_nli", "roberta_nli")
BIENCODER_ORDER = ("minilm", "mpnet", "e5", "bge")


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
        raise ValueError("W23 prototype width changed")
    return raw.reshape(classes, PROTOTYPES_PER_CLASS).mean(dim=1)


def _record(logits: Tensor, gold: int) -> dict[str, object]:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W23 requires finite logits")
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
            "probability_mass_max_error": max(
                [float(row["probability_mass_error"]) for row in raw] or [0.0]
            ),
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
def evaluate_w23(
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w23_cache(cache)
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
                _prototype_raw_logits(scorer, severity_tokens, schema["severity_prototypes"]),
                _prototype_raw_logits(scorer, confidence_tokens, schema["confidence_prototypes"]),
            )
        )

    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    return {
        "per_domain": {domain: _summarize(rows) for domain, rows in sorted(by_domain.items())},
        "pooled": _summarize(raw),
    }


def reference_evaluation_from_scores(
    cases: list[dict],
    score_lookup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    raw = []
    predictions: dict[str, dict[str, object]] = {}
    for case in cases:
        key = str(case["case_id"])
        row = score_lookup.get(key)
        if row is None:
            raise ValueError(f"missing W23 reference scores for {key}")
        record = _case_from_logits(
            case,
            torch.tensor(row["abstract_severity"], dtype=torch.float32),
            torch.tensor(row["abstract_confidence"], dtype=torch.float32),
            torch.tensor(row["severity_prototypes"], dtype=torch.float32),
            torch.tensor(row["confidence_prototypes"], dtype=torch.float32),
        )
        raw.append(record)
        predictions[key] = {
            "domain_id": str(record["domain_id"]),
            "severity_gold": int(record["severity_gold"]),
            "confidence_gold": int(record["confidence_gold"]),
            "severity_pred": int(record["prototype_severity"]["pred"]),
            "confidence_pred": int(record["prototype_confidence"]["pred"]),
        }

    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    summary = {
        "per_domain": {domain: _summarize(rows) for domain, rows in sorted(by_domain.items())},
        "pooled": _summarize(raw),
    }
    return {"summary": summary, "predictions": predictions}



def prototype_reference_evaluation_from_scores(
    cases: list[dict],
    score_lookup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    raw = []
    predictions: dict[str, dict[str, object]] = {}
    for case in cases:
        key = str(case["case_id"])
        row = score_lookup.get(key)
        if row is None:
            raise ValueError(f"missing W23 prototype reference scores for {key}")
        severity_gold = int(case["severity"])
        confidence_gold = int(case["confidence_index"])
        severity_raw = torch.tensor(row["severity_prototypes"], dtype=torch.float32)
        confidence_raw = torch.tensor(row["confidence_prototypes"], dtype=torch.float32)
        severity_primary = _aggregate_prototypes(severity_raw, 4)
        confidence_primary = _aggregate_prototypes(confidence_raw, 3)
        severity_record = _record(severity_primary, severity_gold)
        confidence_record = _record(confidence_primary, confidence_gold)
        s_probs = torch.softmax(severity_primary, dim=0)
        c_probs = torch.softmax(confidence_primary, dim=0)
        record = {
            "case_id": key,
            "domain_id": str(case["domain_id"]),
            "severity_gold": severity_gold,
            "confidence_gold": confidence_gold,
            "prototype_severity": severity_record,
            "prototype_confidence": confidence_record,
            "probability_mass_error": max(
                abs(float(s_probs.sum()) - 1.0),
                abs(float(c_probs.sum()) - 1.0),
            ),
        }
        raw.append(record)
        predictions[key] = {
            "domain_id": str(case["domain_id"]),
            "severity_gold": severity_gold,
            "confidence_gold": confidence_gold,
            "severity_pred": int(severity_record["pred"]),
            "confidence_pred": int(confidence_record["pred"]),
        }

    def summarize(rows):
        n = len(rows)
        severity = [row["prototype_severity"] for row in rows]
        confidence = [row["prototype_confidence"] for row in rows]
        joint = sum(
            bool(s["correct"]) and bool(c["correct"])
            for s, c in zip(severity, confidence)
        ) / max(1, n)
        return {
            "case_count": n,
            "prototype": {
                "severity": _summary(severity),
                "confidence": _summary(confidence),
                "joint_severity_confidence_top1": joint,
                "probability_mass_max_error": max(
                    [float(row["probability_mass_error"]) for row in rows] or [0.0]
                ),
            },
        }

    by_domain = defaultdict(list)
    for row in raw:
        by_domain[str(row["domain_id"])].append(row)
    return {
        "summary": {
            "per_domain": {
                domain: summarize(rows)
                for domain, rows in sorted(by_domain.items())
            },
            "pooled": summarize(raw),
        },
        "predictions": predictions,
    }

def _individual_reference_pass(summary: Mapping[str, object]) -> bool:
    proto = summary["prototype"]
    return (
        float(proto["severity"]["top1"]) >= .80
        and float(proto["confidence"]["top1"]) >= .85
        and float(proto["joint_severity_confidence_top1"]) >= .68
        and float(proto["probability_mass_max_error"]) <= 1e-6
    )


def _domain_case_ids(
    predictions: Mapping[str, Mapping[str, Mapping[str, object]]],
    reference_name: str,
    domain: str,
) -> list[str]:
    return sorted(
        case_id
        for case_id, row in predictions[reference_name].items()
        if str(row["domain_id"]) == domain
    )


def _cross_encoder_domain(
    domain: str,
    references: Mapping[str, Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    for name in CROSS_ENCODER_ORDER:
        if name not in references or name not in predictions:
            raise ValueError(f"W23 missing frozen cross-encoder {name}")

    case_ids = _domain_case_ids(predictions, CROSS_ENCODER_ORDER[0], domain)
    if not case_ids:
        raise ValueError(f"W23 no cross-encoder predictions for {domain}")

    individual = {
        name: _individual_reference_pass(references[name]["per_domain"][domain])
        for name in CROSS_ENCODER_ORDER
    }

    severity_agree = 0
    confidence_agree = 0
    severity_correct = 0
    confidence_correct = 0
    joint_correct = 0
    severity_undefined = 0
    confidence_undefined = 0

    for case_id in case_ids:
        a = predictions[CROSS_ENCODER_ORDER[0]][case_id]
        b = predictions[CROSS_ENCODER_ORDER[1]][case_id]
        s_same = int(a["severity_pred"]) == int(b["severity_pred"])
        c_same = int(a["confidence_pred"]) == int(b["confidence_pred"])
        severity_agree += int(s_same)
        confidence_agree += int(c_same)
        if not s_same:
            severity_undefined += 1
        if not c_same:
            confidence_undefined += 1
        s_ok = s_same and int(a["severity_pred"]) == int(a["severity_gold"])
        c_ok = c_same and int(a["confidence_pred"]) == int(a["confidence_gold"])
        severity_correct += int(s_ok)
        confidence_correct += int(c_ok)
        joint_correct += int(s_ok and c_ok)

    n = len(case_ids)
    severity_agreement = severity_agree / n
    confidence_agreement = confidence_agree / n
    consensus_severity = severity_correct / n
    consensus_confidence = confidence_correct / n
    consensus_joint = joint_correct / n
    adequate = (
        all(individual.values())
        and severity_agreement >= .85
        and confidence_agreement >= .90
        and consensus_severity >= .82
        and consensus_confidence >= .87
        and consensus_joint >= .72
    )
    return {
        "case_count": n,
        "individual_cross_encoder_pass": individual,
        "severity_prediction_agreement": severity_agreement,
        "confidence_prediction_agreement": confidence_agreement,
        "consensus_severity_top1": consensus_severity,
        "consensus_confidence_top1": consensus_confidence,
        "consensus_joint_top1": consensus_joint,
        "undefined_consensus_severity_count": severity_undefined,
        "undefined_consensus_confidence_count": confidence_undefined,
        "cross_encoder_authority_adequate": adequate,
    }


def _biencoder_domain(
    domain: str,
    references: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    individual = {}
    for name in BIENCODER_ORDER:
        if name not in references:
            raise ValueError(f"W23 missing frozen bi-encoder control {name}")
        individual[name] = _individual_reference_pass(
            references[name]["per_domain"][domain]
        )
    return {
        "individual_biencoder_pass": individual,
        "biencoder_pass_count": sum(individual.values()),
    }


def classify_w23(
    metrics: dict[str, object],
    cross_references: Mapping[str, Mapping[str, object]],
    cross_predictions: Mapping[str, Mapping[str, Mapping[str, object]]],
    biencoder_references: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    domains = sorted(metrics["per_domain"])
    per_domain = {}

    for domain in domains:
        cross = _cross_encoder_domain(domain, cross_references, cross_predictions)
        bi = _biencoder_domain(domain, biencoder_references)
        proto = metrics["per_domain"][domain]["prototype"]
        hira_adequate = (
            float(proto["severity"]["top1"]) >= .75
            and float(proto["confidence"]["top1"]) >= .80
            and float(proto["joint_severity_confidence_top1"]) >= .62
            and float(proto["probability_mass_max_error"]) <= 1e-6
        )
        integrity = (
            float(proto["probability_mass_max_error"]) <= 1e-6
            and metrics["per_domain"][domain]["representation_accounting"] == {
                "logical_state_compiles_per_case": 1,
                "a13_query_invocations_per_case": 1,
                "encoded_query_sequences_per_case": 2,
                "isolated_fields": ["severity", "confidence"],
                "prototypes_per_class": 3,
                "prototype_schema_scope": "shared-per-domain",
            }
        )

        authority_adequate = bool(cross["cross_encoder_authority_adequate"])
        biencoder_limit = authority_adequate and int(bi["biencoder_pass_count"]) < 2

        if not authority_adequate:
            classification = "W23_CROSS_ENCODER_REFERENCE_INADEQUATE"
        elif hira_adequate:
            classification = "HIRA_LATENT_AUTHORITY_ADEQUATE"
        else:
            classification = "HIRA_LATENT_GEOMETRY_LIMIT"

        per_domain[domain] = {
            **cross,
            **bi,
            "biencoder_reference_family_limit": biencoder_limit,
            "hira_adequate": hira_adequate,
            "integrity": integrity,
            "classification": classification,
        }

    counts = Counter(row["classification"] for row in per_domain.values())
    stable_hira = None
    for name in ("HIRA_LATENT_AUTHORITY_ADEQUATE", "HIRA_LATENT_GEOMETRY_LIMIT"):
        if counts[name] >= 3:
            stable_hira = name
            break

    authority_adequate_count = sum(
        bool(row["cross_encoder_authority_adequate"]) for row in per_domain.values()
    )
    biencoder_limit_count = sum(
        bool(row["biencoder_reference_family_limit"]) for row in per_domain.values()
    )
    authority_inadequate_count = len(per_domain) - authority_adequate_count

    if stable_hira is not None:
        outcome = "STABLE_CROSS_ENCODER_LOCALIZATION"
    elif authority_adequate_count >= 3 and biencoder_limit_count >= 3:
        outcome = "STABLE_BIENCODER_REFERENCE_FAMILY_LIMIT"
    elif authority_inadequate_count >= 3:
        outcome = "AUTHORITY_REFERENCE_UNRESOLVED"
    else:
        outcome = "CROSS_ENCODER_REFERENCE_MIXED"

    return {
        "outcome": outcome,
        "stable_classification": stable_hira,
        "classification_counts": dict(sorted(counts.items())),
        "cross_encoder_adequate_domain_count": authority_adequate_count,
        "biencoder_family_limit_domain_count": biencoder_limit_count,
        "per_domain": per_domain,
    }


__all__ = [
    "BIENCODER_ORDER",
    "CROSS_ENCODER_ORDER",
    "classify_w23",
    "evaluate_w23",
    "prototype_reference_evaluation_from_scores",
    "reference_evaluation_from_scores",
]
