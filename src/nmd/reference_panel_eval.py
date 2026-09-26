from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Mapping

import torch
from torch import Tensor

from .anchor_preserving_residual_training import _margin, _rank_metrics
from .competitive import CompetitiveCoarseScorer
from .interface_decomposition_eval import _q0_symmetric
from .reference_panel_authority import PROTOTYPES_PER_CLASS, VIEW_IDS
from .reference_panel_cache import validate_w22_cache

REFERENCE_ORDER = ("minilm", "mpnet", "e5", "bge")
NONLEGACY_REFERENCES = ("mpnet", "e5", "bge")


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
        raise ValueError("W22 prototype width changed")
    return raw.reshape(classes, PROTOTYPES_PER_CLASS).mean(dim=1)


def _record(logits: Tensor, gold: int) -> dict[str, object]:
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("W22 requires finite logits")
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
def evaluate_w22(
    scorer: CompetitiveCoarseScorer,
    cache: dict,
) -> dict[str, object]:
    validate_w22_cache(cache)
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
            raise ValueError(f"missing W22 reference scores for {key}")
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
        "per_domain": {
            domain: _summarize(rows)
            for domain, rows in sorted(by_domain.items())
        },
        "pooled": _summarize(raw),
    }
    return {"summary": summary, "predictions": predictions}


def _individual_reference_pass(summary: dict[str, object]) -> bool:
    proto = summary["prototype"]
    return (
        float(proto["severity"]["top1"]) >= .80
        and float(proto["confidence"]["top1"]) >= .85
        and float(proto["joint_severity_confidence_top1"]) >= .68
        and float(proto["probability_mass_max_error"]) <= 1e-6
    )


def _nonlegacy_majority(values: Mapping[str, int]) -> int | None:
    counter = Counter(int(values[name]) for name in NONLEGACY_REFERENCES)
    top = counter.most_common()
    if not top or top[0][1] < 2:
        return None
    return int(top[0][0])


def _panel_majority(values: Mapping[str, int]) -> int | None:
    counter = Counter(int(values[name]) for name in REFERENCE_ORDER)
    top = counter.most_common()
    if top and (len(top) == 1 or top[0][1] > top[1][1]):
        return int(top[0][0])
    return _nonlegacy_majority(values)


def _pairwise_agreement(
    predictions: Mapping[str, Mapping[str, Mapping[str, object]]],
    case_ids: list[str],
    field: str,
) -> float:
    key = f"{field}_pred"
    values = []
    for a, b in combinations(NONLEGACY_REFERENCES, 2):
        if not case_ids:
            values.append(0.0)
            continue
        values.append(
            sum(
                int(predictions[a][case_id][key])
                == int(predictions[b][case_id][key])
                for case_id in case_ids
            )
            / len(case_ids)
        )
    return sum(values) / max(1, len(values))


def _panel_domain_metrics(
    domain: str,
    references: Mapping[str, Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    for name in REFERENCE_ORDER:
        if name not in references or name not in predictions:
            raise ValueError(f"W22 missing frozen reference {name}")

    case_ids = sorted(
        case_id
        for case_id, row in predictions[REFERENCE_ORDER[0]].items()
        if str(row["domain_id"]) == domain
    )
    if not case_ids:
        raise ValueError(f"W22 no panel predictions for {domain}")

    individual = {
        name: _individual_reference_pass(
            references[name]["per_domain"][domain]
        )
        for name in REFERENCE_ORDER
    }

    severity_correct = 0
    confidence_correct = 0
    joint_correct = 0
    undefined_severity = 0
    undefined_confidence = 0

    for case_id in case_ids:
        base = predictions[REFERENCE_ORDER[0]][case_id]
        s_values = {
            name: int(predictions[name][case_id]["severity_pred"])
            for name in REFERENCE_ORDER
        }
        c_values = {
            name: int(predictions[name][case_id]["confidence_pred"])
            for name in REFERENCE_ORDER
        }
        s_pred = _panel_majority(s_values)
        c_pred = _panel_majority(c_values)
        if s_pred is None:
            undefined_severity += 1
        if c_pred is None:
            undefined_confidence += 1
        s_ok = s_pred is not None and s_pred == int(base["severity_gold"])
        c_ok = c_pred is not None and c_pred == int(base["confidence_gold"])
        severity_correct += int(s_ok)
        confidence_correct += int(c_ok)
        joint_correct += int(s_ok and c_ok)

    n = len(case_ids)
    majority_severity = severity_correct / n
    majority_confidence = confidence_correct / n
    majority_joint = joint_correct / n
    severity_agreement = _pairwise_agreement(
        predictions, case_ids, "severity"
    )
    confidence_agreement = _pairwise_agreement(
        predictions, case_ids, "confidence"
    )
    panel_adequate = (
        sum(individual.values()) >= 3
        and sum(individual[name] for name in NONLEGACY_REFERENCES) >= 2
        and majority_severity >= .85
        and majority_confidence >= .88
        and majority_joint >= .75
        and severity_agreement >= .82
        and confidence_agreement >= .85
    )

    return {
        "case_count": n,
        "individual_reference_pass": individual,
        "reference_pass_count": sum(individual.values()),
        "nonlegacy_reference_pass_count": sum(
            individual[name] for name in NONLEGACY_REFERENCES
        ),
        "majority_severity_top1": majority_severity,
        "majority_confidence_top1": majority_confidence,
        "majority_joint_top1": majority_joint,
        "nonlegacy_pairwise_severity_agreement": severity_agreement,
        "nonlegacy_pairwise_confidence_agreement": confidence_agreement,
        "undefined_majority_severity_count": undefined_severity,
        "undefined_majority_confidence_count": undefined_confidence,
        "panel_adequate": panel_adequate,
        "legacy_minilm_reference_limit": (
            panel_adequate and not individual["minilm"]
        ),
    }


def classify_w22(
    metrics: dict[str, object],
    references: Mapping[str, Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    domains = sorted(metrics["per_domain"])
    per_domain = {}
    for domain in domains:
        panel = _panel_domain_metrics(domain, references, predictions)
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

        if not bool(panel["panel_adequate"]):
            classification = "W22_REFERENCE_PANEL_INADEQUATE"
        elif hira_adequate:
            classification = "HIRA_LATENT_AUTHORITY_ADEQUATE"
        else:
            classification = "HIRA_LATENT_GEOMETRY_LIMIT"

        per_domain[domain] = {
            **panel,
            "integrity": integrity,
            "hira_adequate": hira_adequate,
            "classification": classification,
        }

    counts = Counter(row["classification"] for row in per_domain.values())
    stable_hira = None
    for name in ("HIRA_LATENT_AUTHORITY_ADEQUATE", "HIRA_LATENT_GEOMETRY_LIMIT"):
        if counts[name] >= 3:
            stable_hira = name
            break

    panel_adequate_count = sum(
        bool(row["panel_adequate"]) for row in per_domain.values()
    )
    legacy_limit_count = sum(
        bool(row["legacy_minilm_reference_limit"]) for row in per_domain.values()
    )
    panel_inadequate_count = sum(
        not bool(row["panel_adequate"]) for row in per_domain.values()
    )

    if stable_hira is not None:
        outcome = "STABLE_REFERENCE_PANEL_LOCALIZATION"
    elif panel_adequate_count >= 3 and legacy_limit_count >= 3:
        outcome = "STABLE_LEGACY_REFERENCE_LIMIT"
    elif panel_inadequate_count >= 3:
        outcome = "AUTHORITY_REFERENCE_UNRESOLVED"
    else:
        outcome = "REFERENCE_PANEL_MIXED"

    return {
        "outcome": outcome,
        "stable_classification": stable_hira,
        "classification_counts": dict(sorted(counts.items())),
        "panel_adequate_domain_count": panel_adequate_count,
        "legacy_minilm_limit_domain_count": legacy_limit_count,
        "per_domain": per_domain,
    }


__all__ = [
    "NONLEGACY_REFERENCES",
    "REFERENCE_ORDER",
    "classify_w22",
    "evaluate_w22",
    "reference_evaluation_from_scores",
]
