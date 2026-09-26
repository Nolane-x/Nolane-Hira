from __future__ import annotations

from nmd.competitive import CompetitiveCoarseScorer
from nmd.cross_encoder_authority import (
    DOMAINS_ORDER,
    all_w23_prototype_texts,
    all_w23_query_texts,
    all_w23_text_atoms,
    confidence_prototypes,
    generate_all_w23,
    severity_prototypes,
)
from nmd.cross_encoder_cache import compile_w23_cache
from nmd.cross_encoder_eval import classify_w23, evaluate_w23
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w23_authority_balance_and_fresh_prototype_contract():
    rows = generate_all_w23()
    assert len(rows) == 384
    assert {row.domain_id for row in rows} == set(DOMAINS_ORDER)
    assert len(all_w23_text_atoms()) > 180
    assert not (all_w23_query_texts() & all_w23_prototype_texts())
    for domain in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 96
        assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
        assert [sum(row.confidence_index == c for row in subset) for c in range(3)] == [32] * 3
        assert [len(x) for x in severity_prototypes(domain)] == [3] * 4
        assert [len(x) for x in confidence_prototypes(domain)] == [3] * 3


def test_w23_download_free_cache_and_hira_evaluator_contract():
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    cache = compile_w23_cache(model, [generate_all_w23()[0]])
    assert cache["metadata"]["case_count"] == 1
    assert cache["metadata"]["a13_query_invocations_per_case"] == 1
    assert cache["metadata"]["encoded_query_sequences_per_case"] == 2
    assert cache["metadata"]["prototypes_per_class"] == 3
    assert set(cache["schemas"]) == set(DOMAINS_ORDER)

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()
    result = evaluate_w23(scorer, cache)
    pooled = result["pooled"]
    assert pooled["prototype"]["probability_mass_max_error"] <= 1e-6
    assert pooled["representation_accounting"] == {
        "logical_state_compiles_per_case": 1,
        "a13_query_invocations_per_case": 1,
        "encoded_query_sequences_per_case": 2,
        "isolated_fields": ["severity", "confidence"],
        "prototypes_per_class": 3,
        "prototype_schema_scope": "shared-per-domain",
    }


def _metric(top1: float, mae: float = .2):
    return {
        "n": 96,
        "top1": top1,
        "mrr": .95,
        "mean_margin": .3,
        "mae": mae,
    }


def _summary(*, severity=.90, confidence=.92, joint=.84):
    return {
        "case_count": 96,
        "abstract": {
            "severity": _metric(.50),
            "confidence": _metric(.60),
            "joint_severity_confidence_top1": .30,
        },
        "prototype": {
            "severity": _metric(severity),
            "confidence": _metric(confidence),
            "joint_severity_confidence_top1": joint,
            "probability_mass_max_error": 0.0,
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


def _refs(domains, *, passing=True, names=("deberta_nli", "roberta_nli")):
    values = (0.90, 0.92, 0.84) if passing else (0.65, 0.72, 0.48)
    return {
        name: {
            "per_domain": {
                domain: _summary(
                    severity=values[0],
                    confidence=values[1],
                    joint=values[2],
                )
                for domain in domains
            }
        }
        for name in names
    }


def _biencoder_refs(domains, *, passing=False):
    return _refs(
        domains,
        passing=passing,
        names=("minilm", "mpnet", "e5", "bge"),
    )


def _cross_predictions(domains, *, perfect=True):
    result = {name: {} for name in ("deberta_nli", "roberta_nli")}
    for domain in domains:
        for i in range(24):
            case_id = f"{domain.lower()}-{i}"
            s_gold = i % 4
            c_gold = i % 3
            for name in result:
                result[name][case_id] = {
                    "domain_id": domain,
                    "severity_gold": s_gold,
                    "confidence_gold": c_gold,
                    "severity_pred": s_gold if perfect else (s_gold + 1) % 4,
                    "confidence_pred": c_gold if perfect else (c_gold + 1) % 3,
                }
    return result


def test_w23_cross_encoder_gate_precedes_hira_interpretation():
    domains = ["DL"]
    metrics = {"per_domain": {"DL": _summary(severity=.9, confidence=.9, joint=.8)}}
    cross_refs = _refs(domains, passing=False)
    cross_preds = _cross_predictions(domains)
    bi_refs = _biencoder_refs(domains, passing=False)
    out = classify_w23(metrics, cross_refs, cross_preds, bi_refs)
    row = out["per_domain"]["DL"]
    assert row["cross_encoder_authority_adequate"] is False
    assert row["classification"] == "W23_CROSS_ENCODER_REFERENCE_INADEQUATE"


def test_w23_cross_encoder_authority_can_localize_hira_geometry_limit():
    domains = ["DL", "DM", "DN", "DO"]
    metrics = {
        "per_domain": {
            d: _summary(severity=.68, confidence=.74, joint=.50)
            for d in domains
        }
    }
    cross_refs = _refs(domains, passing=True)
    cross_preds = _cross_predictions(domains, perfect=True)
    bi_refs = _biencoder_refs(domains, passing=False)
    out = classify_w23(metrics, cross_refs, cross_preds, bi_refs)
    assert out["outcome"] == "STABLE_CROSS_ENCODER_LOCALIZATION"
    assert out["stable_classification"] == "HIRA_LATENT_GEOMETRY_LIMIT"
    assert out["classification_counts"]["HIRA_LATENT_GEOMETRY_LIMIT"] == 4
    assert out["biencoder_family_limit_domain_count"] == 4


def test_w23_biencoder_family_limit_is_orthogonal_to_mixed_hira_result():
    domains = ["DL", "DM", "DN", "DO"]
    metrics = {
        "per_domain": {
            "DL": _summary(severity=.80, confidence=.85, joint=.66),
            "DM": _summary(severity=.80, confidence=.85, joint=.66),
            "DN": _summary(severity=.68, confidence=.74, joint=.50),
            "DO": _summary(severity=.68, confidence=.74, joint=.50),
        }
    }
    cross_refs = _refs(domains, passing=True)
    cross_preds = _cross_predictions(domains, perfect=True)
    bi_refs = _biencoder_refs(domains, passing=False)
    out = classify_w23(metrics, cross_refs, cross_preds, bi_refs)
    assert out["cross_encoder_adequate_domain_count"] == 4
    assert out["biencoder_family_limit_domain_count"] == 4
    assert out["stable_classification"] is None
    assert out["outcome"] == "STABLE_BIENCODER_REFERENCE_FAMILY_LIMIT"
