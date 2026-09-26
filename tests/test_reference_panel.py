from __future__ import annotations

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.reference_panel_authority import (
    DOMAINS_ORDER,
    all_w22_prototype_texts,
    all_w22_query_texts,
    all_w22_text_atoms,
    confidence_prototypes,
    generate_all_w22,
    severity_prototypes,
)
from nmd.reference_panel_cache import compile_w22_cache
from nmd.reference_panel_eval import classify_w22, evaluate_w22
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w22_authority_balance_and_fresh_prototype_contract():
    rows = generate_all_w22()
    assert len(rows) == 384
    assert {row.domain_id for row in rows} == set(DOMAINS_ORDER)
    assert len(all_w22_text_atoms()) > 180
    assert not (all_w22_query_texts() & all_w22_prototype_texts())
    for domain in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 96
        assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
        assert [sum(row.confidence_index == c for row in subset) for c in range(3)] == [32] * 3
        assert [len(x) for x in severity_prototypes(domain)] == [3] * 4
        assert [len(x) for x in confidence_prototypes(domain)] == [3] * 3


def test_w22_download_free_cache_and_hira_evaluator_contract():
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    cache = compile_w22_cache(model, [generate_all_w22()[0]])
    assert cache["metadata"]["case_count"] == 1
    assert cache["metadata"]["a13_query_invocations_per_case"] == 1
    assert cache["metadata"]["encoded_query_sequences_per_case"] == 2
    assert cache["metadata"]["prototypes_per_class"] == 3
    assert set(cache["schemas"]) == set(DOMAINS_ORDER)

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()
    result = evaluate_w22(scorer, cache)
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


def _hira_summary(*, severity=.78, confidence=.83, joint=.65):
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


def _reference_domain(*, pass_gate=True):
    if pass_gate:
        severity, confidence, joint = .90, .92, .84
    else:
        severity, confidence, joint = .65, .72, .48
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
        "representation_accounting": {},
    }


def _panel_predictions(domains, *, perfect=True):
    result = {name: {} for name in ("minilm", "mpnet", "e5", "bge")}
    for domain in domains:
        for i in range(12):
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


def _references(domains, *, minilm=True, others=True):
    return {
        "minilm": {
            "per_domain": {d: _reference_domain(pass_gate=minilm) for d in domains}
        },
        "mpnet": {
            "per_domain": {d: _reference_domain(pass_gate=others) for d in domains}
        },
        "e5": {
            "per_domain": {d: _reference_domain(pass_gate=others) for d in domains}
        },
        "bge": {
            "per_domain": {d: _reference_domain(pass_gate=others) for d in domains}
        },
    }


def test_w22_reference_panel_gate_precedes_hira_interpretation():
    domains = ["DH"]
    metrics = {"per_domain": {"DH": _hira_summary()}}
    refs = _references(domains, minilm=False, others=False)
    preds = _panel_predictions(domains)
    out = classify_w22(metrics, refs, preds)
    row = out["per_domain"]["DH"]
    assert row["panel_adequate"] is False
    assert row["classification"] == "W22_REFERENCE_PANEL_INADEQUATE"


def test_w22_panel_adequate_can_localize_hira_geometry_limit():
    domains = ["DH", "DI", "DJ", "DK"]
    metrics = {
        "per_domain": {
            d: _hira_summary(severity=.70, confidence=.75, joint=.55)
            for d in domains
        }
    }
    refs = _references(domains)
    preds = _panel_predictions(domains)
    out = classify_w22(metrics, refs, preds)
    assert out["outcome"] == "STABLE_REFERENCE_PANEL_LOCALIZATION"
    assert out["stable_classification"] == "HIRA_LATENT_GEOMETRY_LIMIT"
    assert out["classification_counts"]["HIRA_LATENT_GEOMETRY_LIMIT"] == 4


def test_w22_legacy_minilm_limit_is_orthogonal_to_mixed_hira_result():
    domains = ["DH", "DI", "DJ", "DK"]
    metrics = {
        "per_domain": {
            "DH": _hira_summary(severity=.80, confidence=.85, joint=.66),
            "DI": _hira_summary(severity=.80, confidence=.85, joint=.66),
            "DJ": _hira_summary(severity=.68, confidence=.74, joint=.50),
            "DK": _hira_summary(severity=.68, confidence=.74, joint=.50),
        }
    }
    refs = _references(domains, minilm=False, others=True)
    preds = _panel_predictions(domains)
    out = classify_w22(metrics, refs, preds)
    assert out["panel_adequate_domain_count"] == 4
    assert out["legacy_minilm_limit_domain_count"] == 4
    assert out["stable_classification"] is None
    assert out["outcome"] == "STABLE_LEGACY_REFERENCE_LIMIT"
    assert all(
        row["legacy_minilm_reference_limit"]
        for row in out["per_domain"].values()
    )
