from __future__ import annotations

from nmd.atomic_severity_authority import (
    DOMAINS_ORDER,
    FACTOR_IDS,
    all_w24_prototype_texts,
    all_w24_query_texts,
    all_w24_text_atoms,
    compose_severity,
    generate_all_w24,
    severity_factors,
)
from nmd.atomic_severity_cache import compile_w24_cache
from nmd.atomic_severity_eval import classify_w24, evaluate_w24
from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w24_authority_balance_factor_chain_and_fresh_contract():
    rows = generate_all_w24()
    assert len(rows) == 384
    assert {row.domain_id for row in rows} == set(DOMAINS_ORDER)
    assert len(all_w24_text_atoms()) > 200
    assert not (all_w24_query_texts() & all_w24_prototype_texts())
    assert severity_factors(0) == (0, 0, 0)
    assert severity_factors(1) == (1, 0, 0)
    assert severity_factors(2) == (1, 1, 0)
    assert severity_factors(3) == (1, 1, 1)
    assert compose_severity((0, 0, 0)) == 0
    assert compose_severity((1, 0, 0)) == 1
    assert compose_severity((1, 1, 0)) == 2
    assert compose_severity((1, 1, 1)) == 3
    assert compose_severity((0, 1, 0)) is None
    for domain in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 96
        assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
        assert [sum(row.confidence_index == c for row in subset) for c in range(3)] == [32] * 3


def test_w24_download_free_cache_and_hira_evaluator_contract():
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    cache = compile_w24_cache(model, [generate_all_w24()[0]])
    assert cache["metadata"]["case_count"] == 1
    assert cache["metadata"]["a13_query_invocations_per_case"] == 1
    assert cache["metadata"]["encoded_query_sequences_per_case"] == 2
    assert cache["metadata"]["factor_ids"] == ["F0", "F1", "F2"]
    assert set(cache["schemas"]) == set(DOMAINS_ORDER)

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()
    result = evaluate_w24(scorer, cache)
    pooled = result["pooled"]
    assert pooled["factor_probability_mass_max_error"] <= 1e-6
    assert pooled["representation_accounting"] == {
        "logical_state_compiles_per_case": 1,
        "a13_query_invocations_per_case": 1,
        "encoded_query_sequences_per_case": 2,
        "isolated_fields": ["severity", "confidence"],
        "factor_ids": ["F0", "F1", "F2"],
        "factor_decoder": {"000": 0, "100": 1, "110": 2, "111": 3},
    }


def _factor_metric(top1=.95, balanced=.95):
    return {
        "n": 96,
        "top1": top1,
        "mrr": .97,
        "mean_margin": .4,
        "mae": 1.0 - top1,
        "balanced_accuracy": balanced,
    }


def _atomic_reference_domain(*, passing=True):
    if passing:
        top1, balanced, vector, composed, invalid = .96, .95, .90, .90, .02
    else:
        top1, balanced, vector, composed, invalid = .70, .70, .50, .55, .12
    return {
        "case_count": 96,
        "factors": {
            factor_id: _factor_metric(top1, balanced)
            for factor_id in FACTOR_IDS
        },
        "factor_vector_top1": vector,
        "invalid_factor_vector_rate": invalid,
        "composed_severity_top1": composed,
        "composed_severity_mae": .2,
        "factor_probability_mass_max_error": 0.0,
    }


def _hira_domain(*, atomic_good=True, direct=.45, gain=.20):
    if atomic_good:
        ftop, vector, composed, invalid = .85, .70, direct + gain, .08
    else:
        ftop, vector, composed, invalid = .65, .40, .45, .25
    return {
        "case_count": 96,
        "direct_severity": {
            "n": 96,
            "top1": direct,
            "mrr": .7,
            "mean_margin": .1,
            "mae": .6,
        },
        "confidence": {
            "n": 96,
            "top1": .75,
            "mrr": .85,
            "mean_margin": .2,
            "mae": .3,
        },
        "factors": {
            factor_id: _factor_metric(ftop, ftop)
            for factor_id in FACTOR_IDS
        },
        "factor_vector_top1": vector,
        "invalid_factor_vector_rate": invalid,
        "composed_severity_top1": composed,
        "composed_severity_mae": .3,
        "factor_probability_mass_max_error": 0.0,
        "factorization_gain": gain if atomic_good else composed - direct,
        "direct_to_composed_wrong_to_right": 20,
        "direct_to_composed_right_to_wrong": 2,
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_query_invocations_per_case": 1,
            "encoded_query_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
            "factor_ids": ["F0", "F1", "F2"],
            "factor_decoder": {"000": 0, "100": 1, "110": 2, "111": 3},
        },
    }


def _control_summary(domains, *, deberta=.98, roberta=.94):
    def row(value):
        return {
            "case_count": 96,
            "prototype": {
                "severity": {"top1": .6},
                "confidence": {"top1": value},
                "joint_severity_confidence_top1": .55,
                "probability_mass_max_error": 0.0,
            },
        }
    return {
        "deberta_nli": {"per_domain": {d: row(deberta) for d in domains}},
        "roberta_nli": {"per_domain": {d: row(roberta) for d in domains}},
    }


def _control_predictions(domains, *, agree=True):
    out = {"deberta_nli": {}, "roberta_nli": {}}
    for domain in domains:
        for index in range(24):
            case_id = f"{domain.lower()}-{index}"
            gold = index % 3
            out["deberta_nli"][case_id] = {
                "domain_id": domain,
                "confidence_gold": gold,
                "confidence_pred": gold,
                "severity_gold": index % 4,
                "severity_pred": index % 4,
            }
            out["roberta_nli"][case_id] = {
                "domain_id": domain,
                "confidence_gold": gold,
                "confidence_pred": gold if agree else (gold + 1) % 3,
                "severity_gold": index % 4,
                "severity_pred": index % 4,
            }
    return out


def test_w24_reference_environment_precedes_atomic_interpretation():
    domains = ["DP"]
    metrics = {"per_domain": {"DP": _hira_domain()}}
    atomic = {"per_domain": {"DP": _atomic_reference_domain(passing=True)}}
    controls = _control_summary(domains, deberta=.80, roberta=.80)
    preds = _control_predictions(domains)
    out = classify_w24(metrics, atomic, controls, preds)
    assert out["per_domain"]["DP"]["classification"] == "W24_REFERENCE_ENVIRONMENT_UNSTABLE"


def test_w24_atomic_reference_gate_precedes_hira_interpretation():
    domains = ["DP"]
    metrics = {"per_domain": {"DP": _hira_domain()}}
    atomic = {"per_domain": {"DP": _atomic_reference_domain(passing=False)}}
    controls = _control_summary(domains)
    preds = _control_predictions(domains)
    out = classify_w24(metrics, atomic, controls, preds)
    assert out["per_domain"]["DP"]["classification"] == "W24_ATOMIC_REFERENCE_INADEQUATE"


def test_w24_can_localize_atomic_interface_limit():
    domains = list(DOMAINS_ORDER)
    metrics = {
        "per_domain": {d: _hira_domain(atomic_good=True, direct=.45, gain=.20) for d in domains}
    }
    atomic = {
        "per_domain": {d: _atomic_reference_domain(passing=True) for d in domains}
    }
    controls = _control_summary(domains)
    preds = _control_predictions(domains)
    out = classify_w24(metrics, atomic, controls, preds)
    assert out["outcome"] == "STABLE_ATOMIC_SEVERITY_LOCALIZATION"
    assert out["stable_classification"] == "ATOMIC_SEVERITY_INTERFACE_LIMIT"
    assert out["classification_counts"]["ATOMIC_SEVERITY_INTERFACE_LIMIT"] == 4


def test_w24_can_localize_hira_atomic_geometry_limit():
    domains = list(DOMAINS_ORDER)
    metrics = {
        "per_domain": {d: _hira_domain(atomic_good=False, direct=.45, gain=.0) for d in domains}
    }
    atomic = {
        "per_domain": {d: _atomic_reference_domain(passing=True) for d in domains}
    }
    controls = _control_summary(domains)
    preds = _control_predictions(domains)
    out = classify_w24(metrics, atomic, controls, preds)
    assert out["outcome"] == "STABLE_ATOMIC_SEVERITY_LOCALIZATION"
    assert out["stable_classification"] == "HIRA_ATOMIC_SEVERITY_GEOMETRY_LIMIT"
