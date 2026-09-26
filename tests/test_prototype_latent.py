from __future__ import annotations

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.prototype_latent_authority import (
    DOMAINS_ORDER,
    all_w21_prototype_texts,
    all_w21_query_texts,
    all_w21_text_atoms,
    confidence_prototypes,
    generate_all_w21,
    severity_prototypes,
)
from nmd.prototype_latent_cache import compile_w21_cache
from nmd.prototype_latent_eval import classify_w21, evaluate_w21
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w21_authority_balance_and_prototype_contract():
    rows = generate_all_w21()
    assert len(rows) == 384
    assert {row.domain_id for row in rows} == set(DOMAINS_ORDER)
    assert len(all_w21_text_atoms()) > 180
    assert not (all_w21_query_texts() & all_w21_prototype_texts())
    for domain in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 96
        assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
        assert [sum(row.confidence_index == c for row in subset) for c in range(3)] == [32] * 3
        assert [len(x) for x in severity_prototypes(domain)] == [3] * 4
        assert [len(x) for x in confidence_prototypes(domain)] == [3] * 3


def test_w21_download_free_cache_and_evaluator_contract():
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    cache = compile_w21_cache(model, [generate_all_w21()[0]])
    assert cache["metadata"]["case_count"] == 1
    assert cache["metadata"]["a13_query_invocations_per_case"] == 1
    assert cache["metadata"]["encoded_query_sequences_per_case"] == 2
    assert cache["metadata"]["prototypes_per_class"] == 3
    assert cache["metadata"]["prototype_schema_scope"] == "shared-per-domain"
    assert set(cache["schemas"]) == set(DOMAINS_ORDER)
    for pack in cache["schemas"].values():
        assert len(pack["severity_prototypes"]["option_ids"]) == 12
        assert len(pack["confidence_prototypes"]["option_ids"]) == 9

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()
    result = evaluate_w21(scorer, cache)
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


def _metric(top1, mae=.2):
    return {
        "n": 96,
        "top1": top1,
        "mrr": .95,
        "mean_margin": .3,
        "mae": mae,
    }


def _summary(
    *,
    abstract_s=.66,
    abstract_c=.76,
    abstract_joint=.52,
    proto_s=.84,
    proto_c=.88,
    proto_joint=.73,
    s_agree=.94,
    c_agree=.95,
):
    return {
        "abstract": {
            "severity": _metric(abstract_s, .60),
            "confidence": _metric(abstract_c, .30),
            "joint_severity_confidence_top1": abstract_joint,
        },
        "prototype": {
            "severity": _metric(proto_s, .35),
            "confidence": _metric(proto_c, .18),
            "joint_severity_confidence_top1": proto_joint,
            "severity_leave_one_out_agreement": s_agree,
            "confidence_leave_one_out_agreement": c_agree,
            "severity_single_prototype_position": [_metric(.80)] * 3,
            "confidence_single_prototype_position": [_metric(.84)] * 3,
            "probability_mass_max_error": 0.0,
        },
        "transitions": {},
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_query_invocations_per_case": 1,
            "encoded_query_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
            "prototypes_per_class": 3,
            "prototype_schema_scope": "shared-per-domain",
        },
    }


def _reference(*, adequate=True):
    value = .95 if adequate else .85
    return {
        "prototype": {
            "severity": _metric(value),
            "confidence": _metric(value),
            "joint_severity_confidence_top1": .90 if adequate else .75,
            "probability_mass_max_error": 0.0,
        }
    }


def test_w21_classifier_stable_prototype_grounding_limit():
    metrics = {
        "per_domain": {
            "DD": _summary(),
            "DE": _summary(),
            "DF": _summary(),
            "DG": _summary(proto_joint=.60),
        }
    }
    reference = {"per_domain": {d: _reference() for d in metrics["per_domain"]}}
    out = classify_w21(metrics, reference)
    assert out["outcome"] == "STABLE_PROTOTYPE_LATENT_LOCALIZATION"
    assert out["stable_classification"] == "PROTOTYPE_GROUNDING_INTERFACE_LIMIT"
    assert out["classification_counts"]["PROTOTYPE_GROUNDING_INTERFACE_LIMIT"] == 3


def test_w21_reference_gate_precedes_hira_interpretation():
    metrics = {"per_domain": {"DD": _summary()}}
    reference = {"per_domain": {"DD": _reference(adequate=False)}}
    out = classify_w21(metrics, reference)
    assert out["per_domain"]["DD"]["classification"] == "W21_REFERENCE_INADEQUATE"


def test_w21_abstract_adequate_when_prototypes_do_not_materially_help():
    metrics = {
        "per_domain": {
            "DD": _summary(
                abstract_s=.82,
                abstract_c=.87,
                abstract_joint=.72,
                proto_s=.83,
                proto_c=.88,
                proto_joint=.77,
                s_agree=.95,
                c_agree=.96,
            )
        }
    }
    reference = {"per_domain": {"DD": _reference()}}
    out = classify_w21(metrics, reference)
    assert out["per_domain"]["DD"]["classification"] == "ABSTRACT_LATENT_INTERFACE_ADEQUATE"
