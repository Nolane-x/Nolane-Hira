from __future__ import annotations

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.latent_composition_authority import (
    CONFIDENCE_KEYS,
    DOMAINS_ORDER,
    generate_all_w18,
)
from nmd.latent_composition_cache import compile_w18_cache
from nmd.latent_composition_eval import (
    DIRECT_FIELDS,
    PATHS,
    classify_w18,
    evaluate_w18,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w18_authority_shape_and_field_separation():
    rows = generate_all_w18()
    assert len(rows) == 384
    assert {row.domain_id for row in rows} == set(DOMAINS_ORDER)
    for domain in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 96
        assert {k: sum(row.diagnosis_k == k for row in subset) for k in (4, 8, 16)} == {
            4: 32,
            8: 32,
            16: 32,
        }
    for row in rows[:24]:
        assert len(row.typed.decisions) == 5
        assert row.confidence in CONFIDENCE_KEYS
        assert row.intent_field not in row.severity_field
        assert row.intent_field not in row.confidence_field
        assert len(row.option_definitions) == 5


def test_w18_direct_field_mapping_is_frozen():
    assert DIRECT_FIELDS == {
        "diagnosis": ("intent",),
        "response": ("severity",),
        "needs_review": ("severity", "confidence"),
        "risk": ("severity",),
        "urgency": ("severity", "confidence"),
    }
    assert PATHS == (
        "FIELD_DIRECT",
        "LATENT_COMPOSED_HARD",
        "LATENT_COMPOSED_SOFT",
        "ORACLE_LATENT_COMPOSED",
    )


def test_w18_download_free_cache_and_oracle_execution():
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    cache = compile_w18_cache(model, [generate_all_w18()[0]])
    assert cache["metadata"]["a13_invocations_per_case"] == 1
    assert cache["metadata"]["encoded_sequences_per_case"] == 3
    assert cache["metadata"]["trainable_parameter_count"] == 0

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()
    result = evaluate_w18(scorer, cache)
    pooled = result["pooled"]
    assert set(pooled["paths"]) == set(PATHS)
    assert pooled["paths"]["ORACLE_LATENT_COMPOSED"]["accuracy"] == 1.0
    for path in PATHS:
        assert pooled["paths"][path]["decision_count"] == 5
        assert pooled["paths"][path]["probability_mass_max_error"] <= 1e-6
    assert pooled["representation_accounting"] == {
        "logical_state_compiles_per_case": 1,
        "a13_invocations_per_case": 1,
        "encoded_sequences_per_case": 3,
        "isolated_fields": ["intent", "severity", "confidence"],
    }


def _domain_metrics(*, one_field=True, multi_field=True, atomic=True):
    direct_q = {
        "diagnosis": 0.8,
        "response": 0.55,
        "needs_review": 0.60,
        "risk": 0.68,
        "urgency": 0.35,
    }
    hard_q = {
        "diagnosis": 0.85,
        "response": 0.86 if one_field else 0.70,
        "needs_review": 0.86 if multi_field else 0.70,
        "risk": 0.86 if one_field else 0.72,
        "urgency": 0.80 if multi_field else 0.60,
    }
    atomic_block = {
        "intent_per_k": {
            "4": {"top1": 0.90 if atomic else 0.70},
            "8": {"top1": 0.80},
            "16": {"top1": 0.65 if atomic else 0.40},
        },
        "severity": {"top1": 0.90 if atomic else 0.70},
        "confidence": {"top1": 0.90 if atomic else 0.70},
        "joint_severity_confidence_top1": 0.80 if atomic else 0.50,
    }
    direct = {
        "accuracy": 0.55,
        "non_diagnosis_accuracy": 0.55,
        "question_accuracy": direct_q,
        "probability_mass_max_error": 0.0,
    }
    hard = {
        "accuracy": 0.86,
        "non_diagnosis_accuracy": 0.87,
        "question_accuracy": hard_q,
        "probability_mass_max_error": 0.0,
    }
    soft = {
        "accuracy": 0.85,
        "non_diagnosis_accuracy": 0.85,
        "question_accuracy": hard_q,
        "probability_mass_max_error": 0.0,
    }
    oracle = {
        "accuracy": 1.0,
        "non_diagnosis_accuracy": 1.0,
        "question_accuracy": {k: 1.0 for k in direct_q},
        "probability_mass_max_error": 0.0,
    }
    return {
        "atomic": atomic_block,
        "paths": {
            "FIELD_DIRECT": direct,
            "LATENT_COMPOSED_HARD": hard,
            "LATENT_COMPOSED_SOFT": soft,
            "ORACLE_LATENT_COMPOSED": oracle,
        },
        "conditioned_hard_accuracy": {
            "severity_correct": {
                "response_accuracy": 1.0,
                "risk_accuracy": 1.0,
            },
            "severity_confidence_correct": {
                "needs_review_accuracy": 1.0,
                "urgency_accuracy": 1.0,
            },
        },
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 3,
            "isolated_fields": ["intent", "severity", "confidence"],
        },
    }


def _reference():
    return {
        "intent_per_k": {
            "4": {"top1": 0.90},
            "8": {"top1": 0.85},
            "16": {"top1": 0.75},
        },
        "severity": {"top1": 0.95},
        "confidence": {"top1": 0.95},
    }


def test_w18_frozen_classifier_stable_composition_limit():
    metrics = {
        "per_domain": {
            "CR": _domain_metrics(),
            "CS": _domain_metrics(),
            "CT": _domain_metrics(),
            "CU": _domain_metrics(one_field=False, multi_field=False),
        }
    }
    reference = {"per_domain": {d: _reference() for d in metrics["per_domain"]}}
    out = classify_w18(metrics, reference)
    assert out["outcome"] == "STABLE_TYPED_COMPOSITION_LOCALIZATION"
    assert out["stable_classification"] == "TYPED_COMPOSITION_LIMIT"
    assert out["classification_counts"]["TYPED_COMPOSITION_LIMIT"] == 3


def test_w18_frozen_classifier_extraction_limit_precedes_composition():
    metrics = {"per_domain": {"CR": _domain_metrics(atomic=False)}}
    reference = {"per_domain": {"CR": _reference()}}
    out = classify_w18(metrics, reference)
    assert out["per_domain"]["CR"]["classification"] == "LATENT_FIELD_EXTRACTION_LIMIT"
