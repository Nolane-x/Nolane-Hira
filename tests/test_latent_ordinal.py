from __future__ import annotations

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.latent_ordinal_authority import (
    DOMAINS_ORDER,
    all_w19_text_atoms,
    generate_all_w19,
)
from nmd.latent_ordinal_cache import compile_w19_cache
from nmd.latent_ordinal_eval import classify_w19, evaluate_w19
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w19_authority_is_balanced_and_freshly_structured():
    rows = generate_all_w19()
    assert len(rows) == 288
    assert {row.domain_id for row in rows} == set(DOMAINS_ORDER)
    assert len(all_w19_text_atoms()) > 100
    for domain in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 72
        assert [sum(row.severity == s for row in subset) for s in range(4)] == [18] * 4
        assert [sum(row.confidence_index == c for row in subset) for c in range(3)] == [24] * 3
    assert len({row.severity_field for row in rows}) == 96
    assert len({row.confidence_field for row in rows}) == 72


def test_w19_download_free_cache_and_evaluator_contract():
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    cache = compile_w19_cache(model, [generate_all_w19()[0]])
    assert cache["metadata"]["case_count"] == 1
    assert cache["metadata"]["a13_invocations_per_case"] == 1
    assert cache["metadata"]["encoded_sequences_per_case"] == 2
    assert cache["metadata"]["isolated_fields"] == ["severity", "confidence"]

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()
    result = evaluate_w19(scorer, cache)
    pooled = result["pooled"]
    assert pooled["case_count"] == 1
    assert pooled["ordinal"]["probability_mass_max_error"] <= 1e-6
    assert len(pooled["ordinal"]["severity_thresholds"]) == 3
    assert len(pooled["ordinal"]["confidence_thresholds"]) == 2
    assert pooled["representation_accounting"] == {
        "logical_state_compiles_per_case": 1,
        "a13_invocations_per_case": 1,
        "encoded_sequences_per_case": 2,
        "isolated_fields": ["severity", "confidence"],
    }


def _summary(
    *,
    flat_s=.65,
    flat_c=.90,
    flat_joint=.58,
    ord_s=.86,
    ord_c=.90,
    ord_joint=.76,
    s_mono=.96,
    c_mono=.98,
    flat_s_mae=.70,
    ord_s_mae=.40,
):
    return {
        "flat": {
            "severity": {"top1": flat_s, "mae": flat_s_mae},
            "confidence": {"top1": flat_c, "mae": .15},
            "joint_severity_confidence_top1": flat_joint,
        },
        "ordinal": {
            "severity": {"top1": ord_s, "mae": ord_s_mae},
            "confidence": {"top1": ord_c, "mae": .12},
            "joint_severity_confidence_top1": ord_joint,
            "severity_monotonicity": s_mono,
            "confidence_monotonicity": c_mono,
            "probability_mass_max_error": 0.0,
        },
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
        },
    }


def _reference(*, adequate=True):
    return {
        "ordinal": {
            "severity": {"top1": .95 if adequate else .85},
            "confidence": {"top1": .95},
            "severity_monotonicity": .98,
            "confidence_monotonicity": .99,
        }
    }


def test_w19_classifier_stable_ordinal_interface_limit():
    metrics = {
        "per_domain": {
            "CV": _summary(),
            "CW": _summary(),
            "CX": _summary(),
            "CY": _summary(ord_joint=.67),
        }
    }
    reference = {"per_domain": {d: _reference() for d in metrics["per_domain"]}}
    out = classify_w19(metrics, reference)
    assert out["outcome"] == "STABLE_LATENT_AXIS_LOCALIZATION"
    assert out["stable_classification"] == "ORDINAL_LATENT_INTERFACE_LIMIT"
    assert out["classification_counts"]["ORDINAL_LATENT_INTERFACE_LIMIT"] == 3


def test_w19_reference_gate_precedes_hira_interpretation():
    metrics = {"per_domain": {"CV": _summary()}}
    reference = {"per_domain": {"CV": _reference(adequate=False)}}
    out = classify_w19(metrics, reference)
    assert out["per_domain"]["CV"]["classification"] == "W19_REFERENCE_INADEQUATE"


def test_w19_flat_adequate_classification_when_ordinal_does_not_help_joint():
    metrics = {
        "per_domain": {
            "CV": _summary(
                flat_s=.84,
                flat_c=.90,
                flat_joint=.75,
                ord_s=.85,
                ord_c=.90,
                ord_joint=.80,
                flat_s_mae=.35,
                ord_s_mae=.30,
            )
        }
    }
    reference = {"per_domain": {"CV": _reference()}}
    out = classify_w19(metrics, reference)
    assert out["per_domain"]["CV"]["classification"] == "FLAT_LATENT_INTERFACE_ADEQUATE"
