from __future__ import annotations

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.pairwise_latent_authority import (
    DOMAINS_ORDER,
    all_w20_text_atoms,
    generate_all_w20,
)
from nmd.pairwise_latent_cache import compile_w20_cache
from nmd.pairwise_latent_eval import classify_w20, evaluate_w20
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w20_authority_is_balanced_and_freshly_structured():
    rows = generate_all_w20()
    assert len(rows) == 384
    assert {row.domain_id for row in rows} == set(DOMAINS_ORDER)
    assert len(all_w20_text_atoms()) > 150
    for domain in DOMAINS_ORDER:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 96
        assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
        assert [sum(row.confidence_index == c for row in subset) for c in range(3)] == [32] * 3
    assert len({row.severity_field for row in rows}) == 128
    assert len({row.confidence_field for row in rows}) == 96


def test_w20_download_free_cache_and_evaluator_contract():
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    model = NolaneHira(encoder, HIRACore(d_model=256, dropout=0.0))
    model.eval()
    cache = compile_w20_cache(model, [generate_all_w20()[0]])
    assert cache["metadata"]["case_count"] == 1
    assert cache["metadata"]["a13_invocations_per_case"] == 1
    assert cache["metadata"]["encoded_sequences_per_case"] == 2
    assert cache["metadata"]["isolated_fields"] == ["severity", "confidence"]
    schemas = cache["cases"][0]["schemas"]
    assert len(schemas["severity_pairs"]) == 3
    assert len(schemas["confidence_pairs"]) == 2
    for pair in schemas["severity_pairs"] + schemas["confidence_pairs"]:
        a = pair["canonical"]["D0"]["option_ids"]
        b = pair["swapped"]["D0"]["option_ids"]
        assert tuple(b) == tuple(reversed(a))

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()
    result = evaluate_w20(scorer, cache)
    pooled = result["pooled"]
    assert pooled["pairwise"]["probability_mass_max_error"] <= 1e-6
    assert pooled["representation_accounting"] == {
        "logical_state_compiles_per_case": 1,
        "a13_invocations_per_case": 1,
        "encoded_sequences_per_case": 2,
        "isolated_fields": ["severity", "confidence"],
    }


def _pair(top1=.90, swap=1.0):
    return {
        "n": 32,
        "top1": top1,
        "mrr": .95,
        "mean_margin": .3,
        "mae": 1.0 - top1,
        "balanced_accuracy": top1,
        "swap_prediction_identity": swap,
        "swap_score_max_abs_diff": 0.0,
        "probability_mass_max_error": 0.0,
    }


def _cat(top1=.80):
    return {
        "n": 32,
        "top1": top1,
        "mrr": .90,
        "mean_margin": .2,
        "mae": 1.0 - top1,
    }


def _summary(
    *,
    s_pair=.90,
    c_pair=.92,
    s_flat=.78,
    c_flat=.84,
    swap=1.0,
):
    s_pairs = [_pair(s_pair, swap) for _ in range(3)]
    c_pairs = [_pair(c_pair, swap) for _ in range(2)]
    s_flat_rows = [_cat(s_flat) for _ in range(3)]
    c_flat_rows = [_cat(c_flat) for _ in range(2)]
    return {
        "flat": {
            "severity": _cat(.70),
            "confidence": _cat(.75),
            "joint_severity_confidence_top1": .55,
            "severity_local_pairs": s_flat_rows,
            "confidence_local_pairs": c_flat_rows,
            "severity_local_mean_top1": s_flat,
            "confidence_local_mean_top1": c_flat,
        },
        "pairwise": {
            "severity_pairs": s_pairs,
            "confidence_pairs": c_pairs,
            "severity_mean_top1": s_pair,
            "confidence_mean_top1": c_pair,
            "severity_min_top1": s_pair,
            "confidence_min_top1": c_pair,
            "probability_mass_max_error": 0.0,
        },
        "global_reconstruction": {
            "severity": _cat(.72),
            "confidence": _cat(.78),
            "joint_severity_confidence_top1": .58,
            "severity_transitions": {},
            "confidence_transitions": {},
        },
        "representation_accounting": {
            "logical_state_compiles_per_case": 1,
            "a13_invocations_per_case": 1,
            "encoded_sequences_per_case": 2,
            "isolated_fields": ["severity", "confidence"],
        },
    }


def _reference(*, adequate=True, swap=1.0):
    top1 = .95 if adequate else .85
    return {
        "pairwise": {
            "severity_pairs": [_pair(top1, swap) for _ in range(3)],
            "confidence_pairs": [_pair(top1, swap) for _ in range(2)],
            "severity_mean_top1": top1,
            "confidence_mean_top1": top1,
            "severity_min_top1": top1,
            "confidence_min_top1": top1,
            "probability_mass_max_error": 0.0,
        }
    }


def test_w20_classifier_stable_local_pairwise_interface_limit():
    metrics = {
        "per_domain": {
            "CZ": _summary(),
            "DA": _summary(),
            "DB": _summary(),
            "DC": _summary(s_pair=.80),
        }
    }
    reference = {"per_domain": {d: _reference() for d in metrics["per_domain"]}}
    out = classify_w20(metrics, reference)
    assert out["outcome"] == "STABLE_LATENT_BOUNDARY_LOCALIZATION"
    assert out["stable_classification"] == "LOCAL_PAIRWISE_INTERFACE_LIMIT"
    assert out["classification_counts"]["LOCAL_PAIRWISE_INTERFACE_LIMIT"] == 3


def test_w20_reference_gate_precedes_hira_interpretation():
    metrics = {"per_domain": {"CZ": _summary()}}
    reference = {"per_domain": {"CZ": _reference(adequate=False)}}
    out = classify_w20(metrics, reference)
    assert out["per_domain"]["CZ"]["classification"] == "W20_REFERENCE_INADEQUATE"


def test_w20_swap_integrity_is_part_of_reference_gate():
    metrics = {"per_domain": {"CZ": _summary()}}
    reference = {"per_domain": {"CZ": _reference(swap=.98)}}
    out = classify_w20(metrics, reference)
    assert out["per_domain"]["CZ"]["classification"] == "W20_REFERENCE_INADEQUATE"


def test_w20_flat_local_adequate_when_pairwise_has_no_material_gain():
    metrics = {
        "per_domain": {
            "CZ": _summary(
                s_pair=.84,
                c_pair=.87,
                s_flat=.83,
                c_flat=.86,
            )
        }
    }
    reference = {"per_domain": {"CZ": _reference()}}
    out = classify_w20(metrics, reference)
    assert out["per_domain"]["CZ"]["classification"] == "FLAT_LOCAL_INTERFACE_ADEQUATE"
