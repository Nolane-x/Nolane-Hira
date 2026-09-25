import torch

from nmd.anchor_residual import (
    ANCHOR_MARGIN_UNRELIABLE,
    CONFIDENCE_GATED_RESIDUAL_SIGNAL,
    OUTCOME_STABLE,
    ConfidenceRow,
    classify_domain,
    cross_domain_outcome,
    deterministic_quartiles,
    normalized_anchor_margin,
)
from nmd.anchor_residual_authority import BASES_PER_DOMAIN, generate_w12_domain
from nmd.anchor_residual_cache import compile_w12_cache
from nmd.anchor_residual_eval import _summary
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w12_generator_has_paired_nested_identity():
    rows = generate_w12_domain("BN")
    assert len(rows) == BASES_PER_DOMAIN * 3 * 2
    assert len({row.base_id for row in rows}) == BASES_PER_DOMAIN

    base_id = rows[0].base_id
    base = [row for row in rows if row.base_id == base_id]
    assert len(base) == 6

    by_k = {}
    for k in (4, 8, 16):
        pair = [row for row in base if row.diagnosis_k == k]
        assert {row.view_id for row in pair} == {"definition", "label"}
        assert pair[0].option_ids == pair[1].option_ids
        assert pair[0].gold_index == pair[1].gold_index
        by_k[k] = pair[0].option_ids

    assert set(by_k[4]) < set(by_k[8]) < set(by_k[16])
    assert [x for x in by_k[8] if x in by_k[4]] == list(by_k[4])
    assert [x for x in by_k[16] if x in by_k[8]] == list(by_k[8])


def test_w12_anchor_margin_is_label_free_and_positive():
    assert normalized_anchor_margin([0.1, 0.2, 0.6, 0.3]) > 0
    assert normalized_anchor_margin([0.5, 0.5, 0.5, 0.5]) == 0


def test_w12_quartiles_are_deterministic_16_32_16():
    rows = [ConfidenceRow(base_id=f"b-{i:02d}", margin=float(i % 7)) for i in range(64)]
    a = deterministic_quartiles(rows)
    b = deterministic_quartiles(list(reversed(rows)))
    assert a == b
    assert sum(v == "LOW" for v in a.values()) == 16
    assert sum(v == "MIDDLE" for v in a.values()) == 32
    assert sum(v == "HIGH" for v in a.values()) == 16


def _fixture_gated():
    return {
        "reference": {"4": {"top1": .92}},
        "anchor": {
            "4": {"top1": .80},
            "8": {"top1": .72},
            "16": {"top1": .62},
        },
        "coarse": {
            "4": {"top1": .72},
            "8": {"top1": .65},
            "16": {"top1": .52},
        },
        "final": {
            "4": {"top1": .70},
            "8": {"top1": .66},
            "16": {"top1": .50},
        },
        "strata": {
            "HIGH": {
                "4": {"anchor_top1": .94, "final_top1": .75},
                "8": {"anchor_top1": .88, "final_top1": .75},
                "16": {"anchor_top1": .81, "final_top1": .62},
                "pooled_primary_transition": {
                    "correct_to_wrong_rate": .20,
                    "wrong_to_correct_rate": .04,
                },
            },
            "LOW": {
                "4": {"anchor_top1": .50, "final_top1": .50},
                "8": {"anchor_top1": .44, "final_top1": .48},
                "16": {"anchor_top1": .31, "final_top1": .34},
                "pooled_primary_transition": {
                    "correct_to_wrong_rate": .06,
                    "wrong_to_correct_rate": .14,
                },
            },
            "MIDDLE": {
                "4": {"anchor_top1": .76, "final_top1": .70},
                "8": {"anchor_top1": .68, "final_top1": .65},
                "16": {"anchor_top1": .57, "final_top1": .50},
                "pooled_primary_transition": {
                    "correct_to_wrong_rate": .10,
                    "wrong_to_correct_rate": .05,
                },
            },
        },
        "guards": {
            "G_high_anchor": {
                "4": {"top1": .77},
                "8": {"top1": .72},
                "16": {"top1": .56},
            },
            "G_low_anchor_control": {
                "4": {"top1": .73},
                "8": {"top1": .69},
                "16": {"top1": .52},
            },
        },
    }


def test_w12_classifier_detects_confidence_gated_signal():
    row = classify_domain(_fixture_gated())
    assert row["classification"] == CONFIDENCE_GATED_RESIDUAL_SIGNAL
    outcome = cross_domain_outcome({
        "BN": row,
        "BO": row,
        "BP": row,
        "BQ": {"classification": ANCHOR_MARGIN_UNRELIABLE},
    })
    assert outcome["outcome"] == OUTCOME_STABLE
    assert outcome["stable_classification"] == CONFIDENCE_GATED_RESIDUAL_SIGNAL


def test_w12_cache_preserves_state_once_for_one_base():
    rows = generate_w12_domain("BN")
    base_id = rows[0].base_id
    rows = [row for row in rows if row.base_id == base_id]

    encoder = TrainableSemanticEncoder(
        vocab_size=512,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=64,
    )
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    cache = compile_w12_cache(model, rows)
    assert cache["metadata"]["base_count"] == 1
    assert cache["metadata"]["view_count"] == 6
    assert cache["metadata"]["state_encode_calls"] == 1
    assert cache["metadata"]["state_encode_calls_per_base"] == 1.0


def test_w12_definition_only_guard_summary_is_valid():
    rows = []
    for domain in ("BN", "BO", "BP", "BQ"):
        for k in (4, 8, 16):
            rows.append({
                "base_id": f"{domain.lower()}-{k}",
                "case_id": f"{domain.lower()}-{k}-definition",
                "domain_id": domain,
                "view_id": "definition",
                "k": k,
                "stage": "guard",
                "correct": True,
                "rank": 1,
                "mrr": 1.0,
                "top5": True,
                "margin": 1.0,
                "order": [0],
            })
    summary = _summary(rows, views=("definition",))
    assert summary["per_domain"]["BN"]["definition"]["4"]["top1"] == 1.0
    assert "label" not in summary["per_domain"]["BN"]
