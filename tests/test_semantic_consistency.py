import torch

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.semantic_consistency import (
    CONSISTENCY_BASELINE_INADEQUATE,
    OUTCOME_STABLE,
    PARAPHRASE_CONSISTENCY_RELIABLE,
    SEMANTIC_CONSISTENCY_UNRESOLVED,
    classify_domain,
    countmatched_margin_ids,
    cross_domain_outcome,
    mean_pairwise_rank_stability,
    rank_order_spearman,
    top1_consistency,
)
from nmd.semantic_consistency_authority import (
    BASES_PER_DOMAIN,
    generate_w13_domain,
)
from nmd.semantic_consistency_cache import compile_w13_cache


def test_w13_generator_has_three_paired_views_and_nested_identity():
    rows = generate_w13_domain("BR")
    assert len(rows) == BASES_PER_DOMAIN * 3 * 3
    assert len({row.base_id for row in rows}) == BASES_PER_DOMAIN

    base_id = rows[0].base_id
    base = [row for row in rows if row.base_id == base_id]
    assert len(base) == 9

    by_k = {}
    for k in (4, 8, 16):
        triple = [row for row in base if row.diagnosis_k == k]
        assert {row.view_id for row in triple} == {"D0", "D1", "D2"}
        assert len({row.option_ids for row in triple}) == 1
        assert len({row.gold_index for row in triple}) == 1
        by_k[k] = triple[0].option_ids

    assert set(by_k[4]) < set(by_k[8]) < set(by_k[16])
    assert [item for item in by_k[8] if item in by_k[4]] == list(by_k[4])
    assert [item for item in by_k[16] if item in by_k[8]] == list(by_k[8])


def test_w13_consistency_categories_are_label_free():
    assert top1_consistency(["a", "a", "a"]) == "STRICT"
    assert top1_consistency(["a", "b", "a"]) == "MAJORITY"
    assert top1_consistency(["a", "b", "c"]) == "SPLIT"


def test_w13_rank_stability_is_exact_for_identical_and_reverse_orders():
    same = rank_order_spearman((0, 1, 2, 3), (0, 1, 2, 3))
    reverse = rank_order_spearman((0, 1, 2, 3), (3, 2, 1, 0))
    assert abs(same - 1.0) < 1e-9
    assert abs(reverse + 1.0) < 1e-9
    mean = mean_pairwise_rank_stability(
        ((0, 1, 2, 3), (0, 1, 2, 3), (3, 2, 1, 0))
    )
    assert abs(mean - (-1.0 / 3.0)) < 1e-9


def test_w13_countmatched_margin_control_is_deterministic():
    rows = [(f"b-{i:02d}", float(i % 5)) for i in range(64)]
    selected_a = countmatched_margin_ids(rows, 17)
    selected_b = countmatched_margin_ids(list(reversed(rows)), 17)
    assert selected_a == selected_b
    assert len(selected_a) == 17


def _fixture_reliable():
    return {
        "reference": {
            "4": {"top1": .95},
            "8": {"top1": .90},
            "16": {"top1": .85},
        },
        "ensemble": {
            "4": {"top1": .80},
            "8": {"top1": .70},
            "16": {"top1": .60},
        },
        "final": {
            "4": {"top1": .68},
            "8": {"top1": .58},
            "16": {"top1": .48},
        },
        "strata": {
            "STRICT": {
                "4": {
                    "coverage": .50,
                    "ensemble_top1": .95,
                    "final_top1": .78,
                },
                "8": {
                    "coverage": .50,
                    "ensemble_top1": .88,
                    "final_top1": .73,
                },
                "16": {
                    "coverage": .50,
                    "ensemble_top1": .80,
                    "final_top1": .65,
                },
                "pooled_primary_transition": {
                    "correct_to_wrong_rate": .20,
                    "wrong_to_correct_rate": .04,
                },
            },
            "NON_STRICT": {
                "4": {
                    "coverage": .50,
                    "ensemble_top1": .65,
                    "final_top1": .66,
                },
                "8": {
                    "coverage": .50,
                    "ensemble_top1": .56,
                    "final_top1": .57,
                },
                "16": {
                    "coverage": .50,
                    "ensemble_top1": .55,
                    "final_top1": .56,
                },
                "pooled_primary_transition": {
                    "correct_to_wrong_rate": .06,
                    "wrong_to_correct_rate": .10,
                },
            },
        },
        "guards": {
            "G_strict_ensemble": {
                "4": {"top1": .75},
                "8": {"top1": .65},
                "16": {"top1": .54},
            },
            "G_margin_countmatched": {
                "4": {"top1": .71},
                "8": {"top1": .63},
                "16": {"top1": .52},
            },
        },
    }


def test_w13_classifier_detects_paraphrase_consistency_signal():
    row = classify_domain(_fixture_reliable())
    assert row["classification"] == PARAPHRASE_CONSISTENCY_RELIABLE

    outcome = cross_domain_outcome(
        {
            "BR": row,
            "BS": row,
            "BT": row,
            "BU": {"classification": SEMANTIC_CONSISTENCY_UNRESOLVED},
        }
    )
    assert outcome["outcome"] == OUTCOME_STABLE
    assert outcome["stable_classification"] == PARAPHRASE_CONSISTENCY_RELIABLE


def test_w13_classifier_requires_non_strict_evaluability():
    fixture = _fixture_reliable()
    fixture["strata"]["STRICT"]["4"]["coverage"] = 1.0
    fixture["strata"]["STRICT"]["16"]["coverage"] = 1.0
    fixture["strata"]["NON_STRICT"]["4"]["coverage"] = 0.0
    fixture["strata"]["NON_STRICT"]["16"]["coverage"] = 0.0
    row = classify_domain(fixture)
    assert row["classification"] == CONSISTENCY_BASELINE_INADEQUATE


def test_w13_cache_preserves_state_once_for_one_base():
    rows = generate_w13_domain("BR")
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
    cache = compile_w13_cache(model, rows)
    assert cache["metadata"]["base_count"] == 1
    assert cache["metadata"]["view_count"] == 9
    assert cache["metadata"]["state_encode_calls"] == 1
    assert cache["metadata"]["state_encode_calls_per_base"] == 1.0
