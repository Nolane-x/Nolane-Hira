from __future__ import annotations

from nmd.continuous_reliability import (
    CONTINUOUS_CONSISTENCY_RELIABLE,
    CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE,
    OUTCOME_STABLE,
    classify_domain,
    continuous_reliability,
    cross_domain_outcome,
    fixed_tertiles,
    mean_rank_stability,
    mean_top3_overlap,
    top_margin_ids,
    vote_strength,
)
from nmd.continuous_reliability_authority import (
    BASES_PER_DOMAIN,
    K_VALUES,
    PARAPHRASE_VIEWS,
    generate_w14_domain,
)


def test_w14_fresh_domain_shape_and_paired_identity():
    rows = generate_w14_domain("BV")
    assert len(rows) == BASES_PER_DOMAIN * len(K_VALUES) * len(PARAPHRASE_VIEWS)
    by_base = {}
    for row in rows:
        by_base.setdefault(row.base_id, []).append(row)
    assert len(by_base) == BASES_PER_DOMAIN
    for views in by_base.values():
        assert len(views) == 9
        by_k = {}
        for view in views:
            by_k.setdefault(view.diagnosis_k, []).append(view)
        assert set(by_k) == {4, 8, 16}
        for k, group in by_k.items():
            assert {view.view_id for view in group} == {"D0", "D1", "D2"}
            option_ids = {view.option_ids for view in group}
            gold_indices = {view.gold_index for view in group}
            assert len(option_ids) == 1
            assert len(gold_indices) == 1
            assert len(next(iter(option_ids))) == k


def test_w14_vote_strength_contract():
    assert vote_strength(("a", "b", "c")) == 0.0
    assert vote_strength(("a", "a", "b")) == 0.5
    assert vote_strength(("a", "a", "a")) == 1.0


def test_w14_rank_stability_and_top3_overlap_contracts():
    identity = ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))
    reverse = ((0, 1, 2, 3), (3, 2, 1, 0), (3, 2, 1, 0))
    assert mean_rank_stability(identity) == 1.0
    assert 0.0 <= mean_rank_stability(reverse) <= 1.0
    assert mean_top3_overlap(identity) == 1.0

    disjoint_top3 = (
        (0, 1, 2, 3, 4, 5),
        (3, 4, 5, 0, 1, 2),
        (0, 1, 2, 3, 4, 5),
    )
    overlap = mean_top3_overlap(disjoint_top3)
    assert 0.0 <= overlap <= 1.0


def test_w14_continuous_score_is_fixed_mean():
    orders = ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))
    x = continuous_reliability(("a", "a", "a"), orders)
    assert x == {"V": 1.0, "S": 1.0, "O": 1.0, "R": 1.0}


def test_w14_fixed_tertiles_and_margin_control_are_deterministic():
    rows = [(f"case-{i:02d}", 1.0) for i in range(64)]
    tertiles = fixed_tertiles(rows)
    assert len(tertiles["HIGH"]) == 21
    assert len(tertiles["MIDDLE"]) == 22
    assert len(tertiles["LOW"]) == 21
    assert tertiles["HIGH"] == {f"case-{i:02d}" for i in range(21)}
    assert top_margin_ids(rows) == tertiles["HIGH"]


def _metrics(*, anchor_dominance: bool = False):
    ensemble = {
        "4": {"top1": .90},
        "16": {"top1": .75},
    }
    final = {
        "4": {"top1": .72 if anchor_dominance else .80},
        "16": {"top1": .55 if anchor_dominance else .64},
    }
    high = {
        "4": {"ensemble_top1": .95, "final_top1": .80},
        "16": {"ensemble_top1": .85, "final_top1": .72},
    }
    low = {
        "4": {"ensemble_top1": .65, "final_top1": .64 if not anchor_dominance else .55},
        "16": {"ensemble_top1": .50, "final_top1": .49 if not anchor_dominance else .40},
        "pooled_primary_transition": {
            "wrong_to_correct_rate": .10,
            "correct_to_wrong_rate": .08,
        },
    }
    guards = {
        "G_consistency_high": {
            "4": {"top1": .86},
            "16": {"top1": .70},
        },
        "G_margin_high": {
            "4": {"top1": .80},
            "16": {"top1": .64},
        },
    }
    return {
        "reference": {"4": {"top1": .95}},
        "ensemble": ensemble,
        "final": final,
        "tertiles": {"HIGH": high, "LOW": low},
        "guards": guards,
        "correlations": {
            "4": {"spearman_R_ensemble_correct": .40},
            "16": {"spearman_R_ensemble_correct": .35},
        },
    }


def test_w14_reliable_classifier_contract():
    result = classify_domain(_metrics(anchor_dominance=False))
    assert result["classification"] == CONTINUOUS_CONSISTENCY_RELIABLE


def test_w14_anchor_dominance_classifier_contract():
    metrics = _metrics(anchor_dominance=True)
    metrics["tertiles"]["LOW"]["4"]["final_top1"] = .50
    metrics["tertiles"]["LOW"]["16"]["final_top1"] = .35
    result = classify_domain(metrics)
    assert result["classification"] == CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE


def test_w14_cross_domain_stability_contract():
    per_domain = {
        "BV": {"classification": CONTINUOUS_CONSISTENCY_RELIABLE},
        "BW": {"classification": CONTINUOUS_CONSISTENCY_RELIABLE},
        "BX": {"classification": CONTINUOUS_CONSISTENCY_RELIABLE},
        "BY": {"classification": CONTINUOUS_MULTIVIEW_ANCHOR_DOMINANCE},
    }
    result = cross_domain_outcome(per_domain)
    assert result["outcome"] == OUTCOME_STABLE
    assert result["stable_classification"] == CONTINUOUS_CONSISTENCY_RELIABLE
