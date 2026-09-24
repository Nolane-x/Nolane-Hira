from __future__ import annotations

import math

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.second_order_localization import (
    aggregate_role_context,
    compile_w6g_cache,
    diagnose_w6g_view,
    isolated_value_probe,
    load_w6g_cache,
    pair_context_trajectory,
    role_localization_classification,
    save_w6g_cache,
)
from nmd.second_order_localization_authority import (
    ROLE_KEYS,
    generate_w6g_domain,
)
from nmd.semantic import TrainableSemanticEncoder


def _small_runtime():
    torch.manual_seed(6101)
    encoder = TrainableSemanticEncoder(
        vocab_size=8192,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    hira = HIRACore(d_model=256, dropout=0.0)
    model = NolaneHira(encoder, hira)
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    return model, hira, scorer


def _one_base_views():
    rows = generate_w6g_domain("Q")
    base_id = rows[0].base_id
    selected = [row for row in rows if row.base_id == base_id]
    assert len(selected) == 10
    return selected


def _view_lookup(cache):
    return {row["view_id"]: row for row in cache["views"]}


def test_w6g_cache_is_base_centric_and_state_once_per_base():
    model, _, _ = _small_runtime()
    views = _one_base_views()
    before = model.state_encode_calls
    cache = compile_w6g_cache(model, views)
    after = model.state_encode_calls

    assert after - before == 1
    assert cache["metadata"]["schema_version"] == "r8-w6g-second-order-cache-v1"
    assert cache["metadata"]["base_count"] == 1
    assert cache["metadata"]["view_count"] == 10
    assert cache["metadata"]["state_encode_calls"] == 1
    assert cache["metadata"]["state_encode_calls_per_base"] == 1.0
    assert cache["metadata"]["state_encode_calls_per_view"] == 0.1
    assert len(cache["metadata"]["case_id_sha256"]) == 64
    assert len(cache["metadata"]["semantic_view_sha256"]) == 64
    assert set(cache["metadata"]["domain_semantic_view_sha256"]) == {"Q", "R", "S"}
    assert len(cache["bases"]) == 1
    assert len(cache["views"]) == 10

    base = cache["bases"][0]
    assert set(base["field_probes"]) == set(ROLE_KEYS)
    for role in ROLE_KEYS:
        probe = base["field_probes"][role]
        for key in (
            "gold",
            "negative",
            "role",
            "gold_phrase",
            "negative_phrase",
        ):
            assert probe[key]["tokens"].shape[-1] == 256
            assert probe[key]["tokens"].shape[0] >= 1
            assert probe[key]["token_ids"].shape[0] == probe[key]["tokens"].shape[0]


def test_w6g_cache_roundtrip_is_weights_only_safe(tmp_path):
    model, _, _ = _small_runtime()
    cache = compile_w6g_cache(model, _one_base_views())
    path = save_w6g_cache(cache, tmp_path / "w6g.pt")
    restored = load_w6g_cache(path)

    assert restored["metadata"] == cache["metadata"]
    assert len(restored["bases"]) == 1
    assert len(restored["views"]) == 10
    assert torch.equal(
        restored["bases"][0]["state_content_tokens"],
        cache["bases"][0]["state_content_tokens"],
    )


def test_core_reference_pathways_reproduce_native_pair_margin():
    model, hira, scorer = _small_runtime()
    cache = compile_w6g_cache(model, _one_base_views())
    base = cache["bases"][0]
    views = _view_lookup(cache)
    core = views["core-k8"]

    result = diagnose_w6g_view(
        hira,
        scorer,
        base,
        core,
        core,
    )
    assert result["diagnosis_k"] == 8
    assert set(result["role_pairs"]) == set(ROLE_KEYS)
    assert result["probability_mass_error"] <= 1e-6

    for role in ROLE_KEYS:
        pair = result["role_pairs"][role]
        native = float(pair["native_margin"])
        assert math.isfinite(native)
        assert abs(
            float(pair["aggregation_idf_reference_margin"]) - native
        ) < 1e-5
        assert abs(
            float(pair["common_mode_idf_reference_margin"]) - native
        ) < 1e-5
        assert abs(
            float(pair["full_idf_reference_margin"]) - native
        ) < 1e-5


def test_pair_and_dense_views_expose_same_target_role_pair():
    model, hira, scorer = _small_runtime()
    cache = compile_w6g_cache(model, _one_base_views())
    base = cache["bases"][0]
    views = _view_lookup(cache)
    core = views["core-k8"]

    for role in ROLE_KEYS:
        pair_result = diagnose_w6g_view(
            hira,
            scorer,
            base,
            views[f"pair-{role}"],
            core,
        )
        dense_result = diagnose_w6g_view(
            hira,
            scorer,
            base,
            views[f"dense-{role}64"],
            core,
        )
        assert role in pair_result["role_pairs"]
        assert role in dense_result["role_pairs"]
        assert (
            pair_result["role_pairs"][role]["target_option_id"]
            == dense_result["role_pairs"][role]["target_option_id"]
        )


def test_isolated_value_probe_is_finite_and_nonmutating():
    model, _, scorer = _small_runtime()
    cache = compile_w6g_cache(model, _one_base_views())
    probe = cache["bases"][0]["field_probes"]["entity"]

    before = {
        key: value.detach().clone()
        for key, value in scorer.state_dict().items()
    }
    result = isolated_value_probe(scorer, probe)
    assert isinstance(result["gold_wins"], bool)
    assert math.isfinite(float(result["margin"]))
    assert math.isfinite(float(result["gold_score"]))
    assert math.isfinite(float(result["negative_score"]))
    for key, value in scorer.state_dict().items():
        assert torch.equal(value, before[key])


def _metrics(
    *,
    acc,
    margin=0.0,
    isolated=0.95,
    agg_acc=None,
    common_acc=None,
    agg_margin=None,
    common_margin=None,
):
    return {
        "native_pair_accuracy": acc,
        "native_margin_mean": margin,
        "native_margin_median": margin,
        "aggregation_idf_reference_accuracy": (
            acc if agg_acc is None else agg_acc
        ),
        "aggregation_idf_reference_margin_mean": (
            margin if agg_margin is None else agg_margin
        ),
        "common_mode_idf_reference_accuracy": (
            acc if common_acc is None else common_acc
        ),
        "common_mode_idf_reference_margin_mean": (
            margin if common_margin is None else common_margin
        ),
        "full_idf_reference_accuracy": acc,
        "full_idf_reference_margin_mean": margin,
        "isolated_value_accuracy": isolated,
        "isolated_value_margin_mean": 1.0,
    }


def test_role_localization_rules_do_not_overclaim():
    field = role_localization_classification(
        pair=_metrics(acc=0.80, isolated=0.80),
        far=_metrics(acc=0.79),
        dense=_metrics(acc=0.78),
    )
    assert field["classification"] == "FIELD_SEMANTIC_COLLAPSE"

    binding = role_localization_classification(
        pair=_metrics(acc=0.80, isolated=0.95),
        far=_metrics(acc=0.80),
        dense=_metrics(acc=0.80),
    )
    assert binding["classification"] == "ROLE_BINDING_COLLAPSE"

    aggregation = role_localization_classification(
        pair=_metrics(acc=0.95, margin=0.4),
        far=_metrics(acc=0.94, margin=0.35),
        dense=_metrics(
            acc=0.75,
            margin=0.0,
            agg_acc=0.85,
            common_acc=0.77,
            agg_margin=0.20,
            common_margin=0.04,
        ),
    )
    assert aggregation["classification"] == "IDF_AGGREGATION_INTERFERENCE"

    common = role_localization_classification(
        pair=_metrics(acc=0.95, margin=0.4),
        far=_metrics(acc=0.94, margin=0.35),
        dense=_metrics(
            acc=0.75,
            margin=0.0,
            agg_acc=0.77,
            common_acc=0.85,
            agg_margin=0.04,
            common_margin=0.20,
        ),
    )
    assert common["classification"] == "IDF_COMMON_MODE_INTERFERENCE"

    mixed = role_localization_classification(
        pair=_metrics(acc=0.95),
        far=_metrics(acc=0.94),
        dense=_metrics(
            acc=0.75,
            agg_acc=0.85,
            common_acc=0.84,
            agg_margin=0.18,
            common_margin=0.17,
        ),
    )
    assert mixed["classification"] == "MIXED_IDF_PATHWAY_INTERFERENCE"

    density = role_localization_classification(
        pair=_metrics(acc=0.95),
        far=_metrics(acc=0.94),
        dense=_metrics(
            acc=0.75,
            agg_acc=0.79,
            common_acc=0.78,
            agg_margin=0.05,
            common_margin=0.04,
        ),
    )
    assert density["classification"] == "DENSITY_NEAR_NEIGHBOR_LIMIT"

    unresolved = role_localization_classification(
        pair=_metrics(acc=0.90),
        far=_metrics(acc=0.84),
        dense=_metrics(acc=0.82),
    )
    assert unresolved["classification"] == "NO_SECOND_ORDER_LOCALIZATION"


def test_aggregate_role_context_reports_pair_recovery_metrics():
    records = []
    for index in range(4):
        records.append(
            {
                "view_id": "pair-entity",
                "role_pairs": {
                    "entity": {
                        "native_margin": 0.2 if index < 3 else -0.1,
                        "aggregation_idf_reference_margin": 0.3,
                        "common_mode_idf_reference_margin": 0.25,
                        "full_idf_reference_margin": 0.35,
                        "isolated_value": {"margin": 1.0},
                        "role_value_phrase": {"margin": 0.6},
                    }
                },
            }
        )
    out = aggregate_role_context(records, "entity", "pair-entity")
    assert out["n"] == 4
    assert out["native_pair_accuracy"] == 0.75
    assert out["aggregation_idf_reference_accuracy"] == 1.0
    assert out["common_mode_idf_reference_accuracy"] == 1.0
    assert out["full_idf_reference_accuracy"] == 1.0
    assert out["isolated_value_accuracy"] == 1.0
    assert out["role_value_phrase_accuracy"] == 1.0
    assert out["role_value_phrase_margin_mean"] == 0.6
    assert out["value_to_phrase_margin_delta"] == -0.4
    assert abs(out["phrase_to_full_margin_delta"] + 0.05) < 1e-12


def test_pair_view_exposes_tokenizer_robust_field_counterfactuals():
    model, hira, scorer = _small_runtime()
    cache = compile_w6g_cache(model, _one_base_views())
    base = cache["bases"][0]
    views = _view_lookup(cache)
    core = views["core-k8"]
    result = diagnose_w6g_view(
        hira,
        scorer,
        base,
        views["pair-entity"],
        core,
    )
    role = result["role_pairs"]["entity"]
    value = role["isolated_value"]
    phrase = role["role_value_phrase"]
    assert isinstance(value["gold_wins"], bool)
    assert isinstance(phrase["gold_wins"], bool)
    assert math.isfinite(float(value["margin"]))
    assert math.isfinite(float(phrase["margin"]))


def test_stable_role_target_requires_primary_replica_and_noncontradiction():
    from nmd.second_order_localization import stable_role_target

    stable = stable_role_target(
        scorer_only={
            "Q": "NO_SECOND_ORDER_LOCALIZATION",
            "R": "IDF_AGGREGATION_INTERFERENCE",
            "S": "NO_SECOND_ORDER_LOCALIZATION",
        },
        joint_primary={
            "Q": "IDF_AGGREGATION_INTERFERENCE",
            "R": "IDF_AGGREGATION_INTERFERENCE",
            "S": "NO_SECOND_ORDER_LOCALIZATION",
        },
        joint_replica={
            "Q": "IDF_AGGREGATION_INTERFERENCE",
            "R": "IDF_AGGREGATION_INTERFERENCE",
            "S": "NO_SECOND_ORDER_LOCALIZATION",
        },
    )
    assert stable["stable"] is True
    assert stable["targets"][0]["classification"] == "IDF_AGGREGATION_INTERFERENCE"
    assert stable["targets"][0]["domains"] == ["Q", "R"]

    contradicted = stable_role_target(
        scorer_only={
            "Q": "ROLE_BINDING_COLLAPSE",
            "R": "NO_SECOND_ORDER_LOCALIZATION",
            "S": "NO_SECOND_ORDER_LOCALIZATION",
        },
        joint_primary={
            "Q": "IDF_AGGREGATION_INTERFERENCE",
            "R": "IDF_AGGREGATION_INTERFERENCE",
            "S": "NO_SECOND_ORDER_LOCALIZATION",
        },
        joint_replica={
            "Q": "IDF_AGGREGATION_INTERFERENCE",
            "R": "IDF_AGGREGATION_INTERFERENCE",
            "S": "NO_SECOND_ORDER_LOCALIZATION",
        },
    )
    assert contradicted["stable"] is False


def test_pair_context_trajectory_and_wrong_winner_anatomy_are_recorded():
    model, hira, scorer = _small_runtime()
    cache = compile_w6g_cache(model, _one_base_views())
    base = cache["bases"][0]
    views = _view_lookup(cache)
    core = views["core-k8"]
    records = [
        diagnose_w6g_view(hira, scorer, base, view, core)
        for view in cache["views"]
    ]
    trajectory = pair_context_trajectory(records, "entity")
    assert trajectory["complete_base_count"] == 1
    assert sum(trajectory["first_pair_loss_context"].values()) == 1
    for key in (
        "mean_margin_delta_pair_to_core",
        "mean_margin_delta_core_to_far",
        "mean_margin_delta_far_to_dense",
    ):
        assert math.isfinite(float(trajectory[key]))

    for row in records:
        for key in ("native_wrong_winner", "final_wrong_winner"):
            winner = row[key]
            if winner is not None:
                assert winner["distance"] >= 1
                assert len(winner["changed_roles"]) >= 1
                assert isinstance(winner["option_id"], str)
