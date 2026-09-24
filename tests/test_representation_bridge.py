from __future__ import annotations

import math

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.representation_bridge import (
    aggregate_w6i_records,
    compile_w6i_cache,
    diagnostic_stability,
    diagnose_w6i_base,
    load_w6i_cache,
    representation_classification,
    save_w6i_cache,
)
from nmd.representation_bridge_authority import (
    ROLE_KEYS,
    generate_w6i_domain,
)
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def _small_runtime():
    torch.manual_seed(6201)
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
    rows = generate_w6i_domain("AA")
    base_id = rows[0].base_id
    selected = [row for row in rows if row.base_id == base_id]
    assert len(selected) == 6
    return selected


def _view_lookup(cache):
    return {row["view_id"]: row for row in cache["views"]}


def test_w6i_cache_is_base_centric_and_accounts_extra_probe_encoding():
    model, _, _ = _small_runtime()
    before = model.state_encode_calls
    cache = compile_w6i_cache(model, _one_base_views())
    after = model.state_encode_calls

    assert after - before == 1
    metadata = cache["metadata"]
    assert metadata["schema_version"] == "r8-w6i-representation-cache-v1"
    assert metadata["base_count"] == 1
    assert metadata["view_count"] == 6
    assert metadata["state_encode_calls"] == 1
    assert metadata["state_encodes_per_base"] == 1.0
    assert metadata["representation_encoder_batches"] == 1
    assert metadata["representation_text_count"] == 380
    assert metadata["representation_texts_per_base"] == 380.0
    assert len(metadata["case_id_sha256"]) == 64
    assert len(metadata["semantic_view_sha256"]) == 64

    base = cache["bases"][0]
    assert set(base["field_probes"]) == set(ROLE_KEYS)
    for role in ROLE_KEYS:
        probe = base["field_probes"][role]
        assert set(probe) == {
            "gold",
            "negative",
            "gold_phrase",
            "negative_phrase",
            "swapped_role_gold_phrase",
        }
        for value in probe.values():
            assert value["tokens"].shape[-1] == 256
            assert value["tokens"].shape[0] >= 1

    views = _view_lookup(cache)
    for view_id, k in (("core-k8", 8), ("master-k64", 64)):
        view = views[view_id]
        assert view["canonical"]["option_embeddings"].shape == (k, 256)
        assert view["canonical"]["option_tokens"].shape[0] == k
        assert view["canonical"]["option_content_mask"].shape[0] == k
        assert view["factorized"]["tokens"].shape[:2] == (k, 4)
        assert view["factorized"]["mask"].shape[:2] == (k, 4)


def test_w6i_cache_roundtrip_is_weights_only_safe(tmp_path):
    model, _, _ = _small_runtime()
    cache = compile_w6i_cache(model, _one_base_views())
    path = save_w6i_cache(cache, tmp_path / "w6i.pt")
    restored = load_w6i_cache(path)
    assert restored["metadata"] == cache["metadata"]
    assert torch.equal(
        restored["bases"][0]["state_content_tokens"],
        cache["bases"][0]["state_content_tokens"],
    )


def test_w6i_same_case_bridge_evaluator_is_finite():
    model, hira, scorer = _small_runtime()
    cache = compile_w6i_cache(model, _one_base_views())
    base = cache["bases"][0]
    views = _view_lookup(cache)

    record = diagnose_w6i_base(hira, scorer, base, views)
    assert record["domain_id"] == "AA"
    assert set(record["roles"]) == set(ROLE_KEYS)
    for role in ROLE_KEYS:
        row = record["roles"][role]
        for key in (
            "p2_structured_k2_margin",
            "p3_k8_pair_margin",
            "p3_k64_pair_margin",
        ):
            assert math.isfinite(float(row[key]))
        assert isinstance(
            row["p0_isolated_value"]["gold_wins"],
            bool,
        )
        assert isinstance(
            row["p1_role_value_phrase"]["gold_wins"],
            bool,
        )
        assert isinstance(
            row["role_swap_rejection"]["rejected"],
            bool,
        )
        assert math.isfinite(
            float(row["role_swap_rejection"]["margin"])
        )

    for path in ("production", "canonical"):
        for view_id in (
            ("core-k8", "master-k64")
            if path == "canonical"
            else record[path].keys()
        ):
            if path == "production" and view_id.startswith("pair-"):
                pass
            row = record[path][view_id]
            assert row["probability_mass_error"] <= 1e-6

    for path in ("factorized_mean", "factorized_min"):
        for view_id in ("core-k8", "master-k64"):
            rank = record[path][view_id]
            assert 1 <= int(rank["rank"]) <= (
                8 if view_id == "core-k8" else 64
            )
            assert math.isfinite(float(rank["margin"]))


def test_w6i_aggregate_exposes_bridge_and_representation_metrics():
    model, hira, scorer = _small_runtime()
    cache = compile_w6i_cache(model, _one_base_views())
    record = diagnose_w6i_base(
        hira,
        scorer,
        cache["bases"][0],
        _view_lookup(cache),
    )
    metrics = aggregate_w6i_records([record])
    assert metrics["base_count"] == 1
    assert metrics["role_pair_count"] == 4
    for key in (
        "p0_accuracy",
        "p1_accuracy",
        "p2_accuracy",
        "role_swap_rejection_accuracy",
        "p3_k64_fixed_pair_accuracy",
        "p0_wrong_p2_right_rate",
        "p0_right_p2_wrong_rate",
    ):
        assert 0.0 <= float(metrics[key]) <= 1.0
    for key in (
        "canonical_k64_gain",
        "factorized_mean_k64_gain",
        "factorized_min_k64_gain",
    ):
        assert math.isfinite(float(metrics[key]))


def _metrics(
    *,
    p0,
    p2,
    mismatch=0.0,
    production_k64=0.7,
    fixed_pair=0.9,
    canonical_gain=0.0,
    mean_gain=0.0,
    min_gain=0.0,
):
    return {
        "p0_accuracy": p0,
        "p2_accuracy": p2,
        "p0_wrong_p2_right_rate": mismatch,
        "production_k64": {"top1": production_k64},
        "p3_k64_fixed_pair_accuracy": fixed_pair,
        "canonical_k64_gain": canonical_gain,
        "factorized_mean_k64_gain": mean_gain,
        "factorized_min_k64_gain": min_gain,
    }


def test_frozen_representation_classification_rules_do_not_overclaim():
    proxy = representation_classification(
        _metrics(p0=0.60, p2=0.95, mismatch=0.30)
    )
    assert proxy["classification"] == "STRUCTURED_PAIR_PROXY_MISMATCH"

    conjunction = representation_classification(
        _metrics(
            p0=0.92,
            p2=0.95,
            production_k64=0.40,
            fixed_pair=0.90,
            canonical_gain=0.12,
        )
    )
    assert conjunction["classification"] == "FREE_FORM_CONJUNCTION_LIMIT"

    global_rank = representation_classification(
        _metrics(
            p0=0.92,
            p2=0.95,
            production_k64=0.40,
            fixed_pair=0.90,
            canonical_gain=0.05,
            mean_gain=0.04,
            min_gain=0.03,
        )
    )
    assert (
        global_rank["classification"]
        == "GLOBAL_RANKING_WITHOUT_INTERFACE_GAIN"
    )

    shift = representation_classification(
        _metrics(p0=0.85, p2=0.84)
    )
    assert shift["classification"] == "DOMAIN_DIFFICULTY_SHIFT"

    unresolved = representation_classification(
        _metrics(p0=0.82, p2=0.88)
    )
    assert (
        unresolved["classification"]
        == "REPRESENTATION_BRIDGE_UNRESOLVED"
    )


def _domain_row(label):
    return {
        "metrics": {},
        "classification": {"classification": label},
    }


def test_cross_checkpoint_stability_requires_joint_and_w6h_agreement():
    stable = {
        "AA": _domain_row("STRUCTURED_PAIR_PROXY_MISMATCH"),
        "AB": _domain_row("STRUCTURED_PAIR_PROXY_MISMATCH"),
        "AC": _domain_row("REPRESENTATION_BRIDGE_UNRESOLVED"),
    }
    unresolved = {
        "AA": _domain_row("REPRESENTATION_BRIDGE_UNRESOLVED"),
        "AB": _domain_row("REPRESENTATION_BRIDGE_UNRESOLVED"),
        "AC": _domain_row("REPRESENTATION_BRIDGE_UNRESOLVED"),
    }
    results = {
        "w6e-joint-primary": {"per_domain": stable},
        "w6h-projection-retune": {"per_domain": stable},
        "w6h-semantic-adapter": {"per_domain": unresolved},
    }
    out = diagnostic_stability(results)
    assert out["outcome"] == "STABLE_REPRESENTATION_BRIDGE"
    assert (
        out["stable_classification"]
        == "STRUCTURED_PAIR_PROXY_MISMATCH"
    )
    assert out["rescue_lane_authorized"] is False

    opposite = {
        "AA": _domain_row("FREE_FORM_CONJUNCTION_LIMIT"),
        "AB": _domain_row("FREE_FORM_CONJUNCTION_LIMIT"),
        "AC": _domain_row("REPRESENTATION_BRIDGE_UNRESOLVED"),
    }
    results["w6h-semantic-adapter"] = {"per_domain": opposite}
    out = diagnostic_stability(results)
    assert out["outcome"] == "MIXED_REPRESENTATION_FAILURE"
    assert out["stable_classification"] is None
