from __future__ import annotations

import math

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.high_cardinality_decomposition_eval import (
    aggregate_w6j_records,
    compile_w6j_cache,
    diagnose_w6j_base,
    load_w6j_cache,
    save_w6j_cache,
)
from nmd.high_cardinality_decomposition_authority import generate_w6j_domain
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def _small_runtime():
    torch.manual_seed(6301)
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
    rows = generate_w6j_domain("AD")
    base_id = rows[0].base_id
    selected = [row for row in rows if row.base_id == base_id]
    assert len(selected) == 4
    return selected


def test_w6j_cache_is_base_centric_and_state_once(tmp_path):
    model, _, _ = _small_runtime()
    before = model.state_encode_calls
    cache = compile_w6j_cache(model, _one_base_views())
    after = model.state_encode_calls

    assert after - before == 1
    metadata = cache["metadata"]
    assert metadata["schema_version"] == "r8-w6j-decomposition-cache-v1"
    assert metadata["base_count"] == 1
    assert metadata["view_count"] == 4
    assert metadata["state_encode_calls"] == 1
    assert metadata["state_encodes_per_base"] == 1.0
    assert len(metadata["case_id_sha256"]) == 64
    assert len(metadata["semantic_view_sha256"]) == 64

    base = cache["bases"][0]
    assert base["master_option_embeddings"].shape == (64, 256)
    assert set(base["view_indices"]) == {"8", "16", "32", "64"}
    assert [len(base["view_indices"][str(k)]) for k in (8, 16, 32, 64)] == [
        8,
        16,
        32,
        64,
    ]

    path = save_w6j_cache(cache, tmp_path / "w6j.pt")
    loaded = load_w6j_cache(path)
    assert loaded["metadata"]["semantic_view_sha256"] == metadata["semantic_view_sha256"]


def test_w6j_diagnosis_runs_exhaustive_pairs_and_oracle_probe():
    model, hira, scorer = _small_runtime()
    cache = compile_w6j_cache(model, _one_base_views())
    record = diagnose_w6j_base(hira, scorer, cache["bases"][0])

    assert set(record["views"]) == {"8", "16", "32", "64"}
    assert len(record["pairs"]) == 63
    assert len({
        int(pair["negative_master_index"])
        for pair in record["pairs"]
    }) == 63
    assert {int(pair["distance"]) for pair in record["pairs"]} == {1, 2, 3, 4}

    oracle = record["oracle"]
    assert oracle["coarse"]["top1"] is True
    assert math.isfinite(float(oracle["final"]["margin"]))

    for k in ("8", "16", "32", "64"):
        row = record["views"][k]
        assert math.isfinite(float(row["coarse"]["margin"]))
        assert math.isfinite(float(row["final"]["margin"]))


def test_w6j_aggregation_exposes_frozen_classification_metrics():
    model, hira, scorer = _small_runtime()
    cache = compile_w6j_cache(model, _one_base_views())
    record = diagnose_w6j_base(hira, scorer, cache["bases"][0])
    out = aggregate_w6j_records([record])

    metrics = out["metrics"]
    required = {
        "k64_coarse_top1",
        "k64_final_top1",
        "relation_rescue_rate",
        "relation_damage_rate",
        "relation_damage_given_coarse_correct_rate",
        "global_error_has_pair_loss_rate",
        "winner_reversal_rate",
        "winner_k2_final_to_k64_final_sign_reversal_rate",
        "global_error_winner_beats_gold_k2_coarse_rate",
        "all_pair_win_case_rate",
        "all_pair_win_but_k64_fail_rate",
        "oracle_final_top1",
    }
    assert required.issubset(metrics)
    assert out["classification"]["classification"] in {
        "PAIRWISE_MULTIPLICITY_LIMIT",
        "SET_CONTEXT_RANK_REVERSAL",
        "COARSE_CONJUNCTION_LIMIT",
        "RELATION_DECOMPOSITION_DAMAGE",
        "MIXED_HIGH_CARDINALITY_ARCHITECTURE",
        "HIGH_CARDINALITY_DECOMPOSITION_UNRESOLVED",
    }
