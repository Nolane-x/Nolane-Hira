from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "hira_v0_m5_jev_btzsc.py"
    spec = importlib.util.spec_from_file_location("hira_v0_m5_jev_btzsc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_jev_btzsc_authority_constants_are_exact():
    m = load_module()
    assert m.JEV_COMMIT == "0d610cc53e79bcbec691312b0c4adb4a0e371642"
    assert m.JEV_DATA_BLOB == "efdf58ab89dddc7dff7687f6625747c06ec9551f"
    assert m.JEV_ADAPTER_BLOB == "328a5536bde2b4e0809b4b1727ccbbb6bd6da3fe"
    assert m.JEV_METRICS_BLOB == "eab6060c3aa54a1ab771dbf2a05dab526b5a9be8"
    assert m.DATASET_REVISION == "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"
    assert m.SEED == 20260917
    assert m.SAMPLES_PER_DATASET == 100
    assert m.EXPECTED_MANIFEST_SHA256 == (
        "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
    )


def test_jev_btzsc_option_and_state_contract():
    m = load_module()
    assert m.QUESTION == "Which single label best describes the input text?"
    assert m.canonical_state("hello") == '{"text":"hello"}'
    assert m.CONDITIONS == (
        ("agnews", "topic", 4),
        ("emotiondair", "emotion", 6),
        ("banking77", "intent", 72),
    )


def test_jev_balanced_indices_are_seeded_and_sorted():
    m = load_module()
    targets = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    a = m._balanced_indices(targets, 6, 13)
    b = m._balanced_indices(targets, 6, 13)
    assert a == b
    assert a == sorted(a)
    assert len(a) == 6
    chosen = [targets[i] for i in a]
    assert chosen.count(0) == 2
    assert chosen.count(1) == 2
    assert chosen.count(2) == 2


def test_jev_metric_contract_matches_probability_scoring():
    m = load_module()
    rows = [
        {
            "target_index": 0,
            "predicted_index": 0,
            "probabilities": [0.9, 0.1],
            "latency_seconds": 0.01,
        },
        {
            "target_index": 1,
            "predicted_index": 0,
            "probabilities": [0.6, 0.4],
            "latency_seconds": 0.02,
        },
    ]
    scores = m.score_rows(rows, ece_bins=10, error_budget=0.05)
    assert scores["n"] == 2
    assert scores["valid"] == 2
    assert scores["failures"] == 0
    assert scores["accuracy"] == 0.5
    assert scores["macro_f1"] >= 0.0
    assert scores["brier"] > 0.0
    assert scores["nll"] > 0.0
    assert scores["coverage_at_error_budget"] == 0.5


def test_jev_directional_status_is_explicit():
    m = load_module()
    assert m.status(0.92, 0.91, "higher") == "WIN"
    assert m.status(0.90, 0.91, "higher") == "LOSS"
    assert m.status(0.10, 0.15, "lower") == "WIN"
    assert m.status(0.20, 0.15, "lower") == "LOSS"
    assert m.status(0.91, 0.91, "higher") == "TIE"
