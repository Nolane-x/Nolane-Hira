from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "hira_v0_m5_banking77_direct.py"
    spec = importlib.util.spec_from_file_location("hira_v0_m5_banking77_direct", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(gold: int, pred: int, confidence: float, latency: float) -> dict:
    correct = gold == pred
    return {
        "gold_index": gold,
        "predicted_index": pred,
        "confidence": confidence,
        "correct": correct,
        "hard_brier": 0.1 if correct else 1.1,
        "nll": 0.2 if correct else 2.0,
        "latency_ms": latency,
        "probability_mass_error": 0.0,
        "candidate_budget": 3,
        "relation_delta_max_abs": 0.0,
    }


def test_m5_banking77_summary_preserves_full_k_and_metrics():
    module = load_module()
    rows = [
        row(0, 0, 0.9, 10.0),
        row(1, 1, 0.8, 20.0),
        row(2, 0, 0.6, 30.0),
    ]
    metrics = module.summarize_rows(rows, label_count=3)
    assert metrics["case_count"] == 3
    assert metrics["label_count"] == 3
    assert metrics["accuracy"] == 2 / 3
    assert metrics["candidate_budget_min"] == 3
    assert metrics["candidate_budget_max"] == 3
    assert metrics["probability_mass_max_error"] == 0.0
    assert metrics["relation_delta_max_abs"] == 0.0
    assert metrics["p50_ms_cpu_ci"] == 20.0


def test_m5_banking77_state_hash_is_deterministic_and_content_sensitive():
    module = load_module()
    a = module.state_sha256('{"message":"hello"}')
    b = module.state_sha256('{"message":"hello"}')
    c = module.state_sha256('{"message":"world"}')
    assert a == b
    assert a != c
    assert len(a) == 64


def test_m5_banking77_constants_match_frozen_laya_direct_lane():
    module = load_module()
    assert module.DATASET_ID == "mteb/banking77"
    assert module.DATASET_REVISION == "18072d2685ea682290f7b8924d94c62acc19c0b2"
    assert module.DATASET_SPLIT == "test"
    assert module.EXPECTED_EXAMPLES == 400
    assert module.EXPECTED_LABELS == 77
    assert module.QUESTION == "Which banking intent does `message` express?"
    assert module.LAYA_TARGET == 0.492


def test_m5_banking77_contract_artifact_is_frozen():
    module = load_module()
    assert module.EXPECTED_CONTRACT_ARTIFACT_ID == 10943744649
    assert module.EXPECTED_CONTRACT_DIGEST == (
        "sha256:856031dec357cb0a0d0f1b0fff82a5d7661832a5c67aebc7c2b45d309ec45230"
    )
