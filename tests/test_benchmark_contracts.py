from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"


def load(name: str) -> dict:
    return json.loads((BENCH / name).read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_headline_board_is_exactly_53_unique_cells():
    targets = load("laya_jev_targets.json")
    rows = targets["targets"]
    ids = [row["id"] for row in rows]
    assert len(rows) == 53
    assert len(ids) == len(set(ids))
    assert all(row["direction"] in {"higher", "lower"} for row in rows)


def test_scorecard_preserves_missing_as_missing_not_zero():
    scorecard = load_module("r8_scorecard", BENCH / "scorecard.py")
    targets = {
        "rules": {"win_tolerance": 1e-9},
        "targets": [
            {
                "id": "a",
                "group": "x",
                "direction": "higher",
                "target": 0.5,
            },
            {
                "id": "b",
                "group": "x",
                "direction": "lower",
                "target": 0.2,
            },
        ],
    }
    result = scorecard.build_scorecard(
        targets,
        {"candidate": "fixture", "metrics": {"a": 0.6}},
    )
    assert result["counts"] == {
        "WIN": 1,
        "TIE": 0,
        "LOSS": 0,
        "MISSING": 1,
    }
    row_b = next(row for row in result["rows"] if row["id"] == "b")
    assert row_b["status"] == "MISSING"
    assert row_b["value"] is None


def test_training_allowance_keeps_direct_generalization_held_out():
    allowance = load("training_allowance.json")["direct_lanes"]
    assert allowance["typed-decisions-finetuned"]["task_training_allowed"] is True
    assert allowance["typed-decisions-zero-shot"]["task_training_allowed"] is False
    assert allowance["massive"]["task_training_allowed"] is False
    assert allowance["xnli"]["task_training_allowed"] is False
    assert allowance["banking77"]["task_training_allowed"] is False


def test_laya_systems_authority_stays_t4_q1_q5_q10_q50():
    manifest = load("laya_protocol_manifest.json")
    systems = manifest["systems"]
    assert systems["hardware"] == "Tesla T4 for direct T4 comparison"
    assert systems["question_counts"] == [1, 5, 10, 50]


def test_pinned_sources_never_use_mutable_aliases():
    registry = load("source_registry.json")
    mutable = {"main", "master", "latest"}
    for key, source in registry["sources"].items():
        revision = source.get("revision") or source.get("sha")
        if source.get("status") == "pinned":
            assert revision, key
            assert str(revision).lower() not in mutable, key


def test_contract_preflight_is_valid_even_if_final_sources_remain_blocked():
    preflight = load_module(
        "r8_contract_preflight",
        BENCH / "contract_preflight.py",
    )
    report = preflight.run_preflight()
    assert report["contract_valid"] is True
    assert report["headline_targets"] == 53
    assert report["unique_target_ids"] == 53
    # Final readiness is deliberately separate from contract validity.
    assert isinstance(report["sources_blocking_final"], list)
