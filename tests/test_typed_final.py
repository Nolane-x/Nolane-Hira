import json

import pytest

from nmd.typed_final import (
    EXPECTED_MARKER,
    compare_laya_typed,
    laya_typed_metric_map,
    load_and_validate_marker,
)


def write_marker(tmp_path, marker):
    path = tmp_path / "marker.json"
    path.write_text(
        json.dumps(marker, sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_marker_accepts_exact_frozen_authority(tmp_path):
    path = write_marker(tmp_path, dict(EXPECTED_MARKER))
    assert load_and_validate_marker(path) == EXPECTED_MARKER


def test_marker_rejects_mismatch_missing_and_extra_fields(tmp_path):
    bad = dict(EXPECTED_MARKER)
    bad["selected_head_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="marker mismatch"):
        load_and_validate_marker(write_marker(tmp_path, bad))

    missing = dict(EXPECTED_MARKER)
    missing.pop("dataset_revision")
    with pytest.raises(ValueError, match="missing fields"):
        load_and_validate_marker(write_marker(tmp_path, missing))

    extra = dict(EXPECTED_MARKER)
    extra["allow_retrain"] = True
    with pytest.raises(ValueError, match="unauthorized fields"):
        load_and_validate_marker(write_marker(tmp_path, extra))


def test_laya_typed_mapping_populates_exactly_eight_specialist_cells():
    raw = {
        "accuracy": 0.7,
        "soft_accuracy": 0.4,
        "hard_brier": 0.2,
        "ece": 0.1,
        "score_mae": 0.3,
        "primitive_accuracy": {
            "noul": 0.8,
            "choice": 0.6,
            "score": 0.5,
        },
    }
    mapped = laya_typed_metric_map(raw)

    assert set(mapped) == {
        "laya.typed.accuracy",
        "laya.typed.soft_accuracy",
        "laya.typed.brier",
        "laya.typed.ece_raw",
        "laya.typed.score_mae",
        "laya.typed.noul_accuracy",
        "laya.typed.choice_accuracy",
        "laya.typed.score_accuracy",
    }
    assert all(not key.startswith("jev.") for key in mapped)


def test_typed_comparison_uses_direction_and_never_populates_jev():
    mapped = {
        "laya.typed.accuracy": 0.8,
        "laya.typed.soft_accuracy": 0.4,
        "laya.typed.brier": 0.05,
        "laya.typed.ece_raw": 0.2,
        "laya.typed.score_mae": 0.3,
        "laya.typed.noul_accuracy": 0.9,
        "laya.typed.choice_accuracy": 0.7,
        "laya.typed.score_accuracy": 0.8,
    }
    targets = {
        "targets": [
            {"id": "laya.typed.accuracy", "direction": "higher", "target": 0.766},
            {"id": "laya.typed.soft_accuracy", "direction": "higher", "target": 0.471},
            {"id": "laya.typed.brier", "direction": "lower", "target": 0.061},
            {"id": "laya.typed.ece_raw", "direction": "lower", "target": 0.213},
            {"id": "laya.typed.score_mae", "direction": "lower", "target": 0.242},
            {"id": "laya.typed.noul_accuracy", "direction": "higher", "target": 0.857},
            {"id": "laya.typed.choice_accuracy", "direction": "higher", "target": 0.733},
            {"id": "laya.typed.score_accuracy", "direction": "higher", "target": 0.723},
            {"id": "jev.typed.accuracy", "direction": "higher", "target": 0.727},
        ]
    }
    out = compare_laya_typed(targets, mapped)

    assert len(out["rows"]) == 8
    assert out["jev_cells_populated"] is False
    assert all(row["id"].startswith("laya.typed.") for row in out["rows"])
    statuses = {row["id"]: row["status"] for row in out["rows"]}
    assert statuses["laya.typed.accuracy"] == "WIN"
    assert statuses["laya.typed.soft_accuracy"] == "LOSS"
    assert statuses["laya.typed.brier"] == "WIN"
    assert statuses["laya.typed.score_mae"] == "LOSS"
