from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

W3A_RUN_ID = 35748778854
W3A_SELECTED_ARTIFACT_ID = 10706135649
W3A_SELECTED_ARTIFACT_DIGEST = "sha256:8a2efd31a6ccea3b987957d328c4b73267ab00ff9024c469bcb7f3cd578d5d92"
W3A_SELECTED_HEAD_SHA256 = "2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c"

FINAL_DATASET_ID = "LocalLLaMA/typed-decisions"
FINAL_DATASET_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
FINAL_CONFIG = "all"
FINAL_SPLIT = "test"
FINAL_CASES = 400
FINAL_DECISIONS = 2000

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
A13_MAX_LENGTH = 256
STATE_SEGMENT_TOKENS = 32

EXPECTED_MARKER = {
    "schema_version": "r8-w3b-final-authority-v1",
    "authorization": "AUTHORIZED_ONE_SHOT",
    "w3a_run_id": W3A_RUN_ID,
    "selected_artifact_id": W3A_SELECTED_ARTIFACT_ID,
    "selected_artifact_digest": W3A_SELECTED_ARTIFACT_DIGEST,
    "selected_head_sha256": W3A_SELECTED_HEAD_SHA256,
    "dataset_id": FINAL_DATASET_ID,
    "dataset_revision": FINAL_DATASET_REVISION,
    "config": FINAL_CONFIG,
    "split": FINAL_SPLIT,
    "selection": "all",
    "expected_cases": FINAL_CASES,
    "expected_decisions": FINAL_DECISIONS,
    "a13_model": A13_MODEL,
    "a13_revision": A13_REVISION,
    "a13_weight_sha256": A13_WEIGHT_SHA256,
}


def load_and_validate_marker(path: str | Path) -> dict:
    marker = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("W3b marker must be a JSON object")
    missing = sorted(set(EXPECTED_MARKER) - set(marker))
    if missing:
        raise ValueError(f"W3b marker missing fields: {missing}")
    mismatches = {
        key: {"expected": value, "actual": marker.get(key)}
        for key, value in EXPECTED_MARKER.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(f"W3b marker mismatch: {mismatches}")
    extra = sorted(set(marker) - set(EXPECTED_MARKER))
    if extra:
        raise ValueError(f"W3b marker has unauthorized fields: {extra}")
    return marker


def laya_typed_metric_map(metrics: dict[str, object]) -> dict[str, float]:
    primitive = metrics["primitive_accuracy"]
    mapping = {
        "laya.typed.accuracy": metrics["accuracy"],
        "laya.typed.soft_accuracy": metrics["soft_accuracy"],
        "laya.typed.brier": metrics["hard_brier"],
        "laya.typed.ece_raw": metrics["ece"],
        "laya.typed.score_mae": metrics["score_mae"],
        "laya.typed.noul_accuracy": primitive["noul"],
        "laya.typed.choice_accuracy": primitive["choice"],
        "laya.typed.score_accuracy": primitive["score"],
    }
    if any(value is None for value in mapping.values()):
        raise ValueError("W3b final metrics contain a missing typed value")
    return {key: float(value) for key, value in mapping.items()}


def compare_laya_typed(
    targets: dict,
    metrics: dict[str, float],
    *,
    tolerance: float = 1e-9,
) -> dict:
    rows = []
    for target in targets.get("targets", []):
        target_id = target.get("id")
        if not isinstance(target_id, str) or not target_id.startswith("laya.typed."):
            continue
        if target_id not in metrics:
            raise ValueError(f"missing final metric for {target_id}")
        value = float(metrics[target_id])
        reference = float(target["target"])
        direction = str(target["direction"])
        delta = value - reference
        if abs(delta) <= tolerance:
            status = "TIE"
        elif direction == "higher":
            status = "WIN" if value > reference else "LOSS"
        elif direction == "lower":
            status = "WIN" if value < reference else "LOSS"
        else:
            raise ValueError(f"unknown direction: {direction}")
        rows.append({
            "id": target_id,
            "direction": direction,
            "target": reference,
            "value": value,
            "status": status,
        })
    if len(rows) != 8:
        raise ValueError(f"expected exactly 8 Laya typed targets, got {len(rows)}")
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("WIN", "TIE", "LOSS")
    }
    return {
        "schema_version": "r8-w3b-laya-typed-comparison-v1",
        "counts": counts,
        "rows": rows,
        "jev_cells_populated": False,
    }
