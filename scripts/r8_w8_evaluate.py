from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.semantic_transfer import (
    CHECKPOINTS,
    OUTCOME_STABLE,
    transfer_stability,
)
from nmd.semantic_transfer_cache import load_w8_cache
from nmd.semantic_transfer_eval import evaluate_w8_checkpoint
from nmd.typed_competitive_cache import file_sha256


FREEZE_SCHEMA = "r8-w7b-freeze-v1"


def _load_freeze(path: Path, checkpoints: Path) -> dict[str, dict]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != FREEZE_SCHEMA:
        raise RuntimeError("unexpected W7b freeze schema")
    if freeze.get("status") != "PASS":
        raise RuntimeError("W7b freeze is not PASS")
    if freeze.get("all_candidates_frozen_before_confirm") is not True:
        raise RuntimeError("W7b checkpoints were not frozen before confirm")

    rows = {
        row["candidate"]: row
        for row in freeze.get("candidates", [])
        if row.get("candidate") in CHECKPOINTS
    }
    if set(rows) != set(CHECKPOINTS):
        raise RuntimeError("W8 frozen checkpoint set changed")

    for name, row in rows.items():
        hira_path = checkpoints / f"{name}-hira.pt"
        scorer_path = checkpoints / f"{name}-scorer.pt"
        if file_sha256(hira_path) != row["hira_sha256"]:
            raise RuntimeError(f"W8 {name} HIRA SHA mismatch")
        if file_sha256(scorer_path) != row["scorer_sha256"]:
            raise RuntimeError(f"W8 {name} scorer SHA mismatch")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w8_cache(args.cache)
    rows = _load_freeze(args.freeze, args.checkpoints)

    results: dict[str, dict[str, object]] = {}
    provenance: dict[str, dict[str, object]] = {}

    for name in CHECKPOINTS:
        row = rows[name]
        hira_path = args.checkpoints / f"{name}-hira.pt"
        scorer_path = args.checkpoints / f"{name}-scorer.pt"

        hira = HIRACore(d_model=256, dropout=0.0)
        hira.load_state_dict(
            torch.load(hira_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
        scorer.load_state_dict(
            torch.load(scorer_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        results[name] = evaluate_w8_checkpoint(hira, scorer, cache)
        provenance[name] = {
            "selected_epoch": row["selected_epoch"],
            "optimization_seed": row["optimization_seed"],
            "hira_sha256": row["hira_sha256"],
            "scorer_sha256": row["scorer_sha256"],
            "trainable_parameter_count": row["trainable_parameter_count"],
        }

    checkpoint_domains = {
        name: {
            domain: results[name]["per_domain"][domain]["classification"]
            for domain in ("AU", "AV", "AW", "AX")
        }
        for name in CHECKPOINTS
    }
    stability = transfer_stability(checkpoint_domains)

    max_mass_error = max(
        float(results[name]["probability_mass_max_error"])
        for name in CHECKPOINTS
    )

    output = {
        "schema_version": "r8-w8-semantic-transfer-v1",
        "status": "PASS",
        "scope": "diagnostic semantic/schema transfer only; no training",
        "training_performed": False,
        "results": results,
        "candidate_provenance": provenance,
        "stability": stability,
        "outcome": stability["outcome"],
        "stable_classification": stability["stable_classification"],
        "mechanism_lane_authorized": (
            stability["outcome"] == OUTCOME_STABLE
        ),
        "probability_mass_max_error": max_mass_error,
        "state_encodes_per_base": 1.0,
        "cached_base_count": cache["metadata"]["base_count"],
        "cached_view_count": cache["metadata"]["view_count"],
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "w6h_confirm_rows_used": False,
        "w6i_diagnostic_rows_used": False,
        "w6j_diagnostic_rows_used": False,
        "w7_confirm_rows_used": False,
        "w7b_confirm_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
