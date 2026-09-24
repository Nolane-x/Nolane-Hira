from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.field_semantic_rescue import (
    SemanticAdaptedCompetitiveScorer,
    SemanticResidualAdapter,
)
from nmd.high_cardinality_decomposition import diagnostic_stability
from nmd.high_cardinality_decomposition_eval import (
    aggregate_w6j_checkpoint,
    diagnose_w6j_base,
    load_w6j_cache,
)
from nmd.hira import HIRACore
from nmd.typed_competitive_cache import file_sha256


CHECKPOINTS = (
    "w6e-joint-primary",
    "w6h-projection-retune",
    "w6h-semantic-adapter",
)


def _load_w6e(path: Path):
    receipt = json.loads(
        (path / "receipt.json").read_text(encoding="utf-8")
    )
    if receipt.get("schema_version") != "r8-w6e-candidate-receipt-v1":
        raise RuntimeError("W6j unexpected W6e receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("W6j W6e checkpoint is not PASS")
    if receipt.get("candidate") != "multi-source-joint-primary":
        raise RuntimeError("W6j requires exact W6e joint-primary")
    if receipt.get("confirm_l_exposed") is not False:
        raise RuntimeError("W6j upstream W6e receipt exposed L")
    if receipt.get("confirm_m_exposed") is not False:
        raise RuntimeError("W6j upstream W6e receipt exposed M")

    hira_path = path / "hira.pt"
    scorer_path = path / "scorer.pt"
    if file_sha256(hira_path) != receipt["hira_sha256"]:
        raise RuntimeError("W6j W6e HIRA SHA mismatch")
    if file_sha256(scorer_path) != receipt["scorer_sha256"]:
        raise RuntimeError("W6j W6e scorer SHA mismatch")

    if receipt["hira_sha256"] != (
        "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
    ):
        raise RuntimeError("W6j unexpected frozen W6e HIRA identity")
    if receipt["scorer_sha256"] != (
        "6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e"
    ):
        raise RuntimeError("W6j unexpected frozen W6e scorer identity")

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
    hira.eval()
    scorer.eval()
    return receipt, hira, scorer


def _load_w6h(path: Path, expected_candidate: str):
    receipt = json.loads(
        (path / "receipt.json").read_text(encoding="utf-8")
    )
    if receipt.get("schema_version") != "r8-w6h-candidate-receipt-v1":
        raise RuntimeError(
            f"W6j unexpected W6h receipt schema for {expected_candidate}"
        )
    if receipt.get("status") != "PASS":
        raise RuntimeError(f"{expected_candidate}: W6h checkpoint not PASS")
    if receipt.get("candidate") != expected_candidate:
        raise RuntimeError(f"{expected_candidate}: identity mismatch")
    if receipt.get("confirm_y_exposed") is not False:
        raise RuntimeError(f"{expected_candidate}: receipt exposed Y")
    if receipt.get("confirm_z_exposed") is not False:
        raise RuntimeError(f"{expected_candidate}: receipt exposed Z")

    hira_path = path / "hira.pt"
    scorer_path = path / "scorer.pt"
    if file_sha256(hira_path) != receipt["hira_sha256"]:
        raise RuntimeError(f"{expected_candidate}: HIRA SHA mismatch")
    if file_sha256(scorer_path) != receipt["scorer_sha256"]:
        raise RuntimeError(f"{expected_candidate}: scorer SHA mismatch")
    if receipt["hira_sha256"] != (
        "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
    ):
        raise RuntimeError(f"{expected_candidate}: unexpected HIRA identity")

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(hira_path, map_location="cpu", weights_only=True),
        strict=True,
    )

    if expected_candidate == "projection-retune-control":
        if receipt["scorer_sha256"] != (
            "39235c425d22d01adcea47a7d9dca3330d022488191e5d0ea33bfbb4ee8af9e2"
        ):
            raise RuntimeError("W6j unexpected projection-retune scorer identity")
        scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    elif expected_candidate == "semantic-residual-adapter":
        if receipt["scorer_sha256"] != (
            "9d34e50152706b7a2164b78427abf27dd98966938c3bcce58f71e36163e03469"
        ):
            raise RuntimeError("W6j unexpected semantic-adapter scorer identity")
        scorer = SemanticAdaptedCompetitiveScorer(
            CompetitiveCoarseScorer(d_model=256, d_rel=128),
            SemanticResidualAdapter(),
        )
    else:
        raise ValueError("unsupported W6h checkpoint")

    scorer.load_state_dict(
        torch.load(scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    hira.eval()
    scorer.eval()
    return receipt, hira, scorer


def _evaluate_checkpoint(hira, scorer, cache):
    records = []
    max_probability_error = 0.0
    for base in sorted(cache["bases"], key=lambda row: row["base_id"]):
        record = diagnose_w6j_base(hira, scorer, base)
        records.append(record)
        for view in record["views"].values():
            max_probability_error = max(
                max_probability_error,
                float(view["probability_mass_error"]),
            )
        max_probability_error = max(
            max_probability_error,
            float(record["oracle"]["probability_mass_error"]),
        )
    return records, max_probability_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--w6e-joint-primary", type=Path, required=True)
    parser.add_argument("--w6h-projection-retune", type=Path, required=True)
    parser.add_argument("--w6h-semantic-adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w6j_cache(args.cache)
    if cache["metadata"]["base_count"] != 192:
        raise RuntimeError("W6j cache must contain 192 bases")
    if cache["metadata"]["view_count"] != 768:
        raise RuntimeError("W6j cache must contain 768 nested views")
    if cache["metadata"]["state_encodes_per_base"] != 1.0:
        raise RuntimeError("W6j state-once cache contract failed")

    loaders = {
        "w6e-joint-primary": lambda: _load_w6e(
            args.w6e_joint_primary
        ),
        "w6h-projection-retune": lambda: _load_w6h(
            args.w6h_projection_retune,
            "projection-retune-control",
        ),
        "w6h-semantic-adapter": lambda: _load_w6h(
            args.w6h_semantic_adapter,
            "semantic-residual-adapter",
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    provenance = {}
    max_probability_error = 0.0
    stability_input = {}

    for name in CHECKPOINTS:
        receipt, hira, scorer = loaders[name]()
        records, checkpoint_probability_error = _evaluate_checkpoint(
            hira,
            scorer,
            cache,
        )
        max_probability_error = max(
            max_probability_error,
            checkpoint_probability_error,
        )
        aggregate = aggregate_w6j_checkpoint(records)
        results[name] = aggregate
        stability_input[name] = {
            domain: {
                "classification": row["classification"]["classification"]
            }
            for domain, row in aggregate["per_domain"].items()
        }
        provenance[name] = {
            "candidate": receipt["candidate"],
            "hira_sha256": receipt["hira_sha256"],
            "scorer_sha256": receipt["scorer_sha256"],
            "selected_epoch": receipt["selected_epoch"],
            "trainable_parameter_count": receipt[
                "trainable_parameter_count"
            ],
            "total_parameter_count": receipt[
                "total_parameter_count"
            ],
        }
        (args.out / f"{name}-records.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stability = diagnostic_stability(stability_input)
    audit = {
        "schema_version": "r8-w6j-high-cardinality-decomposition-v1",
        "status": "PASS",
        "scope": "architectural diagnostic only; no training/no promotion",
        "cache_sha256": file_sha256(args.cache),
        "base_count": cache["metadata"]["base_count"],
        "view_count": cache["metadata"]["view_count"],
        "state_encodes_per_base": cache[
            "metadata"
        ]["state_encodes_per_base"],
        "candidate_provenance": provenance,
        "results": results,
        "stability": stability,
        "probability_mass_max_error": max_probability_error,
        "training_performed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "w6h_confirm_rows_used": False,
        "w6h_train_dev_rows_used": False,
        "w6i_diagnostic_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "rescue_lane_authorized": bool(
            stability["rescue_lane_authorized"]
        ),
    }
    (args.out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
