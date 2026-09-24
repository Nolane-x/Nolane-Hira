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
from nmd.hira import HIRACore
from nmd.representation_bridge import (
    aggregate_w6i_records,
    diagnostic_stability,
    diagnose_w6i_base,
    load_w6i_cache,
    representation_classification,
)
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
        raise RuntimeError("W6i unexpected W6e receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("W6i W6e checkpoint is not PASS")
    if receipt.get("candidate") != "multi-source-joint-primary":
        raise RuntimeError("W6i requires exact W6e joint-primary")
    if receipt.get("confirm_l_exposed") is not False:
        raise RuntimeError("W6i upstream W6e receipt exposed L")
    if receipt.get("confirm_m_exposed") is not False:
        raise RuntimeError("W6i upstream W6e receipt exposed M")

    hira_path = path / "hira.pt"
    scorer_path = path / "scorer.pt"
    if file_sha256(hira_path) != receipt["hira_sha256"]:
        raise RuntimeError("W6i W6e HIRA SHA mismatch")
    if file_sha256(scorer_path) != receipt["scorer_sha256"]:
        raise RuntimeError("W6i W6e scorer SHA mismatch")

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
            f"W6i unexpected W6h receipt schema for {expected_candidate}"
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

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(hira_path, map_location="cpu", weights_only=True),
        strict=True,
    )

    if expected_candidate == "projection-retune-control":
        scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    elif expected_candidate == "semantic-residual-adapter":
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
    bases = {row["base_id"]: row for row in cache["bases"]}
    by_base: dict[str, dict[str, dict]] = {}
    for view in cache["views"]:
        by_base.setdefault(view["base_id"], {})[view["view_id"]] = view

    records = []
    for base_id in sorted(by_base):
        records.append(
            diagnose_w6i_base(
                hira,
                scorer,
                bases[base_id],
                by_base[base_id],
            )
        )
    return records


def _per_domain(records):
    output = {}
    for domain in ("AA", "AB", "AC"):
        rows = [
            record
            for record in records
            if record["domain_id"] == domain
        ]
        if len(rows) != 64:
            raise RuntimeError(
                f"W6i domain {domain} expected 64 base records"
            )
        metrics = aggregate_w6i_records(rows)
        output[domain] = {
            "metrics": metrics,
            "classification": representation_classification(metrics),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--w6e-joint-primary", type=Path, required=True)
    parser.add_argument("--w6h-projection-retune", type=Path, required=True)
    parser.add_argument("--w6h-semantic-adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w6i_cache(args.cache)
    if cache["metadata"]["base_count"] != 192:
        raise RuntimeError("W6i cache must contain 192 bases")
    if cache["metadata"]["view_count"] != 1152:
        raise RuntimeError("W6i cache must contain 1,152 views")
    if cache["metadata"]["state_encodes_per_base"] != 1.0:
        raise RuntimeError("W6i state-once cache contract failed")

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

    for name in CHECKPOINTS:
        receipt, hira, scorer = loaders[name]()
        records = _evaluate_checkpoint(hira, scorer, cache)
        per_domain = _per_domain(records)
        pooled_metrics = aggregate_w6i_records(records)
        results[name] = {
            "per_domain": per_domain,
            "pooled": {
                "metrics": pooled_metrics,
                "classification": representation_classification(
                    pooled_metrics
                ),
            },
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

        for record in records:
            for row in record["production"].values():
                max_probability_error = max(
                    max_probability_error,
                    float(row["probability_mass_error"]),
                )
            for row in record["canonical"].values():
                max_probability_error = max(
                    max_probability_error,
                    float(row["probability_mass_error"]),
                )

        (args.out / f"{name}-records.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stability = diagnostic_stability(results)
    audit = {
        "schema_version": "r8-w6i-representation-bridge-v1",
        "status": "PASS",
        "scope": "diagnostic representation bridge only; no training/no promotion",
        "cache_sha256": file_sha256(args.cache),
        "base_count": cache["metadata"]["base_count"],
        "view_count": cache["metadata"]["view_count"],
        "state_encodes_per_base": cache[
            "metadata"
        ]["state_encodes_per_base"],
        "representation_encoder_batches": cache[
            "metadata"
        ]["representation_encoder_batches"],
        "representation_text_count": cache[
            "metadata"
        ]["representation_text_count"],
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
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "rescue_lane_authorized": False,
    }
    (args.out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
