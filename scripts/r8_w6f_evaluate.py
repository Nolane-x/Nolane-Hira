from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.high_k_localization import (
    aggregate_w6f_records,
    diagnose_cached_view,
)
from nmd.hira import HIRACore
from nmd.typed_competitive_cache import file_sha256, load_w6b_cache


CANDIDATES = (
    "multi-source-scorer-only",
    "multi-source-joint-primary",
    "multi-source-joint-replica",
)


def _load_candidate(path: Path, expected: str):
    receipt = json.loads(
        (path / "receipt.json").read_text(encoding="utf-8")
    )
    if receipt.get("schema_version") != "r8-w6e-candidate-receipt-v1":
        raise RuntimeError(f"{expected}: unexpected W6e receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError(f"{expected}: W6e candidate is not PASS")
    if receipt.get("candidate") != expected:
        raise RuntimeError(f"{expected}: candidate identity mismatch")
    if receipt.get("confirm_l_exposed") is not False:
        raise RuntimeError(f"{expected}: candidate receipt exposed CONFIRM-L")
    if receipt.get("confirm_m_exposed") is not False:
        raise RuntimeError(f"{expected}: candidate receipt exposed CONFIRM-M")

    hira_path = path / "hira.pt"
    scorer_path = path / "scorer.pt"
    if file_sha256(hira_path) != receipt["hira_sha256"]:
        raise RuntimeError(f"{expected}: HIRA SHA mismatch")
    if file_sha256(scorer_path) != receipt["scorer_sha256"]:
        raise RuntimeError(f"{expected}: scorer SHA mismatch")

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


def _evaluate_candidate(hira, scorer, cases):
    records = [
        diagnose_cached_view(hira, scorer, case)
        for case in cases
    ]
    domains = sorted({row["domain_id"] for row in records})
    per_domain = {
        domain: aggregate_w6f_records(
            [row for row in records if row["domain_id"] == domain]
        )
        for domain in domains
    }
    pooled = aggregate_w6f_records(records)
    return records, per_domain, pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--scorer-only", type=Path, required=True)
    parser.add_argument("--joint-primary", type=Path, required=True)
    parser.add_argument("--joint-replica", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w6b_cache(
        args.cache,
        expected_split="diagnostic",
    )
    if cache["metadata"].get("w6f_schema_version") != "r8-w6f-high-k-cache-v1":
        raise RuntimeError("unexpected W6f cache extension schema")
    if cache["metadata"].get("view_count") != 1152:
        raise RuntimeError("W6f cache must contain 1,152 views")
    if cache["metadata"].get("base_count") != 288:
        raise RuntimeError("W6f cache must contain 288 bases")

    candidate_dirs = {
        "multi-source-scorer-only": args.scorer_only,
        "multi-source-joint-primary": args.joint_primary,
        "multi-source-joint-replica": args.joint_replica,
    }

    results = {}
    provenance = {}
    for name in CANDIDATES:
        receipt, hira, scorer = _load_candidate(
            candidate_dirs[name],
            name,
        )
        records, per_domain, pooled = _evaluate_candidate(
            hira,
            scorer,
            cache["cases"],
        )
        results[name] = {
            "per_domain": per_domain,
            "pooled": pooled,
        }
        provenance[name] = {
            "hira_sha256": receipt["hira_sha256"],
            "scorer_sha256": receipt["scorer_sha256"],
            "selected_epoch": receipt["selected_epoch"],
            "global_seed": receipt["global_seed"],
            "trainable_parameter_count": receipt[
                "trainable_parameter_count"
            ],
            "total_parameter_count": receipt["total_parameter_count"],
        }

        # Raw records are preserved separately for failure anatomy.
        raw_path = args.out / f"{name}-records.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    pooled_labels = {
        name: result["pooled"]["classification"]["classification"]
        for name, result in results.items()
    }
    domain_labels = {
        name: {
            domain: metrics["classification"]["classification"]
            for domain, metrics in result["per_domain"].items()
        }
        for name, result in results.items()
    }

    receipt = {
        "schema_version": "r8-w6f-localization-v1",
        "status": "PASS",
        "scope": "diagnostic localization only; no training/no promotion",
        "cache_sha256": file_sha256(args.cache),
        "view_count": cache["metadata"]["view_count"],
        "base_count": cache["metadata"]["base_count"],
        "candidate_provenance": provenance,
        "results": results,
        "pooled_classifications": pooled_labels,
        "per_domain_classifications": domain_labels,
        "training_performed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "localization.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
