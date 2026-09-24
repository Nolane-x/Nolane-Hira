from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.second_order_localization import (
    aggregate_fullset_rank,
    aggregate_role_context,
    diagnose_w6g_view,
    load_w6g_cache,
    role_localization_classification,
    stable_role_target,
)
from nmd.second_order_localization_authority import ROLE_KEYS
from nmd.typed_competitive_cache import file_sha256


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


def _evaluate_candidate(hira, scorer, cache):
    bases = {row["base_id"]: row for row in cache["bases"]}
    by_base = {}
    for view in cache["views"]:
        by_base.setdefault(view["base_id"], {})[view["view_id"]] = view

    records = []
    for base_id in sorted(by_base):
        base = bases[base_id]
        views = by_base[base_id]
        core = views["core-k8"]
        for view_id in sorted(views):
            records.append(
                diagnose_w6g_view(
                    hira,
                    scorer,
                    base,
                    views[view_id],
                    core,
                )
            )
    return records


def _role_results(records):
    domains = sorted({row["domain_id"] for row in records})
    output = {}
    for domain in domains:
        domain_records = [
            row for row in records if row["domain_id"] == domain
        ]
        role_rows = {}
        for role in ROLE_KEYS:
            pair = aggregate_role_context(
                domain_records,
                role,
                f"pair-{role}",
            )
            far = aggregate_role_context(
                domain_records,
                role,
                "far64",
            )
            dense = aggregate_role_context(
                domain_records,
                role,
                f"dense-{role}64",
            )
            role_rows[role] = {
                "pair": pair,
                "far64": far,
                "dense64": dense,
                "classification": role_localization_classification(
                    pair=pair,
                    far=far,
                    dense=dense,
                ),
            }

        output[domain] = {
            "roles": role_rows,
            "fullset": {
                "core-k8": aggregate_fullset_rank(
                    domain_records,
                    "core-k8",
                ),
                "far64": aggregate_fullset_rank(
                    domain_records,
                    "far64",
                ),
                **{
                    f"dense-{role}64": aggregate_fullset_rank(
                        domain_records,
                        f"dense-{role}64",
                    )
                    for role in ROLE_KEYS
                },
            },
        }
    return output


def _pooled_results(records):
    roles = {}
    for role in ROLE_KEYS:
        pair = aggregate_role_context(records, role, f"pair-{role}")
        far = aggregate_role_context(records, role, "far64")
        dense = aggregate_role_context(records, role, f"dense-{role}64")
        roles[role] = {
            "pair": pair,
            "far64": far,
            "dense64": dense,
            "classification": role_localization_classification(
                pair=pair,
                far=far,
                dense=dense,
            ),
        }
    return {
        "roles": roles,
        "fullset": {
            "core-k8": aggregate_fullset_rank(records, "core-k8"),
            "far64": aggregate_fullset_rank(records, "far64"),
            **{
                f"dense-{role}64": aggregate_fullset_rank(
                    records,
                    f"dense-{role}64",
                )
                for role in ROLE_KEYS
            },
        },
    }


def _stability(results):
    stable = {}
    for role in ROLE_KEYS:
        scorer_only = {
            domain: metrics["roles"][role]["classification"][
                "classification"
            ]
            for domain, metrics in results[
                "multi-source-scorer-only"
            ]["per_domain"].items()
        }
        primary = {
            domain: metrics["roles"][role]["classification"][
                "classification"
            ]
            for domain, metrics in results[
                "multi-source-joint-primary"
            ]["per_domain"].items()
        }
        replica = {
            domain: metrics["roles"][role]["classification"][
                "classification"
            ]
            for domain, metrics in results[
                "multi-source-joint-replica"
            ]["per_domain"].items()
        }
        stable[role] = stable_role_target(
            scorer_only=scorer_only,
            joint_primary=primary,
            joint_replica=replica,
        )
    targets = {
        role: result
        for role, result in stable.items()
        if result["stable"]
    }
    return {
        "per_role": stable,
        "stable_target_count": len(targets),
        "stable_targets": targets,
        "rescue_lane_authorized": bool(targets),
        "diagnostic_outcome": (
            "STABLE_SECOND_ORDER_LOCALIZATION"
            if targets
            else "NO_STABLE_SECOND_ORDER_LOCALIZATION"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--scorer-only", type=Path, required=True)
    parser.add_argument("--joint-primary", type=Path, required=True)
    parser.add_argument("--joint-replica", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache = load_w6g_cache(args.cache)
    if cache["metadata"].get("base_count") != 192:
        raise RuntimeError("W6g cache must contain 192 bases")
    if cache["metadata"].get("view_count") != 1920:
        raise RuntimeError("W6g cache must contain 1,920 views")

    candidate_dirs = {
        "multi-source-scorer-only": args.scorer_only,
        "multi-source-joint-primary": args.joint_primary,
        "multi-source-joint-replica": args.joint_replica,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    provenance = {}
    max_probability_error = 0.0

    for name in CANDIDATES:
        receipt, hira, scorer = _load_candidate(
            candidate_dirs[name],
            name,
        )
        records = _evaluate_candidate(
            hira,
            scorer,
            cache,
        )
        max_probability_error = max(
            max_probability_error,
            max(
                float(row["probability_mass_error"])
                for row in records
            ),
        )
        per_domain = _role_results(records)
        pooled = _pooled_results(records)
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
        (args.out / f"{name}-records.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stability = _stability(results)
    receipt = {
        "schema_version": "r8-w6g-second-order-localization-v1",
        "status": "PASS",
        "scope": "diagnostic localization only; no training/no promotion",
        "cache_sha256": file_sha256(args.cache),
        "base_count": cache["metadata"]["base_count"],
        "view_count": cache["metadata"]["view_count"],
        "state_encodes_per_base": cache[
            "metadata"
        ]["state_encode_calls_per_base"],
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
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "localization.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
