from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MUTABLE = {"main", "master", "latest"}


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def source_revision(source: dict) -> str | None:
    value = source.get("revision") or source.get("sha")
    return None if value is None else str(value)


def run_preflight() -> dict:
    registry = load("source_registry.json")
    targets = load("laya_jev_targets.json")
    allowance = load("training_allowance.json")
    laya = load("laya_protocol_manifest.json")
    jev = load("jev_protocol_manifest.json")

    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    rows = targets.get("targets", [])
    ids = [str(row.get("id")) for row in rows]
    if len(rows) != 53:
        errors.append({
            "code": "headline_target_count",
            "expected": 53,
            "observed": len(rows),
        })
    if len(set(ids)) != len(ids):
        errors.append({"code": "duplicate_target_ids"})
    for row in rows:
        if row.get("direction") not in {"higher", "lower"}:
            errors.append({
                "code": "invalid_target_direction",
                "id": row.get("id"),
                "direction": row.get("direction"),
            })
        if not isinstance(row.get("target"), (int, float)):
            errors.append({
                "code": "invalid_target_value",
                "id": row.get("id"),
            })

    sources = registry.get("sources", {})
    blocked: list[str] = []
    pinned = 0
    for key, source in sorted(sources.items()):
        revision = source_revision(source)
        if source.get("status") != "pinned" or not revision:
            blocked.append(key)
            continue
        pinned += 1
        if revision.lower() in MUTABLE:
            errors.append({
                "code": "mutable_pinned_revision",
                "source": key,
                "revision": revision,
            })

    rules = targets.get("rules", {})
    if rules.get("no_cross_protocol_winner_claims") is not True:
        errors.append({"code": "cross_protocol_rule_not_frozen"})
    if rules.get("final_test_selection_forbidden") is not True:
        errors.append({"code": "final_selection_rule_not_frozen"})
    if rules.get("missing_metrics_count_as_missing_not_zero") is not True:
        errors.append({"code": "missing_semantics_not_frozen"})

    lanes = allowance.get("direct_lanes", {})
    frozen_allowance = {
        "typed-decisions-finetuned": True,
        "typed-decisions-zero-shot": False,
        "massive": False,
        "xnli": False,
        "banking77": False,
    }
    for lane, expected in frozen_allowance.items():
        observed = (lanes.get(lane) or {}).get("task_training_allowed")
        if observed is not expected:
            errors.append({
                "code": "training_allowance_mismatch",
                "lane": lane,
                "expected": expected,
                "observed": observed,
            })

    if laya.get("schema_version") != "r8-laya-protocol-v1":
        errors.append({"code": "laya_manifest_schema"})
    if jev.get("schema_version") != "r8-jev-protocol-v1":
        errors.append({"code": "jev_manifest_schema"})
    systems = laya.get("systems", {})
    if systems.get("hardware") != "Tesla T4 for direct T4 comparison":
        errors.append({"code": "laya_systems_hardware_changed"})
    if systems.get("question_counts") != [1, 5, 10, 50]:
        errors.append({"code": "laya_question_counts_changed"})

    if blocked:
        warnings.append({
            "code": "sources_block_final_run",
            "sources": blocked,
            "count": len(blocked),
        })

    return {
        "schema_version": "r8-benchmark-contract-preflight-v1",
        "headline_targets": len(rows),
        "unique_target_ids": len(set(ids)),
        "sources_total": len(sources),
        "sources_pinned": pinned,
        "sources_blocking_final": blocked,
        "errors": errors,
        "warnings": warnings,
        "contract_valid": not errors,
        "ready_for_final_campaign": not errors and not blocked,
        "note": (
            "This preflight validates benchmark contracts only. "
            "It does not evaluate HIRA or expose final benchmark labels/results."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--strict-final", action="store_true")
    args = parser.parse_args()

    report = run_preflight()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")

    if not report["contract_valid"]:
        raise SystemExit(1)
    if args.strict_final and not report["ready_for_final_campaign"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
