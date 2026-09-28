from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hira-v0-mainline-m5-final-scorecard-v1"
OUTCOME = "HIRA_V0_M5_FINAL_SCORECARD_READY"
VALID_STATUSES = {
    "WIN", "TIE", "LOSS", "MISSING", "UNSUPPORTED", "NOT_COMPARABLE"
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(value: float, target: float, direction: str, tolerance: float) -> str:
    delta = float(value) - float(target)
    if abs(delta) <= tolerance:
        return "TIE"
    if direction == "higher":
        return "WIN" if delta > 0 else "LOSS"
    if direction == "lower":
        return "WIN" if delta < 0 else "LOSS"
    raise ValueError(f"unknown direction: {direction}")


def add_headline(
    observed: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    source: str,
) -> None:
    target_id = str(row["id"])
    if target_id in observed:
        raise RuntimeError(f"duplicate observed target: {target_id}")
    observed[target_id] = {
        "value": float(row["value"]),
        "status": str(row["status"]),
        "source": source,
        "matched": bool(row.get("matched", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--banking77", type=Path, required=True)
    parser.add_argument("--english", type=Path, required=True)
    parser.add_argument("--xnli", type=Path, required=True)
    parser.add_argument("--jev", type=Path, required=True)
    parser.add_argument("--massive", type=Path, required=True)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--confirmatory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    targets = read_json(args.targets)
    banking77 = read_json(args.banking77)
    english = read_json(args.english)
    xnli = read_json(args.xnli)
    jev = read_json(args.jev)
    massive = read_json(args.massive)
    systems = read_json(args.systems)
    confirmatory = read_json(args.confirmatory)

    expected_outcomes = {
        "banking77": (banking77, "HIRA_V0_M5_BANKING77_DIRECT_READY"),
        "english": (english, "HIRA_V0_M5_ENGLISH_HELDOUT_READY"),
        "xnli": (xnli, "HIRA_V0_M5_XNLI_READY"),
        "jev": (jev, "HIRA_V0_M5_JEV_BTZSC_READY"),
        "massive": (massive, "HIRA_V0_M5_MASSIVE_INTENT_READY"),
        "systems": (systems, "HIRA_V0_M5_SYSTEMS_CPU_READY"),
        "confirmatory": (confirmatory, "HIRA_V0_M5_CONFIRMATORY_READY"),
    }
    for label, (row, outcome) in expected_outcomes.items():
        if row.get("status") != "PASS" or row.get("outcome") != outcome:
            raise RuntimeError(f"{label} authority is not qualified")
        if row.get("production_ready_claimed") is not False:
            raise RuntimeError(f"{label} authority changed production claim")

    observed: dict[str, dict[str, Any]] = {}
    add_headline(observed, banking77["headline"], source="m5-banking77-direct")
    for row in english["headlines"]:
        add_headline(observed, row, source="m5-english-heldout")
    for row in xnli["headlines"]:
        add_headline(observed, row, source="m5-xnli")
    for row in jev["headlines"]:
        add_headline(observed, row, source="m5-jev-btzsc")
    for row in massive["headlines"]:
        add_headline(observed, row, source="m5-massive-intent")

    # Protocol-separated systems targets are intentionally not compared across
    # GitHub-hosted CPU and Laya's T4 GPU authority.
    for target_id, status in systems["systems_target_cells"].items():
        if status != "NOT_COMPARABLE":
            raise RuntimeError(f"systems target unexpectedly comparable: {target_id}")
        if target_id in observed:
            raise RuntimeError(f"duplicate systems target: {target_id}")
        observed[target_id] = {
            "value": None,
            "status": "NOT_COMPARABLE",
            "source": "m5-systems-cpu",
            "matched": False,
            "reason": "hardware/runtime boundary differs from Laya T4 authority",
        }

    adapted_id = "jev.banking77.24shot.fulltest"
    adapted = banking77["jev_banking77_24shot"]
    if adapted["status"] != "NOT_COMPARABLE":
        raise RuntimeError("Jev retrieved24 lane status changed")
    observed[adapted_id] = {
        "value": None,
        "status": "NOT_COMPARABLE",
        "source": "m5-banking77-direct",
        "matched": False,
        "reason": adapted["reason"],
    }

    tolerance = float(targets.get("rules", {}).get("win_tolerance", 1e-9))
    rows = []
    for target in targets["targets"]:
        target_id = target["id"]
        evidence = observed.get(target_id)
        if evidence is None:
            value = None
            status = "MISSING"
            source = None
            matched = False
            reason = "No qualified executable M5 authority populated this preregistered cell."
        else:
            value = evidence["value"]
            status = evidence["status"]
            source = evidence["source"]
            matched = evidence.get("matched", False)
            reason = evidence.get("reason")
            if status in {"WIN", "TIE", "LOSS"}:
                expected_status = compare(
                    float(value),
                    float(target["target"]),
                    str(target["direction"]),
                    tolerance,
                )
                if status != expected_status:
                    raise RuntimeError(
                        f"score status mismatch for {target_id}: {status} != {expected_status}"
                    )
        if status not in VALID_STATUSES:
            raise RuntimeError(f"invalid final status: {status}")
        rows.append(
            {
                **target,
                "value": value,
                "status": status,
                "source": source,
                "matched": matched,
                "reason": reason,
            }
        )

    counts = Counter(row["status"] for row in rows)
    for status in VALID_STATUSES:
        counts.setdefault(status, 0)

    scored = [row for row in rows if row["status"] in {"WIN", "TIE", "LOSS"}]
    matched_scored = [row for row in scored if row["matched"]]
    if not matched_scored:
        raise RuntimeError("final scorecard has no matched scored cells")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "candidate": "hira-v0-m4-frozen",
        "target_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "matched_scored_cells": len(matched_scored),
        "rows": rows,
        "confirmatory": {
            "suite_sha256": confirmatory["suite_sha256"],
            "scores": confirmatory["scores"],
            "claim_scope": confirmatory["claim_scope"],
        },
        "systems_observation": {
            "hira_cpu_latency": systems["latency"],
            "memory": systems["memory"],
            "runtime": systems["runtime"],
            "cross_hardware_laya_t4_comparison": "NOT_COMPARABLE",
        },
        "scientific_conclusion": (
            "The frozen Hira v0 candidate does not establish Laya/Jev parity on "
            "the matched quality evidence executed in M5. Runtime/state-once "
            "mechanics are reproducible, while semantic quality, multilingual "
            "quality, reliability/OOD and production readiness remain provisional."
        ),
        "release_class": "research_release_evidence",
        "global_winner_claimed": False,
        "public_targets_used_for_model_selection": False,
        "model_weights_changed_after_final_exposure": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final-scorecard.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Hira v0 M5 final scorecard",
        "",
        f"Outcome: **{OUTCOME}**",
        "",
        "This is a bounded research evidence release. It does not claim general "
        "Laya/Jev parity or production readiness.",
        "",
        "## Counts",
        "",
    ]
    for status in ("WIN", "TIE", "LOSS", "NOT_COMPARABLE", "UNSUPPORTED", "MISSING"):
        lines.append(f"- {status}: {counts[status]}")
    lines += [
        "",
        "## Populated / protocol-separated cells",
        "",
        "| target | value | reference | status | source |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        if row["status"] == "MISSING":
            continue
        value = "—" if row["value"] is None else f'{float(row["value"]):.6g}'
        lines.append(
            f'| {row["id"]} | {value} | {row["target"]} | '
            f'{row["status"]} | {row["source"] or "—"} |'
        )
    lines += [
        "",
        "## Confirmatory",
        "",
        f'- suite SHA256: `{confirmatory["suite_sha256"]}`',
        f'- original accuracy: {confirmatory["scores"]["original_accuracy"]:.6f}',
        f'- reversed accuracy: {confirmatory["scores"]["reversed_accuracy"]:.6f}',
        f'- order flip rate: {confirmatory["scores"]["order_flip_rate"]:.6f}',
        "",
        "## Scientific conclusion",
        "",
        result["scientific_conclusion"],
        "",
    ]
    (args.out / "FINAL-SCORECARD.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    receipt = {
        "schema_version": "hira-v0-mainline-m5-final-receipt-v1",
        "status": "PASS",
        "outcome": OUTCOME,
        "target_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "matched_scored_cells": len(matched_scored),
        "confirmatory_suite_sha256": confirmatory["suite_sha256"],
        "global_winner_claimed": False,
        "production_ready_claimed": False,
        "release_class": "research_release_evidence",
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M5_FINAL_SCORECARD_RECEIPT=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
