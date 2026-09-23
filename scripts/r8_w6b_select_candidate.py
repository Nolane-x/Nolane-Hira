from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from nmd.typed_competitive_cache import (
    CANDIDATES,
    COMPETITIVE_CANDIDATES,
    dev_selection_key,
    file_sha256,
)


LEGACY = "legacy-w3-joint"
ORDER = {name: index for index, name in enumerate(CANDIDATES)}
COMPETITIVE_TIE_ORDER = {
    "competitive-w5i-scorer-only": 0,
    "competitive-w5i-joint": 1,
}


def candidate_key(receipt: dict) -> tuple:
    metrics = receipt["selected_dev_metrics"]
    epoch = int(receipt["selected_epoch"])
    return (
        *dev_selection_key(metrics, epoch),
        ORDER[receipt["candidate"]],
    )


def competitive_key(receipt: dict) -> tuple:
    metrics = receipt["selected_dev_metrics"]
    epoch = int(receipt["selected_epoch"])
    return (
        *dev_selection_key(metrics, epoch),
        COMPETITIVE_TIE_ORDER[receipt["candidate"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path, Path | None]] = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w6b-candidate-receipt-v1":
            continue
        hira_path = receipt_path.parent / "hira.pt"
        if file_sha256(hira_path) != receipt["hira_sha256"]:
            raise RuntimeError("W6b HIRA checkpoint SHA mismatch")
        scorer_path = receipt_path.parent / "scorer.pt"
        if receipt["candidate"] in COMPETITIVE_CANDIDATES:
            if not scorer_path.exists():
                raise RuntimeError("competitive candidate scorer missing")
            if file_sha256(scorer_path) != receipt["scorer_sha256"]:
                raise RuntimeError("W6b scorer checkpoint SHA mismatch")
            scorer: Path | None = scorer_path
        else:
            if receipt.get("scorer_sha256") is not None:
                raise RuntimeError("legacy candidate unexpectedly has scorer")
            scorer = None
        if receipt.get("confirm_exposed") is not False:
            raise RuntimeError("candidate receipt exposed CONFIRM")
        rows.append((receipt, hira_path, scorer))

    names = {receipt["candidate"] for receipt, _, _ in rows}
    if names != set(CANDIDATES) or len(rows) != 3:
        raise RuntimeError(
            f"expected exactly three W6b candidates, got {sorted(names)}"
        )

    legacy_receipt, legacy_hira, _ = next(
        row for row in rows if row[0]["candidate"] == LEGACY
    )
    competitive_rows = [
        row for row in rows if row[0]["candidate"] in COMPETITIVE_CANDIDATES
    ]
    selected_receipt, selected_hira, selected_scorer = min(
        competitive_rows,
        key=lambda row: competitive_key(row[0]),
    )
    if selected_scorer is None:
        raise RuntimeError("selected competitive scorer missing")

    args.out.mkdir(parents=True, exist_ok=True)
    out_selected_hira = args.out / "competitive-hira.pt"
    out_selected_scorer = args.out / "competitive-scorer.pt"
    out_legacy_hira = args.out / "legacy-hira.pt"
    shutil.copyfile(selected_hira, out_selected_hira)
    shutil.copyfile(selected_scorer, out_selected_scorer)
    shutil.copyfile(legacy_hira, out_legacy_hira)

    summary = {
        "schema_version": "r8-w6b-selection-receipt-v1",
        "status": "PASS",
        "selection_authority": "DEV_ONLY",
        "selected_competitive_candidate": selected_receipt["candidate"],
        "selected_competitive_epoch": selected_receipt["selected_epoch"],
        "selected_competitive_dev_metrics": selected_receipt[
            "selected_dev_metrics"
        ],
        "selected_competitive_hira_sha256": file_sha256(out_selected_hira),
        "selected_competitive_scorer_sha256": file_sha256(
            out_selected_scorer
        ),
        "selected_competitive_trainable_parameter_count": selected_receipt[
            "trainable_parameter_count"
        ],
        "selected_competitive_total_parameter_count": selected_receipt[
            "total_parameter_count"
        ],
        "legacy_candidate": LEGACY,
        "legacy_epoch": legacy_receipt["selected_epoch"],
        "legacy_dev_metrics": legacy_receipt["selected_dev_metrics"],
        "legacy_hira_sha256": file_sha256(out_legacy_hira),
        "legacy_trainable_parameter_count": legacy_receipt[
            "trainable_parameter_count"
        ],
        "legacy_total_parameter_count": legacy_receipt[
            "total_parameter_count"
        ],
        "base_w3_head_sha256": legacy_receipt["base_w3_head_sha256"],
        "base_w5i_scorer_sha256": selected_receipt[
            "base_w5i_scorer_sha256"
        ],
        "confirm_exposed": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "candidates": [
            {
                "candidate": receipt["candidate"],
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt["selected_dev_metrics"],
                "trainable_parameter_count": receipt[
                    "trainable_parameter_count"
                ],
                "total_parameter_count": receipt["total_parameter_count"],
                "hira_sha256": receipt["hira_sha256"],
                "scorer_sha256": receipt["scorer_sha256"],
                "history": receipt["history"],
            }
            for receipt, _, _ in sorted(
                rows,
                key=lambda row: ORDER[row[0]["candidate"]],
            )
        ],
    }
    (args.out / "selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_competitive_candidate": summary[
                    "selected_competitive_candidate"
                ],
                "selected_competitive_epoch": summary[
                    "selected_competitive_epoch"
                ],
                "legacy_epoch": summary["legacy_epoch"],
                "confirm_exposed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
