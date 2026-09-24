from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from nmd.typed_reliability_cache import (
    CANDIDATES,
    TRAINABLE_CANDIDATES,
    dev_selection_key,
    file_sha256,
)


CONTROL = "frozen-production-control"
TIE_ORDER = {
    "primitive-temperature": 0,
    "primitive-temperature-noul-bias": 1,
}


def candidate_key(receipt: dict) -> tuple:
    return (
        *dev_selection_key(
            receipt["selected_dev_metrics"],
            int(receipt["selected_epoch"]),
        ),
        TIE_ORDER[receipt["candidate"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path | None]] = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8")
        )
        if (
            receipt.get("schema_version")
            != "r8-w6c-candidate-receipt-v1"
        ):
            continue
        candidate = receipt["candidate"]
        calibrator_path = receipt_path.parent / "calibrator.pt"
        if candidate in TRAINABLE_CANDIDATES:
            if not calibrator_path.exists():
                raise RuntimeError(
                    "W6c trainable candidate calibrator missing"
                )
            if (
                file_sha256(calibrator_path)
                != receipt["calibrator_sha256"]
            ):
                raise RuntimeError(
                    "W6c calibrator checkpoint SHA mismatch"
                )
            checkpoint: Path | None = calibrator_path
        else:
            if receipt.get("calibrator_sha256") is not None:
                raise RuntimeError(
                    "W6c control unexpectedly has calibrator SHA"
                )
            checkpoint = None
        if receipt.get("confirm_exposed") is not False:
            raise RuntimeError(
                "W6c candidate receipt exposed CONFIRM"
            )
        rows.append((receipt, checkpoint))

    names = {receipt["candidate"] for receipt, _ in rows}
    if names != set(CANDIDATES) or len(rows) != 3:
        raise RuntimeError(
            f"expected exactly three W6c candidates, got {sorted(names)}"
        )

    cache_pairs = {
        (
            receipt.get("train_cache_sha256"),
            receipt.get("dev_cache_sha256"),
        )
        for receipt, _ in rows
    }
    if len(cache_pairs) != 1:
        raise RuntimeError(
            "W6c candidates did not use identical TRAIN/DEV caches"
        )
    train_cache_sha256, dev_cache_sha256 = next(
        iter(cache_pairs)
    )
    if not train_cache_sha256 or not dev_cache_sha256:
        raise RuntimeError("W6c cache provenance missing")

    base_pairs = {
        (
            receipt.get("base_w6b_hira_sha256"),
            receipt.get("base_w6b_scorer_sha256"),
        )
        for receipt, _ in rows
    }
    if len(base_pairs) != 1:
        raise RuntimeError(
            "W6c candidates did not use identical base production checkpoint"
        )
    base_hira_sha256, base_scorer_sha256 = next(
        iter(base_pairs)
    )

    control_receipt, _ = next(
        row
        for row in rows
        if row[0]["candidate"] == CONTROL
    )
    trainable_rows = [
        row
        for row in rows
        if row[0]["candidate"] in TRAINABLE_CANDIDATES
    ]
    selected_receipt, selected_path = min(
        trainable_rows,
        key=lambda row: candidate_key(row[0]),
    )
    if selected_path is None:
        raise RuntimeError(
            "selected W6c trainable calibrator missing"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    out_calibrator = args.out / "calibrator.pt"
    shutil.copyfile(selected_path, out_calibrator)

    summary = {
        "schema_version": "r8-w6c-selection-receipt-v1",
        "status": "PASS",
        "selection_authority": "DEV_ONLY",
        "selected_candidate": selected_receipt["candidate"],
        "selected_mode": selected_receipt["calibration_mode"],
        "selected_epoch": selected_receipt["selected_epoch"],
        "selected_dev_metrics": selected_receipt[
            "selected_dev_metrics"
        ],
        "selected_parameters": selected_receipt[
            "selected_parameters"
        ],
        "selected_calibrator_sha256": file_sha256(
            out_calibrator
        ),
        "selected_trainable_parameter_count": selected_receipt[
            "trainable_parameter_count"
        ],
        "selected_total_parameter_count": selected_receipt[
            "total_parameter_count"
        ],
        "control_candidate": CONTROL,
        "control_dev_metrics": control_receipt[
            "selected_dev_metrics"
        ],
        "control_trainable_parameter_count": 0,
        "control_total_parameter_count": control_receipt[
            "total_parameter_count"
        ],
        "base_w6b_hira_sha256": base_hira_sha256,
        "base_w6b_scorer_sha256": base_scorer_sha256,
        "train_cache_sha256": train_cache_sha256,
        "dev_cache_sha256": dev_cache_sha256,
        "confirm_exposed": False,
        "w6b_confirm_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "candidates": [
            {
                "candidate": receipt["candidate"],
                "calibration_mode": receipt["calibration_mode"],
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt[
                    "selected_dev_metrics"
                ],
                "selected_parameters": receipt[
                    "selected_parameters"
                ],
                "calibrator_sha256": receipt[
                    "calibrator_sha256"
                ],
                "trainable_parameter_count": receipt[
                    "trainable_parameter_count"
                ],
                "total_parameter_count": receipt[
                    "total_parameter_count"
                ],
                "train_cache_sha256": receipt[
                    "train_cache_sha256"
                ],
                "dev_cache_sha256": receipt[
                    "dev_cache_sha256"
                ],
                "history": receipt["history"],
            }
            for receipt, _ in sorted(
                rows,
                key=lambda row: CANDIDATES.index(
                    row[0]["candidate"]
                ),
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
                "selected_candidate": summary[
                    "selected_candidate"
                ],
                "selected_epoch": summary["selected_epoch"],
                "selected_trainable_parameter_count": summary[
                    "selected_trainable_parameter_count"
                ],
                "confirm_exposed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
