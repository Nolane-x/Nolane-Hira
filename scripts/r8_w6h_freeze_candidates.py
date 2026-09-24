from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from nmd.field_semantic_rescue import (
    ADAPTER_PARAMETER_COUNT,
    CANDIDATES,
    SCORER_PARAMETER_COUNT,
)
from nmd.typed_competitive_cache import HIRA_PARAMETER_COUNT, file_sha256


def expected_counts(candidate: str) -> tuple[int, int]:
    if candidate == "frozen-joint-control":
        return 0, HIRA_PARAMETER_COUNT + SCORER_PARAMETER_COUNT
    if candidate == "projection-retune-control":
        return SCORER_PARAMETER_COUNT, HIRA_PARAMETER_COUNT + SCORER_PARAMETER_COUNT
    if candidate == "semantic-residual-adapter":
        return (
            ADAPTER_PARAMETER_COUNT,
            HIRA_PARAMETER_COUNT + SCORER_PARAMETER_COUNT + ADAPTER_PARAMETER_COUNT,
        )
    raise ValueError(candidate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w6h-candidate-receipt-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W6h candidate receipt is not PASS")
        if receipt.get("confirm_y_exposed") is not False or receipt.get("confirm_z_exposed") is not False:
            raise RuntimeError("W6h candidate exposed CONFIRM")
        for key in (
            "w6e_confirm_rows_used",
            "w6f_diagnostic_rows_used",
            "w6g_diagnostic_rows_used",
            "typed_decisions_final_or_test_used",
        ):
            if receipt.get(key) is not False:
                raise RuntimeError(f"W6h forbidden exposure: {key}")

        hira = receipt_path.parent / "hira.pt"
        scorer = receipt_path.parent / "scorer.pt"
        if file_sha256(hira) != receipt["hira_sha256"]:
            raise RuntimeError("W6h HIRA checkpoint SHA mismatch")
        if file_sha256(scorer) != receipt["scorer_sha256"]:
            raise RuntimeError("W6h scorer checkpoint SHA mismatch")

        expected_trainable, expected_total = expected_counts(receipt["candidate"])
        if receipt["trainable_parameter_count"] != expected_trainable:
            raise RuntimeError("W6h trainable parameter count changed")
        if receipt["total_parameter_count"] != expected_total:
            raise RuntimeError("W6h total parameter count changed")
        rows.append((receipt, hira, scorer))

    names = {receipt["candidate"] for receipt, _, _ in rows}
    if names != set(CANDIDATES) or len(rows) != len(CANDIDATES):
        raise RuntimeError(f"expected three W6h candidates, got {sorted(names)}")

    train_hashes = {row[0]["train_cache_sha256"] for row in rows}
    dev_hashes = {row[0]["dev_cache_sha256"] for row in rows}
    if len(train_hashes) != 1 or len(dev_hashes) != 1:
        raise RuntimeError("W6h candidates do not share exact cache provenance")

    args.out.mkdir(parents=True, exist_ok=True)
    frozen = []
    for receipt, hira, scorer in rows:
        name = receipt["candidate"]
        out_hira = args.out / f"{name}-hira.pt"
        out_scorer = args.out / f"{name}-scorer.pt"
        shutil.copyfile(hira, out_hira)
        shutil.copyfile(scorer, out_scorer)
        frozen.append(
            {
                "candidate": name,
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt["selected_dev_metrics"],
                "trainable_parameter_count": receipt["trainable_parameter_count"],
                "total_parameter_count": receipt["total_parameter_count"],
                "hira_sha256": file_sha256(out_hira),
                "scorer_sha256": file_sha256(out_scorer),
                "train_cache_sha256": receipt["train_cache_sha256"],
                "dev_cache_sha256": receipt["dev_cache_sha256"],
                "history": receipt["history"],
                "source_train_metrics": receipt["source_train_metrics"],
            }
        )

    summary = {
        "schema_version": "r8-w6h-freeze-v1",
        "status": "PASS",
        "selection_authority": "DEV_X_INDEPENDENT_PER_CANDIDATE",
        "all_candidates_frozen_before_confirm": True,
        "train_cache_sha256": next(iter(train_hashes)),
        "dev_cache_sha256": next(iter(dev_hashes)),
        "confirm_y_exposed": False,
        "confirm_z_exposed": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "candidates": sorted(frozen, key=lambda row: CANDIDATES.index(row["candidate"])),
    }
    (args.out / "freeze.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
