from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from nmd.typed_competitive_cache import file_sha256
from nmd.typed_domain_generalization import (
    CANDIDATES,
    expected_parameter_counts,
)


ORDER = {name: index for index, name in enumerate(CANDIDATES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path, Path]] = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w6d-candidate-receipt-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W6d candidate receipt is not PASS")
        if receipt.get("confirm_exposed") is not False:
            raise RuntimeError("W6d candidate exposed CONFIRM")
        if receipt.get("w6b_confirm_rows_used") is not False:
            raise RuntimeError("W6d candidate used W6b CONFIRM")
        if receipt.get("w6c_confirm_rows_used") is not False:
            raise RuntimeError("W6d candidate used W6c CONFIRM")
        hira = receipt_path.parent / "hira.pt"
        scorer = receipt_path.parent / "scorer.pt"
        if file_sha256(hira) != receipt["hira_sha256"]:
            raise RuntimeError("W6d HIRA checkpoint SHA mismatch")
        if file_sha256(scorer) != receipt["scorer_sha256"]:
            raise RuntimeError("W6d scorer checkpoint SHA mismatch")
        expected_trainable, expected_total = expected_parameter_counts(
            receipt["candidate"]
        )
        if receipt["trainable_parameter_count"] != expected_trainable:
            raise RuntimeError("W6d trainable parameter receipt mismatch")
        if receipt["total_parameter_count"] != expected_total:
            raise RuntimeError("W6d total parameter receipt mismatch")
        rows.append((receipt, hira, scorer))

    names = {receipt["candidate"] for receipt, _, _ in rows}
    if names != set(CANDIDATES) or len(rows) != len(CANDIDATES):
        raise RuntimeError(
            f"expected four W6d candidates, got {sorted(names)}"
        )

    single_hashes = {r["single_cache_sha256"] for r, _, _ in rows}
    multi_hashes = {r["multi_cache_sha256"] for r, _, _ in rows}
    dev_hashes = {r["dev_cache_sha256"] for r, _, _ in rows}
    if len(single_hashes) != 1 or len(multi_hashes) != 1 or len(dev_hashes) != 1:
        raise RuntimeError("W6d candidates did not share exact cache provenance")

    by_name = {receipt["candidate"]: receipt for receipt, _, _ in rows}
    single = by_name["single-source-scorer-only"]
    multi = by_name["multi-source-scorer-only"]
    joint = by_name["multi-source-joint"]
    if single["train_case_count"] != 384 or multi["train_case_count"] != 384:
        raise RuntimeError("W6d scorer-only case-budget parity changed")
    if single["optimizer_case_steps"] != multi["optimizer_case_steps"]:
        raise RuntimeError("W6d scorer-only optimizer-step parity changed")
    if single["optimizer_case_steps"] != 384 * 6:
        raise RuntimeError("unexpected W6d optimizer step budget")
    if single["trainable_parameter_count"] != multi["trainable_parameter_count"]:
        raise RuntimeError("W6d scorer-only parameter parity changed")
    if joint["train_case_count"] != 384:
        raise RuntimeError("W6d joint candidate case budget changed")

    args.out.mkdir(parents=True, exist_ok=True)
    frozen_rows = []
    for receipt, hira, scorer in sorted(
        rows, key=lambda row: ORDER[row[0]["candidate"]]
    ):
        name = receipt["candidate"]
        out_hira = args.out / f"{name}-hira.pt"
        out_scorer = args.out / f"{name}-scorer.pt"
        shutil.copyfile(hira, out_hira)
        shutil.copyfile(scorer, out_scorer)
        frozen_rows.append({
            "candidate": name,
            "selected_epoch": receipt["selected_epoch"],
            "selected_dev_metrics": receipt["selected_dev_metrics"],
            "trainable_parameter_count": receipt[
                "trainable_parameter_count"
            ],
            "total_parameter_count": receipt["total_parameter_count"],
            "hira_sha256": file_sha256(out_hira),
            "scorer_sha256": file_sha256(out_scorer),
            "train_cache_kind": receipt["train_cache_kind"],
            "train_cache_sha256": receipt["train_cache_sha256"],
            "single_cache_sha256": receipt["single_cache_sha256"],
            "multi_cache_sha256": receipt["multi_cache_sha256"],
            "dev_cache_sha256": receipt["dev_cache_sha256"],
            "train_case_count": receipt["train_case_count"],
            "optimizer_case_steps": receipt["optimizer_case_steps"],
            "source_train_metrics": receipt["source_train_metrics"],
            "history": receipt["history"],
        })

    summary = {
        "schema_version": "r8-w6d-freeze-v1",
        "status": "PASS",
        "selection_authority": "DEV_E_INDEPENDENT_PER_CANDIDATE",
        "all_candidates_frozen_before_confirm": True,
        "single_cache_sha256": next(iter(single_hashes)),
        "multi_cache_sha256": next(iter(multi_hashes)),
        "dev_cache_sha256": next(iter(dev_hashes)),
        "scorer_only_equal_case_budget": True,
        "scorer_only_equal_optimizer_steps": True,
        "scorer_only_equal_trainable_parameters": True,
        "confirm_exposed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
        "candidates": frozen_rows,
    }
    (args.out / "freeze.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
