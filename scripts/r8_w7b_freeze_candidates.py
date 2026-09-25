from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from nmd.freeform_attribution_training import (
    CANDIDATES,
    EPOCHS,
    candidate_seed,
    expected_trainable_parameters,
)
from nmd.typed_competitive_cache import file_sha256


ORDER = {name: index for index, name in enumerate(CANDIDATES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path, Path]] = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w7b-candidate-receipt-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W7b candidate receipt is not PASS")
        if receipt.get("candidate") not in CANDIDATES:
            raise RuntimeError("W7b candidate identity changed")
        if receipt.get("confirm_as_exposed") is not False:
            raise RuntimeError("W7b candidate exposed CONFIRM-AS")
        if receipt.get("confirm_at_exposed") is not False:
            raise RuntimeError("W7b candidate exposed CONFIRM-AT")
        for key in (
            "w6b_confirm_rows_used",
            "w6c_confirm_rows_used",
            "w6d_confirm_rows_used",
            "w6e_confirm_rows_used",
            "w6f_diagnostic_rows_used",
            "w6g_diagnostic_rows_used",
            "w6h_confirm_rows_used",
            "w6i_diagnostic_rows_used",
            "w6j_diagnostic_rows_used",
            "typed_decisions_final_or_test_used",
        ):
            if receipt.get(key) is not False:
                raise RuntimeError(f"W7b forbidden exposure: {key}")

        hira = receipt_path.parent / "hira.pt"
        scorer = receipt_path.parent / "scorer.pt"
        if file_sha256(hira) != receipt["hira_sha256"]:
            raise RuntimeError("W7b HIRA checkpoint SHA mismatch")
        if file_sha256(scorer) != receipt["scorer_sha256"]:
            raise RuntimeError("W7b scorer checkpoint SHA mismatch")

        expected_trainable = expected_trainable_parameters(
            receipt["candidate"]
        )
        if receipt["trainable_parameter_count"] != expected_trainable:
            raise RuntimeError("W7b trainable parameter receipt mismatch")
        if receipt["optimization_seed"] != candidate_seed(
            receipt["candidate"]
        ):
            raise RuntimeError("W7b optimization seed changed")
        if receipt.get("hira_frozen") is not True:
            raise RuntimeError("W7b HIRACore must remain frozen")
        rows.append((receipt, hira, scorer))

    names = {receipt["candidate"] for receipt, _, _ in rows}
    if names != set(CANDIDATES) or len(rows) != len(CANDIDATES):
        raise RuntimeError(
            f"expected five W7b candidates, got {sorted(names)}"
        )

    train_hashes = {r["train_cache_sha256"] for r, _, _ in rows}
    dev_hashes = {r["dev_cache_sha256"] for r, _, _ in rows}
    if len(train_hashes) != 1 or len(dev_hashes) != 1:
        raise RuntimeError("W7b candidates did not share exact cache provenance")

    by_name = {receipt["candidate"]: receipt for receipt, _, _ in rows}
    for name in CANDIDATES[1:]:
        row = by_name[name]
        if row["train_case_count"] != 384:
            raise RuntimeError(f"W7b {name} case budget changed")
        if row["optimizer_case_steps"] != 384 * EPOCHS:
            raise RuntimeError(f"W7b {name} optimizer budget changed")
        if row["selected_epoch"] < 1 or row["selected_epoch"] > EPOCHS:
            raise RuntimeError(f"W7b {name} selected epoch invalid")

    if by_name["frozen-w6e-control"]["selected_epoch"] != 0:
        raise RuntimeError("W7b frozen control must remain epoch 0")
    if (
        by_name["typed-plus-pair-primary"]["optimization_seed"]
        == by_name["typed-plus-pair-replica"]["optimization_seed"]
    ):
        raise RuntimeError("W7b primary and replica seeds must differ")

    args.out.mkdir(parents=True, exist_ok=True)
    frozen_rows = []
    for receipt, hira, scorer in sorted(
        rows,
        key=lambda row: ORDER[row[0]["candidate"]],
    ):
        name = receipt["candidate"]
        out_hira = args.out / f"{name}-hira.pt"
        out_scorer = args.out / f"{name}-scorer.pt"
        shutil.copyfile(hira, out_hira)
        shutil.copyfile(scorer, out_scorer)
        frozen_rows.append(
            {
                "candidate": name,
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt["selected_dev_metrics"],
                "trainable_parameter_count": receipt[
                    "trainable_parameter_count"
                ],
                "scorer_parameter_count": receipt[
                    "scorer_parameter_count"
                ],
                "total_parameter_count": receipt[
                    "total_parameter_count"
                ],
                "hira_sha256": file_sha256(out_hira),
                "scorer_sha256": file_sha256(out_scorer),
                "train_cache_sha256": receipt["train_cache_sha256"],
                "dev_cache_sha256": receipt["dev_cache_sha256"],
                "train_case_count": receipt["train_case_count"],
                "optimizer_case_steps": receipt["optimizer_case_steps"],
                "optimization_seed": receipt["optimization_seed"],
                "source_train_metrics": receipt["source_train_metrics"],
                "history": receipt["history"],
                "factor_branch_enabled": receipt[
                    "factor_branch_enabled"
                ],
            }
        )

    summary = {
        "schema_version": "r8-w7b-freeze-v1",
        "status": "PASS",
        "selection_authority": "DEV_AR_INDEPENDENT_PER_CANDIDATE",
        "all_candidates_frozen_before_confirm": True,
        "train_cache_sha256": next(iter(train_hashes)),
        "dev_cache_sha256": next(iter(dev_hashes)),
        "equal_train_case_budget": True,
        "equal_optimizer_steps": True,
        "primary_replica_seed_independence": True,
        "confirm_as_exposed": False,
        "confirm_at_exposed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "w6h_confirm_rows_used": False,
        "w6i_diagnostic_rows_used": False,
        "w6j_diagnostic_rows_used": False,
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
