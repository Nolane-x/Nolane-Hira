from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from nmd.typed_competitive_cache import file_sha256
from nmd.typed_joint_replication import (
    CANDIDATES,
    candidate_seed,
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
        if receipt.get("schema_version") != "r8-w6e-candidate-receipt-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W6e candidate receipt is not PASS")
        if receipt.get("confirm_l_exposed") is not False:
            raise RuntimeError("W6e candidate exposed CONFIRM-L")
        if receipt.get("confirm_m_exposed") is not False:
            raise RuntimeError("W6e candidate exposed CONFIRM-M")
        for key in (
            "w6b_confirm_rows_used",
            "w6c_confirm_rows_used",
            "w6d_confirm_rows_used",
            "typed_decisions_final_or_test_used",
        ):
            if receipt.get(key) is not False:
                raise RuntimeError(f"W6e forbidden exposure: {key}")

        hira = receipt_path.parent / "hira.pt"
        scorer = receipt_path.parent / "scorer.pt"
        if file_sha256(hira) != receipt["hira_sha256"]:
            raise RuntimeError("W6e HIRA checkpoint SHA mismatch")
        if file_sha256(scorer) != receipt["scorer_sha256"]:
            raise RuntimeError("W6e scorer checkpoint SHA mismatch")

        expected_trainable, expected_total = expected_parameter_counts(
            receipt["candidate"]
        )
        if receipt["trainable_parameter_count"] != expected_trainable:
            raise RuntimeError("W6e trainable parameter receipt mismatch")
        if receipt["total_parameter_count"] != expected_total:
            raise RuntimeError("W6e total parameter receipt mismatch")
        if receipt["global_seed"] != candidate_seed(receipt["candidate"]):
            raise RuntimeError("W6e candidate seed identity changed")
        rows.append((receipt, hira, scorer))

    names = {receipt["candidate"] for receipt, _, _ in rows}
    if names != set(CANDIDATES) or len(rows) != len(CANDIDATES):
        raise RuntimeError(
            f"expected four W6e candidates, got {sorted(names)}"
        )

    train_hashes = {r["train_cache_sha256"] for r, _, _ in rows}
    dev_hashes = {r["dev_cache_sha256"] for r, _, _ in rows}
    if len(train_hashes) != 1 or len(dev_hashes) != 1:
        raise RuntimeError("W6e candidates did not share exact cache provenance")

    by_name = {receipt["candidate"]: receipt for receipt, _, _ in rows}
    for name in CANDIDATES[1:]:
        row = by_name[name]
        if row["train_case_count"] != 384:
            raise RuntimeError(f"W6e {name} case budget changed")
        if row["optimizer_case_steps"] != 384 * 6:
            raise RuntimeError(f"W6e {name} optimizer budget changed")

    if (
        by_name["multi-source-joint-primary"]["global_seed"]
        == by_name["multi-source-joint-replica"]["global_seed"]
    ):
        raise RuntimeError("W6e primary and replica seeds must differ")

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
            "train_cache_sha256": receipt["train_cache_sha256"],
            "dev_cache_sha256": receipt["dev_cache_sha256"],
            "train_case_count": receipt["train_case_count"],
            "optimizer_case_steps": receipt["optimizer_case_steps"],
            "global_seed": receipt["global_seed"],
            "source_train_metrics": receipt["source_train_metrics"],
            "history": receipt["history"],
        })

    summary = {
        "schema_version": "r8-w6e-freeze-v1",
        "status": "PASS",
        "selection_authority": "DEV_K_INDEPENDENT_PER_CANDIDATE",
        "all_candidates_frozen_before_confirm": True,
        "train_cache_sha256": next(iter(train_hashes)),
        "dev_cache_sha256": next(iter(dev_hashes)),
        "equal_train_case_budget": True,
        "equal_optimizer_steps": True,
        "primary_replica_seed_independence": True,
        "confirm_l_exposed": False,
        "confirm_m_exposed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
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
