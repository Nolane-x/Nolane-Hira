from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from nmd.semantic_alignment_training import (
    CANDIDATES,
    EPOCHS,
    candidate_seed,
    expected_trainable_parameters,
)
from nmd.typed_competitive_cache import file_sha256


ORDER = {name: index for index, name in enumerate(CANDIDATES)}
EXPECTED_SCORER_COUNTS = {
    "frozen-w6e-control": 32769,
    "projection-semantic-control": 32769,
    "shared-bridge-semantic-control": 32769 + 4096,
    "asymmetric-bridge-primary": 32769 + 8192,
    "asymmetric-bridge-replica": 32769 + 8192,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path, Path, Path]] = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w9-candidate-receipt-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W9 candidate receipt is not PASS")
        candidate = receipt.get("candidate")
        if candidate not in CANDIDATES:
            raise RuntimeError("W9 candidate identity changed")

        for key in (
            "confirm_bd_exposed",
            "confirm_be_exposed",
            "w7_confirm_rows_used",
            "w7b_confirm_rows_used",
            "w8_diagnostic_rows_used",
            "banking77_rows_used",
            "typed_decisions_final_or_test_used",
        ):
            if receipt.get(key) is not False:
                raise RuntimeError(f"W9 forbidden exposure: {key}")
        if receipt.get("campaign_cells_populated") != 0:
            raise RuntimeError("W9 candidate populated campaign cells")
        if receipt.get("hira_frozen") is not True:
            raise RuntimeError("W9 HIRACore must remain frozen")
        if receipt.get("a13_frozen") is not True:
            raise RuntimeError("W9 A13 must remain frozen")

        hira = receipt_path.parent / "hira.pt"
        scorer = receipt_path.parent / "scorer.pt"
        if file_sha256(hira) != receipt["hira_sha256"]:
            raise RuntimeError("W9 HIRA checkpoint SHA mismatch")
        if file_sha256(scorer) != receipt["scorer_sha256"]:
            raise RuntimeError("W9 scorer checkpoint SHA mismatch")

        if receipt["trainable_parameter_count"] != expected_trainable_parameters(
            candidate
        ):
            raise RuntimeError("W9 trainable parameter receipt mismatch")
        if receipt["optimization_seed"] != candidate_seed(candidate):
            raise RuntimeError("W9 optimization seed changed")
        if receipt["scorer_parameter_count"] != EXPECTED_SCORER_COUNTS[candidate]:
            raise RuntimeError("W9 scorer total parameter count changed")
        if receipt.get("typed_loss_enabled") is not False:
            raise RuntimeError("W9 typed loss must remain disabled")
        if receipt.get("pair_margin_loss_enabled") is not False:
            raise RuntimeError("W9 pair margin must remain disabled")

        rows.append((receipt, hira, scorer, receipt_path))

    names = {receipt["candidate"] for receipt, _, _, _ in rows}
    if names != set(CANDIDATES) or len(rows) != len(CANDIDATES):
        raise RuntimeError(
            f"expected exact W9 candidate set, got {sorted(names)}"
        )

    train_hashes = {receipt["train_cache_sha256"] for receipt, *_ in rows}
    dev_hashes = {receipt["dev_cache_sha256"] for receipt, *_ in rows}
    if len(train_hashes) != 1 or len(dev_hashes) != 1:
        raise RuntimeError("W9 candidates did not share exact cache provenance")

    by_name = {receipt["candidate"]: receipt for receipt, *_ in rows}
    for candidate in CANDIDATES[1:]:
        row = by_name[candidate]
        if row["train_base_count"] != 256:
            raise RuntimeError(f"W9 {candidate} train base budget changed")
        if row["optimizer_steps"] != 256 * EPOCHS:
            raise RuntimeError(f"W9 {candidate} optimizer budget changed")
        if not 1 <= int(row["selected_epoch"]) <= EPOCHS:
            raise RuntimeError(f"W9 {candidate} selected epoch invalid")
        if row.get("alignment_loss_enabled") is not True:
            raise RuntimeError(f"W9 {candidate} alignment loss disabled")

    frozen = by_name["frozen-w6e-control"]
    if frozen["selected_epoch"] != 0:
        raise RuntimeError("W9 frozen control must remain epoch 0")
    if frozen["optimizer_steps"] != 0:
        raise RuntimeError("W9 frozen control optimizer steps changed")
    if frozen.get("alignment_loss_enabled") is not False:
        raise RuntimeError("W9 frozen control cannot train alignment")

    if (
        by_name["asymmetric-bridge-primary"]["optimization_seed"]
        == by_name["asymmetric-bridge-replica"]["optimization_seed"]
    ):
        raise RuntimeError("W9 primary/replica seeds must differ")

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints_root = args.out / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    frozen_rows = []
    for receipt, hira, scorer, _ in sorted(
        rows,
        key=lambda item: ORDER[item[0]["candidate"]],
    ):
        candidate = receipt["candidate"]
        target = checkpoints_root / candidate
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hira, target / "hira.pt")
        shutil.copy2(scorer, target / "scorer.pt")
        shutil.copy2(
            Path(hira).parent / "receipt.json",
            target / "candidate-receipt.json",
        )
        frozen_rows.append(
            {
                "candidate": candidate,
                "selected_epoch": receipt["selected_epoch"],
                "optimization_seed": receipt["optimization_seed"],
                "trainable_parameter_count": receipt[
                    "trainable_parameter_count"
                ],
                "scorer_parameter_count": receipt[
                    "scorer_parameter_count"
                ],
                "hira_sha256": receipt["hira_sha256"],
                "scorer_sha256": receipt["scorer_sha256"],
                "checkpoint_dir": f"checkpoints/{candidate}",
            }
        )

    freeze = {
        "schema_version": "r8-w9-freeze-v1",
        "status": "PASS",
        "selection_authority": "DEV_BC_INDEPENDENT_PER_CANDIDATE",
        "all_candidates_frozen_before_confirm": True,
        "candidates": frozen_rows,
        "train_cache_sha256": next(iter(train_hashes)),
        "dev_cache_sha256": next(iter(dev_hashes)),
        "equal_train_base_budget": True,
        "equal_optimizer_steps": True,
        "trainable_optimizer_steps": 256 * EPOCHS,
        "primary_replica_seed_independence": True,
        "confirm_bd_exposed": False,
        "confirm_be_exposed": False,
        "w8_diagnostic_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(freeze, sort_keys=True))


if __name__ == "__main__":
    main()
