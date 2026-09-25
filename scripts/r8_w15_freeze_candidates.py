from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from nmd.anchor_preserving_residual_training import (
    CANDIDATES,
    EPOCHS,
    TRAINABLE_CANDIDATES,
    candidate_bounded,
    candidate_seed,
)
from nmd.typed_competitive_cache import file_sha256


ORDER = {name: index for index, name in enumerate(CANDIDATES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[dict, Path | None]] = []
    for receipt_path in sorted(args.candidates.rglob("receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "r8-w15-candidate-receipt-v1":
            continue
        if receipt.get("status") != "PASS":
            raise RuntimeError("W15 candidate receipt is not PASS")
        name = receipt.get("candidate")
        if name not in CANDIDATES:
            raise RuntimeError("W15 candidate identity changed")
        if receipt.get("confirm_ce_exposed") is not False:
            raise RuntimeError("W15 candidate exposed CONFIRM-CE")
        if receipt.get("confirm_cf_exposed") is not False:
            raise RuntimeError("W15 candidate exposed CONFIRM-CF")
        if receipt.get("hira_frozen") is not True:
            raise RuntimeError("W15 HIRACore must remain frozen")
        if receipt.get("scorer_frozen") is not True:
            raise RuntimeError("W15 semantic/competitive scorer must remain frozen")
        if receipt.get("a13_frozen") is not True:
            raise RuntimeError("W15 A13 must remain frozen")
        if receipt.get("forced_full_k") is not True:
            raise RuntimeError("W15 must use full-K relation evaluation")
        if receipt.get("adaptive_budget") is not False:
            raise RuntimeError("W15 adaptive budget must remain false")
        if receipt.get("typed_decisions_final_or_test_used") is not False:
            raise RuntimeError("W15 forbidden typed final/test exposure")
        if receipt.get("banking77_rows_used") is not False:
            raise RuntimeError("W15 forbidden Banking77 exposure")
        if int(receipt.get("campaign_cells_populated", -1)) != 0:
            raise RuntimeError("W15 campaign cells must remain empty")

        mixer_path = None
        if name in TRAINABLE_CANDIDATES:
            if int(receipt.get("trainable_parameter_count", -1)) != 6:
                raise RuntimeError("W15 residual trainable count changed")
            if int(receipt.get("optimizer_case_steps", -1)) != 384 * EPOCHS:
                raise RuntimeError("W15 optimizer case-step budget changed")
            if int(receipt.get("train_case_count", -1)) != 384:
                raise RuntimeError("W15 train case budget changed")
            if int(receipt.get("optimization_seed", -1)) != candidate_seed(name):
                raise RuntimeError("W15 optimization seed changed")
            if receipt.get("mixer_bounded") is not candidate_bounded(name):
                raise RuntimeError("W15 mixer bounded identity changed")
            selected_epoch = int(receipt.get("selected_epoch", -1))
            if not 1 <= selected_epoch <= EPOCHS:
                raise RuntimeError("W15 selected epoch invalid")
            mixer_path = receipt_path.parent / "mixer.pt"
            if not mixer_path.exists():
                raise RuntimeError("W15 trainable candidate mixer missing")
            if file_sha256(mixer_path) != receipt.get("mixer_sha256"):
                raise RuntimeError("W15 candidate mixer SHA mismatch")
        else:
            if int(receipt.get("trainable_parameter_count", -1)) != 0:
                raise RuntimeError("W15 frozen control gained trainable params")
            if int(receipt.get("selected_epoch", -1)) != 0:
                raise RuntimeError("W15 frozen control must be epoch zero")
            if receipt.get("mixer_sha256") is not None:
                raise RuntimeError("W15 frozen control must not have mixer")

        rows.append((receipt, mixer_path))

    names = {row[0]["candidate"] for row in rows}
    if names != set(CANDIDATES) or len(rows) != len(CANDIDATES):
        raise RuntimeError(
            f"W15 expected five candidate receipts, got {sorted(names)}"
        )

    train_hashes = {row[0]["train_cache_sha256"] for row in rows}
    dev_hashes = {row[0]["dev_cache_sha256"] for row in rows}
    hira_hashes = {row[0]["hira_sha256"] for row in rows}
    scorer_hashes = {row[0]["scorer_sha256"] for row in rows}
    if len(train_hashes) != 1 or len(dev_hashes) != 1:
        raise RuntimeError("W15 candidates did not share exact cache provenance")
    if len(hira_hashes) != 1 or len(scorer_hashes) != 1:
        raise RuntimeError("W15 candidates did not share exact frozen base")

    by_name = {row[0]["candidate"]: row[0] for row in rows}
    if (
        by_name["bounded-residual-primary"]["optimization_seed"]
        == by_name["bounded-residual-replica"]["optimization_seed"]
    ):
        raise RuntimeError("W15 bounded primary/replica seeds must differ")

    args.out.mkdir(parents=True, exist_ok=True)
    frozen_rows = []
    for receipt, mixer_path in sorted(
        rows,
        key=lambda row: ORDER[row[0]["candidate"]],
    ):
        name = receipt["candidate"]
        out_mixer = None
        mixer_sha = None
        if mixer_path is not None:
            out_mixer = args.out / f"{name}-mixer.pt"
            shutil.copyfile(mixer_path, out_mixer)
            mixer_sha = file_sha256(out_mixer)

        frozen_rows.append(
            {
                "candidate": name,
                "selected_epoch": receipt["selected_epoch"],
                "selected_dev_metrics": receipt["selected_dev_metrics"],
                "selection_key": receipt["selection_key"],
                "trainable_parameter_count": receipt[
                    "trainable_parameter_count"
                ],
                "optimizer_case_steps": receipt["optimizer_case_steps"],
                "optimization_seed": receipt["optimization_seed"],
                "mixer_bounded": receipt["mixer_bounded"],
                "mixer_sha256": mixer_sha,
                "hira_sha256": receipt["hira_sha256"],
                "scorer_sha256": receipt["scorer_sha256"],
            }
        )

    summary = {
        "schema_version": "r8-w15-freeze-v1",
        "status": "PASS",
        "selection_authority": "DEV_CD_INDEPENDENT_PER_CANDIDATE",
        "all_candidates_frozen_before_confirm": True,
        "train_cache_sha256": next(iter(train_hashes)),
        "dev_cache_sha256": next(iter(dev_hashes)),
        "hira_sha256": next(iter(hira_hashes)),
        "scorer_sha256": next(iter(scorer_hashes)),
        "equal_train_case_budget": True,
        "equal_optimizer_steps": True,
        "primary_replica_seed_independence": True,
        "confirm_ce_exposed": False,
        "confirm_cf_exposed": False,
        "typed_decisions_final_or_test_used": False,
        "banking77_rows_used": False,
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
