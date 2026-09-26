from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.compositional_projection_cache import load_w28_cache
from nmd.compositional_projection_eval import (
    CANDIDATE_SEEDS,
    GRAD_CLIP,
    PROJECTION_EPOCHS,
    PROJECTION_LR,
    PROJECTION_TEMPERATURE,
    PROJECTION_WEIGHT_DECAY,
    TRAIN_BATCH_SIZE,
    train_projection,
)
from nmd.typed_competitive_cache import file_sha256

W9_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W9_SCORER_SHA256 = "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"


def _load_w9_projection(freeze_path: Path, checkpoints: Path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r8-w9-freeze-v1" or freeze.get("status") != "PASS":
        raise RuntimeError("W28 unexpected W9 freeze")
    rows = {row["candidate"]: row for row in freeze["candidates"]}
    row = rows.get("projection-semantic-control")
    if row is None:
        raise RuntimeError("W28 W9 projection-semantic-control missing")
    root = checkpoints / row["checkpoint_dir"]
    hira_path = root / "hira.pt"
    scorer_path = root / "scorer.pt"
    if file_sha256(hira_path) != W9_HIRA_SHA256:
        raise RuntimeError("W28 frozen HIRA SHA mismatch")
    if file_sha256(scorer_path) != W9_SCORER_SHA256:
        raise RuntimeError("W28 frozen scorer SHA mismatch")
    state = torch.load(scorer_path, map_location="cpu", weights_only=True)
    projection = state.get("projection.weight")
    if not isinstance(projection, torch.Tensor) or tuple(projection.shape) != (128,256):
        raise RuntimeError("W28 W9 projection shape changed")
    return freeze, row, projection.detach().float().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--w9-freeze", type=Path, required=True)
    parser.add_argument("--w9-checkpoints", type=Path, required=True)
    parser.add_argument("--candidate", choices=("T0","T1"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_cache = load_w28_cache(args.train_cache, expected_partition="train")
    dev_cache = load_w28_cache(args.dev_cache, expected_partition="dev")
    freeze, row, w9_projection = _load_w9_projection(
        args.w9_freeze,
        args.w9_checkpoints,
    )

    projection, selected, history = train_projection(
        args.candidate,
        train_cache,
        dev_cache,
        w9_projection,
    )

    checkpoint = {
        "schema_version": "r8-w28-candidate-checkpoint-v1",
        "candidate": args.candidate,
        "kind": "projection",
        "projection_weight": projection,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out/"candidate.pt"
    torch.save(checkpoint, checkpoint_path)

    receipt = {
        "schema_version": "r8-w28-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "kind": "projection",
        "selected_dev_epoch": int(selected["epoch"]),
        "selected_dev_metrics": selected["dev"],
        "selected_train_loss": float(selected["train_factor_ce"]),
        "selection_key": selected["selection_key"],
        "history": history,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "train_cache_sha256": file_sha256(args.train_cache),
        "dev_cache_sha256": file_sha256(args.dev_cache),
        "w9_freeze_status": freeze["status"],
        "w9_projection_scorer_sha256": row["scorer_sha256"],
        "w9_hira_sha256": W9_HIRA_SHA256,
        "w9_scorer_sha256": W9_SCORER_SHA256,
        "initialization": "exact-frozen-w9-projection",
        "optimization_seed": CANDIDATE_SEEDS[args.candidate],
        "epochs": PROJECTION_EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "optimizer_steps": 12 * PROJECTION_EPOCHS,
        "lr": PROJECTION_LR,
        "weight_decay": PROJECTION_WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "projection_temperature": PROJECTION_TEMPERATURE,
        "objective": "equal-sum-three-factor-ce",
        "trainable_parameter_count": 32768,
        "a13_frozen": True,
        "selection_performed": True,
        "selection_partition": "ET",
        "confirm_exposed": False,
        "reference_outputs_used_as_training_targets": False,
        "qualification_rows_used_for_hira_selection": False,
        "w27_rows_used": False,
        "w26_rows_used": False,
        "w25_rows_used": False,
        "w24_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out/"receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt,sort_keys=True))


if __name__ == "__main__":
    main()
