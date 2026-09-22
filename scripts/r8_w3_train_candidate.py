from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from nmd.hira import HIRACore, count_parameters
from nmd.losses import LossWeights
from nmd.typed_feature_cache import (
    W3_EPOCHS,
    W3_GLOBAL_SEED,
    cache_role_cases,
    dev_selection_key,
    load_w3_feature_cache,
    train_w3_candidate,
)


R15_HEAD_SHA256 = "007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f"
EXPECTED_HEAD_PARAMS = 422_159


LOSS_FAMILIES = {
    "balanced": LossWeights(
        hard_ce=1.0,
        teacher_kl=0.5,
        brier=0.1,
        soft_brier=0.5,
        ordinal_mae=0.2,
    ),
    "soft": LossWeights(
        hard_ce=0.5,
        teacher_kl=1.0,
        brier=0.05,
        soft_brier=1.0,
        ordinal_mae=0.2,
    ),
}


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--r15-head", type=Path, required=True)
    parser.add_argument(
        "--init",
        choices=["r15", "fresh13"],
        required=True,
    )
    parser.add_argument(
        "--loss-family",
        choices=sorted(LOSS_FAMILIES),
        required=True,
    )
    parser.add_argument(
        "--lr",
        type=float,
        choices=[3e-4, 1e-3],
        required=True,
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.r15_head) != R15_HEAD_SHA256:
        raise RuntimeError("R15 selected head SHA-256 mismatch")

    cache = load_w3_feature_cache(args.cache)
    metadata = cache["metadata"]
    if metadata.get("dataset_split") != "train":
        raise RuntimeError("W3a cache is not TRAIN-only")
    train_cases = cache_role_cases(cache, "train")
    dev_cases = cache_role_cases(cache, "dev")
    if len(train_cases) != 960 or len(dev_cases) != 240:
        raise RuntimeError("W3 train/dev authority count mismatch")

    if args.init == "fresh13":
        torch.manual_seed(13)
        hira = HIRACore(d_model=256, dropout=0.05)
    else:
        torch.manual_seed(0)
        hira = HIRACore(d_model=256, dropout=0.05)
        state = torch.load(
            args.r15_head,
            map_location="cpu",
            weights_only=True,
        )
        hira.load_state_dict(state, strict=True)

    if count_parameters(hira) != EXPECTED_HEAD_PARAMS:
        raise RuntimeError("HIRA head parameter count changed")

    random.seed(W3_GLOBAL_SEED)
    torch.manual_seed(W3_GLOBAL_SEED)

    weights = LOSS_FAMILIES[args.loss_family]
    history, best_state, best_metrics = train_w3_candidate(
        hira,
        train_cases,
        dev_cases,
        weights=weights,
        lr=args.lr,
        epochs=W3_EPOCHS,
        seed=W3_GLOBAL_SEED,
        weight_decay=0.01,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    head_path = args.out / "hira-head.pt"
    torch.save(best_state, head_path)
    selected_epoch = int(best_metrics["epoch"])
    selection_key = list(
        dev_selection_key(
            best_metrics,
            epoch=selected_epoch,
        )
    )
    receipt = {
        "schema_version": "r8-w3-candidate-receipt-v1",
        "status": "PASS",
        "candidate_name": args.name,
        "scope": "typed-decisions TRAIN/DEV only; test forbidden",
        "init": args.init,
        "loss_family": args.loss_family,
        "loss_weights": weights.__dict__,
        "lr": args.lr,
        "weight_decay": 0.01,
        "epochs": W3_EPOCHS,
        "global_seed": W3_GLOBAL_SEED,
        "train_case_count": len(train_cases),
        "dev_case_count": len(dev_cases),
        "train_case_id_sha256": metadata["train_case_id_sha256"],
        "dev_case_id_sha256": metadata["dev_case_id_sha256"],
        "selected_epoch": selected_epoch,
        "selected_dev_metrics": best_metrics,
        "selection_key": selection_key,
        "history": history,
        "head_sha256": file_sha256(head_path),
        "head_parameter_count": count_parameters(hira),
        "full_k_only": True,
        "adaptive_budget": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_name": args.name,
                "selected_epoch": selected_epoch,
                "selected_dev_metrics": best_metrics,
                "head_sha256": receipt["head_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
