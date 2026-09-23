from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.semantic_late_interaction import (
    GLOBAL_SEED,
    LateInteractionMatcher,
    dev_selection_key,
    evaluate_matcher,
    load_late_interaction_cache,
    train_projected_matcher,
)


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def projection_dim(name: str) -> int | None:
    if name == "raw-maxsim":
        return None
    if name == "proj64-maxsim":
        return 64
    if name == "proj128-maxsim":
        return 128
    raise ValueError(f"unknown W5f candidate: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=["raw-maxsim", "proj64-maxsim", "proj128-maxsim"],
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_cache = load_late_interaction_cache(args.train_cache)
    dev_cache = load_late_interaction_cache(args.dev_cache)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    matcher = LateInteractionMatcher(projection_dim(args.candidate))
    if args.candidate == "raw-maxsim":
        history = []
        best_metrics = evaluate_matcher(matcher, dev_cache)
        best_metrics["epoch"] = 0
        state = matcher.state_dict()
        selected_epoch = 0
    else:
        history, state, best_metrics = train_projected_matcher(
            matcher,
            train_cache,
            dev_cache,
        )
        selected_epoch = int(best_metrics["epoch"])

    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "matcher.pt"
    torch.save(state, state_path)
    params = sum(p.numel() for p in matcher.parameters())

    receipt = {
        "schema_version": "r8-w5f-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "projection_dim": projection_dim(args.candidate),
        "trainable_parameter_count": params,
        "selected_epoch": selected_epoch,
        "selected_dev_metrics": best_metrics,
        "selection_key": list(dev_selection_key(best_metrics, selected_epoch)),
        "history": history,
        "matcher_sha256": file_sha256(state_path),
        "confirm_exposed": False,
        "forbidden_benchmark_data_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
