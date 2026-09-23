from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.semantic_cross_candidate_binding import (
    CANDIDATES,
    GLOBAL_SEED,
    CrossCandidateBindingMatcher,
    dev_selection_key,
    load_binding_cache,
    train_matcher,
)


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_cache = load_binding_cache(args.train_cache)
    dev_cache = load_binding_cache(args.dev_cache)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    matcher = CrossCandidateBindingMatcher(args.candidate)
    history, state, best_metrics = train_matcher(
        matcher,
        train_cache,
        dev_cache,
    )
    selected_epoch = int(best_metrics["epoch"])

    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "matcher.pt"
    torch.save(state, state_path)

    receipt = {
        "schema_version": "r8-w5i-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "projection_dim": 128,
        "trainable_parameter_count": sum(p.numel() for p in matcher.parameters()),
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
