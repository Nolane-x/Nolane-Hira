from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.hira import HIRACore, count_parameters
from nmd.semantic_token_curriculum import (
    EPOCHS,
    GLOBAL_SEED,
    RELATION_MODES,
    dev_selection_key,
    load_token_cache,
    train_token_candidate,
)

W3_SHA = "2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c"
EXPECTED_PARAMS = 422_159
LR = 3e-4


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
    parser.add_argument("--w3-head", type=Path, required=True)
    parser.add_argument("--mode", choices=RELATION_MODES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.w3_head) != W3_SHA:
        raise RuntimeError("W3 selected head SHA mismatch")

    train_cache = load_token_cache(args.train_cache)
    dev_cache = load_token_cache(args.dev_cache)

    torch.manual_seed(0)
    hira = HIRACore(d_model=256, dropout=0.05)
    state = torch.load(
        args.w3_head,
        map_location="cpu",
        weights_only=True,
    )
    hira.load_state_dict(state, strict=True)
    if count_parameters(hira) != EXPECTED_PARAMS:
        raise RuntimeError("HIRA parameter count changed")

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    history, best_state, best_metrics = train_token_candidate(
        hira,
        train_cache,
        dev_cache,
        relation_mode=args.mode,
        lr=LR,
        epochs=EPOCHS,
        seed=GLOBAL_SEED,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    head_path = args.out / "hira-head.pt"
    torch.save(best_state, head_path)
    epoch = int(best_metrics["epoch"])
    receipt = {
        "schema_version": "r8-w5b-candidate-v1",
        "status": "PASS",
        "relation_mode": args.mode,
        "init": "w3",
        "lr": LR,
        "epochs": EPOCHS,
        "selected_epoch": epoch,
        "selected_dev_metrics": best_metrics,
        "selection_key": list(dev_selection_key(best_metrics, epoch)),
        "head_sha256": file_sha256(head_path),
        "head_parameter_count": count_parameters(hira),
        "confirm_exposed": False,
        "forbidden_benchmark_data_used": False,
        "history": history,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "relation_mode": args.mode,
                "selected_epoch": epoch,
                "selected_dev_metrics": best_metrics,
                "head_sha256": receipt["head_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
