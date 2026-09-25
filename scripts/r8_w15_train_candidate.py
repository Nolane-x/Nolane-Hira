from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.anchor_preserving_residual import (
    AnchorPreservingResidualMixer,
    count_anchor_residual_parameters,
)
from nmd.anchor_preserving_residual_cache import load_w15_cache
from nmd.anchor_preserving_residual_training import (
    CANDIDATES,
    EPOCHS,
    LR,
    TRAINABLE_CANDIDATES,
    WEIGHT_DECAY,
    candidate_bounded,
    candidate_seed,
    dev_selection_key,
    evaluate_w15,
    train_w15_candidate,
)
from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore
from nmd.typed_competitive_cache import file_sha256


W9_HIRA_SHA256 = (
    "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
)
W9_SCORER_SHA256 = (
    "8078acd153be4cadf109713faa1efede0c58c31397f41db7f5aff3a12ab74102"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--w9-hira", type=Path, required=True)
    parser.add_argument("--w9-scorer", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.w9_hira) != W9_HIRA_SHA256:
        raise RuntimeError("W15 frozen HIRA SHA mismatch")
    if file_sha256(args.w9_scorer) != W9_SCORER_SHA256:
        raise RuntimeError("W15 W9 semantic scorer SHA mismatch")

    train_cache = load_w15_cache(
        args.train_cache,
        expected_split="train",
    )
    dev_cache = load_w15_cache(
        args.dev_cache,
        expected_split="dev-cd",
    )
    if len(train_cache["cases"]) != 384:
        raise RuntimeError("W15 TRAIN cache must contain 384 cases")
    if len(dev_cache["cases"]) != 96:
        raise RuntimeError("W15 DEV-CD cache must contain 96 cases")
    if train_cache["metadata"].get("confirm_exposed") is not False:
        raise RuntimeError("W15 TRAIN cache unexpectedly marked confirm")
    if dev_cache["metadata"].get("confirm_exposed") is not False:
        raise RuntimeError("W15 DEV cache unexpectedly marked confirm")

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(args.w9_hira, map_location="cpu", weights_only=True),
        strict=True,
    )
    hira.eval()
    for parameter in hira.parameters():
        parameter.requires_grad_(False)

    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(args.w9_scorer, map_location="cpu", weights_only=True),
        strict=True,
    )
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)

    seed = candidate_seed(args.candidate)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mixer = None
    history: list[dict[str, object]] = []
    selected_epoch = 0
    mixer_state = None

    if args.candidate in TRAINABLE_CANDIDATES:
        mixer = AnchorPreservingResidualMixer(
            bounded=candidate_bounded(args.candidate)
        )
        if count_anchor_residual_parameters(mixer) != 6:
            raise RuntimeError("W15 residual mixer parameter count changed")
        history, mixer_state, selected_metrics = train_w15_candidate(
            args.candidate,
            hira,
            scorer,
            mixer,
            train_cache["cases"],
            dev_cache["cases"],
            epochs=EPOCHS,
        )
        selected_epoch = int(selected_metrics["epoch"])
    else:
        selected_metrics = evaluate_w15(
            args.candidate,
            hira,
            scorer,
            dev_cache["cases"],
            mixer=None,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    mixer_path = None
    mixer_sha = None
    if mixer_state is not None:
        mixer_path = args.out / "mixer.pt"
        torch.save(mixer_state, mixer_path)
        mixer_sha = file_sha256(mixer_path)

    receipt = {
        "schema_version": "r8-w15-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "selected_epoch": selected_epoch,
        "selected_dev_metrics": selected_metrics,
        "selection_key": (
            None
            if args.candidate not in TRAINABLE_CANDIDATES
            else list(dev_selection_key(selected_metrics, selected_epoch))
        ),
        "history": history,
        "mixer_bounded": (
            None
            if args.candidate not in TRAINABLE_CANDIDATES
            else candidate_bounded(args.candidate)
        ),
        "mixer_sha256": mixer_sha,
        "trainable_parameter_count": (
            0 if args.candidate not in TRAINABLE_CANDIDATES else 6
        ),
        "optimizer_case_steps": (
            0
            if args.candidate not in TRAINABLE_CANDIDATES
            else len(train_cache["cases"]) * EPOCHS
        ),
        "train_case_count": (
            0
            if args.candidate not in TRAINABLE_CANDIDATES
            else len(train_cache["cases"])
        ),
        "epochs": (
            0
            if args.candidate not in TRAINABLE_CANDIDATES
            else EPOCHS
        ),
        "lr": (
            None
            if args.candidate not in TRAINABLE_CANDIDATES
            else LR
        ),
        "weight_decay": (
            None
            if args.candidate not in TRAINABLE_CANDIDATES
            else WEIGHT_DECAY
        ),
        "optimization_seed": seed,
        "hira_sha256": W9_HIRA_SHA256,
        "scorer_sha256": W9_SCORER_SHA256,
        "train_cache_sha256": file_sha256(args.train_cache),
        "dev_cache_sha256": file_sha256(args.dev_cache),
        "hira_frozen": True,
        "scorer_frozen": True,
        "a13_frozen": True,
        "forced_full_k": True,
        "adaptive_budget": False,
        "confirm_ce_exposed": False,
        "confirm_cf_exposed": False,
        "typed_decisions_final_or_test_used": False,
        "banking77_rows_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
