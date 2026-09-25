from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore, count_parameters
from nmd.semantic_alignment_bridge import SemanticAlignmentBridgeScorer
from nmd.semantic_alignment_cache import load_w9_cache
from nmd.semantic_alignment_eval import (
    ALIGNMENT_TEMPERATURE,
    evaluate_w9_checkpoint,
)
from nmd.semantic_alignment_training import (
    CANDIDATES,
    EPOCHS,
    GRAD_CLIP,
    LR,
    WEIGHT_DECAY,
    candidate_seed,
    dev_selection_key,
    expected_trainable_parameters,
    train_w9_candidate,
)
from nmd.typed_competitive_cache import (
    HIRA_PARAMETER_COUNT,
    SCORER_PARAMETER_COUNT,
    file_sha256,
)


W6E_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W6E_SCORER_SHA256 = "6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e"


def _build_scorer(candidate: str, base: CompetitiveCoarseScorer):
    if candidate in {
        "frozen-w6e-control",
        "projection-semantic-control",
    }:
        return base
    if candidate == "shared-bridge-semantic-control":
        return SemanticAlignmentBridgeScorer(
            base,
            mode="shared",
        )
    if candidate in {
        "asymmetric-bridge-primary",
        "asymmetric-bridge-replica",
    }:
        return SemanticAlignmentBridgeScorer(
            base,
            mode="asymmetric",
        )
    raise ValueError(f"unknown W9 candidate: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--w6e-hira", type=Path, required=True)
    parser.add_argument("--w6e-scorer", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.w6e_hira) != W6E_HIRA_SHA256:
        raise RuntimeError("W9 base W6e HIRA SHA mismatch")
    if file_sha256(args.w6e_scorer) != W6E_SCORER_SHA256:
        raise RuntimeError("W9 base W6e scorer SHA mismatch")

    train_cache = load_w9_cache(args.train_cache)
    dev_cache = load_w9_cache(args.dev_cache)

    if set(train_cache["metadata"]["domains"]) != {"AY", "AZ", "BA", "BB"}:
        raise RuntimeError("W9 TRAIN cache domain set changed")
    if set(dev_cache["metadata"]["domains"]) != {"BC"}:
        raise RuntimeError("W9 DEV cache domain set changed")
    if int(train_cache["metadata"]["base_count"]) != 256:
        raise RuntimeError("W9 TRAIN must contain 256 bases")
    if int(dev_cache["metadata"]["base_count"]) != 64:
        raise RuntimeError("W9 DEV-BC must contain 64 bases")

    torch.manual_seed(0)
    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(args.w6e_hira, map_location="cpu", weights_only=True),
        strict=True,
    )
    if count_parameters(hira) != HIRA_PARAMETER_COUNT:
        raise RuntimeError("W9 HIRA parameter count changed")

    seed = candidate_seed(args.candidate)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    base_scorer.load_state_dict(
        torch.load(args.w6e_scorer, map_location="cpu", weights_only=True),
        strict=True,
    )
    if sum(p.numel() for p in base_scorer.parameters()) != SCORER_PARAMETER_COUNT:
        raise RuntimeError("W9 base scorer parameter count changed")

    scorer = _build_scorer(args.candidate, base_scorer)

    if args.candidate == "frozen-w6e-control":
        for parameter in hira.parameters():
            parameter.requires_grad_(False)
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
        history = []
        selected_metrics = evaluate_w9_checkpoint(
            hira,
            scorer,
            dev_cache,
        )
        selected_metrics["epoch"] = 0
        selected_epoch = 0
        scorer_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in scorer.state_dict().items()
        }
    else:
        history, scorer_state, selected_metrics = train_w9_candidate(
            args.candidate,
            hira,
            scorer,
            train_cache,
            dev_cache,
            epochs=EPOCHS,
        )
        selected_epoch = int(selected_metrics["epoch"])

    args.out.mkdir(parents=True, exist_ok=True)
    hira_path = args.out / "hira.pt"
    scorer_path = args.out / "scorer.pt"
    torch.save(
        {
            name: tensor.detach().cpu().clone()
            for name, tensor in hira.state_dict().items()
        },
        hira_path,
    )
    torch.save(scorer_state, scorer_path)

    total_scorer_count = sum(p.numel() for p in scorer.parameters())
    trainable_count = expected_trainable_parameters(args.candidate)

    receipt = {
        "schema_version": "r8-w9-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "scorer_kind": (
            "competitive"
            if isinstance(scorer, CompetitiveCoarseScorer)
            else f"semantic-bridge-{scorer.mode}"
        ),
        "selected_epoch": selected_epoch,
        "selected_dev_metrics": selected_metrics,
        "selection_key": (
            None
            if args.candidate == "frozen-w6e-control"
            else list(dev_selection_key(selected_metrics, selected_epoch))
        ),
        "history": history,
        "hira_sha256": file_sha256(hira_path),
        "scorer_sha256": file_sha256(scorer_path),
        "base_w6e_hira_sha256": W6E_HIRA_SHA256,
        "base_w6e_scorer_sha256": W6E_SCORER_SHA256,
        "train_cache_sha256": file_sha256(args.train_cache),
        "dev_cache_sha256": file_sha256(args.dev_cache),
        "train_base_count": (
            0 if args.candidate == "frozen-w6e-control" else 256
        ),
        "optimizer_steps": (
            0
            if args.candidate == "frozen-w6e-control"
            else 256 * EPOCHS
        ),
        "trainable_parameter_count": trainable_count,
        "scorer_parameter_count": total_scorer_count,
        "total_parameter_count": count_parameters(hira) + total_scorer_count,
        "epochs": (
            0 if args.candidate == "frozen-w6e-control" else EPOCHS
        ),
        "lr": None if args.candidate == "frozen-w6e-control" else LR,
        "weight_decay": (
            None
            if args.candidate == "frozen-w6e-control"
            else WEIGHT_DECAY
        ),
        "grad_clip": (
            None
            if args.candidate == "frozen-w6e-control"
            else GRAD_CLIP
        ),
        "alignment_temperature": ALIGNMENT_TEMPERATURE,
        "optimization_seed": seed,
        "hira_frozen": True,
        "a13_frozen": True,
        "typed_loss_enabled": False,
        "pair_margin_loss_enabled": False,
        "alignment_loss_enabled": args.candidate != "frozen-w6e-control",
        "projection_log_scale_frozen": (
            args.candidate == "projection-semantic-control"
        ),
        "confirm_bd_exposed": False,
        "confirm_be_exposed": False,
        "w7_confirm_rows_used": False,
        "w7b_confirm_rows_used": False,
        "w8_diagnostic_rows_used": False,
        "banking77_rows_used": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }

    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
