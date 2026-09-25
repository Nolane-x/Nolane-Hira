from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.conjunctive_cache import load_w7_cache
from nmd.freeform_attribution_training import (
    CANDIDATES,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    candidate_seed,
    dev_selection_key,
    evaluate_w7,
    expected_trainable_parameters,
    train_w7b_candidate,
)
from nmd.hira import HIRACore, count_parameters
from nmd.typed_competitive_cache import (
    HIRA_PARAMETER_COUNT,
    SCORER_PARAMETER_COUNT,
    file_sha256,
)


W6E_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W6E_SCORER_SHA256 = "6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e"


def _domain_metrics(hira, scorer, cases):
    result = {}
    for domain in ("AN", "AO", "AP", "AQ"):
        rows = [row for row in cases if row.get("domain_id") == domain]
        if len(rows) != 96:
            raise RuntimeError(
                f"W7b TRAIN domain {domain} expected 96 cases, got {len(rows)}"
            )
        result[domain] = evaluate_w7(hira, scorer, rows)
    return result


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
        raise RuntimeError("W7b base W6e HIRA SHA mismatch")
    if file_sha256(args.w6e_scorer) != W6E_SCORER_SHA256:
        raise RuntimeError("W7b base W6e scorer SHA mismatch")

    train_cache = load_w7_cache(
        args.train_cache,
        expected_split="train",
    )
    dev_cache = load_w7_cache(
        args.dev_cache,
        expected_split="dev-ar",
    )
    if len(train_cache["cases"]) != 384:
        raise RuntimeError("W7b TRAIN must contain 384 cases")
    if len(dev_cache["cases"]) != 192:
        raise RuntimeError("W7b DEV-AR must contain 192 cases")

    torch.manual_seed(0)
    hira = HIRACore(d_model=256, dropout=0.05)
    hira.load_state_dict(
        torch.load(args.w6e_hira, map_location="cpu", weights_only=True),
        strict=True,
    )
    if count_parameters(hira) != HIRA_PARAMETER_COUNT:
        raise RuntimeError("W7b HIRA parameter count changed")

    base_scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    base_scorer.load_state_dict(
        torch.load(args.w6e_scorer, map_location="cpu", weights_only=True),
        strict=True,
    )
    if sum(p.numel() for p in base_scorer.parameters()) != SCORER_PARAMETER_COUNT:
        raise RuntimeError("W7b base scorer parameter count changed")

    scorer = base_scorer

    seed = candidate_seed(args.candidate)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.candidate == "frozen-w6e-control":
        history = []
        selected_metrics = evaluate_w7(
            hira,
            scorer,
            dev_cache["cases"],
        )
        selected_metrics["epoch"] = 0
        selected_epoch = 0
        scorer_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in scorer.state_dict().items()
        }
        source_metrics = {}
    else:
        history, scorer_state, selected_metrics = train_w7b_candidate(
            args.candidate,
            hira,
            scorer,
            train_cache["cases"],
            dev_cache["cases"],
            epochs=EPOCHS,
        )
        selected_epoch = int(selected_metrics["epoch"])
        source_metrics = _domain_metrics(
            hira,
            scorer,
            train_cache["cases"],
        )

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

    trainable_count = expected_trainable_parameters(args.candidate)
    total_scorer_count = sum(p.numel() for p in scorer.parameters())
    expected_total_scorer = SCORER_PARAMETER_COUNT
    if total_scorer_count != expected_total_scorer:
        raise RuntimeError("W7b total scorer parameter contract changed")

    receipt = {
        "schema_version": "r8-w7b-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "selected_epoch": selected_epoch,
        "selected_dev_metrics": selected_metrics,
        "selection_key": (
            None
            if args.candidate == "frozen-w6e-control"
            else list(dev_selection_key(selected_metrics, selected_epoch))
        ),
        "history": history,
        "source_train_metrics": source_metrics,
        "hira_sha256": file_sha256(hira_path),
        "scorer_sha256": file_sha256(scorer_path),
        "base_w6e_hira_sha256": W6E_HIRA_SHA256,
        "base_w6e_scorer_sha256": W6E_SCORER_SHA256,
        "train_cache_sha256": file_sha256(args.train_cache),
        "dev_cache_sha256": file_sha256(args.dev_cache),
        "train_case_count": (
            0 if args.candidate == "frozen-w6e-control" else 384
        ),
        "optimizer_case_steps": (
            0
            if args.candidate == "frozen-w6e-control"
            else 384 * EPOCHS
        ),
        "trainable_parameter_count": trainable_count,
        "scorer_parameter_count": total_scorer_count,
        "total_parameter_count": count_parameters(hira) + total_scorer_count,
        "epochs": (
            0 if args.candidate == "frozen-w6e-control" else EPOCHS
        ),
        "lr": (
            None if args.candidate == "frozen-w6e-control" else LR
        ),
        "weight_decay": (
            None
            if args.candidate == "frozen-w6e-control"
            else WEIGHT_DECAY
        ),
        "optimization_seed": seed,
        "hira_frozen": True,
        "full_k_only": True,
        "adaptive_budget": False,
        "factor_branch_enabled": False,
        "loss_mode": args.candidate,
        "confirm_as_exposed": False,
        "confirm_at_exposed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
        "w6d_confirm_rows_used": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
        "w6h_confirm_rows_used": False,
        "w6i_diagnostic_rows_used": False,
        "w6j_diagnostic_rows_used": False,
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
