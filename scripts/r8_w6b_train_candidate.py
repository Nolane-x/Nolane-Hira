from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.hira import HIRACore, count_parameters
from nmd.typed_competitive_cache import (
    CANDIDATES,
    HIRA_PARAMETER_COUNT,
    JOINT_PARAMETER_COUNT,
    SCORER_PARAMETER_COUNT,
    configure_candidate_trainability,
    dev_selection_key,
    file_sha256,
    load_w6b_cache,
    train_w6b_candidate,
)


W3_HEAD_SHA256 = "2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c"
W5I_SCORER_SHA256 = "5ce9cdceb5a87c8b18b8d50e68396f23184e1dfb9acad338f25f24277d438a0d"
GLOBAL_SEED = 809


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--w3-head", type=Path, required=True)
    parser.add_argument("--w5i-scorer", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.w3_head) != W3_HEAD_SHA256:
        raise RuntimeError("W3 selected head SHA-256 mismatch")
    if file_sha256(args.w5i_scorer) != W5I_SCORER_SHA256:
        raise RuntimeError("W5i forward scorer SHA-256 mismatch")

    train_cache = load_w6b_cache(
        args.train_cache,
        expected_split="train",
    )
    dev_cache = load_w6b_cache(
        args.dev_cache,
        expected_split="dev",
    )

    torch.manual_seed(0)
    hira = HIRACore(d_model=256, dropout=0.05)
    hira_state = torch.load(
        args.w3_head,
        map_location="cpu",
        weights_only=True,
    )
    hira.load_state_dict(hira_state, strict=True)
    if count_parameters(hira) != HIRA_PARAMETER_COUNT:
        raise RuntimeError("unexpected HIRA parameter count")

    scorer = None
    if args.candidate != "legacy-w3-joint":
        scorer = CompetitiveCoarseScorer(
            d_model=256,
            d_rel=128,
        )
        scorer_state = torch.load(
            args.w5i_scorer,
            map_location="cpu",
            weights_only=True,
        )
        scorer.load_state_dict(scorer_state, strict=True)
        scorer_count = sum(p.numel() for p in scorer.parameters())
        if scorer_count != SCORER_PARAMETER_COUNT:
            raise RuntimeError("unexpected competitive scorer parameter count")

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    history, best_hira, best_scorer, best_metrics = train_w6b_candidate(
        args.candidate,
        hira,
        scorer,
        train_cache["cases"],
        dev_cache["cases"],
    )
    selected_epoch = int(best_metrics["epoch"])

    # Trainable count reflects the frozen candidate contract after
    # configure_candidate_trainability has been applied.
    _, trainable = configure_candidate_trainability(
        args.candidate,
        hira,
        scorer,
    )
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = count_parameters(hira) + (
        0 if scorer is None else sum(p.numel() for p in scorer.parameters())
    )
    expected_trainable = {
        "legacy-w3-joint": HIRA_PARAMETER_COUNT,
        "competitive-w5i-joint": JOINT_PARAMETER_COUNT,
        "competitive-w5i-scorer-only": SCORER_PARAMETER_COUNT,
    }[args.candidate]
    expected_total = (
        HIRA_PARAMETER_COUNT
        if args.candidate == "legacy-w3-joint"
        else JOINT_PARAMETER_COUNT
    )
    if trainable_count != expected_trainable:
        raise RuntimeError("W6b trainable parameter contract changed")
    if total_count != expected_total:
        raise RuntimeError("W6b total parameter contract changed")

    args.out.mkdir(parents=True, exist_ok=True)
    hira_path = args.out / "hira.pt"
    torch.save(best_hira, hira_path)
    scorer_path = None
    if best_scorer is not None:
        scorer_path = args.out / "scorer.pt"
        torch.save(best_scorer, scorer_path)

    receipt = {
        "schema_version": "r8-w6b-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "selected_epoch": selected_epoch,
        "selected_dev_metrics": best_metrics,
        "selection_key": list(
            dev_selection_key(best_metrics, selected_epoch)
        ),
        "history": history,
        "hira_sha256": file_sha256(hira_path),
        "train_cache_sha256": file_sha256(args.train_cache),
        "dev_cache_sha256": file_sha256(args.dev_cache),
        "scorer_sha256": (
            None
            if scorer_path is None
            else file_sha256(scorer_path)
        ),
        "base_w3_head_sha256": W3_HEAD_SHA256,
        "base_w5i_scorer_sha256": (
            None
            if scorer is None
            else W5I_SCORER_SHA256
        ),
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "full_k_only": True,
        "adaptive_budget": False,
        "epochs": 6,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "global_seed": GLOBAL_SEED,
        "loss_weights": {
            "hard_ce": 1.0,
            "teacher_kl": 0.5,
            "brier": 0.1,
            "soft_brier": 0.5,
            "ordinal_mae": 0.2,
        },
        "confirm_exposed": False,
        "typed_decisions_final_or_test_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "selected_epoch": selected_epoch,
                "selected_dev_metrics": best_metrics,
                "hira_sha256": receipt["hira_sha256"],
                "scorer_sha256": receipt["scorer_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
