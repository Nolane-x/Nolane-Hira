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
    HIRA_PARAMETER_COUNT,
    JOINT_PARAMETER_COUNT,
    SCORER_PARAMETER_COUNT,
    evaluate_w6b_cases,
    file_sha256,
    load_w6b_cache,
)
from nmd.typed_domain_generalization import (
    CANDIDATES,
    GLOBAL_SEED,
    LR,
    WEIGHT_DECAY,
    dev_selection_key,
    expected_parameter_counts,
    train_w6d_candidate,
)


W6B_HIRA_SHA256 = "925f74094ac4ae583c015ea2a0be32ec692d885ec64dcf9cf3d94b64be0ccf42"
W6B_SCORER_SHA256 = "50abb2e8136599bcaf5c41d61036e3c335a7589c6244536cc0292dcea15b1ef0"


def domain_id_from_case_id(case_id: str) -> str | None:
    parts = case_id.split("-")
    if len(parts) >= 4 and parts[0] == "w6d":
        return parts[3].upper()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-cache", type=Path, required=True)
    parser.add_argument("--multi-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--w6b-hira", type=Path, required=True)
    parser.add_argument("--w6b-scorer", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.w6b_hira) != W6B_HIRA_SHA256:
        raise RuntimeError("W6b selected HIRA SHA mismatch")
    if file_sha256(args.w6b_scorer) != W6B_SCORER_SHA256:
        raise RuntimeError("W6b selected scorer SHA mismatch")

    single_cache = load_w6b_cache(
        args.single_cache,
        expected_split="train-single",
    )
    multi_cache = load_w6b_cache(
        args.multi_cache,
        expected_split="train-multi",
    )
    dev_cache = load_w6b_cache(
        args.dev_cache,
        expected_split="dev",
    )
    if len(single_cache["cases"]) != 384 or len(multi_cache["cases"]) != 384:
        raise RuntimeError("W6d TRAIN budget must be 384 cases for both sources")
    if len(dev_cache["cases"]) != 192:
        raise RuntimeError("W6d DEV-E must contain 192 cases")

    torch.manual_seed(0)
    hira = HIRACore(d_model=256, dropout=0.05)
    hira.load_state_dict(
        torch.load(args.w6b_hira, map_location="cpu", weights_only=True),
        strict=True,
    )
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.load_state_dict(
        torch.load(args.w6b_scorer, map_location="cpu", weights_only=True),
        strict=True,
    )
    if count_parameters(hira) != HIRA_PARAMETER_COUNT:
        raise RuntimeError("unexpected HIRA parameter count")
    if sum(p.numel() for p in scorer.parameters()) != SCORER_PARAMETER_COUNT:
        raise RuntimeError("unexpected scorer parameter count")

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    if args.candidate == "frozen-w6b-control":
        history = []
        best_hira = {
            name: tensor.detach().cpu().clone()
            for name, tensor in hira.state_dict().items()
        }
        best_scorer = {
            name: tensor.detach().cpu().clone()
            for name, tensor in scorer.state_dict().items()
        }
        best_metrics = evaluate_w6b_cases(
            hira, scorer, dev_cache["cases"], competitive=True
        )
        best_metrics["epoch"] = 0
        selected_epoch = 0
        train_cache_path = None
        source_metrics = {}
    else:
        train_cache = (
            single_cache
            if args.candidate == "single-source-scorer-only"
            else multi_cache
        )
        history, best_hira, best_scorer, best_metrics = train_w6d_candidate(
            args.candidate,
            hira,
            scorer,
            train_cache["cases"],
            dev_cache["cases"],
        )
        selected_epoch = int(best_metrics["epoch"])
        train_cache_path = (
            args.single_cache
            if args.candidate == "single-source-scorer-only"
            else args.multi_cache
        )
        source_metrics = {}
        if args.candidate == "single-source-scorer-only":
            source_metrics["A"] = evaluate_w6b_cases(
                hira, scorer, train_cache["cases"], competitive=True
            )
        else:
            for domain in ("A", "B", "C", "D"):
                rows = [
                    row for row in train_cache["cases"]
                    if domain_id_from_case_id(str(row["case_id"])) == domain
                ]
                if len(rows) != 96:
                    raise RuntimeError(
                        f"W6d multi-source domain {domain} expected 96 cases"
                    )
                source_metrics[domain] = evaluate_w6b_cases(
                    hira, scorer, rows, competitive=True
                )

    args.out.mkdir(parents=True, exist_ok=True)
    hira_path = args.out / "hira.pt"
    scorer_path = args.out / "scorer.pt"
    torch.save(best_hira, hira_path)
    torch.save(best_scorer, scorer_path)

    trainable_count, total_count = expected_parameter_counts(args.candidate)
    if total_count != JOINT_PARAMETER_COUNT:
        raise RuntimeError("W6d total parameter contract changed")

    receipt = {
        "schema_version": "r8-w6d-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "selected_epoch": selected_epoch,
        "selected_dev_metrics": best_metrics,
        "selection_key": (
            None
            if args.candidate == "frozen-w6b-control"
            else list(dev_selection_key(best_metrics, selected_epoch))
        ),
        "history": history,
        "source_train_metrics": source_metrics,
        "hira_sha256": file_sha256(hira_path),
        "scorer_sha256": file_sha256(scorer_path),
        "base_w6b_hira_sha256": W6B_HIRA_SHA256,
        "base_w6b_scorer_sha256": W6B_SCORER_SHA256,
        "train_cache_kind": (
            "none"
            if train_cache_path is None
            else (
                "single"
                if args.candidate == "single-source-scorer-only"
                else "multi"
            )
        ),
        "train_cache_sha256": (
            None
            if train_cache_path is None
            else file_sha256(train_cache_path)
        ),
        "single_cache_sha256": file_sha256(args.single_cache),
        "multi_cache_sha256": file_sha256(args.multi_cache),
        "dev_cache_sha256": file_sha256(args.dev_cache),
        "train_case_count": (
            0 if train_cache_path is None else 384
        ),
        "optimizer_case_steps": (
            0 if train_cache_path is None else 384 * 6
        ),
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "epochs": 0 if train_cache_path is None else 6,
        "lr": None if train_cache_path is None else LR,
        "weight_decay": (
            None if train_cache_path is None else WEIGHT_DECAY
        ),
        "global_seed": GLOBAL_SEED,
        "full_k_only": True,
        "adaptive_budget": False,
        "confirm_exposed": False,
        "w6b_confirm_rows_used": False,
        "w6c_confirm_rows_used": False,
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
