from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from nmd.competitive import CompetitiveCoarseScorer
from nmd.field_semantic_rescue import (
    ADAPTER_PARAMETER_COUNT,
    CANDIDATES,
    EPOCHS,
    GLOBAL_SEED,
    LR,
    SCORER_PARAMETER_COUNT,
    WEIGHT_DECAY,
    SemanticAdaptedCompetitiveScorer,
    SemanticResidualAdapter,
    configure_trainability,
    dev_selection_key,
    evaluate_w6h,
    train_w6h_candidate,
)
from nmd.hira import HIRACore, count_parameters
from nmd.typed_competitive_cache import (
    HIRA_PARAMETER_COUNT,
    file_sha256,
    load_w6b_cache,
)

W6E_HIRA_SHA256 = "d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588"
W6E_SCORER_SHA256 = "6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e"


def domain_from_case_id(case_id: str) -> str | None:
    parts = case_id.split("-")
    if len(parts) >= 4 and parts[0] == "w6h" and parts[1] == "train":
        return parts[2].upper()
    return None


def make_models(hira_path: Path, scorer_path: Path, candidate: str):
    if file_sha256(hira_path) != W6E_HIRA_SHA256:
        raise RuntimeError("W6e joint-primary HIRA SHA mismatch")
    if file_sha256(scorer_path) != W6E_SCORER_SHA256:
        raise RuntimeError("W6e joint-primary scorer SHA mismatch")

    hira = HIRACore(d_model=256, dropout=0.0)
    hira.load_state_dict(
        torch.load(hira_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    base = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    base.load_state_dict(
        torch.load(scorer_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    if count_parameters(hira) != HIRA_PARAMETER_COUNT:
        raise RuntimeError("unexpected HIRA parameter count")
    if sum(p.numel() for p in base.parameters()) != SCORER_PARAMETER_COUNT:
        raise RuntimeError("unexpected scorer parameter count")

    if candidate == "semantic-residual-adapter":
        scorer = SemanticAdaptedCompetitiveScorer(
            base,
            SemanticResidualAdapter(),
        )
    else:
        scorer = base
    return hira, scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--w6e-hira", type=Path, required=True)
    parser.add_argument("--w6e-scorer", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_cache = load_w6b_cache(args.train_cache, expected_split="train")
    dev_cache = load_w6b_cache(args.dev_cache, expected_split="dev-x")
    if len(train_cache["cases"]) != 384:
        raise RuntimeError("W6h TRAIN must contain 384 cases")
    if len(dev_cache["cases"]) != 192:
        raise RuntimeError("W6h DEV-X must contain 192 cases")

    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    hira, scorer = make_models(args.w6e_hira, args.w6e_scorer, args.candidate)

    if args.candidate == "frozen-joint-control":
        configure_trainability(args.candidate, hira, scorer)
        history = []
        best_metrics = evaluate_w6h(hira, scorer, dev_cache["cases"])
        best_metrics["epoch"] = 0
        selected_epoch = 0
    else:
        history, _, best_metrics = train_w6h_candidate(
            args.candidate,
            hira,
            scorer,
            train_cache["cases"],
            dev_cache["cases"],
            epochs=EPOCHS,
            seed=GLOBAL_SEED,
        )
        selected_epoch = int(best_metrics["epoch"])

    source_metrics = {}
    for domain in ("T", "U", "V", "W"):
        rows = [
            row
            for row in train_cache["cases"]
            if domain_from_case_id(str(row["case_id"])) == domain
        ]
        if len(rows) != 96:
            raise RuntimeError(f"W6h source domain {domain} expected 96 cases")
        source_metrics[domain] = evaluate_w6h(hira, scorer, rows)

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
    torch.save(
        {
            name: tensor.detach().cpu().clone()
            for name, tensor in scorer.state_dict().items()
        },
        scorer_path,
    )

    if args.candidate == "frozen-joint-control":
        trainable = 0
        total = HIRA_PARAMETER_COUNT + SCORER_PARAMETER_COUNT
    elif args.candidate == "projection-retune-control":
        trainable = SCORER_PARAMETER_COUNT
        total = HIRA_PARAMETER_COUNT + SCORER_PARAMETER_COUNT
    else:
        trainable = ADAPTER_PARAMETER_COUNT
        total = HIRA_PARAMETER_COUNT + SCORER_PARAMETER_COUNT + ADAPTER_PARAMETER_COUNT

    receipt = {
        "schema_version": "r8-w6h-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "selected_epoch": selected_epoch,
        "selected_dev_metrics": best_metrics,
        "selection_key": (
            None
            if args.candidate == "frozen-joint-control"
            else list(dev_selection_key(best_metrics, selected_epoch))
        ),
        "history": history,
        "source_train_metrics": source_metrics,
        "hira_sha256": file_sha256(hira_path),
        "scorer_sha256": file_sha256(scorer_path),
        "base_w6e_hira_sha256": W6E_HIRA_SHA256,
        "base_w6e_scorer_sha256": W6E_SCORER_SHA256,
        "train_cache_sha256": file_sha256(args.train_cache),
        "dev_cache_sha256": file_sha256(args.dev_cache),
        "train_case_count": 0 if args.candidate == "frozen-joint-control" else 384,
        "optimizer_case_steps": 0 if args.candidate == "frozen-joint-control" else 384 * EPOCHS,
        "trainable_parameter_count": trainable,
        "total_parameter_count": total,
        "epochs": 0 if args.candidate == "frozen-joint-control" else EPOCHS,
        "lr": None if args.candidate == "frozen-joint-control" else LR,
        "weight_decay": None if args.candidate == "frozen-joint-control" else WEIGHT_DECAY,
        "global_seed": 0 if args.candidate == "frozen-joint-control" else GLOBAL_SEED,
        "hira_frozen": True,
        "full_k_only": True,
        "adaptive_budget": False,
        "confirm_y_exposed": False,
        "confirm_z_exposed": False,
        "w6e_confirm_rows_used": False,
        "w6f_diagnostic_rows_used": False,
        "w6g_diagnostic_rows_used": False,
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
