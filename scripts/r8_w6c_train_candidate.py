from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.calibration import TypedReliabilityCalibrator
from nmd.typed_competitive_cache import (
    HIRA_PARAMETER_COUNT,
    SCORER_PARAMETER_COUNT,
)
from nmd.typed_reliability_cache import (
    CANDIDATES,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    W6B_HIRA_SHA256,
    W6B_SCORER_SHA256,
    candidate_to_mode,
    file_sha256,
    load_w6c_logit_cache,
    train_w6c_candidate,
)


BASE_PRODUCTION_PARAMETER_COUNT = (
    HIRA_PARAMETER_COUNT + SCORER_PARAMETER_COUNT
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_cache = load_w6c_logit_cache(
        args.train_cache,
        expected_split="train",
    )
    dev_cache = load_w6c_logit_cache(
        args.dev_cache,
        expected_split="dev",
    )
    history, best_state, best_metrics = train_w6c_candidate(
        args.candidate,
        train_cache,
        dev_cache,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    mode = candidate_to_mode(args.candidate)
    calibrator_path = None
    calibrator_sha = None
    selected_parameters = None
    trainable_count = 0

    if mode is not None:
        if best_state is None:
            raise RuntimeError(
                "trainable W6c candidate has no calibrator checkpoint"
            )
        calibrator = TypedReliabilityCalibrator(mode)
        calibrator.load_state_dict(best_state, strict=True)
        trainable_count = calibrator.trainable_parameter_count
        expected = (
            3
            if args.candidate == "primitive-temperature"
            else 4
        )
        if trainable_count != expected:
            raise RuntimeError(
                "unexpected W6c trainable parameter count"
            )
        calibrator_path = args.out / "calibrator.pt"
        torch.save(best_state, calibrator_path)
        calibrator_sha = file_sha256(calibrator_path)
        temperatures = [
            float(value)
            for value in calibrator.temperatures().detach().cpu()
        ]
        selected_parameters = {
            "temperatures": temperatures,
            "noul_true_bias": (
                None
                if calibrator.noul_true_bias is None
                else float(
                    calibrator.noul_true_bias.detach().cpu()
                )
            ),
        }
    elif best_state is not None:
        raise RuntimeError(
            "frozen production control unexpectedly has a checkpoint"
        )

    receipt = {
        "schema_version": "r8-w6c-candidate-receipt-v1",
        "status": "PASS",
        "candidate": args.candidate,
        "calibration_mode": mode,
        "selected_epoch": int(best_metrics["epoch"]),
        "selected_dev_metrics": best_metrics,
        "selected_parameters": selected_parameters,
        "history": history,
        "calibrator_sha256": calibrator_sha,
        "trainable_parameter_count": trainable_count,
        "base_production_parameter_count": BASE_PRODUCTION_PARAMETER_COUNT,
        "total_parameter_count": (
            BASE_PRODUCTION_PARAMETER_COUNT + trainable_count
        ),
        "base_w6b_hira_sha256": W6B_HIRA_SHA256,
        "base_w6b_scorer_sha256": W6B_SCORER_SHA256,
        "train_cache_sha256": file_sha256(args.train_cache),
        "dev_cache_sha256": file_sha256(args.dev_cache),
        "epochs": 0 if mode is None else EPOCHS,
        "optimizer": None if mode is None else "Adam",
        "lr": None if mode is None else LR,
        "weight_decay": None if mode is None else WEIGHT_DECAY,
        "objective": (
            "teacher_kl_1.0_plus_soft_brier_1.0"
            if mode is not None
            else "none"
        ),
        "hira_frozen": True,
        "competitive_scorer_frozen": True,
        "confirm_exposed": False,
        "w6b_confirm_rows_used": False,
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
