from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.calibration import TypedReliabilityCalibrator
from nmd.mainline import (
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m1_mechanism,
)
from nmd.mainline_m1_authority import generate_m1_authority
from nmd.mainline_m1_cache import compile_m1_frozen_cache
from nmd.mainline_m1_training import (
    TinyOODHead,
    m1_dev_qualification,
    select_selective_threshold,
    train_calibration_tournament,
    train_ood_tournament,
)
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M1 A13 weight SHA mismatch: {actual}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=256,
    )


def _validate_t0(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w28-candidate-receipt-v1":
        raise RuntimeError("M1 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M1 T0 receipt identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1 T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M1 unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M1 W34 receipt did not pass")
    if receipt.get("qualification_outcome") != "W34_REFERENCE_QUALIFIED":
        raise RuntimeError("M1 W34 qualification provenance changed")
    if int(receipt.get("candidate_parameter_count", -1)) != 8192:
        raise RuntimeError("M1 W34 parameter count changed")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M1 W34 receipt checkpoint SHA changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M1 W34 checkpoint bytes changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M1 W34 training receipt contains confirm leakage")
    return checkpoint


def _selected_calibrator(selection: dict[str, object]):
    mode = selection["selected_candidate"]
    state = selection["selected_state_dict"]
    if mode == "control":
        if state is not None:
            raise RuntimeError("M1 control calibration unexpectedly has state")
        return None
    calibrator = TypedReliabilityCalibrator(mode)
    calibrator.load_state_dict(state, strict=True)
    calibrator.eval()
    return calibrator


def _calibration_summary(selection: dict[str, object]) -> dict[str, object]:
    return {
        "selected_candidate": selection["selected_candidate"],
        "selected_parameter_count": selection["selected_parameter_count"],
        "selected_epoch": selection["selected_epoch"],
        "selected_dev": selection["selected_dev"],
        "control_dev": selection["control_dev"],
        "candidate_summaries": [
            {
                "candidate": row["candidate"],
                "parameter_count": row["parameter_count"],
                "selected_epoch": row["selected_epoch"],
                "selected_dev": row["selected_dev"],
            }
            for row in selection["candidates"]
        ],
    }


def _ood_summary(selection: dict[str, object]) -> dict[str, object]:
    return {
        "selected_candidate": selection["selected_candidate"],
        "selected_parameter_count": selection["selected_parameter_count"],
        "selected_epoch": selection["selected_epoch"],
        "selected_threshold": selection["selected_threshold"],
        "selected_dev": selection["selected_dev"],
        "selected_feature_indices": list(selection["selected_feature_indices"]),
        "selected_feature_names": list(selection["selected_feature_names"]),
        "candidate_summaries": [
            {
                "candidate": row["candidate"],
                "parameter_count": row["parameter_count"],
                "selected_epoch": row["selected_epoch"],
                "selected_threshold": row["selected_threshold"],
                "selected_dev": row["selected_dev"],
                "feature_names": list(row["feature_names"]),
            }
            for row in selection["candidates"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    encoder = _load_a13()
    model = build_hira_v0_m1_mechanism(encoder, t0, w34)

    # Only non-sealed M1 authority is generated here.
    cal_train_rows = generate_m1_authority("cal_train")
    cal_dev_rows = generate_m1_authority("cal_dev")
    ood_train_rows = generate_m1_authority("ood_train")
    ood_dev_rows = generate_m1_authority("ood_dev")

    cal_train = compile_m1_frozen_cache(
        model, cal_train_rows, partition="cal_train"
    )
    cal_dev = compile_m1_frozen_cache(
        model, cal_dev_rows, partition="cal_dev"
    )
    ood_train = compile_m1_frozen_cache(
        model, ood_train_rows, partition="ood_train"
    )
    ood_dev = compile_m1_frozen_cache(
        model, ood_dev_rows, partition="ood_dev"
    )

    calibration = train_calibration_tournament(cal_train, cal_dev)
    calibrator = _selected_calibrator(calibration)
    selective = select_selective_threshold(cal_dev, calibrator)
    ood = train_ood_tournament(
        cal_train,
        ood_train,
        cal_dev,
        ood_dev,
    )
    dev_qualification = m1_dev_qualification(
        calibration,
        selective,
        ood,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    cache_paths = {}
    for name, cache in (
        ("cal-train", cal_train),
        ("cal-dev", cal_dev),
        ("ood-train", ood_train),
        ("ood-dev", ood_dev),
    ):
        path = args.out / f"{name}-cache.pt"
        torch.save(cache, path)
        cache_paths[name] = path

    calibrator_path = None
    calibrator_sha = None
    if calibrator is not None:
        calibrator_path = args.out / "calibrator.pt"
        torch.save(
            {
                "schema_version": "hira-v0-m1-calibrator-v1",
                "candidate": calibration["selected_candidate"],
                "parameter_count": calibration["selected_parameter_count"],
                "state_dict": calibration["selected_state_dict"],
            },
            calibrator_path,
        )
        calibrator_sha = file_sha256(calibrator_path)

    ood_path = args.out / "ood-head.pt"
    torch.save(
        {
            "schema_version": "hira-v0-m1-ood-head-v1",
            "candidate": ood["selected_candidate"],
            "parameter_count": ood["selected_parameter_count"],
            "feature_indices": tuple(ood["selected_feature_indices"]),
            "feature_names": tuple(ood["selected_feature_names"]),
            "feature_mean": ood["selected_feature_mean"],
            "feature_std": ood["selected_feature_std"],
            "threshold": float(ood["selected_threshold"]),
            "state_dict": ood["selected_state_dict"],
        },
        ood_path,
    )
    ood_sha = file_sha256(ood_path)

    # Validate selected OOD checkpoint can be reconstructed without the core.
    selected_head = TinyOODHead(len(ood["selected_feature_indices"]))
    selected_head.load_state_dict(ood["selected_state_dict"], strict=True)
    if selected_head.trainable_parameter_count != int(ood["selected_parameter_count"]):
        raise RuntimeError("M1 selected OOD head parameter count changed")

    total_rows = (
        len(cal_train_rows)
        + len(cal_dev_rows)
        + len(ood_train_rows)
        + len(ood_dev_rows)
    )
    if model.runtime.state_encode_calls != total_rows:
        raise RuntimeError("M1 empirical TRAIN/DEV state-once count changed")

    receipt = {
        "schema_version": "hira-v0-mainline-m1-train-dev-v1",
        "status": "PASS",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "transfer_core_maturity": model.manifest.transfer_core,
        "decision_core_frozen": True,
        "decision_core_trainable_parameter_count": 0,
        "calibration": _calibration_summary(calibration),
        "calibrator_sha256": calibrator_sha,
        "selective_policy": selective,
        "ood": _ood_summary(ood),
        "dev_qualification": dev_qualification,
        "ood_head_sha256": ood_sha,
        "cache_sha256": {
            name: file_sha256(path)
            for name, path in cache_paths.items()
        },
        "case_counts": {
            "cal_train": len(cal_train_rows),
            "cal_dev": len(cal_dev_rows),
            "ood_train": len(ood_train_rows),
            "ood_dev": len(ood_dev_rows),
        },
        "state_encode_count": model.runtime.state_encode_calls,
        "state_encodes_per_case": 1.0,
        "selective_confirm_exposed": False,
        "ood_confirm_exposed": False,
        "sealed_rows_used": False,
        "sealed_exposure_authorized": bool(dev_qualification["pass"]),
        "confidence_used_as_ood_authority": False,
        "quality_claim_made": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M1_TRAIN_DEV=" + json.dumps({
        "status": receipt["status"],
        "calibration": receipt["calibration"],
        "selective_policy": receipt["selective_policy"],
        "ood": receipt["ood"],
        "dev_qualification": receipt["dev_qualification"],
        "case_counts": receipt["case_counts"],
        "sealed_rows_used": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
