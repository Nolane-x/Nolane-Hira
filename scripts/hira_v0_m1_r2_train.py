from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from nmd.calibration import TypedReliabilityCalibrator
from nmd.mainline import (
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m1_mechanism,
)
from nmd.mainline_m1_cache import compile_m1_frozen_cache
from nmd.mainline_m1_r2_authority import generate_m1_r2_authority
from nmd.mainline_m1_r2_training import (
    TinySelectiveRiskHead,
    build_selected_calibrator,
    m1_r2_dev_qualification,
    safe_calibration_tournament,
    train_selective_risk_tournament,
)
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

R1_RUN_ID = 36305969587
R1_ARTIFACT_ID = 10927625673
R1_ARTIFACT_DIGEST = "sha256:79ff7eb67c3ce82ed82562787d813b75d35e7eca1bed0211f01ddf1b09c31dca"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("M1-R2 A13 weight identity changed")
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
        raise RuntimeError("M1-R2 unexpected T0 receipt")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M1-R2 T0 status changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1-R2 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1-R2 T0 bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M1-R2 unexpected W34 receipt")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M1-R2 W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M1-R2 W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M1-R2 W34 checkpoint bytes changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M1-R2 W34 training contains confirm leakage")
    return checkpoint


def _validate_r1(directory: Path) -> tuple[dict[str, object], Path]:
    receipt_path = directory / "receipt.json"
    ood_path = directory / "ood-head.pt"
    if not receipt_path.is_file() or not ood_path.is_file():
        raise FileNotFoundError("M1-R2 R1 artifact incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "hira-v0-mainline-m1-train-dev-v1":
        raise RuntimeError("M1-R2 unexpected R1 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M1-R2 R1 receipt did not pass")
    if receipt.get("dev_qualification", {}).get("pass") is not False:
        raise RuntimeError("M1-R2 expects frozen R1 DEV qualification failure")
    if receipt.get("sealed_exposure_authorized") is not False:
        raise RuntimeError("M1-R2 R1 unexpectedly authorized sealed exposure")
    if receipt.get("selective_confirm_exposed") is not False:
        raise RuntimeError("M1-R2 UE was exposed in R1")
    if receipt.get("ood_confirm_exposed") is not False:
        raise RuntimeError("M1-R2 UI was exposed in R1")
    if receipt.get("sealed_rows_used") is not False:
        raise RuntimeError("M1-R2 R1 leaked sealed rows")
    ood = receipt.get("ood", {})
    if ood.get("selected_candidate") != "semantic-linear":
        raise RuntimeError("M1-R2 frozen R1 OOD candidate changed")
    if int(ood.get("selected_parameter_count", -1)) != 6:
        raise RuntimeError("M1-R2 frozen R1 OOD capacity changed")
    if tuple(ood.get("selected_feature_names", ())) != (
        "state_question_cosine",
        "state_option_max_cosine",
        "state_option_mean_cosine",
        "state_option_std_cosine",
        "normalized_log_k",
    ):
        raise RuntimeError("M1-R2 frozen R1 OOD feature identity changed")
    if abs(float(ood.get("selected_threshold")) - 0.52) > 1e-12:
        raise RuntimeError("M1-R2 frozen R1 OOD threshold changed")
    if file_sha256(ood_path) != receipt.get("ood_head_sha256"):
        raise RuntimeError("M1-R2 frozen R1 OOD checkpoint SHA mismatch")
    return receipt, ood_path


def _calibration_summary(selection: dict[str, object]) -> dict[str, object]:
    return {
        "selected_candidate": selection["selected_candidate"],
        "selected_parameter_count": selection["selected_parameter_count"],
        "selected_epoch": selection["selected_epoch"],
        "selected_dev": selection["selected_dev"],
        "control_dev": selection["control_dev"],
        "candidate_summaries": selection["candidate_summaries"],
    }


def _risk_summary(selection: dict[str, object]) -> dict[str, object]:
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
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    r1_receipt, r1_ood_path = _validate_r1(args.r1_dir)

    encoder = _load_a13()
    model = build_hira_v0_m1_mechanism(encoder, t0, w34)

    train_rows = generate_m1_r2_authority("train")
    dev_rows = generate_m1_r2_authority("dev")
    train_cache = compile_m1_frozen_cache(model, train_rows, partition="train")
    dev_cache = compile_m1_frozen_cache(model, dev_rows, partition="dev")

    calibration = safe_calibration_tournament(train_cache, dev_cache)
    calibrator = build_selected_calibrator(calibration)
    risk = train_selective_risk_tournament(train_cache, dev_cache, calibrator)
    dev_qualification = m1_r2_dev_qualification(calibration, risk)

    args.out.mkdir(parents=True, exist_ok=True)
    train_cache_path = args.out / "r2-train-cache.pt"
    dev_cache_path = args.out / "r2-dev-cache.pt"
    torch.save(train_cache, train_cache_path)
    torch.save(dev_cache, dev_cache_path)

    calibrator_path = None
    calibrator_sha = None
    if calibrator is not None:
        calibrator_path = args.out / "calibrator.pt"
        torch.save(
            {
                "schema_version": "hira-v0-m1-r2-calibrator-v1",
                "candidate": calibration["selected_candidate"],
                "parameter_count": calibration["selected_parameter_count"],
                "state_dict": calibration["selected_state_dict"],
            },
            calibrator_path,
        )
        calibrator_sha = file_sha256(calibrator_path)

    risk_path = args.out / "selective-risk-head.pt"
    torch.save(
        {
            "schema_version": "hira-v0-m1-r2-selective-risk-head-v1",
            "candidate": risk["selected_candidate"],
            "parameter_count": risk["selected_parameter_count"],
            "feature_indices": tuple(risk["selected_feature_indices"]),
            "feature_names": tuple(risk["selected_feature_names"]),
            "feature_mean": risk["selected_feature_mean"],
            "feature_std": risk["selected_feature_std"],
            "threshold": float(risk["selected_threshold"]),
            "state_dict": risk["selected_state_dict"],
        },
        risk_path,
    )
    risk_sha = file_sha256(risk_path)

    selected_risk = TinySelectiveRiskHead(len(risk["selected_feature_indices"]))
    selected_risk.load_state_dict(risk["selected_state_dict"], strict=True)
    if selected_risk.trainable_parameter_count != int(risk["selected_parameter_count"]):
        raise RuntimeError("M1-R2 selected risk head parameter count changed")

    frozen_ood_out = args.out / "frozen-r1-ood-head.pt"
    shutil.copy2(r1_ood_path, frozen_ood_out)
    if file_sha256(frozen_ood_out) != r1_receipt["ood_head_sha256"]:
        raise RuntimeError("M1-R2 frozen OOD copy changed bytes")

    if model.runtime.state_encode_calls != len(train_rows) + len(dev_rows):
        raise RuntimeError("M1-R2 state-once count changed")

    receipt = {
        "schema_version": "hira-v0-mainline-m1-r2-train-dev-v1",
        "status": "PASS",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "decision_core_frozen": True,
        "decision_core_trainable_parameter_count": 0,
        "calibration": _calibration_summary(calibration),
        "calibrator_sha256": calibrator_sha,
        "selective_risk": _risk_summary(risk),
        "selective_risk_head_sha256": risk_sha,
        "dev_qualification": dev_qualification,
        "case_counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
        },
        "state_encode_count": model.runtime.state_encode_calls,
        "state_encodes_per_case": 1.0,
        "r1_ood": {
            "source_run_id": R1_RUN_ID,
            "source_artifact_id": R1_ARTIFACT_ID,
            "source_artifact_digest": R1_ARTIFACT_DIGEST,
            "candidate": r1_receipt["ood"]["selected_candidate"],
            "parameter_count": r1_receipt["ood"]["selected_parameter_count"],
            "threshold": r1_receipt["ood"]["selected_threshold"],
            "feature_indices": r1_receipt["ood"]["selected_feature_indices"],
            "feature_names": r1_receipt["ood"]["selected_feature_names"],
            "head_sha256": r1_receipt["ood_head_sha256"],
            "copied_head_sha256": file_sha256(frozen_ood_out),
            "retrained": False,
            "retuned": False,
        },
        "r1_ua_ud_rows_used_for_r2_fitting": False,
        "r1_uf_uh_rows_used_for_r2_fitting": False,
        "ue_exposed": False,
        "ui_exposed": False,
        "sealed_rows_used": False,
        "sealed_exposure_authorized": bool(dev_qualification["pass"]),
        "confidence_used_as_ood_authority": False,
        "quality_claim_made": False,
        "cache_sha256": {
            "train": file_sha256(train_cache_path),
            "dev": file_sha256(dev_cache_path),
        },
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M1_R2_TRAIN_DEV=" + json.dumps({
        "calibration": receipt["calibration"],
        "selective_risk": receipt["selective_risk"],
        "dev_qualification": receipt["dev_qualification"],
        "r1_ood": receipt["r1_ood"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
