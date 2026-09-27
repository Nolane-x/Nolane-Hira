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
    evaluate_calibration_cache,
    evaluate_final_reliability_policy,
    evaluate_ood_head,
    m1_sealed_qualification,
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
        raise RuntimeError(f"M1 confirm A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M1 confirm unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M1 confirm T0 receipt identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1 confirm T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1 confirm T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M1 confirm unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M1 confirm W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M1 confirm W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M1 confirm W34 checkpoint bytes changed")
    return checkpoint


def _validate_train(directory: Path) -> dict[str, object]:
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "hira-v0-mainline-m1-train-dev-v1":
        raise RuntimeError("M1 confirm unexpected TRAIN DEV receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M1 TRAIN DEV did not pass")
    if receipt.get("t0_checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1 TRAIN DEV T0 identity changed")
    if (
        receipt.get("transfer_checkpoint_sha256")
        != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
    ):
        raise RuntimeError("M1 TRAIN DEV transfer identity changed")
    if receipt.get("decision_core_frozen") is not True:
        raise RuntimeError("M1 TRAIN DEV decision core was not frozen")
    if int(receipt.get("decision_core_trainable_parameter_count", -1)) != 0:
        raise RuntimeError("M1 TRAIN DEV trained decision-core parameters")
    if receipt.get("sealed_rows_used") is not False:
        raise RuntimeError("M1 TRAIN DEV leaked sealed authority")
    if receipt.get("selective_confirm_exposed") is not False:
        raise RuntimeError("M1 selective confirm was exposed before selection")
    if receipt.get("ood_confirm_exposed") is not False:
        raise RuntimeError("M1 OOD confirm was exposed before selection")
    if receipt.get("confidence_used_as_ood_authority") is not False:
        raise RuntimeError("M1 confidence-as-OOD shortcut detected")
    if receipt.get("sealed_exposure_authorized") is not True:
        raise RuntimeError("M1 DEV qualification did not authorize sealed exposure")
    if receipt.get("dev_qualification", {}).get("pass") is not True:
        raise RuntimeError("M1 DEV qualification is not PASS")
    return receipt


def _load_calibrator(directory: Path, receipt: dict[str, object]):
    calibration = receipt["calibration"]
    candidate = calibration["selected_candidate"]
    parameter_count = int(calibration["selected_parameter_count"])
    if candidate == "control":
        if parameter_count != 0:
            raise RuntimeError("M1 control calibration parameter count changed")
        return None

    path = directory / "calibrator.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != receipt["calibrator_sha256"]:
        raise RuntimeError("M1 calibrator checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "hira-v0-m1-calibrator-v1":
        raise RuntimeError("unexpected M1 calibrator schema")
    if payload.get("candidate") != candidate:
        raise RuntimeError("M1 calibrator candidate identity changed")
    if int(payload.get("parameter_count", -1)) != parameter_count:
        raise RuntimeError("M1 calibrator parameter count changed")

    calibrator = TypedReliabilityCalibrator(candidate)
    calibrator.load_state_dict(payload["state_dict"], strict=True)
    if calibrator.trainable_parameter_count != parameter_count:
        raise RuntimeError("M1 calibrator live parameter count changed")
    calibrator.eval()
    return calibrator


def _load_ood(directory: Path, receipt: dict[str, object]):
    path = directory / "ood-head.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != receipt["ood_head_sha256"]:
        raise RuntimeError("M1 OOD checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "hira-v0-m1-ood-head-v1":
        raise RuntimeError("unexpected M1 OOD checkpoint schema")

    selection = receipt["ood"]
    if payload.get("candidate") != selection["selected_candidate"]:
        raise RuntimeError("M1 OOD candidate identity changed")
    if int(payload.get("parameter_count", -1)) != int(
        selection["selected_parameter_count"]
    ):
        raise RuntimeError("M1 OOD parameter count changed")
    if abs(float(payload.get("threshold")) - float(selection["selected_threshold"])) > 1e-12:
        raise RuntimeError("M1 OOD threshold changed")
    if tuple(payload.get("feature_names", ())) != tuple(
        selection["selected_feature_names"]
    ):
        raise RuntimeError("M1 OOD feature identity changed")

    feature_indices = tuple(int(i) for i in payload["feature_indices"])
    head = TinyOODHead(len(feature_indices))
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    return (
        head,
        payload["feature_mean"].float(),
        payload["feature_std"].float(),
        feature_indices,
        float(payload["threshold"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_receipt = _validate_train(args.train_dir)
    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    calibrator = _load_calibrator(args.train_dir, train_receipt)
    (
        ood_head,
        feature_mean,
        feature_std,
        feature_indices,
        ood_threshold,
    ) = _load_ood(args.train_dir, train_receipt)

    selective_threshold = float(
        train_receipt["selective_policy"]["threshold"]
    )

    encoder = _load_a13()
    model = build_hira_v0_m1_mechanism(encoder, t0, w34)

    print("HIRA_V0_M1_UE_UI_SEALED_EXPOSURE_BEGIN", flush=True)
    ue_rows = generate_m1_authority(
        "selective_confirm",
        allow_sealed=True,
    )
    ui_rows = generate_m1_authority(
        "ood_confirm",
        allow_sealed=True,
    )
    ue_cache = compile_m1_frozen_cache(
        model,
        ue_rows,
        partition="selective_confirm",
    )
    ui_cache = compile_m1_frozen_cache(
        model,
        ui_rows,
        partition="ood_confirm",
    )

    selected_calibration = evaluate_calibration_cache(ue_cache, calibrator)
    control_calibration = evaluate_calibration_cache(ue_cache, None)
    ood_metrics = evaluate_ood_head(
        ood_head,
        feature_mean,
        feature_std,
        ue_cache,
        ui_cache,
        feature_indices=feature_indices,
        threshold=ood_threshold,
    )
    final_policy = evaluate_final_reliability_policy(
        ue_cache,
        ui_cache,
        calibrator,
        ood_head,
        feature_mean,
        feature_std,
        feature_indices=feature_indices,
        ood_threshold=ood_threshold,
        selective_threshold=selective_threshold,
    )
    gates = m1_sealed_qualification(
        selected_calibration,
        control_calibration,
        ood_metrics,
        final_policy,
    )
    outcome = (
        "HIRA_V0_M1_RELIABILITY_READY"
        if gates["pass"]
        else "HIRA_V0_M1_RELIABILITY_FAIL"
    )

    if model.runtime.state_encode_calls != len(ue_rows) + len(ui_rows):
        raise RuntimeError("M1 sealed state-once count changed")

    args.out.mkdir(parents=True, exist_ok=True)
    ue_path = args.out / "ue-cache.pt"
    ui_path = args.out / "ui-cache.pt"
    torch.save(ue_cache, ue_path)
    torch.save(ui_cache, ui_path)

    policy = {
        "schema_version": "hira-v0-m1-reliability-policy-v1",
        "status": "qualified" if gates["pass"] else "failed",
        "calibration_candidate": train_receipt["calibration"]["selected_candidate"],
        "calibration_authority": "HIRA-V0-M1-UE",
        "selective_confidence_threshold": selective_threshold,
        "ood_candidate": train_receipt["ood"]["selected_candidate"],
        "ood_authority": "HIRA-V0-M1-UE-UI",
        "ood_threshold": ood_threshold,
        "ood_calibrator_id": train_receipt["ood_head_sha256"],
        "train_dev_receipt_authority": train_receipt["schema_version"],
    }
    (args.out / "policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema_version": "hira-v0-mainline-m1-sealed-confirm-v1",
        "status": "PASS",
        "outcome": outcome,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "train_dev_dev_qualification": train_receipt["dev_qualification"],
        "train_dev_sealed_exposure_authorized": True,
        "calibration_candidate": train_receipt["calibration"]["selected_candidate"],
        "calibrator_sha256": train_receipt["calibrator_sha256"],
        "selective_threshold": selective_threshold,
        "ood_candidate": train_receipt["ood"]["selected_candidate"],
        "ood_head_sha256": train_receipt["ood_head_sha256"],
        "ood_threshold": ood_threshold,
        "selected_calibration": selected_calibration,
        "control_calibration": control_calibration,
        "ood_metrics": ood_metrics,
        "final_policy": final_policy,
        "gates": gates,
        "ue_case_count": len(ue_rows),
        "ui_case_count": len(ui_rows),
        "state_encode_count": model.runtime.state_encode_calls,
        "state_encodes_per_case": 1.0,
        "ue_cache_sha256": file_sha256(ue_path),
        "ui_cache_sha256": file_sha256(ui_path),
        "sealed_rows_used_for_fitting": False,
        "sealed_rows_used_for_threshold_selection": False,
        "sealed_rows_used_for_candidate_ranking": False,
        "decision_core_frozen": True,
        "decision_core_trainable_parameter_count": 0,
        "confidence_used_as_ood_authority": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M1_SEALED_FINAL=" + json.dumps({
        "outcome": outcome,
        "selected_calibration": selected_calibration,
        "control_calibration": control_calibration,
        "ood_metrics": ood_metrics,
        "final_policy": final_policy,
        "gates": gates,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
