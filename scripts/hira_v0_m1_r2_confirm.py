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
from nmd.mainline_m1_r2_training import (
    TinySelectiveRiskHead,
    evaluate_r2_final_policy,
    evaluate_risk_head,
    m1_r2_sealed_qualification,
)
from nmd.mainline_m1_training import (
    TinyOODHead,
    evaluate_calibration_cache,
    evaluate_ood_head,
    m1_sealed_qualification,
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
        raise RuntimeError("M1-R2 confirm A13 identity changed")
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
        raise RuntimeError("M1-R2 confirm unexpected T0 receipt")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1-R2 confirm T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M1-R2 confirm T0 bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M1-R2 confirm unexpected W34 receipt")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M1-R2 confirm W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M1-R2 confirm W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M1-R2 confirm W34 bytes changed")
    return checkpoint


def _validate_r2(directory: Path) -> dict[str, object]:
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "hira-v0-mainline-m1-r2-train-dev-v1":
        raise RuntimeError("M1-R2 confirm unexpected TRAIN DEV schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M1-R2 TRAIN DEV receipt did not pass")
    if receipt.get("dev_qualification", {}).get("pass") is not True:
        raise RuntimeError("M1-R2 DEV qualification did not pass")
    if receipt.get("sealed_exposure_authorized") is not True:
        raise RuntimeError("M1-R2 did not authorize sealed exposure")
    if receipt.get("ue_exposed") is not False or receipt.get("ui_exposed") is not False:
        raise RuntimeError("M1-R2 sealed evidence was exposed before confirmation")
    if receipt.get("sealed_rows_used") is not False:
        raise RuntimeError("M1-R2 sealed rows leaked into selection")
    if receipt.get("decision_core_frozen") is not True:
        raise RuntimeError("M1-R2 decision core was not frozen")
    if int(receipt.get("decision_core_trainable_parameter_count", -1)) != 0:
        raise RuntimeError("M1-R2 trained decision-core parameters")
    if receipt.get("r1_ua_ud_rows_used_for_r2_fitting") is not False:
        raise RuntimeError("M1-R2 reused R1 calibration rows")
    if receipt.get("r1_uf_uh_rows_used_for_r2_fitting") is not False:
        raise RuntimeError("M1-R2 reused R1 OOD rows")
    frozen_ood = receipt.get("r1_ood", {})
    if int(frozen_ood.get("source_run_id", -1)) != R1_RUN_ID:
        raise RuntimeError("M1-R2 frozen OOD source run changed")
    if int(frozen_ood.get("source_artifact_id", -1)) != R1_ARTIFACT_ID:
        raise RuntimeError("M1-R2 frozen OOD artifact changed")
    if frozen_ood.get("source_artifact_digest") != R1_ARTIFACT_DIGEST:
        raise RuntimeError("M1-R2 frozen OOD digest changed")
    if frozen_ood.get("candidate") != "semantic-linear":
        raise RuntimeError("M1-R2 frozen OOD candidate changed")
    if int(frozen_ood.get("parameter_count", -1)) != 6:
        raise RuntimeError("M1-R2 frozen OOD capacity changed")
    if frozen_ood.get("retrained") is not False or frozen_ood.get("retuned") is not False:
        raise RuntimeError("M1-R2 frozen OOD was changed")
    return receipt


def _load_calibrator(directory: Path, receipt: dict[str, object]):
    selection = receipt["calibration"]
    candidate = selection["selected_candidate"]
    if candidate == "control":
        if selection["selected_parameter_count"] != 0:
            raise RuntimeError("M1-R2 control calibration capacity changed")
        return None

    path = directory / "calibrator.pt"
    if file_sha256(path) != receipt["calibrator_sha256"]:
        raise RuntimeError("M1-R2 calibrator SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "hira-v0-m1-r2-calibrator-v1":
        raise RuntimeError("M1-R2 calibrator schema changed")
    if payload.get("candidate") != candidate:
        raise RuntimeError("M1-R2 calibrator candidate changed")
    calibrator = TypedReliabilityCalibrator(candidate)
    calibrator.load_state_dict(payload["state_dict"], strict=True)
    if calibrator.trainable_parameter_count != int(selection["selected_parameter_count"]):
        raise RuntimeError("M1-R2 calibrator capacity changed")
    calibrator.eval()
    return calibrator


def _load_risk(directory: Path, receipt: dict[str, object]):
    path = directory / "selective-risk-head.pt"
    if file_sha256(path) != receipt["selective_risk_head_sha256"]:
        raise RuntimeError("M1-R2 risk head SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "hira-v0-m1-r2-selective-risk-head-v1":
        raise RuntimeError("M1-R2 risk head schema changed")
    selection = receipt["selective_risk"]
    if payload.get("candidate") != selection["selected_candidate"]:
        raise RuntimeError("M1-R2 risk candidate changed")
    if int(payload.get("parameter_count", -1)) != int(selection["selected_parameter_count"]):
        raise RuntimeError("M1-R2 risk capacity changed")
    if abs(float(payload.get("threshold")) - float(selection["selected_threshold"])) > 1e-12:
        raise RuntimeError("M1-R2 risk threshold changed")
    feature_indices = tuple(int(i) for i in payload["feature_indices"])
    head = TinySelectiveRiskHead(len(feature_indices))
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    return (
        head,
        payload["feature_mean"].float(),
        payload["feature_std"].float(),
        feature_indices,
        float(payload["threshold"]),
    )


def _load_frozen_ood(directory: Path, receipt: dict[str, object]):
    path = directory / "frozen-r1-ood-head.pt"
    expected = receipt["r1_ood"]["head_sha256"]
    if file_sha256(path) != expected:
        raise RuntimeError("M1-R2 frozen OOD head SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "hira-v0-m1-ood-head-v1":
        raise RuntimeError("M1-R2 frozen OOD schema changed")
    if payload.get("candidate") != "semantic-linear":
        raise RuntimeError("M1-R2 frozen OOD candidate changed")
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
    parser.add_argument("--r2-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    receipt = _validate_r2(args.r2_dir)
    calibrator = _load_calibrator(args.r2_dir, receipt)
    (
        risk_head,
        risk_mean,
        risk_std,
        risk_indices,
        risk_threshold,
    ) = _load_risk(args.r2_dir, receipt)
    (
        ood_head,
        ood_mean,
        ood_std,
        ood_indices,
        ood_threshold,
    ) = _load_frozen_ood(args.r2_dir, receipt)

    encoder = _load_a13()
    model = build_hira_v0_m1_mechanism(encoder, t0, w34)

    print("HIRA_V0_M1_R2_UE_UI_SEALED_EXPOSURE_BEGIN", flush=True)
    ue_rows = generate_m1_authority("selective_confirm", allow_sealed=True)
    ui_rows = generate_m1_authority("ood_confirm", allow_sealed=True)
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
    risk_metrics = evaluate_risk_head(
        risk_head,
        risk_mean,
        risk_std,
        ue_cache,
        calibrator,
        feature_indices=risk_indices,
        threshold=risk_threshold,
    )
    ood_metrics = evaluate_ood_head(
        ood_head,
        ood_mean,
        ood_std,
        ue_cache,
        ui_cache,
        feature_indices=ood_indices,
        threshold=ood_threshold,
    )
    final_policy = evaluate_r2_final_policy(
        ue_cache,
        ui_cache,
        calibrator,
        risk_head,
        risk_mean,
        risk_std,
        ood_head,
        ood_mean,
        ood_std,
        risk_feature_indices=risk_indices,
        risk_threshold=risk_threshold,
        ood_feature_indices=ood_indices,
        ood_threshold=ood_threshold,
    )
    base_gates = m1_sealed_qualification(
        selected_calibration,
        control_calibration,
        ood_metrics,
        final_policy,
    )
    gates = m1_r2_sealed_qualification(
        base_gates["calibration"],
        risk_metrics,
        base_gates,
    )
    outcome = (
        "HIRA_V0_M1_RELIABILITY_READY"
        if gates["pass"]
        else "HIRA_V0_M1_RELIABILITY_FAIL"
    )

    if model.runtime.state_encode_calls != len(ue_rows) + len(ui_rows):
        raise RuntimeError("M1-R2 sealed state-once count changed")

    args.out.mkdir(parents=True, exist_ok=True)
    ue_path = args.out / "ue-cache.pt"
    ui_path = args.out / "ui-cache.pt"
    torch.save(ue_cache, ue_path)
    torch.save(ui_cache, ui_path)

    policy = {
        "schema_version": "hira-v0-m1-r2-reliability-policy-v1",
        "status": "qualified" if gates["pass"] else "failed",
        "calibration_candidate": receipt["calibration"]["selected_candidate"],
        "calibration_authority": "HIRA-V0-M1-R2-UE",
        "risk_candidate": receipt["selective_risk"]["selected_candidate"],
        "risk_authority": "HIRA-V0-M1-R2-UE",
        "risk_threshold": risk_threshold,
        "ood_candidate": receipt["r1_ood"]["candidate"],
        "ood_authority": "HIRA-V0-M1-R1-UI",
        "ood_threshold": ood_threshold,
        "ood_calibrator_id": receipt["r1_ood"]["head_sha256"],
    }
    (args.out / "policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sealed = {
        "schema_version": "hira-v0-mainline-m1-r2-sealed-confirm-v1",
        "status": "PASS",
        "outcome": outcome,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "r2_dev_qualification": receipt["dev_qualification"],
        "r2_sealed_exposure_authorized": True,
        "calibration_candidate": receipt["calibration"]["selected_candidate"],
        "calibrator_sha256": receipt["calibrator_sha256"],
        "risk_candidate": receipt["selective_risk"]["selected_candidate"],
        "selective_risk_head_sha256": receipt["selective_risk_head_sha256"],
        "risk_threshold": risk_threshold,
        "ood_candidate": receipt["r1_ood"]["candidate"],
        "ood_head_sha256": receipt["r1_ood"]["head_sha256"],
        "ood_threshold": ood_threshold,
        "selected_calibration": selected_calibration,
        "control_calibration": control_calibration,
        "standalone_risk": risk_metrics,
        "ood_metrics": ood_metrics,
        "final_policy": final_policy,
        "base_gates": base_gates,
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
        "r1_ood_retrained": False,
        "r1_ood_retuned": False,
        "confidence_used_as_ood_authority": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M1_R2_SEALED_FINAL=" + json.dumps({
        "outcome": outcome,
        "selected_calibration": selected_calibration,
        "control_calibration": control_calibration,
        "standalone_risk": risk_metrics,
        "ood_metrics": ood_metrics,
        "final_policy": final_policy,
        "gates": gates,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
