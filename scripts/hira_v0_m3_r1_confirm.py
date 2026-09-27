from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.hira import HIRACore
from nmd.mainline import (
    HiraV0Mainline,
    HiraV0Manifest,
    M2_MECHANICS_AUTHORITY,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
)
from nmd.mainline_m3_alignment import (
    AlignedSemanticEncoder,
    M3_ALIGNMENT_PARAMETER_COUNT,
    MultilingualAlignmentAdapter,
)
from nmd.mainline_m3_authority import generate_m3_paired_authority
from nmd.mainline_m3_eval import (
    evaluate_m3_paired_cases,
    m3_multilingual_qualification,
)
from nmd.mainline_m3_r1_training import M3_R1_CHECKPOINT_SCHEMA
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256
from nmd.w34_transfer_core import build_hira_v0_w34_core

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M3_R1_RECEIPT_SCHEMA = "hira-v0-mainline-m3-r1-train-dev-v1"
M3_R1_CONFIRM_SCHEMA = "hira-v0-mainline-m3-r1-sealed-confirm-v1"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M3-R1 confirm A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M3-R1 confirm unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M3-R1 confirm T0 identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M3-R1 confirm T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M3-R1 confirm T0 bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M3-R1 confirm unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M3-R1 confirm W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M3-R1 confirm W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M3-R1 confirm W34 bytes changed")
    return checkpoint


def _validate_r1(directory: Path) -> tuple[dict[str, object], dict[str, object], Path]:
    receipt_path = directory / "receipt.json"
    primary_path = directory / "primary.pt"
    if not receipt_path.is_file() or not primary_path.is_file():
        raise FileNotFoundError("M3-R1 sealed input incomplete")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != M3_R1_RECEIPT_SCHEMA:
        raise RuntimeError("M3-R1 confirm unexpected TRAIN DEV receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M3-R1 TRAIN DEV infrastructure did not pass")
    if receipt.get("outcome") != "HIRA_V0_M3_R1_DEV_QUALIFIED":
        raise RuntimeError("M3-R1 fresh DEV did not qualify")
    if receipt.get("joint_qualification", {}).get("pass") is not True:
        raise RuntimeError("M3-R1 primary+replica joint gate did not pass")
    if receipt.get("sealed_exposure_authorized") is not True:
        raise RuntimeError("M3-R1 did not authorize MVC")
    if receipt.get("mvc_exposed") is not False or receipt.get("mvc_rows_used") is not False:
        raise RuntimeError("M3-R1 MVC was exposed before sealed confirm")
    if int(receipt.get("alignment_candidate_parameter_count", -1)) != M3_ALIGNMENT_PARAMETER_COUNT:
        raise RuntimeError("M3-R1 alignment capacity changed")
    if receipt.get("only_alignment_adapter_parameters_trained") is not True:
        raise RuntimeError("M3-R1 trained outside alignment adapter")
    for key in (
        "a13_gradient_updates",
        "w28_projection_gradient_updates",
        "w34_gradient_updates",
        "hira_core_gradient_updates",
        "reliability_gradient_updates",
        "mva_mvb_rows_used_for_training",
        "mva_mvb_rows_used_for_selection",
        "massive_rows_used",
        "xnli_rows_used",
        "m2_semantic_rows_used",
    ):
        if receipt.get(key) is not False:
            raise RuntimeError(f"M3-R1 forbidden flag changed: {key}")

    expected_sha = receipt.get("primary_checkpoint_sha256")
    actual_sha = file_sha256(primary_path)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError("M3-R1 primary checkpoint SHA mismatch")

    payload = torch.load(primary_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != M3_R1_CHECKPOINT_SCHEMA:
        raise RuntimeError("M3-R1 primary checkpoint schema changed")
    if payload.get("role") != "primary":
        raise RuntimeError("M3-R1 sealed confirm requires primary checkpoint")
    if int(payload.get("parameter_count", -1)) != M3_ALIGNMENT_PARAMETER_COUNT:
        raise RuntimeError("M3-R1 primary adapter capacity changed")
    state = payload.get("candidate_state_dict")
    if not isinstance(state, dict) or set(state) != {"down.weight", "up.weight"}:
        raise RuntimeError("M3-R1 primary adapter state changed")
    if payload.get("alignment_identity") != receipt["primary"]["alignment_identity"]:
        raise RuntimeError("M3-R1 primary alignment identity changed")
    return receipt, payload, primary_path


def _build_frozen_model(
    base_encoder: HFAutoSemanticEncoder,
    adapter_payload: dict[str, object],
    t0: Path,
    w34: Path,
) -> HiraV0Mainline:
    adapter = MultilingualAlignmentAdapter()
    adapter.load_candidate_state_dict(
        adapter_payload["candidate_state_dict"],
        freeze=True,
    )
    aligned = AlignedSemanticEncoder(
        base_encoder,
        adapter,
        alignment_identity=str(adapter_payload["alignment_identity"]),
    )
    for parameter in aligned.parameters():
        parameter.requires_grad_(False)
    aligned.eval()

    runtime = build_hira_v0_w34_core(
        aligned,
        t0,
        w34,
        expected_t0_sha256=W28_T0_CHECKPOINT_SHA256,
        expected_candidate_sha256=W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        hira=HIRACore(d_model=256, dropout=0.0),
        include_unbridged_baseline=False,
    )
    model = HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m3_multilingual_provisional(),
    )
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M3-R1 confirm model must be fully frozen")
    if model.manifest.high_k_mechanics_authority != M2_MECHANICS_AUTHORITY:
        raise RuntimeError("M3-R1 confirm M2 mechanics authority changed")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    r1_receipt, primary_payload, primary_path = _validate_r1(args.r1_dir)

    base_encoder = _load_a13()
    model = _build_frozen_model(base_encoder, primary_payload, t0, w34)

    print("HIRA_V0_M3_R1_MVC_EN_VI_SEALED_EXPOSURE_BEGIN", flush=True)
    rows = generate_m3_paired_authority("confirm", allow_sealed=True)
    evaluation = evaluate_m3_paired_cases(model, rows)
    qualification = m3_multilingual_qualification(evaluation)

    if evaluation["pair_count"] != 36:
        raise RuntimeError("M3-R1 sealed MVC pair count changed")
    if evaluation["language_case_count"] != 72:
        raise RuntimeError("M3-R1 sealed MVC language case count changed")
    if evaluation["state_encode_count"] != 72:
        raise RuntimeError("M3-R1 sealed MVC state-once count changed")

    outcome = (
        "HIRA_V0_M3_MULTILINGUAL_READY"
        if qualification["pass"]
        else "HIRA_V0_M3_MULTILINGUAL_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M3_R1_CONFIRM_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "base_w34_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "primary_checkpoint_sha256": file_sha256(primary_path),
        "primary_alignment_identity": primary_payload["alignment_identity"],
        "primary_selected_epoch": primary_payload["selected_epoch"],
        "alignment_candidate_parameter_count": M3_ALIGNMENT_PARAMETER_COUNT,
        "r1_dev_outcome": r1_receipt["outcome"],
        "r1_joint_qualification": r1_receipt["joint_qualification"],
        "evaluation": evaluation,
        "sealed_qualification": qualification,
        "pair_count": 36,
        "language_case_count": 72,
        "mva_mvb_rows_used_for_fitting": False,
        "mva_mvb_rows_used_for_selection": False,
        "mvc_exposed": True,
        "mvc_rows_used_for_fitting": False,
        "mvc_rows_used_for_selection": False,
        "mvc_rows_used_for_threshold_tuning": False,
        "massive_rows_used": False,
        "xnli_rows_used": False,
        "m2_semantic_rows_used": False,
        "gradient_updates_during_confirm": False,
        "multilingual_promoted": bool(qualification["pass"]),
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HIRA_V0_M3_R1_SEALED_FINAL=" + json.dumps({
        "outcome": outcome,
        "evaluation": {
            "en": evaluation["en"],
            "vi": evaluation["vi"],
            "vi_en_top1_ratio": evaluation["vi_en_top1_ratio"],
            "vi_en_mrr_ratio": evaluation["vi_en_mrr_ratio"],
            "paired_prediction_agreement": evaluation["paired_prediction_agreement"],
            "paired_both_correct_rate": evaluation["paired_both_correct_rate"],
        },
        "sealed_qualification": qualification,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
