from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.mainline import (
    M2_MECHANICS_AUTHORITY,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m2_mechanics,
)
from nmd.mainline_m2_authority import generate_m2_high_k_semantic
from nmd.mainline_m2_eval import (
    evaluate_m2_semantic_cases,
    m2_sealed_qualification,
)
from nmd.mainline_m2_r1_training import install_m2_r1_candidate
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M2_R1_RECEIPT_SCHEMA = "hira-v0-mainline-m2-r1-train-dev-v1"
M2_R1_CHECKPOINT_SCHEMA = "hira-v0-mainline-m2-r1-high-k-scorer-v1"
M2_R1_CONFIRM_SCHEMA = "hira-v0-mainline-m2-r1-sealed-confirm-v1"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M2-R1 confirm A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M2-R1 confirm unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M2-R1 confirm T0 identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 confirm T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 confirm T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M2-R1 confirm unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M2-R1 confirm W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 confirm W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 confirm W34 checkpoint bytes changed")
    if int(receipt.get("candidate_parameter_count", -1)) != 8192:
        raise RuntimeError("M2-R1 confirm W34 candidate capacity changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M2-R1 confirm W34 receipt contains confirm leakage")
    return checkpoint


def _validate_r1(directory: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    receipt_path = directory / "receipt.json"
    primary_path = directory / "primary.pt"
    if not receipt_path.is_file() or not primary_path.is_file():
        raise FileNotFoundError("M2-R1 confirm input is incomplete")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != M2_R1_RECEIPT_SCHEMA:
        raise RuntimeError("M2-R1 confirm unexpected TRAIN DEV receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M2-R1 TRAIN DEV receipt did not pass infrastructure")
    if receipt.get("outcome") != "HIRA_V0_M2_R1_DEV_QUALIFIED":
        raise RuntimeError("M2-R1 fresh DEV did not qualify")
    if receipt.get("dev_qualification", {}).get("pass") is not True:
        raise RuntimeError("M2-R1 fresh DEV gate did not pass")
    if receipt.get("sealed_exposure_authorized") is not True:
        raise RuntimeError("M2-R1 did not authorize HKG exposure")

    if receipt.get("train_partitions") != ["HKA", "HKB", "HKC", "HKD"]:
        raise RuntimeError("M2-R1 TRAIN authority changed")
    if receipt.get("dev_partitions") != ["HKH", "HKI"]:
        raise RuntimeError("M2-R1 DEV authority changed")
    if int(receipt.get("candidate_parameter_count", -1)) != 8192:
        raise RuntimeError("M2-R1 candidate capacity changed")
    if receipt.get("only_w34_candidate_parameters_trained") is not True:
        raise RuntimeError("M2-R1 trained outside W34 candidate surface")
    if int(receipt.get("decision_core_trainable_parameter_count", -1)) != 0:
        raise RuntimeError("M2-R1 decision-core trainable count changed")
    if receipt.get("encoder_gradient_updates") is not False:
        raise RuntimeError("M2-R1 updated semantic encoder")
    if receipt.get("w28_projection_gradient_updates") is not False:
        raise RuntimeError("M2-R1 updated W28 projection")
    if receipt.get("hira_core_gradient_updates") is not False:
        raise RuntimeError("M2-R1 updated HIRACore")
    if receipt.get("hke_hkf_rows_used_for_training") is not False:
        raise RuntimeError("M2-R1 reused exposed HKE/HKF for training")
    if receipt.get("hke_hkf_rows_used_for_selection") is not False:
        raise RuntimeError("M2-R1 reused exposed HKE/HKF for selection")
    if receipt.get("hkg_exposed") is not False or receipt.get("hkg_rows_used") is not False:
        raise RuntimeError("M2-R1 HKG was exposed before sealed confirm")
    if receipt.get("candidate_pruning_used") is not False:
        raise RuntimeError("M2-R1 candidate pruning changed")
    if receipt.get("relation_refinement_used") is not False:
        raise RuntimeError("M2-R1 relation refinement changed")
    if receipt.get("adaptive_budget_used") is not False:
        raise RuntimeError("M2-R1 adaptive budget changed")

    expected_sha = receipt.get("primary_checkpoint_sha256")
    actual_sha = file_sha256(primary_path)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError("M2-R1 primary checkpoint SHA mismatch")

    payload = torch.load(primary_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != M2_R1_CHECKPOINT_SCHEMA:
        raise RuntimeError("M2-R1 primary checkpoint schema changed")
    if payload.get("role") != "primary":
        raise RuntimeError("M2-R1 sealed confirm requires primary checkpoint")
    if payload.get("candidate") != receipt.get("primary_candidate"):
        raise RuntimeError("M2-R1 primary candidate identity changed")
    if payload.get("family") != receipt.get("selected_family"):
        raise RuntimeError("M2-R1 selected family identity changed")
    if int(payload.get("parameter_count", -1)) != 8192:
        raise RuntimeError("M2-R1 primary checkpoint capacity changed")
    if payload.get("base_w34_checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2-R1 primary checkpoint W34 base changed")
    state = payload.get("candidate_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("M2-R1 primary candidate state missing")
    return receipt, primary_path, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    r1_receipt, primary_path, primary_state = _validate_r1(args.r1_dir)

    encoder = _load_a13()
    model = build_hira_v0_m2_mechanics(encoder, t0, w34)

    if model.manifest.high_k_mechanics != "available":
        raise RuntimeError("M2-R1 confirm requires promoted mechanics")
    if model.manifest.high_k_mechanics_authority != M2_MECHANICS_AUTHORITY:
        raise RuntimeError("M2-R1 confirm mechanics authority changed")
    if model.manifest.high_k != "provisional":
        raise RuntimeError("M2-R1 semantic quality was prematurely promoted")
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M2-R1 confirm base model must be frozen")

    install_m2_r1_candidate(model, primary_state)
    if model.runtime.coevidence_symmetric_semantic_scorer is None:
        raise RuntimeError("M2-R1 confirm scorer missing after install")
    if model.runtime.coevidence_symmetric_semantic_scorer.trainable_parameter_count != 0:
        raise RuntimeError("M2-R1 sealed scorer must be frozen")

    print("HIRA_V0_M2_R1_HKG_K255_SEALED_EXPOSURE_BEGIN", flush=True)
    rows = generate_m2_high_k_semantic("confirm", allow_sealed=True)
    evaluation = evaluate_m2_semantic_cases(model, rows)
    qualification = m2_sealed_qualification(evaluation)

    if evaluation["case_count"] != 24:
        raise RuntimeError("M2-R1 sealed case count changed")
    if evaluation["state_encode_count"] != 24:
        raise RuntimeError("M2-R1 sealed state-once count changed")
    if set(evaluation["per_k"]) != {"255"}:
        raise RuntimeError("M2-R1 sealed K surface changed")

    outcome = (
        "HIRA_V0_M2_HIGH_K_SEMANTIC_READY"
        if qualification["pass"]
        else "HIRA_V0_M2_HIGH_K_SEMANTIC_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M2_R1_CONFIRM_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "base_w34_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "selected_family": r1_receipt["selected_family"],
        "primary_candidate": r1_receipt["primary_candidate"],
        "primary_checkpoint_sha256": file_sha256(primary_path),
        "primary_selected_epoch": next(
            row["selected_epoch"]
            for row in r1_receipt["candidates"]
            if row["candidate"] == r1_receipt["primary_candidate"]
        ),
        "candidate_parameter_count": 8192,
        "r1_dev_outcome": r1_receipt["outcome"],
        "r1_dev_qualification": r1_receipt["dev_qualification"],
        "evaluation": evaluation,
        "sealed_qualification": qualification,
        "case_counts": {"HKG_K255": 24, "total": 24},
        "train_partitions_exposed": True,
        "r1_dev_partitions_exposed": True,
        "hke_hkf_rows_used_for_fitting": False,
        "hke_hkf_rows_used_for_selection": False,
        "hkg_exposed": True,
        "hkg_case_count": 24,
        "hkg_rows_used_for_fitting": False,
        "hkg_rows_used_for_selection": False,
        "hkg_rows_used_for_threshold_tuning": False,
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
        "gradient_updates_during_confirm": False,
        "semantic_quality_promoted": bool(qualification["pass"]),
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M2_R1_SEALED_FINAL=" + json.dumps({
        "outcome": outcome,
        "selected_family": receipt["selected_family"],
        "primary_candidate": receipt["primary_candidate"],
        "evaluation": evaluation,
        "sealed_qualification": qualification,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
