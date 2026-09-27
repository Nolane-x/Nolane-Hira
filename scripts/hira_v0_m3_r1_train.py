from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.mainline import W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
from nmd.mainline_m3_r1_authority import generate_m3_r1_authority
from nmd.mainline_m3_r1_cache import compile_m3_r1_base_cache
from nmd.mainline_m3_r1_training import (
    M3_R1_CHECKPOINT_SCHEMA,
    M3_R1_PRIMARY_SEED,
    M3_R1_REPLICA_SEED,
    build_m3_r1_training_runtime,
    m3_r1_joint_qualification,
    train_m3_r1_candidate,
)
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M3A_RUN_ID = 36320497057
M3A_ARTIFACT_ID = 10932028604
M3A_ARTIFACT_DIGEST = (
    "sha256:d9837f68ed0f9a7d216c42a95e3cc9977fa5f3d218aa792493ecfcf63a47d6bd"
)

M3_R1_RECEIPT_SCHEMA = "hira-v0-mainline-m3-r1-train-dev-v1"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M3-R1 A13 weight SHA mismatch: {actual}")

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
    receipt_path = directory / "receipt.json"
    if not checkpoint.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("M3-R1 T0 artifact incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w28-candidate-receipt-v1":
        raise RuntimeError("M3-R1 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M3-R1 T0 identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M3-R1 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M3-R1 T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt_path = directory / "receipt.json"
    if not checkpoint.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("M3-R1 W34 artifact incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M3-R1 unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M3-R1 W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M3-R1 W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M3-R1 W34 checkpoint bytes changed")
    if int(receipt.get("candidate_parameter_count", -1)) != 8192:
        raise RuntimeError("M3-R1 W34 candidate capacity changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M3-R1 W34 receipt contains confirm leakage")
    return checkpoint


def _validate_m3a(directory: Path) -> dict[str, object]:
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("M3-R1 M3-A receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "hira-v0-mainline-m3-paired-dev-v1":
        raise RuntimeError("M3-R1 unexpected M3-A receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M3-R1 M3-A infrastructure did not pass")
    if receipt.get("outcome") != "HIRA_V0_M3_PAIRED_DEV_FAIL":
        raise RuntimeError("M3-R1 requires frozen M3-A DEV failure")
    if receipt.get("dev_qualification", {}).get("pass") is not False:
        raise RuntimeError("M3-R1 expected failed M3-A DEV gate")
    if receipt.get("sealed_exposure_authorized") is not False:
        raise RuntimeError("M3-R1 M3-A unexpectedly authorized MVC")
    if receipt.get("confirm_partition_exposed") is not False:
        raise RuntimeError("M3-R1 MVC was exposed before rescue")
    if int(receipt.get("confirm_pair_count_used", -1)) != 0:
        raise RuntimeError("M3-R1 M3-A used MVC rows")
    if receipt.get("gradient_updates_used") is not False:
        raise RuntimeError("M3-R1 M3-A unexpectedly trained parameters")
    if receipt.get("massive_rows_used") is not False:
        raise RuntimeError("M3-R1 M3-A used MASSIVE rows")
    if receipt.get("xnli_rows_used") is not False:
        raise RuntimeError("M3-R1 M3-A used XNLI rows")
    if receipt.get("semantic_frontend_changed") is not False:
        raise RuntimeError("M3-R1 M3-A front-end was changed")
    return receipt


def _checkpoint_payload(result: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": M3_R1_CHECKPOINT_SCHEMA,
        "role": result["role"],
        "seed": result["seed"],
        "selected_epoch": result["selected_epoch"],
        "parameter_count": result["parameter_count"],
        "alignment_identity": result["alignment_identity"],
        "candidate_state_dict": result["candidate_state_dict"],
        "selected_dev": result["selected_dev"],
    }


def _summary(result: dict[str, object]) -> dict[str, object]:
    return {
        "role": result["role"],
        "seed": result["seed"],
        "selected_epoch": result["selected_epoch"],
        "parameter_count": result["parameter_count"],
        "alignment_identity": result["alignment_identity"],
        "selected_dev": result["selected_dev"],
        "dev_qualification": result["dev_qualification"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--m3a-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    m3a = _validate_m3a(args.m3a_dir)

    base_encoder = _load_a13()
    train_rows = generate_m3_r1_authority("train")
    dev_rows = generate_m3_r1_authority("dev")

    train_cache = compile_m3_r1_base_cache(
        base_encoder,
        train_rows,
        partition="train",
    )
    dev_cache = compile_m3_r1_base_cache(
        base_encoder,
        dev_rows,
        partition="dev",
    )

    primary_runtime, primary_adapter = build_m3_r1_training_runtime(
        base_encoder,
        t0,
        w34,
        seed=M3_R1_PRIMARY_SEED,
        alignment_identity=f"m3-r1-primary-seed-{M3_R1_PRIMARY_SEED}",
    )
    primary = train_m3_r1_candidate(
        primary_runtime,
        primary_adapter,
        train_cache,
        dev_cache,
        seed=M3_R1_PRIMARY_SEED,
        role="primary",
    )

    replica_runtime, replica_adapter = build_m3_r1_training_runtime(
        base_encoder,
        t0,
        w34,
        seed=M3_R1_REPLICA_SEED,
        alignment_identity=f"m3-r1-replica-seed-{M3_R1_REPLICA_SEED}",
    )
    replica = train_m3_r1_candidate(
        replica_runtime,
        replica_adapter,
        train_cache,
        dev_cache,
        seed=M3_R1_REPLICA_SEED,
        role="replica",
    )

    joint = m3_r1_joint_qualification(primary, replica)
    outcome = (
        "HIRA_V0_M3_R1_DEV_QUALIFIED"
        if joint["pass"]
        else "HIRA_V0_M3_R1_DEV_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    primary_path = args.out / "primary.pt"
    replica_path = args.out / "replica.pt"
    torch.save(_checkpoint_payload(primary), primary_path)
    torch.save(_checkpoint_payload(replica), replica_path)

    primary_sha = file_sha256(primary_path)
    replica_sha = file_sha256(replica_path)

    receipt = {
        "schema_version": M3_R1_RECEIPT_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "base_w34_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "m3a": {
            "source_run_id": M3A_RUN_ID,
            "source_artifact_id": M3A_ARTIFACT_ID,
            "source_artifact_digest": M3A_ARTIFACT_DIGEST,
            "outcome": m3a["outcome"],
        },
        "train_partitions": ["MVD", "MVE", "MVF", "MVG"],
        "train_pair_count": len(train_rows),
        "train_language_case_count": 2 * len(train_rows),
        "dev_partitions": ["MVH", "MVI"],
        "dev_pair_count": len(dev_rows),
        "dev_language_case_count": 2 * len(dev_rows),
        "primary": _summary(primary),
        "replica": _summary(replica),
        "primary_checkpoint_sha256": primary_sha,
        "replica_checkpoint_sha256": replica_sha,
        "joint_qualification": joint,
        "sealed_exposure_authorized": bool(joint["pass"]),
        "alignment_candidate_parameter_count": 8192,
        "only_alignment_adapter_parameters_trained": True,
        "a13_gradient_updates": False,
        "w28_projection_gradient_updates": False,
        "w34_gradient_updates": False,
        "hira_core_gradient_updates": False,
        "reliability_gradient_updates": False,
        "mva_mvb_rows_used_for_training": False,
        "mva_mvb_rows_used_for_selection": False,
        "mvc_exposed": False,
        "mvc_rows_used": False,
        "massive_rows_used": False,
        "xnli_rows_used": False,
        "m2_semantic_rows_used": False,
        "train_cache_state_encode_count": train_cache["metadata"]["state_encode_count"],
        "dev_cache_state_encode_count": dev_cache["metadata"]["state_encode_count"],
        "gradient_updates_during_base_cache": False,
        "quality_claim_made": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HIRA_V0_M3_R1_FINAL=" + json.dumps({
        "outcome": outcome,
        "primary": receipt["primary"],
        "replica": receipt["replica"],
        "joint_qualification": joint,
        "sealed_exposure_authorized": receipt["sealed_exposure_authorized"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
