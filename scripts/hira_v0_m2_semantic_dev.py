from __future__ import annotations

import argparse
import json
from pathlib import Path

from nmd.mainline import (
    M2_MECHANICS_AUTHORITY,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m2_mechanics,
)
from nmd.mainline_m2_authority import generate_m2_high_k_semantic
from nmd.mainline_m2_eval import (
    evaluate_m2_semantic_cases,
    m2_dev_qualification,
)
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M2_DEV_SCHEMA = "hira-v0-mainline-m2-semantic-dev-v1"
M2_MECHANICS_RUN_ID = 36310118240
M2_MECHANICS_ARTIFACT_ID = 10928771833
M2_MECHANICS_ARTIFACT_DIGEST = (
    "sha256:99e8c02aeb48fa32910a591d71b92e3e6f96258c0fa8501f38682ea4a2b7f453"
)


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M2 semantic DEV A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M2 semantic DEV unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M2 semantic DEV T0 receipt identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2 semantic DEV T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2 semantic DEV T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M2 semantic DEV unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M2 semantic DEV W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2 semantic DEV W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2 semantic DEV W34 checkpoint bytes changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M2 semantic DEV W34 training receipt contains confirm leakage")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)

    encoder = _load_a13()
    model = build_hira_v0_m2_mechanics(encoder, t0, w34)

    if model.manifest.high_k_mechanics != "available":
        raise RuntimeError("M2 semantic DEV requires promoted mechanics")
    if model.manifest.high_k_mechanics_authority != M2_MECHANICS_AUTHORITY:
        raise RuntimeError("M2 mechanics authority changed")
    if model.manifest.high_k != "provisional":
        raise RuntimeError("M2 semantic quality was prematurely promoted")

    report = model.parameter_report().to_dict()
    if report["trainable_total"] != 0:
        raise RuntimeError("M2 frozen semantic baseline requires zero trainable params")

    # Critical evidence boundary: evaluate DEV only.
    # HKA-HKD TRAIN remains unexposed for a possible preregistered rescue.
    # HKG K=255 remains sealed.
    dev_rows = generate_m2_high_k_semantic("dev")
    evaluation = evaluate_m2_semantic_cases(model, dev_rows)
    qualification = m2_dev_qualification(evaluation)

    if evaluation["case_count"] != 32:
        raise RuntimeError("M2 semantic DEV case count changed")
    if evaluation["state_encode_count"] != 32:
        raise RuntimeError("M2 semantic DEV state-once count changed")
    if set(evaluation["per_k"]) != {"64", "128"}:
        raise RuntimeError("M2 semantic DEV K surface changed")

    outcome = (
        "HIRA_V0_M2_SEMANTIC_DEV_QUALIFIED"
        if qualification["pass"]
        else "HIRA_V0_M2_SEMANTIC_DEV_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M2_DEV_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "mechanics_authority": M2_MECHANICS_AUTHORITY,
        "mechanics_run_id": M2_MECHANICS_RUN_ID,
        "mechanics_artifact_id": M2_MECHANICS_ARTIFACT_ID,
        "mechanics_artifact_digest": M2_MECHANICS_ARTIFACT_DIGEST,
        "parameter_report": report,
        "evaluation": evaluation,
        "dev_qualification": qualification,
        "case_counts": {
            "HKE_K64": 16,
            "HKF_K128": 16,
            "total": 32,
        },
        "train_partitions_exposed": False,
        "train_case_count_used": 0,
        "confirm_partition_exposed": False,
        "confirm_case_count_used": 0,
        "sealed_exposure_authorized": bool(qualification["pass"]),
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
        "gradient_updates_used": False,
        "semantic_quality_promoted": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HIRA_V0_M2_SEMANTIC_DEV_FINAL=" + json.dumps({
        "outcome": outcome,
        "evaluation": evaluation,
        "dev_qualification": qualification,
        "sealed_exposure_authorized": receipt["sealed_exposure_authorized"],
        "train_partitions_exposed": False,
        "confirm_partition_exposed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
