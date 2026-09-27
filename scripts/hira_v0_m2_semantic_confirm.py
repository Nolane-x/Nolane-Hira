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
    m2_sealed_qualification,
)
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M2_CONFIRM_SCHEMA = "hira-v0-mainline-m2-semantic-confirm-v1"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M2 semantic confirm A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M2 semantic confirm unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M2 semantic confirm T0 receipt identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2 semantic confirm T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2 semantic confirm T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M2 semantic confirm unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M2 semantic confirm W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2 semantic confirm W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2 semantic confirm W34 checkpoint bytes changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M2 semantic confirm W34 training receipt contains confirm leakage")
    return checkpoint


def _validate_dev(directory: Path) -> dict[str, object]:
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "hira-v0-mainline-m2-semantic-dev-v1":
        raise RuntimeError("M2 semantic confirm unexpected DEV receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M2 semantic DEV receipt did not pass infrastructure")
    if receipt.get("outcome") != "HIRA_V0_M2_SEMANTIC_DEV_QUALIFIED":
        raise RuntimeError("M2 semantic DEV did not qualify sealed exposure")
    if receipt.get("dev_qualification", {}).get("pass") is not True:
        raise RuntimeError("M2 semantic DEV gate did not pass")
    if receipt.get("sealed_exposure_authorized") is not True:
        raise RuntimeError("M2 semantic DEV did not authorize HKG")
    if receipt.get("train_partitions_exposed") is not False:
        raise RuntimeError("M2 semantic DEV exposed HKA-HKD")
    if int(receipt.get("train_case_count_used", -1)) != 0:
        raise RuntimeError("M2 semantic DEV used TRAIN rows")
    if receipt.get("confirm_partition_exposed") is not False:
        raise RuntimeError("M2 semantic DEV already exposed HKG")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M2 semantic DEV used HKG rows")
    if receipt.get("gradient_updates_used") is not False:
        raise RuntimeError("M2 semantic DEV unexpectedly trained parameters")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--dev-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    dev_receipt = _validate_dev(args.dev_dir)

    encoder = _load_a13()
    model = build_hira_v0_m2_mechanics(encoder, t0, w34)

    if model.manifest.high_k_mechanics != "available":
        raise RuntimeError("M2 semantic confirm requires promoted mechanics")
    if model.manifest.high_k_mechanics_authority != M2_MECHANICS_AUTHORITY:
        raise RuntimeError("M2 semantic confirm mechanics authority changed")
    if model.manifest.high_k != "provisional":
        raise RuntimeError("M2 semantic quality was prematurely promoted")

    report = model.parameter_report().to_dict()
    if report["trainable_total"] != 0:
        raise RuntimeError("M2 semantic confirm requires zero trainable params")

    print("HIRA_V0_M2_HKG_K255_SEALED_EXPOSURE_BEGIN", flush=True)
    rows = generate_m2_high_k_semantic("confirm", allow_sealed=True)
    evaluation = evaluate_m2_semantic_cases(model, rows)
    qualification = m2_sealed_qualification(evaluation)

    if evaluation["case_count"] != 24:
        raise RuntimeError("M2 sealed semantic case count changed")
    if evaluation["state_encode_count"] != 24:
        raise RuntimeError("M2 sealed semantic state-once count changed")
    if set(evaluation["per_k"]) != {"255"}:
        raise RuntimeError("M2 sealed semantic K surface changed")

    outcome = (
        "HIRA_V0_M2_HIGH_K_SEMANTIC_READY"
        if qualification["pass"]
        else "HIRA_V0_M2_HIGH_K_SEMANTIC_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M2_CONFIRM_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "mechanics_authority": M2_MECHANICS_AUTHORITY,
        "parameter_report": report,
        "dev_outcome": dev_receipt["outcome"],
        "dev_qualification": dev_receipt["dev_qualification"],
        "evaluation": evaluation,
        "sealed_qualification": qualification,
        "case_counts": {
            "HKG_K255": 24,
            "total": 24,
        },
        "train_partitions_exposed": False,
        "train_case_count_used": 0,
        "dev_rows_used_for_fitting": False,
        "dev_rows_used_for_selection": False,
        "confirm_partition_exposed": True,
        "confirm_case_count_used": 24,
        "confirm_rows_used_for_fitting": False,
        "confirm_rows_used_for_selection": False,
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
        "gradient_updates_used": False,
        "semantic_quality_promoted": bool(qualification["pass"]),
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("HIRA_V0_M2_SEMANTIC_CONFIRM_FINAL=" + json.dumps({
        "outcome": outcome,
        "evaluation": evaluation,
        "sealed_qualification": qualification,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
