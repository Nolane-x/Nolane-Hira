from __future__ import annotations

import argparse
import json
from pathlib import Path

from nmd.mainline import (
    M2_MECHANICS_AUTHORITY,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m3_baseline,
)
from nmd.mainline_m3_authority import generate_m3_paired_authority
from nmd.mainline_m3_eval import (
    evaluate_m3_paired_cases,
    m3_multilingual_qualification,
)
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M3_CONFIRM_SCHEMA = "hira-v0-mainline-m3-paired-confirm-v1"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M3 confirm A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M3 confirm unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M3 confirm T0 identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M3 confirm T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M3 confirm T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M3 confirm unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M3 confirm W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M3 confirm W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M3 confirm W34 checkpoint bytes changed")
    return checkpoint


def _validate_dev(directory: Path) -> dict[str, object]:
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "hira-v0-mainline-m3-paired-dev-v1":
        raise RuntimeError("M3 confirm unexpected DEV receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M3 DEV infrastructure did not pass")
    if receipt.get("outcome") != "HIRA_V0_M3_PAIRED_DEV_QUALIFIED":
        raise RuntimeError("M3 paired DEV did not qualify")
    if receipt.get("dev_qualification", {}).get("pass") is not True:
        raise RuntimeError("M3 paired DEV qualification did not pass")
    if receipt.get("sealed_exposure_authorized") is not True:
        raise RuntimeError("M3 DEV did not authorize sealed confirmation")
    if receipt.get("confirm_partition_exposed") is not False:
        raise RuntimeError("M3 confirm partition was exposed before sealed confirm")
    if int(receipt.get("confirm_pair_count_used", -1)) != 0:
        raise RuntimeError("M3 DEV used confirm rows")
    if receipt.get("gradient_updates_used") is not False:
        raise RuntimeError("M3 DEV unexpectedly trained parameters")
    if receipt.get("massive_rows_used") is not False:
        raise RuntimeError("M3 DEV used MASSIVE")
    if receipt.get("xnli_rows_used") is not False:
        raise RuntimeError("M3 DEV used XNLI")
    if receipt.get("semantic_frontend_changed") is not False:
        raise RuntimeError("M3 zero-training front-end changed before confirm")
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
    model = build_hira_v0_m3_baseline(encoder, t0, w34)
    if model.manifest.high_k_mechanics_authority != M2_MECHANICS_AUTHORITY:
        raise RuntimeError("M3 confirm mechanics authority changed")
    if model.manifest.multilingual != "provisional":
        raise RuntimeError("M3 confirm multilingual maturity changed")
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M3 confirm requires zero trainable parameters")

    print("HIRA_V0_M3_MVC_EN_VI_SEALED_EXPOSURE_BEGIN", flush=True)
    rows = generate_m3_paired_authority("confirm", allow_sealed=True)
    evaluation = evaluate_m3_paired_cases(model, rows)
    qualification = m3_multilingual_qualification(evaluation)

    if evaluation["pair_count"] != 36:
        raise RuntimeError("M3 confirm pair count changed")
    if evaluation["language_case_count"] != 72:
        raise RuntimeError("M3 confirm language case count changed")
    if evaluation["state_encode_count"] != 72:
        raise RuntimeError("M3 confirm state-once count changed")

    outcome = (
        "HIRA_V0_M3_MULTILINGUAL_READY"
        if qualification["pass"]
        else "HIRA_V0_M3_MULTILINGUAL_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M3_CONFIRM_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "mechanics_authority": M2_MECHANICS_AUTHORITY,
        "dev_outcome": dev_receipt["outcome"],
        "dev_qualification": dev_receipt["dev_qualification"],
        "evaluation": evaluation,
        "sealed_qualification": qualification,
        "pair_count": 36,
        "language_case_count": 72,
        "dev_rows_used_for_fitting": False,
        "dev_rows_used_for_selection": False,
        "confirm_partition_exposed": True,
        "confirm_rows_used_for_fitting": False,
        "confirm_rows_used_for_selection": False,
        "gradient_updates_used": False,
        "massive_rows_used": False,
        "xnli_rows_used": False,
        "m2_semantic_rows_used": False,
        "semantic_frontend_changed": False,
        "multilingual_promoted": bool(qualification["pass"]),
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M3_PAIRED_CONFIRM_FINAL=" + json.dumps({
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
