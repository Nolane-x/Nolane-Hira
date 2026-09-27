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

M3_DEV_SCHEMA = "hira-v0-mainline-m3-paired-dev-v1"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M3 A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M3 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M3 T0 receipt identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M3 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M3 T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M3 unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M3 W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M3 W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M3 W34 checkpoint bytes changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M3 W34 training receipt contains confirm leakage")
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
    model = build_hira_v0_m3_baseline(encoder, t0, w34)

    if model.manifest.multilingual != "provisional":
        raise RuntimeError("M3 DEV multilingual maturity changed")
    if model.manifest.high_k_mechanics != "available":
        raise RuntimeError("M3 DEV lost M2 mechanics promotion")
    if model.manifest.high_k_mechanics_authority != M2_MECHANICS_AUTHORITY:
        raise RuntimeError("M3 DEV mechanics authority changed")
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("M3 DEV requires zero trainable parameters")

    rows = generate_m3_paired_authority("dev")
    evaluation = evaluate_m3_paired_cases(model, rows)
    qualification = m3_multilingual_qualification(evaluation)
    outcome = (
        "HIRA_V0_M3_PAIRED_DEV_QUALIFIED"
        if qualification["pass"]
        else "HIRA_V0_M3_PAIRED_DEV_FAIL"
    )

    if evaluation["pair_count"] != 72:
        raise RuntimeError("M3 DEV pair count changed")
    if evaluation["language_case_count"] != 144:
        raise RuntimeError("M3 DEV language case count changed")
    if evaluation["state_encode_count"] != 144:
        raise RuntimeError("M3 DEV state-once count changed")

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M3_DEV_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "mechanics_authority": M2_MECHANICS_AUTHORITY,
        "parameter_report": model.parameter_report().to_dict(),
        "evaluation": evaluation,
        "dev_qualification": qualification,
        "pair_count": 72,
        "language_case_count": 144,
        "gradient_updates_used": False,
        "massive_rows_used": False,
        "xnli_rows_used": False,
        "m2_semantic_rows_used": False,
        "w34_confirm_rows_used": False,
        "confirm_partition_exposed": False,
        "confirm_pair_count_used": 0,
        "sealed_exposure_authorized": bool(qualification["pass"]),
        "semantic_frontend_changed": False,
        "multilingual_promoted": False,
        "quality_claim_made": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M3_PAIRED_DEV_FINAL=" + json.dumps({
        "outcome": outcome,
        "evaluation": {
            "en": evaluation["en"],
            "vi": evaluation["vi"],
            "vi_en_top1_ratio": evaluation["vi_en_top1_ratio"],
            "vi_en_mrr_ratio": evaluation["vi_en_mrr_ratio"],
            "paired_prediction_agreement": evaluation["paired_prediction_agreement"],
            "paired_both_correct_rate": evaluation["paired_both_correct_rate"],
            "probability_mass_max_error": evaluation["probability_mass_max_error"],
            "relation_delta_max": evaluation["relation_delta_max"],
            "full_k": evaluation["full_k"],
            "finite": evaluation["finite"],
        },
        "dev_qualification": qualification,
        "sealed_exposure_authorized": receipt["sealed_exposure_authorized"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
