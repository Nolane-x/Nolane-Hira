from __future__ import annotations

import argparse
import json
from pathlib import Path

from nmd.contracts import LogicalOption
from nmd.mainline import (
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_mainline,
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
        raise RuntimeError(f"M0 A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M0 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M0 T0 receipt identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M0 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M0 T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M0 unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M0 W34 training receipt did not pass")
    if receipt.get("qualification_outcome") != "W34_REFERENCE_QUALIFIED":
        raise RuntimeError("M0 W34 qualification provenance changed")
    if int(receipt.get("candidate_parameter_count", -1)) != 8192:
        raise RuntimeError("M0 W34 parameter count changed")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M0 W34 receipt checkpoint SHA changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M0 W34 checkpoint bytes changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M0 W34 training receipt contains confirm leakage")
    return checkpoint


def _binary_options():
    return (
        LogicalOption(
            "wait",
            "a short wait remains acceptable",
            aliases=("brief delay is permitted",),
            exemplars=("the request may pause briefly",),
            value=0.0,
        ),
        LogicalOption(
            "now",
            "action must begin immediately",
            aliases=("no brief delay is permitted",),
            exemplars=("the request must start now",),
            value=1.0,
        ),
    )


def _three_options():
    return (
        LogicalOption("low", "low urgency", value=0.0),
        LogicalOption("medium", "medium urgency", value=0.5),
        LogicalOption("high", "high urgency", value=1.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = _validate_t0(args.t0_dir)
    w34 = _validate_w34(args.w34_dir)
    encoder = _load_a13()

    model = build_hira_v0_mainline(
        encoder,
        t0,
        w34,
    )
    session = model.open_session(
        "The request can tolerate a short pause, but the next handling stage may become urgent."
    )
    encoded_after_open = model.runtime.state_encode_calls

    choice = session.decide(
        primitive="choice",
        question_text="Which timing condition best applies?",
        options=_binary_options(),
        use_schema_cache=False,
    )
    score = session.decide(
        primitive="score",
        question_text="What urgency level best applies?",
        options=_three_options(),
        use_schema_cache=False,
    )
    noul = session.decide(
        primitive="noul",
        question_text="Must action begin immediately?",
        options=_binary_options(),
        use_schema_cache=False,
    )

    if model.runtime.state_encode_calls != encoded_after_open:
        raise RuntimeError("M0 integration violated state-once after opening session")
    if session.query_count != 3:
        raise RuntimeError("M0 integration query count changed")

    outputs = (choice, score, noul)
    expected_k = (2, 3, 2)
    for out, k in zip(outputs, expected_k):
        if int(out.hira.candidate_budget.item()) != k:
            raise RuntimeError("M0 integration full-K invariant changed")
        if not bool(out.hira.selected_mask.all()):
            raise RuntimeError("M0 integration selected mask changed")
        if float(out.hira.relation_delta.abs().max()) != 0.0:
            raise RuntimeError("M0 integration relation delta changed")
        if abs(float(out.probabilities.sum()) - 1.0) > 1e-6:
            raise RuntimeError("M0 integration probability mass changed")

    manifest = model.manifest.to_dict()
    parameters = model.parameter_report().to_dict()
    if manifest["production_ready"] is not False:
        raise RuntimeError("M0 integration cannot claim production readiness")
    if manifest["transfer_core"] != "provisional":
        raise RuntimeError("M0 integration transfer status changed")
    if parameters["transfer_candidate_parameters"] != 8192:
        raise RuntimeError("M0 integration transfer parameter surface changed")
    if parameters["projection_parameters"] != 128 * 256:
        raise RuntimeError("M0 integration projection surface changed")
    if parameters["trainable_total"] != 0:
        raise RuntimeError("M0 packaged inference shell must be fully frozen")

    receipt = {
        "schema_version": "hira-v0-mainline-m0-integration-v1",
        "status": "PASS",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "manifest": manifest,
        "parameter_report": parameters,
        "state_encode_count": model.runtime.state_encode_calls,
        "session_query_count": session.query_count,
        "query_candidate_budgets": [
            int(out.hira.candidate_budget.item())
            for out in outputs
        ],
        "relation_delta_max_abs": max(
            float(out.hira.relation_delta.abs().max())
            for out in outputs
        ),
        "probability_mass_max_error": max(
            abs(float(out.probabilities.sum()) - 1.0)
            for out in outputs
        ),
        "choice_selected_option_id": choice.selected_option_id,
        "score_value": float(score.value),
        "noul_value": float(noul.value),
        "quality_claim_made": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M0_INTEGRATION=" + json.dumps({
        "status": receipt["status"],
        "manifest": manifest,
        "parameter_report": parameters,
        "state_encode_count": receipt["state_encode_count"],
        "session_query_count": receipt["session_query_count"],
        "query_candidate_budgets": receipt["query_candidate_budgets"],
        "quality_claim_made": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
