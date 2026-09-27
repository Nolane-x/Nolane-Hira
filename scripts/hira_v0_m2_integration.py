from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from nmd.contracts import LogicalOption
from nmd.high_cardinality_stress import ACTIONS, COLORS, OBJECTS
from nmd.mainline import (
    HIRA_V0_MAX_K,
    W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    build_hira_v0_m2_mechanics,
)
from nmd.mainline_high_k import M2_K_LADDER, run_high_k_mechanics_suite
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import W28_T0_CHECKPOINT_SHA256
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

M2_INTEGRATION_SCHEMA = "hira-v0-mainline-m2-mechanics-integration-v1"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"M2 A13 weight SHA mismatch: {actual}")

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
        raise RuntimeError("M2 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("M2 T0 receipt identity changed")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("M2 T0 checkpoint bytes changed")
    return checkpoint


def _validate_w34(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w34-coevidence-training-receipt-v1":
        raise RuntimeError("M2 unexpected W34 receipt schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("M2 W34 receipt did not pass")
    if receipt.get("checkpoint_sha256") != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2 W34 checkpoint identity changed")
    if file_sha256(checkpoint) != W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256:
        raise RuntimeError("M2 W34 checkpoint bytes changed")
    if int(receipt.get("confirm_case_count_used", -1)) != 0:
        raise RuntimeError("M2 W34 training receipt contains confirm leakage")
    return checkpoint


def _phrase_space() -> tuple[str, ...]:
    return tuple(
        f"{color} {obj} {action} route"
        for color, obj, action in itertools.product(COLORS, OBJECTS, ACTIONS)
    )


PHRASES = _phrase_space()


def _options(k: int) -> tuple[LogicalOption, ...]:
    if not 4 <= k <= HIRA_V0_MAX_K:
        raise ValueError(f"M2 integration invalid K: {k}")
    return tuple(
        LogicalOption(
            option_id=f"m2-route-{index:03d}",
            criterion_text=PHRASES[index],
        )
        for index in range(k)
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
    model = build_hira_v0_m2_mechanics(encoder, t0, w34)

    parameter_report = model.parameter_report().to_dict()
    if parameter_report["trainable_total"] != 0:
        raise RuntimeError("M2 mechanics integration requires frozen runtime")

    before = model.runtime.state_encode_calls
    suite = run_high_k_mechanics_suite(
        model,
        state_text=(
            "The route record explicitly names amber anchor align route. "
            "The remaining route phrases are alternatives only."
        ),
        question_text="Which route phrase matches the route record?",
        option_factory=_options,
    )
    state_encode_delta = model.runtime.state_encode_calls - before

    if state_encode_delta != 1:
        raise RuntimeError("M2 exact integration violated state-once")
    if suite.k_values != M2_K_LADDER:
        raise RuntimeError("M2 K ladder changed")
    if suite.query_count != 14:
        raise RuntimeError("M2 query count changed")

    outcome = (
        "HIRA_V0_M2_MECHANICS_READY"
        if suite.mechanics_pass
        else "HIRA_V0_M2_MECHANICS_FAIL"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": M2_INTEGRATION_SCHEMA,
        "status": "PASS",
        "outcome": outcome,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "transfer_checkpoint_sha256": W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        "manifest": model.manifest.to_dict(),
        "parameter_report": parameter_report,
        "k_ladder": list(M2_K_LADDER),
        "max_k": HIRA_V0_MAX_K,
        "state_encode_delta": state_encode_delta,
        "query_count": suite.query_count,
        "suite": suite.to_dict(),
        "semantic_quality_claim_made": False,
        "reliability_promotion_claim_made": False,
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V0_M2_MECHANICS_FINAL=" + json.dumps({
        "outcome": outcome,
        "parameter_report": parameter_report,
        "suite": suite.to_dict(),
        "semantic_quality_claim_made": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
