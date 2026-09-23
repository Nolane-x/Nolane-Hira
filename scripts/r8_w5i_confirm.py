from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_cross_candidate_binding import (
    CONFIRM_SEED,
    MAX_LENGTH,
    CrossCandidateBindingMatcher,
    compile_binding_cache,
    confirm_verdict,
    evaluate_matcher,
    evaluate_pooled_baseline,
    generate_binding_authority,
)

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
CONTROL = "idf-competitive-forward-proj128"


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--matcher", type=Path, required=True)
    parser.add_argument("--control-matcher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "r8-w5i-selection-receipt-v1":
        raise RuntimeError("unexpected W5i selection schema")
    if selection.get("status") != "PASS":
        raise RuntimeError("W5i selection is not PASS")
    if selection.get("confirm_exposed") is not False:
        raise RuntimeError("W5i selection already exposed confirm")
    if file_sha256(args.matcher) != selection["selected_matcher_sha256"]:
        raise RuntimeError("selected W5i matcher SHA mismatch")
    if file_sha256(args.control_matcher) != selection["control_matcher_sha256"]:
        raise RuntimeError("control W5i matcher SHA mismatch")
    if selection.get("all_candidate_parameter_counts_equal") is not True:
        raise RuntimeError("W5i candidate capacity equality not verified")
    if int(selection.get("selected_trainable_parameter_count", -1)) != 32769:
        raise RuntimeError("unexpected selected W5i trainable parameter count")
    if int(selection.get("control_trainable_parameter_count", -1)) != 32769:
        raise RuntimeError("unexpected control W5i trainable parameter count")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=MAX_LENGTH,
    )

    print("R8_W5H_CONFIRM_GENERATION_BEGIN", flush=True)
    confirm_cases = generate_binding_authority("confirm", allow_confirm=True)
    confirm_cache = compile_binding_cache(encoder, confirm_cases)

    selected = CrossCandidateBindingMatcher(selection["selected_candidate"])
    selected_state = torch.load(args.matcher, map_location="cpu", weights_only=True)
    selected.load_state_dict(selected_state, strict=True)
    selected.eval()

    control = CrossCandidateBindingMatcher(CONTROL)
    control_state = torch.load(
        args.control_matcher, map_location="cpu", weights_only=True
    )
    control.load_state_dict(control_state, strict=True)
    control.eval()

    selected_metrics = evaluate_matcher(selected, confirm_cache)
    control_metrics = evaluate_matcher(control, confirm_cache)
    pooled_metrics = evaluate_pooled_baseline(confirm_cache)
    verdict, gates = confirm_verdict(selected_metrics, control_metrics)

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "r8-w5i-confirm-v1",
        "status": "PASS",
        "selected_candidate": selection["selected_candidate"],
        "selected_epoch": selection["selected_epoch"],
        "selected_matcher_sha256": selection["selected_matcher_sha256"],
        "control_candidate": CONTROL,
        "control_epoch": selection["control_epoch"],
        "control_matcher_sha256": selection["control_matcher_sha256"],
        "confirm_seed": CONFIRM_SEED,
        "salience_formula": "log((K + 1) / (df(t) + 1)) + 1; valid-token mean normalized to 1",
        "selected_binding_operator": selection["selected_candidate"],
        "control_binding_operator": CONTROL,
        "trainable_parameter_count": selection["selected_trainable_parameter_count"],
        "all_candidate_parameter_counts_equal": selection["all_candidate_parameter_counts_equal"],
        "candidate_dev_histories": selection["candidates"],
        "confirm_generated_after_selection_freeze": True,
        "confirm_case_count": len(confirm_cases),
        "confirm_encoder_calls": confirm_cache["encoder_calls"],
        "state_text_encodes_per_case": confirm_cache["state_text_encodes_per_case"],
        "selected": selected_metrics,
        "control_baseline": control_metrics,
        "pooled_baseline": pooled_metrics,
        "gates": gates,
        "verdict": verdict,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "forbidden_benchmark_data_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "confirm.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
