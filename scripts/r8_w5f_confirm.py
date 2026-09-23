from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_late_interaction import (
    MAX_LENGTH,
    LateInteractionMatcher,
    compile_late_interaction_cache,
    confirm_verdict,
    evaluate_matcher,
    evaluate_pooled_baseline,
    generate_late_interaction_authority,
)

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"


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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "r8-w5f-selection-receipt-v1":
        raise RuntimeError("unexpected W5f selection schema")
    if selection.get("status") != "PASS":
        raise RuntimeError("W5f selection is not PASS")
    if selection.get("confirm_exposed") is not False:
        raise RuntimeError("W5f selection already exposed confirm")
    if file_sha256(args.matcher) != selection["selected_matcher_sha256"]:
        raise RuntimeError("selected W5f matcher SHA mismatch")

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

    # Scientific exposure boundary: no CONFIRM case exists before this line.
    print("R8_W5F_CONFIRM_GENERATION_BEGIN", flush=True)
    confirm_cases = generate_late_interaction_authority("confirm")
    confirm_cache = compile_late_interaction_cache(encoder, confirm_cases)

    projection_dim = selection["projection_dim"]
    matcher = LateInteractionMatcher(
        None if projection_dim is None else int(projection_dim)
    )
    state = torch.load(args.matcher, map_location="cpu", weights_only=True)
    matcher.load_state_dict(state, strict=True)
    matcher.eval()

    selected_metrics = evaluate_matcher(matcher, confirm_cache)
    pooled_metrics = evaluate_pooled_baseline(confirm_cache)
    verdict, gates = confirm_verdict(selected_metrics, pooled_metrics)

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "r8-w5f-confirm-v1",
        "status": "PASS",
        "selected_candidate": selection["selected_candidate"],
        "projection_dim": projection_dim,
        "selected_epoch": selection["selected_epoch"],
        "selected_matcher_sha256": selection["selected_matcher_sha256"],
        "confirm_generated_after_selection_freeze": True,
        "confirm_case_count": len(confirm_cases),
        "confirm_encoder_calls": confirm_cache["encoder_calls"],
        "state_text_encodes_per_case": confirm_cache["state_text_encodes_per_case"],
        "selected": selected_metrics,
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
