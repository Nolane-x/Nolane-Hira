from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_contrastive_salience import (
    MAX_LENGTH,
    ContrastiveSalienceMatcher,
    compile_salience_cache,
    confirm_verdict,
    evaluate_matcher,
    evaluate_pooled_baseline,
    generate_salience_authority,
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
    parser.add_argument("--uniform-matcher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "r8-w5g-selection-receipt-v1":
        raise RuntimeError("unexpected W5g selection schema")
    if selection.get("status") != "PASS":
        raise RuntimeError("W5g selection is not PASS")
    if selection.get("confirm_exposed") is not False:
        raise RuntimeError("W5g selection already exposed confirm")
    if file_sha256(args.matcher) != selection["selected_matcher_sha256"]:
        raise RuntimeError("selected W5g matcher SHA mismatch")
    if file_sha256(args.uniform_matcher) != selection["uniform_matcher_sha256"]:
        raise RuntimeError("uniform W5g matcher SHA mismatch")

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

    print("R8_W5G_CONFIRM_GENERATION_BEGIN", flush=True)
    confirm_cases = generate_salience_authority("confirm")
    confirm_cache = compile_salience_cache(encoder, confirm_cases)

    selected = ContrastiveSalienceMatcher(selection["selected_candidate"])
    selected_state = torch.load(args.matcher, map_location="cpu", weights_only=True)
    selected.load_state_dict(selected_state, strict=True)
    selected.eval()

    uniform = ContrastiveSalienceMatcher("uniform-proj128")
    uniform_state = torch.load(args.uniform_matcher, map_location="cpu", weights_only=True)
    uniform.load_state_dict(uniform_state, strict=True)
    uniform.eval()

    selected_metrics = evaluate_matcher(selected, confirm_cache)
    uniform_metrics = evaluate_matcher(uniform, confirm_cache)
    pooled_metrics = evaluate_pooled_baseline(confirm_cache)
    verdict, gates = confirm_verdict(selected_metrics, uniform_metrics)

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "r8-w5g-confirm-v1",
        "status": "PASS",
        "selected_candidate": selection["selected_candidate"],
        "selected_epoch": selection["selected_epoch"],
        "selected_matcher_sha256": selection["selected_matcher_sha256"],
        "uniform_epoch": selection["uniform_epoch"],
        "uniform_matcher_sha256": selection["uniform_matcher_sha256"],
        "confirm_generated_after_selection_freeze": True,
        "confirm_case_count": len(confirm_cases),
        "confirm_encoder_calls": confirm_cache["encoder_calls"],
        "state_text_encodes_per_case": confirm_cache["state_text_encodes_per_case"],
        "selected": selected_metrics,
        "uniform_baseline": uniform_metrics,
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
