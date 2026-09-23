from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_encoder_adaptation import (
    confirm_verdict,
    evaluate_encoder,
    generate_adaptation_authority,
    load_adaptation_state,
)


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
A13_MAX_LENGTH = 256


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_base(snapshot: Path):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    return HFAutoSemanticEncoder(
        model,
        tokenizer,
        revision=A13_REVISION,
        max_length=A13_MAX_LENGTH,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptation", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "r8-w5d-selection-v1":
        raise RuntimeError("unexpected W5d selection receipt")
    if selection.get("status") != "PASS":
        raise RuntimeError("W5d selection not PASS")
    if selection.get("confirm_exposed") is not False:
        raise RuntimeError("W5d confirm was already exposed")
    if file_sha256(args.adaptation) != selection["selected_adaptation_sha256"]:
        raise RuntimeError("W5d selected adaptation SHA mismatch")

    from huggingface_hub import snapshot_download
    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    if file_sha256(snapshot / "model.safetensors") != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")

    frozen = load_base(snapshot)
    for parameter in frozen.model.parameters():
        parameter.requires_grad_(False)

    adapted = load_base(snapshot)
    state = torch.load(args.adaptation, map_location="cpu", weights_only=True)
    info = load_adaptation_state(
        adapted,
        state,
        top_n=int(selection["selected_config"]["top_n"]),
    )
    adapted.eval()
    for parameter in adapted.model.parameters():
        parameter.requires_grad_(False)

    print("R8_W5D_CONFIRM_GENERATION_BEGIN", flush=True)
    cases = generate_adaptation_authority("confirm")
    if len(cases) != 192:
        raise RuntimeError("W5d CONFIRM count changed")

    frozen_metrics = evaluate_encoder(frozen, cases)
    adapted_metrics = evaluate_encoder(adapted, cases)
    verdict, gates = confirm_verdict(adapted_metrics, frozen_metrics)

    result = {
        "schema_version": "r8-w5d-confirm-v1",
        "status": "PASS",
        "verdict": verdict,
        "confirm_generated_after_selection_freeze": True,
        "confirm_case_count": len(cases),
        "selected_candidate": selection["selected_candidate"],
        "selected_config": selection["selected_config"],
        "selected_adaptation_sha256": selection["selected_adaptation_sha256"],
        "base_a13_revision": A13_REVISION,
        "base_a13_weight_sha256": A13_WEIGHT_SHA256,
        "layer_count": info["layer_count"],
        "trainable_parameter_count": info["trainable_parameter_count"],
        "frozen_metrics": frozen_metrics,
        "adapted_metrics": adapted_metrics,
        "gates": gates,
        "forbidden_benchmark_data_used": False,
        "campaign_cells_populated": 0,
        "a22_capacity_control_authorized": (
            verdict == "A13_ADAPTATION_FAIL_CAPACITY_TRIGGER"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "confirm.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
