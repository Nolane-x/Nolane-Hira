from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.banking77_direct import (
    A13_MAX_LENGTH,
    A13_MODEL,
    A13_REVISION,
    A13_WEIGHT_SHA256,
    EXPECTED_EXAMPLES,
    EXPECTED_LABELS,
    LAYA_TARGET,
    SELECTED_HEAD_SHA256,
    evaluate_banking77_direct,
    load_and_validate_marker,
    load_banking77_authority,
)
from nmd.hira import HIRACore, count_parameters
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder

EXPECTED_HEAD_PARAMS = 422_159


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--selected-head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    marker = load_and_validate_marker(args.marker)
    selected_sha = file_sha256(args.selected_head)
    if selected_sha != SELECTED_HEAD_SHA256:
        raise RuntimeError("selected HIRA head SHA mismatch")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight_path = snapshot / "model.safetensors"
    if file_sha256(weight_path) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    encoder = HFAutoSemanticEncoder(
        base, tokenizer, revision=A13_REVISION, max_length=A13_MAX_LENGTH
    )
    hira = HIRACore(d_model=256, dropout=0.05)
    state = torch.load(args.selected_head, map_location="cpu", weights_only=True)
    hira.load_state_dict(state, strict=True)
    if count_parameters(hira) != EXPECTED_HEAD_PARAMS:
        raise RuntimeError("HIRA head parameter count changed")

    model = NolaneHira(encoder, hira)
    model.eval()

    print("R8_W4A_BANKING77_EXPOSURE_BEGIN", flush=True)
    authority = load_banking77_authority()
    metrics = evaluate_banking77_direct(model, authority)

    if metrics["case_count"] != EXPECTED_EXAMPLES:
        raise RuntimeError("Banking77 case count mismatch")
    if metrics["label_count"] != EXPECTED_LABELS:
        raise RuntimeError("Banking77 label count mismatch")

    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "r8-w4a-banking77-direct-results-v1",
        "status": "PASS",
        "scientific_authority": "ONE_SHOT_HELD_OUT_DIRECT",
        "marker": marker,
        "selected_head_sha256": selected_sha,
        "head_parameter_count": count_parameters(hira),
        "metrics": metrics,
        "headline": {
            "id": "laya.app.banking77_full",
            "direction": "higher",
            "target": LAYA_TARGET,
            "value": metrics["accuracy"],
            "status": metrics["laya_status"],
        },
        "jev_cells_populated": False,
        "banking77_task_training_used": False,
    }
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
