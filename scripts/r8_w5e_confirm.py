from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_capacity_control import (
    MAX_LENGTH,
    MODEL_SPECS,
    capacity_verdict,
    generate_capacity_authority,
)
from nmd.semantic_encoder_adaptation import (
    evaluate_encoder,
    load_adaptation_state,
)


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(model_key: str) -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    spec = MODEL_SPECS[model_key]
    snapshot = Path(
        snapshot_download(
            repo_id=spec["model_id"],
            revision=spec["revision"],
        )
    )
    if file_sha256(snapshot / spec["weight_file"]) != spec["weight_sha256"]:
        raise RuntimeError(f"{model_key} weight SHA mismatch")
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
        revision=str(spec["revision"]),
        max_length=MAX_LENGTH,
    )


def validate_track(
    model_key: str,
    receipt_path: Path,
    adaptation_path: Path,
) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w5e-adapted-track-v1":
        raise RuntimeError("unexpected W5e adapted-track receipt")
    if receipt.get("status") != "PASS":
        raise RuntimeError("W5e adapted track not PASS")
    if receipt.get("model_key") != model_key:
        raise RuntimeError("W5e model-key mismatch")
    if receipt.get("confirm_exposed") is not False:
        raise RuntimeError("W5e confirm was exposed before checkpoint freeze")
    if file_sha256(adaptation_path) != receipt["adaptation_sha256"]:
        raise RuntimeError("W5e adaptation SHA mismatch")
    if int(receipt["top_n"]) != 2 or float(receipt["lr"]) != 1e-5:
        raise RuntimeError("W5e fixed adaptation recipe changed")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a13-adaptation", type=Path, required=True)
    parser.add_argument("--a13-receipt", type=Path, required=True)
    parser.add_argument("--a22-adaptation", type=Path, required=True)
    parser.add_argument("--a22-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a13_receipt = validate_track(
        "a13",
        args.a13_receipt,
        args.a13_adaptation,
    )
    a22_receipt = validate_track(
        "a22",
        args.a22_receipt,
        args.a22_adaptation,
    )

    a13_frozen = load_encoder("a13")
    a22_frozen = load_encoder("a22")
    for encoder in (a13_frozen, a22_frozen):
        for parameter in encoder.model.parameters():
            parameter.requires_grad_(False)

    a13_adapted = load_encoder("a13")
    a13_state = torch.load(
        args.a13_adaptation,
        map_location="cpu",
        weights_only=True,
    )
    a13_info = load_adaptation_state(
        a13_adapted,
        a13_state,
        top_n=2,
    )
    a13_adapted.eval()
    for parameter in a13_adapted.model.parameters():
        parameter.requires_grad_(False)

    a22_adapted = load_encoder("a22")
    a22_state = torch.load(
        args.a22_adaptation,
        map_location="cpu",
        weights_only=True,
    )
    a22_info = load_adaptation_state(
        a22_adapted,
        a22_state,
        top_n=2,
    )
    a22_adapted.eval()
    for parameter in a22_adapted.model.parameters():
        parameter.requires_grad_(False)

    print("R8_W5E_CONFIRM_GENERATION_BEGIN", flush=True)
    cases = generate_capacity_authority("confirm")
    if len(cases) != 192:
        raise RuntimeError("W5e CONFIRM count changed")

    tracks = {
        "a13_frozen": evaluate_encoder(a13_frozen, cases),
        "a22_frozen": evaluate_encoder(a22_frozen, cases),
        "a13_adapted": evaluate_encoder(a13_adapted, cases),
        "a22_adapted": evaluate_encoder(a22_adapted, cases),
    }
    verdict, gates = capacity_verdict(
        tracks["a22_adapted"],
        tracks["a13_adapted"],
    )

    result = {
        "schema_version": "r8-w5e-capacity-confirm-v1",
        "status": "PASS",
        "verdict": verdict,
        "confirm_generated_after_both_checkpoint_freezes": True,
        "confirm_case_count": len(cases),
        "a13": {
            "revision": MODEL_SPECS["a13"]["revision"],
            "weight_sha256": MODEL_SPECS["a13"]["weight_sha256"],
            "adaptation_sha256": a13_receipt["adaptation_sha256"],
            "selected_epoch": a13_receipt["selected_epoch"],
            "trainable_parameter_count": a13_info["trainable_parameter_count"],
            "total_parameter_count": a13_info["total_parameter_count"],
        },
        "a22": {
            "revision": MODEL_SPECS["a22"]["revision"],
            "weight_sha256": MODEL_SPECS["a22"]["weight_sha256"],
            "adaptation_sha256": a22_receipt["adaptation_sha256"],
            "selected_epoch": a22_receipt["selected_epoch"],
            "trainable_parameter_count": a22_info["trainable_parameter_count"],
            "total_parameter_count": a22_info["total_parameter_count"],
        },
        "tracks": tracks,
        "capacity_gates": gates,
        "forbidden_benchmark_data_used": False,
        "campaign_cells_populated": 0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "confirm.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
