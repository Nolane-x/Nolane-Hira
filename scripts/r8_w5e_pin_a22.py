from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


MODEL_ID = "microsoft/xtremedistil-l6-h384-uncased"
EXPECTED_HIDDEN = 384
EXPECTED_LAYERS = 6


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_assets(snapshot: Path) -> dict[str, dict[str, object]]:
    wanted = {
        "pytorch_model.bin",
        "model.safetensors",
        "config.json",
        "vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    out: dict[str, dict[str, object]] = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file() and path.name in wanted:
            out[path.name] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel, AutoTokenizer

    info = HfApi().model_info(MODEL_ID)
    revision = str(info.sha)
    if len(revision) != 40:
        raise RuntimeError(
            f"A22 resolved revision must be 40-char immutable SHA; got {revision!r}"
        )

    root = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
        )
    )
    assets = hash_assets(root)
    if "pytorch_model.bin" not in assets:
        raise RuntimeError(
            "A22 pin authority requires pytorch_model.bin"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(root),
        local_files_only=True,
    )
    model.eval()

    hidden = int(model.config.hidden_size)
    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_attention_heads)
    total_params = sum(p.numel() for p in model.parameters())
    pooler_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith("pooler.")
    )
    encoder_params_no_pooler = total_params - pooler_params

    if hidden != EXPECTED_HIDDEN:
        raise AssertionError(
            f"expected A22 hidden_size={EXPECTED_HIDDEN}, got {hidden}"
        )
    if layers != EXPECTED_LAYERS:
        raise AssertionError(
            f"expected A22 layers={EXPECTED_LAYERS}, got {layers}"
        )

    # Offline-load parity check from the pinned snapshot only.
    ids = tokenizer(
        ["capacity control pin check"],
        return_tensors="pt",
        truncation=True,
        max_length=32,
    )
    output = model(**ids)
    if tuple(output.last_hidden_state.shape)[-1] != EXPECTED_HIDDEN:
        raise AssertionError("A22 offline forward hidden-size mismatch")

    receipt = {
        "schema_version": "r8-w5e-a22-pin-v1",
        "status": "PASS",
        "scope": (
            "PIN_ONLY: no W5e semantic TRAIN/DEV/CONFIRM generation "
            "or evaluation"
        ),
        "model_id": MODEL_ID,
        "resolved_revision": revision,
        "assets": assets,
        "primary_weight_file": "pytorch_model.bin",
        "primary_weight_sha256": assets["pytorch_model.bin"]["sha256"],
        "config_sha256": assets.get("config.json", {}).get("sha256"),
        "vocab_sha256": assets.get("vocab.txt", {}).get("sha256"),
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "model_total_params": total_params,
        "pooler_params": pooler_params,
        "encoder_params_no_pooler": encoder_params_no_pooler,
        "semantic_cases_generated": 0,
        "semantic_cases_evaluated": 0,
        "campaign_cells_populated": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
