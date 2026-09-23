from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_token_curriculum import (
    compile_token_cache,
    generate_token_authority,
    save_token_cache,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=A13_MAX_LENGTH,
    )
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    model.eval()

    train_cases = generate_token_authority("train")
    dev_cases = generate_token_authority("dev")
    if len(train_cases) != 512 or len(dev_cases) != 160:
        raise RuntimeError("W5b TRAIN/DEV authority counts changed")

    train_cache = compile_token_cache(model, train_cases)
    dev_cache = compile_token_cache(model, dev_cases)

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train-cache.pt"
    dev_path = args.out / "dev-cache.pt"
    save_token_cache(train_cache, train_path)
    save_token_cache(dev_cache, dev_path)

    receipt = {
        "schema_version": "r8-w5b-cache-receipt-v1",
        "status": "PASS",
        "scope": "fresh generated W5b TRAIN/DEV only",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "train_case_count": train_cache["case_count"],
        "dev_case_count": dev_cache["case_count"],
        "train_state_encode_calls": train_cache["state_encode_calls"],
        "dev_state_encode_calls": dev_cache["state_encode_calls"],
        "train_state_encode_calls_per_case": train_cache[
            "state_encode_calls_per_case"
        ],
        "dev_state_encode_calls_per_case": dev_cache[
            "state_encode_calls_per_case"
        ],
        "train_cache_sha256": file_sha256(train_path),
        "dev_cache_sha256": file_sha256(dev_path),
        "confirm_exposed": False,
        "forbidden_benchmark_data_used": False,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
