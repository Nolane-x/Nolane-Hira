from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_decisions import (
    TYPED_DECISIONS_DATASET_ID,
    TYPED_DECISIONS_REVISION,
    load_pinned_typed_decisions_train,
)
from nmd.typed_feature_cache import (
    compile_w3_feature_cache,
    save_w3_feature_cache,
)


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"
MAX_LENGTH = 256
SEGMENT_TOKENS = 32


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

    args.out.mkdir(parents=True, exist_ok=True)

    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    weight_path = snapshot / "model.safetensors"
    if file_sha256(weight_path) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

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
        max_length=MAX_LENGTH,
    )
    model = NolaneHira(
        encoder,
        HIRACore(d_model=256, dropout=0.0),
    )
    model.eval()

    cases = load_pinned_typed_decisions_train()
    if len(cases) != 1200:
        raise RuntimeError("W3 source must contain exactly 1200 train cases")

    cache = compile_w3_feature_cache(
        model,
        cases,
        segment_tokens=SEGMENT_TOKENS,
    )
    cache["metadata"].update(
        {
            "dataset_id": TYPED_DECISIONS_DATASET_ID,
            "dataset_revision": TYPED_DECISIONS_REVISION,
            "dataset_split": "train",
            "a13_model": A13_MODEL,
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
            "max_length": MAX_LENGTH,
        }
    )

    cache_path = save_w3_feature_cache(
        cache,
        args.out / "typed-feature-cache.pt",
    )
    receipt = {
        "schema_version": "r8-w3-cache-receipt-v1",
        "status": "PASS",
        "scope": "typed-decisions TRAIN only; no final test exposure",
        "dataset_id": TYPED_DECISIONS_DATASET_ID,
        "dataset_revision": TYPED_DECISIONS_REVISION,
        "dataset_split": "train",
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "max_length": MAX_LENGTH,
        "segment_tokens": SEGMENT_TOKENS,
        "cache_sha256": file_sha256(cache_path),
        "cache_metadata": cache["metadata"],
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
