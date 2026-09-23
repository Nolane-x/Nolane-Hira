from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_encoder_adaptation import (
    EPOCHS,
    GLOBAL_SEED,
    dev_selection_key,
    evaluate_encoder,
    generate_adaptation_authority,
    train_candidate,
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
    parser.add_argument("--top-n", type=int, choices=[1, 2], required=True)
    parser.add_argument("--lr", type=float, choices=[1e-5, 3e-5], required=True)
    parser.add_argument("--name", required=True)
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
    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=A13_MAX_LENGTH,
    )

    train_cases = generate_adaptation_authority("train")
    dev_cases = generate_adaptation_authority("dev")
    if len(train_cases) != 416 or len(dev_cases) != 144:
        raise RuntimeError("W5d TRAIN/DEV counts changed")

    for parameter in encoder.model.parameters():
        parameter.requires_grad_(False)
    frozen_dev = evaluate_encoder(encoder, dev_cases)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    history, state, best, info = train_candidate(
        encoder,
        train_cases,
        dev_cases,
        top_n=args.top_n,
        lr=args.lr,
        epochs=EPOCHS,
        seed=GLOBAL_SEED,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "adaptation.pt"
    torch.save(state, checkpoint)
    epoch = int(best["epoch"])
    receipt = {
        "schema_version": "r8-w5d-candidate-v1",
        "status": "PASS",
        "candidate_name": args.name,
        "base_a13_model": A13_MODEL,
        "base_a13_revision": A13_REVISION,
        "base_a13_weight_sha256": A13_WEIGHT_SHA256,
        "top_n": args.top_n,
        "lr": args.lr,
        "weight_decay": 0.01,
        "epochs": EPOCHS,
        "global_seed": GLOBAL_SEED,
        "selected_epoch": epoch,
        "frozen_dev_metrics": frozen_dev,
        "selected_dev_metrics": best,
        "selection_key": list(dev_selection_key(best, epoch)),
        "history": history,
        "adaptation_sha256": file_sha256(checkpoint),
        "layer_count": info["layer_count"],
        "trainable_parameter_names": info["trainable_parameter_names"],
        "trainable_parameter_count": info["trainable_parameter_count"],
        "frozen_parameter_count": info["frozen_parameter_count"],
        "total_parameter_count": info["total_parameter_count"],
        "confirm_exposed": False,
        "forbidden_benchmark_data_used": False,
        "campaign_cells_populated": 0,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "candidate_name": args.name,
        "selected_epoch": epoch,
        "frozen_dev_metrics": frozen_dev,
        "selected_dev_metrics": best,
        "adaptation_sha256": receipt["adaptation_sha256"],
        "trainable_parameter_count": info["trainable_parameter_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
