from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_capacity_control import (
    EPOCHS,
    GLOBAL_SEED,
    LR,
    TOP_N,
    generate_capacity_authority,
)
from nmd.semantic_encoder_adaptation import (
    dev_selection_key,
    evaluate_encoder,
    train_candidate,
)


MODELS = {
    "a13": {
        "model_id": "microsoft/xtremedistil-l6-h256-uncased",
        "revision": "4226d9e4d2c08703e5cb0491b479bfc6a1607181",
        "weight_file": "model.safetensors",
        "weight_sha256": "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880",
        "hidden_size": 256,
    },
    "a22": {
        "model_id": "microsoft/xtremedistil-l6-h384-uncased",
        "revision": "359df7d52613d4edc15647e6d65e0d87200eb747",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "38bd5f8a7d1b7045de8fee25bfac1777edf5a2ec8cd3399b21bde917b0278e23",
        "hidden_size": 384,
    },
}
MAX_LENGTH = 256


def file_sha256(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    spec = MODELS[args.model]
    snapshot = Path(
        snapshot_download(
            repo_id=spec["model_id"],
            revision=spec["revision"],
        )
    )
    weight_path = snapshot / spec["weight_file"]
    if file_sha256(weight_path) != spec["weight_sha256"]:
        raise RuntimeError(f"{args.model} frozen weight SHA mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    if int(base.config.hidden_size) != int(spec["hidden_size"]):
        raise RuntimeError(f"{args.model} hidden-size mismatch")
    if int(base.config.num_hidden_layers) != 6:
        raise RuntimeError(f"{args.model} layer-count mismatch")

    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=str(spec["revision"]),
        max_length=MAX_LENGTH,
    )

    train_cases = generate_capacity_authority("train")
    dev_cases = generate_capacity_authority("dev")
    if len(train_cases) != 416 or len(dev_cases) != 144:
        raise RuntimeError("W5e TRAIN/DEV authority count changed")

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
        top_n=TOP_N,
        lr=LR,
        epochs=EPOCHS,
        seed=GLOBAL_SEED,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "adaptation.pt"
    torch.save(state, checkpoint)
    selected_epoch = int(best["epoch"])
    receipt = {
        "schema_version": "r8-w5e-adapted-track-v1",
        "status": "PASS",
        "model_key": args.model,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "weight_file": spec["weight_file"],
        "weight_sha256": spec["weight_sha256"],
        "hidden_size": spec["hidden_size"],
        "top_n": TOP_N,
        "lr": LR,
        "weight_decay": 0.01,
        "epochs": EPOCHS,
        "global_seed": GLOBAL_SEED,
        "selected_epoch": selected_epoch,
        "frozen_dev_metrics": frozen_dev,
        "selected_dev_metrics": best,
        "selection_key": list(dev_selection_key(best, selected_epoch)),
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
        "model_key": args.model,
        "selected_epoch": selected_epoch,
        "frozen_dev_metrics": frozen_dev,
        "selected_dev_metrics": best,
        "adaptation_sha256": receipt["adaptation_sha256"],
        "trainable_parameter_count": info["trainable_parameter_count"],
        "total_parameter_count": info["total_parameter_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
