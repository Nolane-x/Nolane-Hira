from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from nmd.hira import HIRACore
from nmd.hira_v0_authority import all_w29_text_atoms
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    build_hira_v0_semantic_core,
    load_rescued_projection_checkpoint,
)
from nmd.semantic_transfer_authority import (
    all_w30_text_atoms,
    generate_w30_partition,
)
from nmd.semantic_transfer_cache import compile_w30_cache
from nmd.semantic_transfer_core import (
    W30_BRIDGE_CHECKPOINT_SCHEMA,
    W30_BRIDGE_PARAMETER_COUNT,
    W30_BRIDGE_RANK,
)
from nmd.semantic_transfer_eval import (
    BRIDGE_BATCH_SIZE,
    BRIDGE_EPOCHS,
    BRIDGE_GRAD_CLIP,
    BRIDGE_LR,
    BRIDGE_SEED,
    BRIDGE_TEMPERATURE,
    BRIDGE_WEIGHT_DECAY,
    build_baseline_scorer,
    build_bridged_scorer,
    evaluate_w30_cache,
    train_transfer_bridge,
)
from nmd.typed_competitive_cache import file_sha256

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"


def prior_text_atoms() -> set[str]:
    spec = importlib.util.spec_from_file_location(
        "w29eval",
        Path(__file__).with_name("r8_w29_evaluate.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load W29 prior-text firewall")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.prior_text_atoms()) | set(all_w29_text_atoms())


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"W30 A13 weight SHA mismatch: {actual}")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=256,
    )


def _validate_t0(directory: Path) -> Path:
    checkpoint = directory / "candidate.pt"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "r8-w28-candidate-receipt-v1":
        raise RuntimeError("W30 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("W30 T0 identity mismatch")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W30 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W30 T0 checkpoint bytes changed")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    overlap = sorted(all_w30_text_atoms() & prior_text_atoms())
    if overlap:
        raise RuntimeError(f"W30 exact text overlaps exposed authorities: {overlap}")

    t0_path = _validate_t0(args.t0_dir)
    projection = load_rescued_projection_checkpoint(t0_path)

    encoder = _load_a13()
    cache_model = build_hira_v0_semantic_core(
        encoder,
        t0_path,
        hira=HIRACore(d_model=256, dropout=0.0),
    )
    cache_model.eval()

    train_rows = generate_w30_partition("train")
    dev_rows = generate_w30_partition("dev")
    before = cache_model.state_encode_calls
    train_cache = compile_w30_cache(cache_model, train_rows, partition="train")
    dev_cache = compile_w30_cache(cache_model, dev_rows, partition="dev")
    encoded = cache_model.state_encode_calls - before
    if encoded != len(train_rows) + len(dev_rows):
        raise RuntimeError("W30 train/dev state encoding count changed")

    baseline = build_baseline_scorer(projection)
    baseline_dev = evaluate_w30_cache(dev_cache, baseline)

    bridge_state, selected, history = train_transfer_bridge(
        train_cache,
        dev_cache,
        projection,
    )
    bridged = build_bridged_scorer(projection, bridge_state, freeze=True)
    bridged_train = evaluate_w30_cache(train_cache, bridged)
    bridged_dev = evaluate_w30_cache(dev_cache, bridged)

    if bridged.projection.weight.requires_grad:
        raise RuntimeError("W30 selected scorer unfroze T0")
    if bridged.bridge_parameter_count != W30_BRIDGE_PARAMETER_COUNT:
        raise RuntimeError("W30 bridge parameter count changed")
    if bridged.trainable_parameter_count != 0:
        raise RuntimeError("W30 selected inference scorer must be frozen")

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "bridge.pt"
    checkpoint = {
        "schema_version": W30_BRIDGE_CHECKPOINT_SCHEMA,
        "kind": "semantic-transfer-bridge",
        "rank": W30_BRIDGE_RANK,
        "parameter_count": W30_BRIDGE_PARAMETER_COUNT,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "selected_dev_epoch": int(selected["epoch"]),
        "bridge_state_dict": bridge_state,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = file_sha256(checkpoint_path)

    receipt = {
        "schema_version": "r8-w30-transfer-training-receipt-v1",
        "status": "PASS",
        "checkpoint_sha256": checkpoint_sha,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "train_case_count": len(train_rows),
        "dev_case_count": len(dev_rows),
        "confirm_case_count_used": 0,
        "w29_rows_used": False,
        "older_authority_rows_used": False,
        "projection_training_performed": False,
        "bridge_training_performed": True,
        "bridge_parameter_count": W30_BRIDGE_PARAMETER_COUNT,
        "bridge_rank": W30_BRIDGE_RANK,
        "seed": BRIDGE_SEED,
        "epochs": BRIDGE_EPOCHS,
        "batch_size": BRIDGE_BATCH_SIZE,
        "lr": BRIDGE_LR,
        "weight_decay": BRIDGE_WEIGHT_DECAY,
        "grad_clip": BRIDGE_GRAD_CLIP,
        "temperature": BRIDGE_TEMPERATURE,
        "selected_dev_epoch": int(selected["epoch"]),
        "selected_dev": selected["dev"],
        "baseline_dev": baseline_dev["pooled"],
        "bridged_dev": bridged_dev["pooled"],
        "bridged_train": bridged_train["pooled"],
        "exact_text_overlap": [],
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("W30_TRAIN=" + json.dumps({
        "checkpoint_sha256": checkpoint_sha,
        "selected_dev_epoch": int(selected["epoch"]),
        "baseline_dev": baseline_dev["pooled"],
        "bridged_dev": bridged_dev["pooled"],
        "bridged_train": bridged_train["pooled"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
