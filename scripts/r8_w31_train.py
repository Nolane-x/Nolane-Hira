from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.compositional_projection_authority import all_w28_text_atoms
from nmd.hira import HIRACore
from nmd.hira_v0_authority import all_w29_text_atoms
from nmd.semantic import HFAutoSemanticEncoder
from nmd.semantic_core import (
    W28_T0_CHECKPOINT_SHA256,
    build_hira_v0_semantic_core,
    load_rescued_projection_checkpoint,
)
from nmd.typed_competitive_cache import file_sha256
from nmd.w30_transfer_authority import all_w30_text_atoms
from nmd.w31_transfer_authority import (
    all_w31_text_atoms,
    generate_w31_partition,
)
from nmd.w31_transfer_cache import compile_w31_cache
from nmd.w31_transfer_core import (
    W31_ADAPTER_CHECKPOINT_SCHEMA,
    W31_ADAPTER_PARAMETER_COUNT,
    W31_ADAPTER_RANK,
)
from nmd.w31_transfer_eval import (
    ADAPTER_BATCH_SIZE,
    ADAPTER_EPOCHS,
    ADAPTER_GRAD_CLIP,
    ADAPTER_LR,
    ADAPTER_SEED,
    ADAPTER_TEMPERATURE,
    ADAPTER_WEIGHT_DECAY,
    ANCHOR_COEFFICIENT,
    ANCHOR_MARGIN_THRESHOLD,
    build_dual_scorer,
    evaluate_w31_cache,
    train_dual_adapter,
)

A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"


def _load_a13() -> HFAutoSemanticEncoder:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual = file_sha256(weight)
    if actual != A13_WEIGHT_SHA256:
        raise RuntimeError(f"W31 A13 weight SHA mismatch: {actual}")
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
        raise RuntimeError("W31 unexpected T0 receipt schema")
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "T0":
        raise RuntimeError("W31 T0 identity mismatch")
    if receipt.get("checkpoint_sha256") != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W31 T0 receipt SHA changed")
    if file_sha256(checkpoint) != W28_T0_CHECKPOINT_SHA256:
        raise RuntimeError("W31 T0 checkpoint bytes changed")
    return checkpoint


def _validate_qualification(directory: Path) -> dict[str, object]:
    path = directory / "qualification.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "r8-w31-reference-qualification-v1":
        raise RuntimeError("W31 qualification schema mismatch")
    if result.get("status") != "PASS":
        raise RuntimeError("W31 qualification execution did not pass")
    if result.get("outcome") != "W31_REFERENCE_QUALIFIED":
        raise RuntimeError("W31 reference authority is not qualified")
    if result.get("domains") != ["QH", "QI"] or int(result.get("case_count", -1)) != 192:
        raise RuntimeError("W31 qualification identity changed")
    if result.get("per_domain") != {"QH": True, "QI": True}:
        raise RuntimeError("W31 qualification domains did not both pass")
    if result.get("hira_candidate_evaluated") is not False:
        raise RuntimeError("W31 qualification candidate leakage detected")
    if result.get("a13_loaded") is not False:
        raise RuntimeError("W31 qualification unexpectedly loaded A13")
    if result.get("w30_rows_used") is not False or result.get("w29_rows_used") is not False:
        raise RuntimeError("W31 qualification prior-row leakage detected")
    if result.get("older_authority_rows_used") is not False:
        raise RuntimeError("W31 qualification older-row leakage detected")
    if result.get("exact_text_overlap") != []:
        raise RuntimeError("W31 qualification freshness firewall failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prior = (
        set(all_w28_text_atoms())
        | set(all_w29_text_atoms())
        | set(all_w30_text_atoms())
    )
    overlap = sorted(all_w31_text_atoms() & prior)
    if overlap:
        raise RuntimeError(f"W31 exact text overlaps exposed authorities: {overlap}")

    qualification = _validate_qualification(args.qualification_dir)
    t0_path = _validate_t0(args.t0_dir)
    projection = load_rescued_projection_checkpoint(t0_path)

    encoder = _load_a13()
    cache_model = build_hira_v0_semantic_core(
        encoder,
        t0_path,
        hira=HIRACore(d_model=256, dropout=0.0),
    )
    cache_model.eval()

    train_rows = generate_w31_partition("train")
    dev_rows = generate_w31_partition("dev")
    before = cache_model.state_encode_calls
    train_cache = compile_w31_cache(cache_model, train_rows, partition="train")
    dev_cache = compile_w31_cache(cache_model, dev_rows, partition="dev")
    encoded = cache_model.state_encode_calls - before
    if encoded != len(train_rows) + len(dev_rows):
        raise RuntimeError("W31 train/dev state encoding count changed")

    adapter_state, selected, history, training_meta = train_dual_adapter(
        train_cache,
        dev_cache,
        projection,
    )
    dual = build_dual_scorer(
        projection,
        adapter_state,
        freeze=True,
    )
    dual_train = evaluate_w31_cache(train_cache, dual)
    dual_dev = evaluate_w31_cache(dev_cache, dual)

    if dual.projection.weight.requires_grad:
        raise RuntimeError("W31 selected scorer unfroze T0")
    if dual.adapter_parameter_count != W31_ADAPTER_PARAMETER_COUNT:
        raise RuntimeError("W31 adapter parameter count changed")
    if dual.trainable_parameter_count != 0:
        raise RuntimeError("W31 selected inference scorer must be frozen")

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "adapter.pt"
    checkpoint = {
        "schema_version": W31_ADAPTER_CHECKPOINT_SCHEMA,
        "kind": "dual-semantic-adapter",
        "rank": W31_ADAPTER_RANK,
        "parameter_count": W31_ADAPTER_PARAMETER_COUNT,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "selected_dev_epoch": int(selected["epoch"]),
        "adapter_state_dict": adapter_state,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = file_sha256(checkpoint_path)

    receipt = {
        "schema_version": "r8-w31-dual-adapter-training-receipt-v1",
        "status": "PASS",
        "qualification_outcome": qualification["outcome"],
        "qualification_domains": qualification["domains"],
        "checkpoint_sha256": checkpoint_sha,
        "t0_checkpoint_sha256": W28_T0_CHECKPOINT_SHA256,
        "a13_model": A13_MODEL,
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "train_case_count": len(train_rows),
        "dev_case_count": len(dev_rows),
        "confirm_case_count_used": 0,
        "qualification_rows_used_for_training": False,
        "w30_rows_used": False,
        "w29_rows_used": False,
        "older_authority_rows_used": False,
        "projection_training_performed": False,
        "adapter_training_performed": True,
        "adapter_parameter_count": W31_ADAPTER_PARAMETER_COUNT,
        "adapter_rank": W31_ADAPTER_RANK,
        "seed": ADAPTER_SEED,
        "epochs": ADAPTER_EPOCHS,
        "batch_size": ADAPTER_BATCH_SIZE,
        "lr": ADAPTER_LR,
        "weight_decay": ADAPTER_WEIGHT_DECAY,
        "grad_clip": ADAPTER_GRAD_CLIP,
        "temperature": ADAPTER_TEMPERATURE,
        "anchor_coefficient": ANCHOR_COEFFICIENT,
        "anchor_margin_threshold": ANCHOR_MARGIN_THRESHOLD,
        "anchor_count": training_meta["anchor_count"],
        "anchor_candidate_count": training_meta["anchor_candidate_count"],
        "anchor_rate": training_meta["anchor_rate"],
        "selected_dev_epoch": int(selected["epoch"]),
        "selected_dev": selected["dev"],
        "baseline_dev": training_meta["baseline_dev"],
        "dual_dev": dual_dev["pooled"],
        "dual_train": dual_train["pooled"],
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
    print("W31_TRAIN=" + json.dumps({
        "checkpoint_sha256": checkpoint_sha,
        "selected_dev_epoch": int(selected["epoch"]),
        "baseline_dev": training_meta["baseline_dev"],
        "dual_dev": dual_dev["pooled"],
        "dual_train": dual_train["pooled"],
        "anchor_rate": training_meta["anchor_rate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
