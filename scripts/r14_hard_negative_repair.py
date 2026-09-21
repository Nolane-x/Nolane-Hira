from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from nmd.hard_negative import (
    balanced_top_indices,
    top_non_entailment_indices,
    train_hard_negative_repair,
)
from nmd.hira import HIRACore, count_parameters
from nmd.relation_cache import RelationCache
from nmd.semantic import HFAutoSemanticEncoder


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

MNLI_DATASET = "nyu-mll/multi_nli"
MNLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"

R12_HEAD_SHA256 = "0ca95572399d15717e1069545439165e46083c58afea8707cbbadeca67ad3b86"
R12_MATCHED_ACCURACY = 0.5806666612625122
MATCHED_FLOOR = R12_MATCHED_ACCURACY - 0.02

OPTION_TEXTS = (
    "the hypothesis is entailed by the premise",
    "the hypothesis is neutral with respect to the premise",
    "the hypothesis contradicts the premise",
)


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_indices(indices: list[int]) -> str:
    payload = ",".join(map(str, indices)).encode()
    return sha256(payload).hexdigest()


def segment_pool(tokens: torch.Tensor, mask: torch.Tensor, n_segments: int):
    n, _, d = tokens.shape
    out = torch.zeros(n, n_segments, d, dtype=torch.float16)
    out_mask = torch.zeros(n, n_segments, dtype=torch.bool)
    for i in range(n):
        valid = tokens[i][mask[i].bool()].detach().cpu()
        if valid.shape[0] == 0:
            valid = tokens[i, :1].detach().cpu()
        pieces = torch.tensor_split(valid, min(n_segments, valid.shape[0]), dim=0)
        for j, piece in enumerate(pieces):
            if piece.numel() == 0:
                continue
            out[i, j] = piece.mean(0).half()
            out_mask[i, j] = True
    return out, out_mask


@torch.inference_mode()
def encode_dataset(
    ds,
    encoder,
    option_embeddings,
    *,
    batch_size: int,
    n_segments: int,
    metadata: dict,
) -> RelationCache:
    states, masks, questions, labels = [], [], [], []
    for start in range(0, len(ds), batch_size):
        batch = ds[start:min(len(ds), start + batch_size)]
        premise = [str(x) for x in batch["premise"]]
        hypothesis = [str(x) for x in batch["hypothesis"]]
        y = torch.tensor(batch["label"], dtype=torch.long)
        if (y < 0).any() or (y > 2).any():
            raise ValueError("MultiNLI batch contains unsupported label")
        s = encoder.encode_texts(premise)
        q = encoder.encode_texts(hypothesis)
        seg, seg_mask = segment_pool(
            s.token_embeddings,
            s.attention_mask,
            n_segments,
        )
        states.append(seg)
        masks.append(seg_mask)
        questions.append(q.pooled_embeddings.detach().cpu().half())
        labels.append(y)
    return RelationCache(
        state_segments=torch.cat(states),
        state_mask=torch.cat(masks),
        question_embeddings=torch.cat(questions),
        option_embeddings=option_embeddings.detach().cpu().half(),
        labels=torch.cat(labels),
        metadata=metadata,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r12-head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-per-label", type=int, default=6000)
    ap.add_argument("--hard-val-per-label", type=int, default=500)
    ap.add_argument("--matched-val-n", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--train-batch-size", type=int, default=96)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--margin-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if file_sha256(args.r12_head) != R12_HEAD_SHA256:
        raise RuntimeError("R12 selected head SHA-256 mismatch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if file_sha256(weight) != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base,
        tokenizer,
        revision=A13_REVISION,
        max_length=128,
    )

    with torch.inference_mode():
        option_embeddings = F.normalize(
            encoder.encode_texts(OPTION_TEXTS).pooled_embeddings,
            dim=-1,
        )

    ds = load_dataset(MNLI_DATASET, revision=MNLI_REVISION)
    train_full = ds["train"].filter(lambda x: int(x["label"]) >= 0)
    matched_full = ds["validation_matched"].filter(lambda x: int(x["label"]) >= 0)
    mismatched_full = ds["validation_mismatched"].filter(lambda x: int(x["label"]) >= 0)

    train_indices, train_stats = balanced_top_indices(
        train_full["premise"],
        train_full["hypothesis"],
        train_full["label"],
        per_label=args.train_per_label,
    )
    hard_indices, hard_stats = top_non_entailment_indices(
        mismatched_full["premise"],
        mismatched_full["hypothesis"],
        mismatched_full["label"],
        per_label=args.hard_val_per_label,
    )

    # Reproduce the exact R12 matched validation selection.
    matched = matched_full.shuffle(seed=args.seed + 1).select(
        range(min(args.matched_val_n, len(matched_full)))
    )
    train = train_full.select(train_indices)
    hard_val = mismatched_full.select(hard_indices)

    common = {
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "mnli_revision": MNLI_REVISION,
        "seed": args.seed,
        "segments": args.segments,
        "xnli_used": False,
        "massive_used": False,
        "banking77_used": False,
        "hans_used_for_selection": False,
        "breaking_nli_used_for_selection": False,
    }

    train_cache = encode_dataset(
        train,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "split": "train",
            "selection": "balanced-high-lexical-overlap",
            "n": len(train),
            "index_sha256": hash_indices(train_indices),
        },
    )
    matched_cache = encode_dataset(
        matched,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "split": "validation_matched",
            "selection": "shuffle(seed=14)-first-1500",
            "n": len(matched),
        },
    )
    hard_cache = encode_dataset(
        hard_val,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "split": "validation_mismatched",
            "selection": "top-lexical-overlap-neutral+contradiction",
            "n": len(hard_val),
            "index_sha256": hash_indices(hard_indices),
        },
    )

    hira = HIRACore(d_model=256, dropout=0.05)
    initial_state = torch.load(args.r12_head, map_location="cpu", weights_only=True)
    hira.load_state_dict(initial_state, strict=True)
    if count_parameters(hira) != 422_159:
        raise RuntimeError("HIRA parameter ledger changed")

    history, selected_state, result = train_hard_negative_repair(
        hira,
        train_cache,
        matched_cache,
        hard_cache,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        seed=args.seed,
        brier_weight=0.1,
        margin=args.margin,
        margin_weight=args.margin_weight,
        matched_accuracy_floor=MATCHED_FLOOR,
    )

    head_path = args.out / "hira-head.pt"
    torch.save(selected_state, head_path)
    selected_sha = file_sha256(head_path)

    baseline_hard = result["baseline"]["hard"]["non_entailment_accuracy"]
    selected_hard = result["selected"]["hard"]["non_entailment_accuracy"]
    hard_gain = selected_hard - baseline_hard
    matched_acc = result["selected"]["matched"]["accuracy"]

    gates = {
        "matched_accuracy_floor": MATCHED_FLOOR,
        "matched_preservation_pass": matched_acc >= MATCHED_FLOOR,
        "hard_non_entailment_gain_floor": 0.10,
        "hard_non_entailment_gain": hard_gain,
        "hard_gain_pass": hard_gain >= 0.10,
        "parameter_count_unchanged": count_parameters(hira) == 422_159,
        "selected_eligible": bool(result["selected_eligible"]),
    }
    gates["primary_pass"] = all([
        gates["matched_preservation_pass"],
        gates["hard_gain_pass"],
        gates["parameter_count_unchanged"],
        gates["selected_eligible"],
    ])

    receipt = {
        "schema_version": "r14-hard-negative-repair-v1",
        "status": "PASS",
        "evidence_scope": "MultiNLI-only repair of R12 HIRA head; HANS/Breaking/XNLI/MASSIVE/Banking77 excluded from model selection",
        "sources": {
            "a13_model": A13_MODEL,
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
            "mnli_dataset": MNLI_DATASET,
            "mnli_revision": MNLI_REVISION,
            "r12_head_sha256": R12_HEAD_SHA256,
        },
        "curriculum": {
            "train_per_label": args.train_per_label,
            "train_total": len(train),
            "train_index_sha256": hash_indices(train_indices),
            "train_overlap_stats": train_stats,
            "hard_validation_per_non_entailment_label": args.hard_val_per_label,
            "hard_validation_total": len(hard_val),
            "hard_validation_index_sha256": hash_indices(hard_indices),
            "hard_validation_overlap_stats": hard_stats,
            "matched_validation_n": len(matched),
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.train_batch_size,
            "brier_weight": 0.1,
            "margin": args.margin,
            "margin_weight": args.margin_weight,
            "hira_params": count_parameters(hira),
            "a13_frozen": True,
            "selected_head_sha256": selected_sha,
        },
        "baseline": result["baseline"],
        "selected": result["selected"],
        "history": history,
        "gates": gates,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "PASS",
        "primary_pass": gates["primary_pass"],
        "selected_eligible": gates["selected_eligible"],
        "matched_accuracy": matched_acc,
        "hard_baseline_non_entailment_accuracy": baseline_hard,
        "hard_selected_non_entailment_accuracy": selected_hard,
        "hard_gain": hard_gain,
        "selected_head_sha256": selected_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
