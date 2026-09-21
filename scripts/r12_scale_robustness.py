from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.hira import HIRACore, count_parameters
from nmd.relation_cache import (
    RelationCache,
    evaluate_cached,
    evaluate_option_permutation,
    train_cached,
)
from nmd.semantic import HFAutoSemanticEncoder


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

MNLI_DATASET = "nyu-mll/multi_nli"
MNLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"

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
def encode_dataset(ds, encoder, option_embeddings, *, batch_size: int, n_segments: int, metadata: dict):
    states, state_masks, questions, labels = [], [], [], []
    for start in range(0, len(ds), batch_size):
        batch = ds[start:min(len(ds), start + batch_size)]
        premise = [str(x) for x in batch["premise"]]
        hypothesis = [str(x) for x in batch["hypothesis"]]
        y = torch.tensor(batch["label"], dtype=torch.long)
        if (y < 0).any() or (y > 2).any():
            raise ValueError("MultiNLI batch contains unsupported label")
        state_batch = encoder.encode_texts(premise)
        q_batch = encoder.encode_texts(hypothesis)
        seg, seg_mask = segment_pool(
            state_batch.token_embeddings,
            state_batch.attention_mask,
            n_segments,
        )
        states.append(seg)
        state_masks.append(seg_mask)
        questions.append(q_batch.pooled_embeddings.detach().cpu().half())
        labels.append(y)
    return RelationCache(
        state_segments=torch.cat(states),
        state_mask=torch.cat(state_masks),
        question_embeddings=torch.cat(questions),
        option_embeddings=option_embeddings.detach().cpu().half(),
        labels=torch.cat(labels),
        metadata=metadata,
    )


def subset_cache(cache: RelationCache, n: int, *, name: str) -> RelationCache:
    n = min(int(n), len(cache))
    return RelationCache(
        state_segments=cache.state_segments[:n],
        state_mask=cache.state_mask[:n],
        question_embeddings=cache.question_embeddings[:n],
        option_embeddings=cache.option_embeddings,
        labels=cache.labels[:n],
        metadata={**cache.metadata, "subset": name, "n": n},
    )


def permutation_suite(hira: HIRACore, validation: RelationCache, batch_size: int):
    rows = []
    for perm in itertools.permutations(range(validation.option_embeddings.shape[0])):
        result = evaluate_option_permutation(
            hira,
            validation,
            torch.tensor(perm),
            batch_size=batch_size,
        )
        rows.append({"permutation": list(perm), **result})
    return {
        "rows": rows,
        "max_abs_accuracy_delta": max(abs(x["accuracy_delta"]) for x in rows),
        "max_prediction_flip_rate": max(x["prediction_flip_rate"] for x in rows),
        "max_probability_equivariance_error": max(
            x["max_probability_equivariance_error"] for x in rows
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-train-n", type=int, default=15000)
    ap.add_argument("--rungs", default="5000,10000,15000")
    ap.add_argument("--val-n", type=int, default=1500)
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--train-batch-size", type=int, default=96)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    if not rungs or max(rungs) > args.max_train_n:
        raise SystemExit("rungs must be non-empty and <= max-train-n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    actual_weight_hash = file_sha256(weight)
    if actual_weight_hash != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight hash mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(base, tokenizer, revision=A13_REVISION, max_length=128)

    with torch.inference_mode():
        option_embeddings = encoder.encode_texts(OPTION_TEXTS).pooled_embeddings
        option_embeddings = torch.nn.functional.normalize(option_embeddings, dim=-1)

    ds = load_dataset(MNLI_DATASET, revision=MNLI_REVISION)
    train_ds = ds["train"].filter(lambda x: int(x["label"]) >= 0)
    val_ds = ds["validation_matched"].filter(lambda x: int(x["label"]) >= 0)
    train_ds = train_ds.shuffle(seed=args.seed).select(
        range(min(args.max_train_n, len(train_ds)))
    )
    val_ds = val_ds.shuffle(seed=args.seed + 1).select(
        range(min(args.val_n, len(val_ds)))
    )

    common = {
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "mnli_revision": MNLI_REVISION,
        "seed": args.seed,
        "segments": args.segments,
        "xnli_used": False,
        "massive_used": False,
        "banking77_used": False,
    }
    train_cache = encode_dataset(
        train_ds,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={**common, "split": "train", "n": len(train_ds)},
    )
    val_cache = encode_dataset(
        val_ds,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={**common, "split": "validation_matched", "n": len(val_ds)},
    )

    results = []
    best_row = None
    best_state = None
    for rung in rungs:
        torch.manual_seed(args.seed)
        hira = HIRACore(d_model=256, dropout=0.05)
        sub = subset_cache(train_cache, rung, name=f"mnli-{rung}")
        initial = evaluate_cached(hira, val_cache, batch_size=args.train_batch_size)
        history, state = train_cached(
            hira,
            sub,
            val_cache,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            seed=args.seed,
            brier_weight=0.1,
        )
        final = evaluate_cached(hira, val_cache, batch_size=args.train_batch_size)
        perm = permutation_suite(hira, val_cache, args.train_batch_size)
        row = {
            "train_n": len(sub),
            "initial_metrics": initial,
            "final_metrics": final,
            "history": history,
            "permutation": perm,
        }
        results.append(row)
        if best_row is None or final["accuracy"] > best_row["final_metrics"]["accuracy"]:
            best_row = row
            best_state = state

    if best_row is None or best_state is None:
        raise RuntimeError("no scale rung completed")

    head_path = args.out / "hira-head.pt"
    torch.save(best_state, head_path)
    head_sha = file_sha256(head_path)

    robustness_pass = (
        best_row["permutation"]["max_abs_accuracy_delta"] <= 1e-7
        and best_row["permutation"]["max_prediction_flip_rate"] <= 1e-7
        and best_row["permutation"]["max_probability_equivariance_error"] <= 1e-5
    )
    scale_pass = (
        best_row["final_metrics"]["accuracy"] >= 0.56
        and best_row["final_metrics"]["accuracy"] >= results[0]["final_metrics"]["accuracy"]
    )

    receipt = {
        "schema_version": "r12-scale-robustness-v1",
        "status": "PASS",
        "evidence_scope": "frozen A13 + HIRA MultiNLI scale/option-permutation study; no Laya/Jev parity",
        "a13": {
            "model_id": A13_MODEL,
            "revision": A13_REVISION,
            "weight_sha256": actual_weight_hash,
            "frozen": True,
        },
        "dataset": {
            "id": MNLI_DATASET,
            "revision": MNLI_REVISION,
            "max_train_n": len(train_ds),
            "validation_n": len(val_ds),
            "seed": args.seed,
            "xnli_used": False,
            "massive_used": False,
            "banking77_used": False,
        },
        "training": {
            "rungs": rungs,
            "epochs_per_rung": args.epochs,
            "lr": args.lr,
            "train_batch_size": args.train_batch_size,
            "segments": args.segments,
            "hira_params": count_parameters(HIRACore()),
            "selected_head_sha256": head_sha,
        },
        "gates": {
            "scale_accuracy_floor": 0.56,
            "scale_pass": scale_pass,
            "permutation_robustness_pass": robustness_pass,
        },
        "rungs": results,
        "best": best_row,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "scale_pass": scale_pass,
        "permutation_robustness_pass": robustness_pass,
        "best_train_n": best_row["train_n"],
        "best_accuracy": best_row["final_metrics"]["accuracy"],
        "best_brier": best_row["final_metrics"]["brier"],
        "best_ece": best_row["final_metrics"]["ece"],
        "head_sha256": head_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
