from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from nmd.hira import HIRACore, count_parameters
from nmd.relation_cache import RelationCache, evaluate_cached, train_cached
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


def segment_pool(tokens: torch.Tensor, mask: torch.Tensor, n_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, _, d = tokens.shape
    out = torch.zeros(n, n_segments, d, dtype=torch.float16)
    out_mask = torch.zeros(n, n_segments, dtype=torch.bool)
    for i in range(n):
        valid = tokens[i][mask[i].bool()].detach().cpu()
        if valid.shape[0] == 0:
            valid = tokens[i, :1].detach().cpu()
        pieces = torch.tensor_split(valid, min(n_segments, valid.shape[0]), dim=0)
        for j, piece in enumerate(pieces):
            if piece.shape[0] == 0:
                continue
            out[i, j] = piece.mean(0).half()
            out_mask[i, j] = True
    return out, out_mask


@torch.inference_mode()
def encode_dataset(ds, encoder: HFAutoSemanticEncoder, option_embeddings: torch.Tensor, *, batch_size: int, n_segments: int, metadata: dict) -> RelationCache:
    states = []
    state_masks = []
    questions = []
    labels = []
    for start in range(0, len(ds), batch_size):
        stop = min(len(ds), start + batch_size)
        batch = ds[start:stop]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-n", type=int, default=5000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--train-batch-size", type=int, default=64)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    weight = snapshot / "model.safetensors"
    if not weight.exists():
        raise RuntimeError("A13 safetensors file missing")
    actual_weight_hash = file_sha256(weight)
    if actual_weight_hash != A13_WEIGHT_SHA256:
        raise RuntimeError(
            f"A13 weight hash mismatch: {actual_weight_hash} != {A13_WEIGHT_SHA256}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base, tokenizer, revision=A13_REVISION, max_length=128
    )

    with torch.inference_mode():
        option_embeddings = encoder.encode_texts(OPTION_TEXTS).pooled_embeddings
        option_embeddings = torch.nn.functional.normalize(option_embeddings, dim=-1)

    ds = load_dataset(MNLI_DATASET, revision=MNLI_REVISION)
    train_ds = ds["train"].filter(lambda x: int(x["label"]) >= 0)
    val_ds = ds["validation_matched"].filter(lambda x: int(x["label"]) >= 0)
    train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.train_n, len(train_ds))))
    val_ds = val_ds.shuffle(seed=args.seed + 1).select(range(min(args.val_n, len(val_ds))))

    common = {
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "mnli_revision": MNLI_REVISION,
        "seed": args.seed,
        "segments": args.segments,
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
    train_cache.save(args.out / "train-cache.pt")
    val_cache.save(args.out / "val-cache.pt")

    torch.manual_seed(args.seed)
    hira = HIRACore(d_model=256, dropout=0.05)
    initial = evaluate_cached(hira, val_cache, batch_size=args.train_batch_size)
    history, best = train_cached(
        hira,
        train_cache,
        val_cache,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        seed=args.seed,
        brier_weight=0.1,
    )
    final = evaluate_cached(hira, val_cache, batch_size=args.train_batch_size)

    head_path = args.out / "hira-head.pt"
    torch.save(best, head_path)
    head_sha = file_sha256(head_path)
    semantic_gate = final["accuracy"] >= 0.50

    receipt = {
        "schema_version": "r11-mnli-pilot-v1",
        "status": "PASS",
        "semantic_gate_threshold": 0.50,
        "semantic_gate_pass": semantic_gate,
        "evidence_scope": "frozen A13 embeddings + trained HIRA head on MultiNLI; not Laya/Jev parity",
        "a13": {
            "model_id": A13_MODEL,
            "revision": A13_REVISION,
            "weight_sha256": actual_weight_hash,
            "frozen": True,
        },
        "dataset": {
            "id": MNLI_DATASET,
            "revision": MNLI_REVISION,
            "train_n": len(train_ds),
            "validation_n": len(val_ds),
            "seed": args.seed,
            "xnli_used": False,
            "massive_used": False,
        },
        "training": {
            "epochs": args.epochs,
            "encode_batch_size": args.encode_batch_size,
            "train_batch_size": args.train_batch_size,
            "segments": args.segments,
            "lr": args.lr,
            "hira_params": count_parameters(hira),
            "head_sha256": head_sha,
        },
        "initial_metrics": initial,
        "final_metrics": final,
        "history": history,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "semantic_gate_pass": semantic_gate,
        "initial_accuracy": initial["accuracy"],
        "final_accuracy": final["accuracy"],
        "final_brier": final["brier"],
        "final_ece": final["ece"],
        "head_sha256": head_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
