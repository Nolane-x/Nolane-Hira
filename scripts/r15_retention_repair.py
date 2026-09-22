from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from nmd.hard_negative import balanced_top_indices, top_non_entailment_indices
from nmd.hira import HIRACore, count_parameters
from nmd.relation_cache import RelationCache
from nmd.retention import (
    RetentionConfig,
    cached_teacher_probabilities,
    train_retention_candidate,
)
from nmd.semantic import HFAutoSemanticEncoder


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

MNLI_DATASET = "nyu-mll/multi_nli"
MNLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"

R12_HEAD_SHA256 = "0ca95572399d15717e1069545439165e46083c58afea8707cbbadeca67ad3b86"
R12_MATCHED_ACCURACY = 0.5806666612625122
MATCHED_FLOOR = R12_MATCHED_ACCURACY - 0.02
R12_HARD_NON_ENTAILMENT = 0.339

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
    return sha256(",".join(map(str, indices)).encode()).hexdigest()


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
    states, masks, questions, labels = [], [], [], []
    for start in range(0, len(ds), batch_size):
        batch = ds[start:min(len(ds), start + batch_size)]
        premise = [str(x) for x in batch["premise"]]
        hypothesis = [str(x) for x in batch["hypothesis"]]
        y = torch.tensor(batch["label"], dtype=torch.long)
        s = encoder.encode_texts(premise)
        q = encoder.encode_texts(hypothesis)
        seg, seg_mask = segment_pool(s.token_embeddings, s.attention_mask, n_segments)
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
    ap.add_argument("--hard-per-label", type=int, default=6000)
    ap.add_argument("--replay-n", type=int, default=15000)
    ap.add_argument("--hard-val-per-label", type=int, default=500)
    ap.add_argument("--matched-val-n", type=int, default=1500)
    ap.add_argument("--kl-weights", default="0.25,0.5,1.0")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--train-batch-size", type=int, default=96)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--margin-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    kl_weights = [float(x) for x in args.kl_weights.split(",") if x.strip()]
    if kl_weights != [0.25, 0.5, 1.0]:
        raise SystemExit("R15 preregistered KL weights are exactly 0.25,0.5,1.0")
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
    if file_sha256(snapshot / "model.safetensors") != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(base, tokenizer, revision=A13_REVISION, max_length=128)

    with torch.inference_mode():
        option_embeddings = F.normalize(
            encoder.encode_texts(OPTION_TEXTS).pooled_embeddings, dim=-1
        )

    ds = load_dataset(MNLI_DATASET, revision=MNLI_REVISION)
    train_full = ds["train"].filter(lambda x: int(x["label"]) >= 0)
    matched_full = ds["validation_matched"].filter(lambda x: int(x["label"]) >= 0)
    mismatched_full = ds["validation_mismatched"].filter(lambda x: int(x["label"]) >= 0)

    hard_indices, hard_stats = balanced_top_indices(
        train_full["premise"],
        train_full["hypothesis"],
        train_full["label"],
        per_label=args.hard_per_label,
    )
    hard_val_indices, hard_val_stats = top_non_entailment_indices(
        mismatched_full["premise"],
        mismatched_full["hypothesis"],
        mismatched_full["label"],
        per_label=args.hard_val_per_label,
    )

    indexed_train = train_full.add_column("_source_index", list(range(len(train_full))))
    replay = indexed_train.shuffle(seed=args.seed).select(
        range(min(args.replay_n, len(indexed_train)))
    )
    replay_indices = [int(x) for x in replay["_source_index"]]
    replay = replay.remove_columns("_source_index")
    hard = train_full.select(hard_indices)
    hard_val = mismatched_full.select(hard_val_indices)
    matched = matched_full.shuffle(seed=args.seed + 1).select(
        range(min(args.matched_val_n, len(matched_full)))
    )

    common = {
        "a13_revision": A13_REVISION,
        "a13_weight_sha256": A13_WEIGHT_SHA256,
        "mnli_revision": MNLI_REVISION,
        "seed": args.seed,
        "segments": args.segments,
        "hans_used_for_selection": False,
        "breaking_nli_used_for_selection": False,
        "xnli_used": False,
        "massive_used": False,
        "banking77_used": False,
    }

    hard_cache = encode_dataset(
        hard, encoder, option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common, "role": "hard-train", "n": len(hard),
            "index_sha256": hash_indices(hard_indices),
        },
    )
    replay_cache = encode_dataset(
        replay, encoder, option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common, "role": "r12-replay", "n": len(replay),
            "index_sha256": hash_indices(replay_indices),
        },
    )
    matched_cache = encode_dataset(
        matched, encoder, option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={**common, "role": "matched-validation", "n": len(matched)},
    )
    hard_val_cache = encode_dataset(
        hard_val, encoder, option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common, "role": "hard-validation", "n": len(hard_val),
            "index_sha256": hash_indices(hard_val_indices),
        },
    )

    r12_state = torch.load(args.r12_head, map_location="cpu", weights_only=True)
    teacher = HIRACore(d_model=256, dropout=0.0)
    teacher.load_state_dict(r12_state, strict=True)
    teacher_probs = cached_teacher_probabilities(
        teacher, replay_cache, batch_size=args.train_batch_size
    )

    candidates = []
    selected = None
    selected_state = None
    fallback = None
    fallback_state = None

    for kl_weight in kl_weights:
        torch.manual_seed(args.seed)
        student = HIRACore(d_model=256, dropout=0.05)
        student.load_state_dict(r12_state, strict=True)
        config = RetentionConfig(
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            brier_weight=0.1,
            margin=args.margin,
            margin_weight=args.margin_weight,
            teacher_kl_weight=kl_weight,
            matched_accuracy_floor=MATCHED_FLOOR,
            seed=args.seed,
        )
        history, state, result = train_retention_candidate(
            student,
            hard_cache,
            replay_cache,
            teacher_probs,
            matched_cache,
            hard_val_cache,
            config=config,
        )
        matched_acc = float(result["selected"]["matched"]["accuracy"])
        hard_ne = float(result["selected"]["hard"]["non_entailment_accuracy"])
        row = {
            "teacher_kl_weight": kl_weight,
            "history": history,
            "baseline": result["baseline"],
            "selected": result["selected"],
            "selected_eligible": bool(result["selected_eligible"]),
            "hard_gain_vs_r12": hard_ne - R12_HARD_NON_ENTAILMENT,
        }
        candidates.append(row)

        fb_key = (matched_acc, hard_ne)
        if fallback is None or fb_key > fallback[0]:
            fallback = (fb_key, row)
            fallback_state = state

        if result["selected_eligible"]:
            key = (hard_ne, matched_acc)
            if selected is None or key > selected[0]:
                selected = (key, row)
                selected_state = state

    selected_eligible = selected is not None
    if selected is None:
        if fallback is None or fallback_state is None:
            raise RuntimeError("R15 produced no checkpoint")
        selected_row = fallback[1]
        selected_state = fallback_state
    else:
        selected_row = selected[1]

    head_path = args.out / "hira-head.pt"
    torch.save(selected_state, head_path)
    selected_sha = file_sha256(head_path)

    matched_acc = float(selected_row["selected"]["matched"]["accuracy"])
    hard_ne = float(selected_row["selected"]["hard"]["non_entailment_accuracy"])
    hard_gain = hard_ne - R12_HARD_NON_ENTAILMENT
    gates = {
        "matched_accuracy_floor": MATCHED_FLOOR,
        "matched_pass": matched_acc >= MATCHED_FLOOR,
        "hard_non_entailment_floor": 0.55,
        "hard_non_entailment_pass": hard_ne >= 0.55,
        "hard_gain_floor": 0.20,
        "hard_gain": hard_gain,
        "hard_gain_pass": hard_gain >= 0.20,
        "parameter_count_unchanged": count_parameters(HIRACore()) == 422_159,
        "selected_eligible": selected_eligible,
    }
    gates["primary_pass"] = all([
        gates["matched_pass"],
        gates["hard_non_entailment_pass"],
        gates["hard_gain_pass"],
        gates["parameter_count_unchanged"],
        gates["selected_eligible"],
    ])

    receipt = {
        "schema_version": "r15-retention-repair-v1",
        "status": "PASS",
        "evidence_scope": "MultiNLI-only replay+teacher-anchor tournament; no HANS/Breaking/XNLI/MASSIVE/Banking77 selection",
        "sources": {
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
            "mnli_revision": MNLI_REVISION,
            "r12_head_sha256": R12_HEAD_SHA256,
        },
        "data": {
            "hard_train_n": len(hard),
            "hard_train_index_sha256": hash_indices(hard_indices),
            "hard_overlap_stats": hard_stats,
            "replay_n": len(replay),
            "replay_index_sha256": hash_indices(replay_indices),
            "hard_validation_n": len(hard_val),
            "hard_validation_index_sha256": hash_indices(hard_val_indices),
            "hard_validation_overlap_stats": hard_val_stats,
            "matched_validation_n": len(matched),
        },
        "tournament": {
            "kl_weights": kl_weights,
            "epochs": args.epochs,
            "lr": args.lr,
            "hard_replay_batch_ratio": "1:1",
            "candidates": candidates,
        },
        "selected": selected_row,
        "selected_head_sha256": selected_sha,
        "gates": gates,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "primary_pass": gates["primary_pass"],
        "selected_eligible": selected_eligible,
        "teacher_kl_weight": selected_row["teacher_kl_weight"],
        "matched_accuracy": matched_acc,
        "hard_non_entailment_accuracy": hard_ne,
        "hard_gain": hard_gain,
        "selected_head_sha256": selected_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
