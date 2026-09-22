from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from nmd.hard_negative import evaluate_repair_slice
from nmd.hira import HIRACore, count_parameters
from nmd.interpolation import (
    balanced_near_structural_non_entailment_indices,
    interpolate_state_dicts,
)
from nmd.relation_cache import RelationCache
from nmd.semantic import HFAutoSemanticEncoder


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

MNLI_DATASET = "nyu-mll/multi_nli"
MNLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"

R15_HEAD_SHA256 = "007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f"
R16_HEAD_SHA256 = "dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75"
MATCHED_FLOOR = 0.5606666612625122
ALPHAS = (0.00, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
NEAR_THRESHOLD = 0.80
MIN_HYPOTHESIS_TOKENS = 3
SUPPORT_PER_LABEL = 100
NEAR_MAX_PER_LABEL = 500

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
    ap.add_argument("--r15-head", type=Path, required=True)
    ap.add_argument("--r16-head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alphas", default="0,0.025,0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--matched-val-n", type=int, default=1500)
    ap.add_argument("--near-max-per-label", type=int, default=500)
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    alphas = tuple(float(x) for x in args.alphas.split(",") if x.strip())
    if alphas != ALPHAS:
        raise SystemExit(f"R17 preregistered alpha grid is exactly {ALPHAS}")
    if args.near_max_per_label != NEAR_MAX_PER_LABEL:
        raise SystemExit(f"R17 near max per label is exactly {NEAR_MAX_PER_LABEL}")
    if file_sha256(args.r15_head) != R15_HEAD_SHA256:
        raise RuntimeError("R15 selected head SHA-256 mismatch")
    if file_sha256(args.r16_head) != R16_HEAD_SHA256:
        raise RuntimeError("R16 failure-analysis head SHA-256 mismatch")

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
    matched_full = ds["validation_matched"].filter(lambda x: int(x["label"]) >= 0)
    mismatched_full = ds["validation_mismatched"].filter(lambda x: int(x["label"]) >= 0)

    near_indices, near_stats = balanced_near_structural_non_entailment_indices(
        mismatched_full["premise"],
        mismatched_full["hypothesis"],
        mismatched_full["label"],
        threshold=NEAR_THRESHOLD,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        max_per_label=args.near_max_per_label,
    )
    available = near_stats["available_by_label"]
    support_valid = (
        int(available.get("1", 0)) >= SUPPORT_PER_LABEL
        and int(available.get("2", 0)) >= SUPPORT_PER_LABEL
    )

    indexed_matched = matched_full.add_column("_source_index", list(range(len(matched_full))))
    matched = indexed_matched.shuffle(seed=args.seed + 1).select(
        range(min(args.matched_val_n, len(indexed_matched)))
    )
    matched_indices = [int(x) for x in matched["_source_index"]]
    matched = matched.remove_columns("_source_index")
    near = mismatched_full.select(near_indices)

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
    matched_cache = encode_dataset(
        matched,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "matched-validation",
            "n": len(matched),
            "index_sha256": hash_indices(matched_indices),
        },
    )
    near_cache = encode_dataset(
        near,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "near-structural-validation",
            "n": len(near),
            "index_sha256": hash_indices(near_indices),
            "near_threshold": NEAR_THRESHOLD,
            "min_hypothesis_tokens": MIN_HYPOTHESIS_TOKENS,
        },
    )

    r15_state = torch.load(args.r15_head, map_location="cpu", weights_only=True)
    r16_state = torch.load(args.r16_head, map_location="cpu", weights_only=True)

    candidates = []
    eligible = []
    baseline_near = None
    selected_state = None

    for alpha in alphas:
        state = interpolate_state_dicts(r15_state, r16_state, alpha)
        model = HIRACore(d_model=256, dropout=0.0)
        model.load_state_dict(state, strict=True)
        matched_metrics = evaluate_repair_slice(
            model, matched_cache, batch_size=args.eval_batch_size
        )
        near_metrics = evaluate_repair_slice(
            model, near_cache, batch_size=args.eval_batch_size
        )
        row = {
            "alpha": alpha,
            "matched": matched_metrics,
            "near_structural": near_metrics,
            "eligible": bool(matched_metrics["accuracy"] >= MATCHED_FLOOR),
        }
        candidates.append(row)
        if alpha == 0.0:
            baseline_near = float(near_metrics["non_entailment_accuracy"])
        if row["eligible"]:
            eligible.append((row, state))

    if baseline_near is None:
        raise RuntimeError("R17 alpha=0 baseline missing")

    selected_eligible = bool(eligible)
    if eligible:
        selected_row, selected_state = max(
            eligible,
            key=lambda pair: (
                float(pair[0]["near_structural"]["non_entailment_accuracy"]),
                float(pair[0]["matched"]["accuracy"]),
                -float(pair[0]["alpha"]),
            ),
        )
    else:
        selected_row = max(
            candidates,
            key=lambda row: (
                float(row["matched"]["accuracy"]),
                float(row["near_structural"]["non_entailment_accuracy"]),
                -float(row["alpha"]),
            ),
        )
        selected_state = interpolate_state_dicts(
            r15_state, r16_state, float(selected_row["alpha"])
        )

    head_path = args.out / "hira-head.pt"
    torch.save(selected_state, head_path)
    selected_sha = file_sha256(head_path)

    matched_acc = float(selected_row["matched"]["accuracy"])
    near_acc = float(selected_row["near_structural"]["non_entailment_accuracy"])
    near_gain = near_acc - baseline_near
    gates = {
        "support_required_per_label": SUPPORT_PER_LABEL,
        "support_valid": support_valid,
        "matched_accuracy_floor": MATCHED_FLOOR,
        "matched_pass": matched_acc >= MATCHED_FLOOR,
        "near_structural_floor": 0.60,
        "near_structural_pass": near_acc >= 0.60,
        "near_structural_gain_floor": 0.05,
        "near_structural_gain": near_gain,
        "near_structural_gain_pass": near_gain >= 0.05,
        "parameter_count_unchanged": count_parameters(HIRACore()) == 422_159,
        "selected_eligible": selected_eligible,
    }
    gates["primary_pass"] = all(
        [
            gates["support_valid"],
            gates["matched_pass"],
            gates["near_structural_pass"],
            gates["near_structural_gain_pass"],
            gates["parameter_count_unchanged"],
            gates["selected_eligible"],
        ]
    )

    receipt = {
        "schema_version": "r17-weight-interpolation-v1",
        "status": "PASS",
        "evidence_scope": (
            "MultiNLI-only frozen endpoint interpolation; "
            "no HANS/Breaking/XNLI/MASSIVE/Banking77 model selection"
        ),
        "sources": {
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
            "mnli_revision": MNLI_REVISION,
            "r15_head_sha256": R15_HEAD_SHA256,
            "r16_failure_analysis_head_sha256": R16_HEAD_SHA256,
        },
        "protocol": {
            "alphas": list(alphas),
            "matched_accuracy_floor": MATCHED_FLOOR,
            "near_threshold": NEAR_THRESHOLD,
            "min_hypothesis_tokens": MIN_HYPOTHESIS_TOKENS,
            "support_required_per_label": SUPPORT_PER_LABEL,
            "near_max_per_label": NEAR_MAX_PER_LABEL,
            "selection_order": [
                "eligible matched accuracy",
                "maximize balanced near-structural non-entailment accuracy",
                "tie-break by matched accuracy",
                "then prefer smaller alpha",
            ],
        },
        "data": {
            "near_validation_n": len(near),
            "near_validation_index_sha256": hash_indices(near_indices),
            "near_validation_stats": near_stats,
            "matched_validation_n": len(matched),
            "matched_validation_index_sha256": hash_indices(matched_indices),
        },
        "candidates": candidates,
        "baseline_alpha0_near_structural_non_entailment_accuracy": baseline_near,
        "selected": selected_row,
        "selected_head_sha256": selected_sha,
        "gates": gates,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "primary_pass": gates["primary_pass"],
                "support_valid": support_valid,
                "selected_eligible": selected_eligible,
                "alpha": selected_row["alpha"],
                "matched_accuracy": matched_acc,
                "near_structural_baseline": baseline_near,
                "near_structural_accuracy": near_acc,
                "near_structural_gain": near_gain,
                "selected_head_sha256": selected_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
