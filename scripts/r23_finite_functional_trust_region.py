from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from nmd.block_interpolation import atomic_group_for_key
from nmd.delta_surgery import (
    balanced_ranked_structural_window_indices,
    flatten_relation_named_tensors,
    relation_delta,
    relation_layout,
)
from nmd.functional_damage import (
    finite_sample_relative_accuracy_pass,
    teacher_kl_loss,
    weighted_batch_gradient_row,
)
from nmd.functional_trust import functional_trust_metrics, trust_safe
from nmd.hard_negative import anti_entailment_margin_loss, evaluate_repair_slice
from nmd.hira import HIRACore, count_parameters
from nmd.relation_cache import RelationCache, forward_cached, minibatches
from nmd.semantic import HFAutoSemanticEncoder
from nmd.structural_tangent import (
    functional_null_structural_tangent,
    normalize_to_reference_l2,
)
from nmd.subspace_projection import (
    apply_relation_update,
    build_retention_subspace,
    non_relation_bit_identical,
    projection_metrics,
    project_onto_retention_subspace,
    rotate_delta_away_from_retention,
)


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

MNLI_DATASET = "nyu-mll/multi_nli"
MNLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"

R15_HEAD_SHA256 = "007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f"
R16_HEAD_SHA256 = "dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75"

R19_STRUCTURAL_PER_LABEL = 2000
R19_RETENTION_N = 6000
R19_RETENTION_SEED = 13

R20_STRUCTURAL_PER_LABEL = 2000
R20_RETENTION_N = 6000
R20_RETENTION_SEED = 17

R21_STRUCTURAL_PER_LABEL = 2000
R21_RETENTION_N = 6000
R21_RETENTION_SEED = 23

R22_STRUCTURAL_PER_LABEL = 2000
R22_RETENTION_N = 6000
R22_RETENTION_SEED = 29

R23_STRUCTURAL_PER_LABEL = 2000
R23_RETENTION_N = 6000
R23_RETENTION_SEED = 31

STRUCTURAL_VAL_PER_LABEL = 250
R18_STRUCTURAL_VAL_START = 0
R19_STRUCTURAL_VAL_START = 250
R20_STRUCTURAL_VAL_START = 500
R21_STRUCTURAL_VAL_START = 750
R22_STRUCTURAL_VAL_START = 1000
R23_STRUCTURAL_VAL_START = 1250

MATCHED_SHUFFLE_SEED = 14
R23_MATCHED_START = 7500
MATCHED_VAL_N = 1500

MIN_HYPOTHESIS_TOKENS = 3
GRAD_BATCH_SIZE = 96

PROBE_BETA = 0.25
DAMAGE_RANK = 32
ETAS = (
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
    2.0,
    4.0,
)

PROBE_KL_FLOOR = 1e-4
MIN_FUNCTIONAL_RANK = 32
RELATIVE_EIGENVALUE_FLOOR = 1e-10
ORTHONORMAL_TOLERANCE = 1e-7
MAX_NULL_LEAKAGE_ENERGY = 1e-8
TRUST_MEAN_KL_CEILING = 1e-4

BASELINE_MATCHED_FLOOR = 0.53
MATCHED_RELATIVE_TOLERANCE = 0.0
MIN_MATCHED_LABEL_COUNT = 400
STRUCTURAL_FLOOR = 0.60
STRUCTURAL_GAIN_FLOOR = 0.02

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
        s = encoder.encode_texts(premise)
        q = encoder.encode_texts(hypothesis)
        seg, seg_mask = segment_pool(
            s.token_embeddings, s.attention_mask, n_segments
        )
        states.append(seg)
        masks.append(seg_mask)
        questions.append(q.pooled_embeddings.detach().cpu().half())
        labels.append(y)
    cache = RelationCache(
        state_segments=torch.cat(states),
        state_mask=torch.cat(masks),
        question_embeddings=torch.cat(questions),
        option_embeddings=option_embeddings.detach().cpu().half(),
        labels=torch.cat(labels),
        metadata=metadata,
    )
    cache.validate()
    return cache


def relation_parameter_map(model: HIRACore) -> dict[str, torch.nn.Parameter]:
    result: dict[str, torch.nn.Parameter] = {}
    for name, parameter in model.named_parameters():
        is_relation = atomic_group_for_key(name) == "relation"
        parameter.requires_grad_(is_relation)
        if is_relation:
            result[name] = parameter
    expected = {span.key for span in relation_layout(model.state_dict())}
    if set(result) != expected:
        raise RuntimeError(
            f"relation parameter/state mismatch: params={sorted(result)}, "
            f"state={sorted(expected)}"
        )
    return result


def collect_structural_gradient(
    model: HIRACore,
    cache: RelationCache,
    *,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    params = relation_parameter_map(model)
    sums = {
        name: torch.zeros_like(param, dtype=torch.float64, device="cpu")
        for name, param in params.items()
    }
    seen = 0
    loss_weighted = 0.0
    model.eval()
    for idx in minibatches(
        len(cache), batch_size, seed=0, epoch=0, shuffle=False
    ):
        model.zero_grad(set_to_none=True)
        out = forward_cached(model, cache, idx)
        labels = cache.labels[idx].long()
        ce = F.cross_entropy(out.logits, labels)
        margin = anti_entailment_margin_loss(out.logits, labels, margin=0.5)
        loss = ce + 0.5 * margin
        loss.backward()
        n = int(len(idx))
        seen += n
        loss_weighted += float(loss.detach().cpu()) * n
        for name, param in params.items():
            if param.grad is None:
                raise RuntimeError(f"missing structural gradient for {name}")
            sums[name].add_(
                param.grad.detach().cpu().to(torch.float64),
                alpha=float(n),
            )
    if seen != len(cache):
        raise RuntimeError("structural gradient pass incomplete")
    for name in sums:
        sums[name].div_(float(seen))
    return sums, {
        "n": float(seen),
        "mean_loss": loss_weighted / max(1, seen),
        "mode": "structural_mean_gradient",
    }


@torch.inference_mode()
def frozen_teacher_probabilities(
    model: HIRACore,
    cache: RelationCache,
    *,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for idx in minibatches(
        len(cache), batch_size, seed=0, epoch=0, shuffle=False
    ):
        out = forward_cached(model, cache, idx)
        chunks.append(torch.softmax(out.logits, dim=-1).detach().cpu())
    result = torch.cat(chunks, dim=0)
    if result.shape != (len(cache), 3):
        raise RuntimeError("teacher probability shape mismatch")
    if not torch.isfinite(result).all():
        raise RuntimeError("teacher probabilities are non-finite")
    return result


@torch.inference_mode()
def cached_logits(
    model: HIRACore,
    cache: RelationCache,
    *,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for idx in minibatches(
        len(cache), batch_size, seed=0, epoch=0, shuffle=False
    ):
        out = forward_cached(model, cache, idx)
        chunks.append(out.logits.detach().cpu())
    result = torch.cat(chunks, dim=0)
    if result.shape != (
        len(cache),
        cache.option_embeddings.shape[0],
    ):
        raise RuntimeError("cached logits shape mismatch")
    return result


def collect_functional_kl_gradient_rows(
    probe_model: HIRACore,
    cache: RelationCache,
    teacher_probabilities: torch.Tensor,
    *,
    batch_size: int,
    layout,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    params = relation_parameter_map(probe_model)
    rows: list[torch.Tensor] = []
    seen = 0
    kl_weighted = 0.0
    total_n = len(cache)
    all_rows_finite = True
    probe_model.eval()

    for idx in minibatches(
        total_n, batch_size, seed=0, epoch=0, shuffle=False
    ):
        probe_model.zero_grad(set_to_none=True)
        out = forward_cached(probe_model, cache, idx)
        teacher = teacher_probabilities[idx].to(out.logits.dtype)
        loss = teacher_kl_loss(out.logits, teacher)
        loss.backward()

        n = int(len(idx))
        seen += n
        kl_weighted += float(loss.detach().cpu()) * n

        grad_map: dict[str, torch.Tensor] = {}
        for name, param in params.items():
            if param.grad is None:
                raise RuntimeError(f"missing functional KL gradient for {name}")
            grad_map[name] = param.grad.detach().cpu()
        flat = flatten_relation_named_tensors(grad_map, layout=layout)
        row = weighted_batch_gradient_row(
            flat,
            batch_n=n,
            total_n=total_n,
        )
        finite = bool(torch.isfinite(row).all())
        all_rows_finite = all_rows_finite and finite
        rows.append(row)

    if seen != total_n:
        raise RuntimeError("functional KL gradient pass incomplete")
    matrix = torch.stack(rows).to(torch.float64)
    return matrix, {
        "n": float(seen),
        "batch_count": float(matrix.shape[0]),
        "mean_teacher_kl": kl_weighted / max(1, seen),
        "all_rows_finite": all_rows_finite,
        "mode": "sqrt_sample_weighted_batch_mean_teacher_KL_gradient_rows",
    }


def evaluate_state(
    state: dict[str, torch.Tensor],
    matched_cache: RelationCache,
    structural_cache: RelationCache,
    *,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    model = HIRACore(d_model=256, dropout=0.0)
    model.load_state_dict(state, strict=True)
    model.eval()
    return (
        evaluate_repair_slice(model, matched_cache, batch_size=batch_size),
        evaluate_repair_slice(model, structural_cache, batch_size=batch_size),
    )


def random_exclusion_sample(
    train_full,
    *,
    seed: int,
    n: int,
    excluded: set[int],
) -> list[int]:
    indexed = train_full.add_column(
        "_source_index", list(range(len(train_full)))
    )
    shuffled = indexed.shuffle(seed=seed)
    result: list[int] = []
    for raw in shuffled["_source_index"]:
        i = int(raw)
        if i in excluded:
            continue
        result.append(i)
        if len(result) == n:
            break
    if len(result) != n:
        raise RuntimeError(f"unable to sample exactly {n} examples")
    return result


def reconstruct_prior_train(train_full) -> dict[str, list[int]]:
    r19_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"],
        train_full["hypothesis"],
        train_full["label"],
        start_per_label=0,
        count_per_label=R19_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
    )
    r19_retention = random_exclusion_sample(
        train_full,
        seed=R19_RETENTION_SEED,
        n=R19_RETENTION_N,
        excluded=set(r19_structural),
    )
    r19_union = set(r19_structural) | set(r19_retention)

    r20_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"],
        train_full["hypothesis"],
        train_full["label"],
        start_per_label=0,
        count_per_label=R20_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        exclude_indices=sorted(r19_union),
    )
    r20_retention = random_exclusion_sample(
        train_full,
        seed=R20_RETENTION_SEED,
        n=R20_RETENTION_N,
        excluded=r19_union | set(r20_structural),
    )
    r20_union = r19_union | set(r20_structural) | set(r20_retention)

    r21_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"],
        train_full["hypothesis"],
        train_full["label"],
        start_per_label=0,
        count_per_label=R21_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        exclude_indices=sorted(r20_union),
    )
    r21_retention = random_exclusion_sample(
        train_full,
        seed=R21_RETENTION_SEED,
        n=R21_RETENTION_N,
        excluded=r20_union | set(r21_structural),
    )
    r21_union = r20_union | set(r21_structural) | set(r21_retention)

    r22_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"],
        train_full["hypothesis"],
        train_full["label"],
        start_per_label=0,
        count_per_label=R22_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        exclude_indices=sorted(r21_union),
    )
    r22_retention = random_exclusion_sample(
        train_full,
        seed=R22_RETENTION_SEED,
        n=R22_RETENTION_N,
        excluded=r21_union | set(r22_structural),
    )

    result = {
        "r19_structural": r19_structural,
        "r19_retention": r19_retention,
        "r20_structural": r20_structural,
        "r20_retention": r20_retention,
        "r21_structural": r21_structural,
        "r21_retention": r21_retention,
        "r22_structural": r22_structural,
        "r22_retention": r22_retention,
    }
    union_size = len(set().union(*(set(v) for v in result.values())))
    if union_size != 40000:
        raise RuntimeError(
            f"prior scored-train authorities are not exactly disjoint: {union_size}"
        )
    return result

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r15-head", type=Path, required=True)
    ap.add_argument("--r16-head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encode-batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

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

    snapshot = Path(snapshot_download(
        repo_id=A13_MODEL,
        revision=A13_REVISION,
    ))
    if file_sha256(snapshot / "model.safetensors") != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base_encoder = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    base_encoder.eval()
    for p in base_encoder.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base_encoder,
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
    matched_full = ds["validation_matched"].filter(
        lambda x: int(x["label"]) >= 0
    )
    mismatched_full = ds["validation_mismatched"].filter(
        lambda x: int(x["label"]) >= 0
    )

    prior = reconstruct_prior_train(train_full)
    prior_train = set().union(*(set(v) for v in prior.values()))

    structural_train_indices, structural_train_stats = (
        balanced_ranked_structural_window_indices(
            train_full["premise"],
            train_full["hypothesis"],
            train_full["label"],
            start_per_label=0,
            count_per_label=R23_STRUCTURAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
            exclude_indices=sorted(prior_train),
        )
    )
    if len(structural_train_indices) != 4000:
        raise RuntimeError("R23 structural train must be exactly 4000")

    structural_train_set = set(structural_train_indices)
    if not structural_train_set.isdisjoint(prior_train):
        raise RuntimeError("R23 structural train overlaps prior scored train")

    retention_indices = random_exclusion_sample(
        train_full,
        seed=R23_RETENTION_SEED,
        n=R23_RETENTION_N,
        excluded=prior_train | structural_train_set,
    )
    retention_set = set(retention_indices)
    train_sets_disjoint = (
        retention_set.isdisjoint(prior_train)
        and retention_set.isdisjoint(structural_train_set)
        and structural_train_set.isdisjoint(prior_train)
    )
    if not train_sets_disjoint:
        raise RuntimeError("R23 train disjointness invariant failed")

    prior_structural_validation: set[int] = set()
    for start in (
        R18_STRUCTURAL_VAL_START,
        R19_STRUCTURAL_VAL_START,
        R20_STRUCTURAL_VAL_START,
        R21_STRUCTURAL_VAL_START,
        R22_STRUCTURAL_VAL_START,
    ):
        indices, _ = balanced_ranked_structural_window_indices(
            mismatched_full["premise"],
            mismatched_full["hypothesis"],
            mismatched_full["label"],
            start_per_label=start,
            count_per_label=STRUCTURAL_VAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        )
        prior_structural_validation.update(indices)

    structural_val_indices, structural_val_stats = (
        balanced_ranked_structural_window_indices(
            mismatched_full["premise"],
            mismatched_full["hypothesis"],
            mismatched_full["label"],
            start_per_label=R23_STRUCTURAL_VAL_START,
            count_per_label=STRUCTURAL_VAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        )
    )
    structural_validation_disjoint = set(
        structural_val_indices
    ).isdisjoint(prior_structural_validation)
    if len(structural_val_indices) != 500:
        raise RuntimeError("R23 structural validation must be 500")
    if not structural_validation_disjoint:
        raise RuntimeError("R23 structural validation overlaps prior windows")

    indexed_matched = matched_full.add_column(
        "_source_index", list(range(len(matched_full)))
    )
    shuffled_matched = indexed_matched.shuffle(seed=MATCHED_SHUFFLE_SEED)
    matched_stop = R23_MATCHED_START + MATCHED_VAL_N
    if len(shuffled_matched) < matched_stop:
        raise RuntimeError("not enough matched validation rows for R23")
    prior_matched_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][:R23_MATCHED_START]
    ]
    matched_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][
            R23_MATCHED_START:matched_stop
        ]
    ]
    matched_validation_disjoint = set(
        prior_matched_indices
    ).isdisjoint(matched_indices)
    if len(matched_indices) != MATCHED_VAL_N:
        raise RuntimeError("R23 matched validation must be 1500")
    if not matched_validation_disjoint:
        raise RuntimeError("R23 matched validation overlaps prior windows")

    structural_train = train_full.select(structural_train_indices)
    retention_train = train_full.select(retention_indices)
    structural_val = mismatched_full.select(structural_val_indices)
    matched_val = matched_full.select(matched_indices)

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
        "laya_final_used": False,
        "jev_final_used": False,
    }

    structural_train_cache = encode_dataset(
        structural_train,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r23-train-structural-exclusion-first-top-2000-per-label",
            "n": len(structural_train),
            "index_sha256": hash_indices(structural_train_indices),
        },
    )
    retention_train_cache = encode_dataset(
        retention_train,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r23-train-finite-functional-trust",
            "n": len(retention_train),
            "index_sha256": hash_indices(retention_indices),
        },
    )
    structural_val_cache = encode_dataset(
        structural_val,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r23-ranked-structural-validation-ranks-1251-1500",
            "n": len(structural_val),
            "index_sha256": hash_indices(structural_val_indices),
        },
    )
    matched_val_cache = encode_dataset(
        matched_val,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r23-matched-validation-window-7500-9000",
            "n": len(matched_val),
            "index_sha256": hash_indices(matched_indices),
        },
    )

    r15_state = torch.load(
        args.r15_head,
        map_location="cpu",
        weights_only=True,
    )
    r16_state = torch.load(
        args.r16_head,
        map_location="cpu",
        weights_only=True,
    )
    if set(r15_state) != set(r16_state):
        raise RuntimeError("R15/R16 state keys differ")

    layout = relation_layout(r15_state)
    delta = relation_delta(r15_state, r16_state, layout=layout)

    structural_model = HIRACore(d_model=256, dropout=0.0)
    structural_model.load_state_dict(r15_state, strict=True)
    structural_grad_map, structural_grad_meta = collect_structural_gradient(
        structural_model,
        structural_train_cache,
        batch_size=GRAD_BATCH_SIZE,
    )
    structural_gradient = flatten_relation_named_tensors(
        structural_grad_map,
        layout=layout,
    )

    teacher_model = HIRACore(d_model=256, dropout=0.0)
    teacher_model.load_state_dict(r15_state, strict=True)
    teacher_probabilities = frozen_teacher_probabilities(
        teacher_model,
        retention_train_cache,
        batch_size=GRAD_BATCH_SIZE,
    )

    probe_state = apply_relation_update(
        r15_state,
        delta,
        alpha=PROBE_BETA,
        layout=layout,
    )
    probe_non_relation_identical = non_relation_bit_identical(
        r15_state,
        probe_state,
        layout=layout,
    )
    if not probe_non_relation_identical:
        raise RuntimeError("R23 probe changed non-relation state")

    probe_model = HIRACore(d_model=256, dropout=0.0)
    probe_model.load_state_dict(probe_state, strict=True)
    functional_rows, functional_meta = collect_functional_kl_gradient_rows(
        probe_model,
        retention_train_cache,
        teacher_probabilities,
        batch_size=GRAD_BATCH_SIZE,
        layout=layout,
    )
    functional_rows_finite = bool(torch.isfinite(functional_rows).all())
    subspace = build_retention_subspace(
        functional_rows,
        relative_eigenvalue_floor=RELATIVE_EIGENVALUE_FLOOR,
    )

    mean_teacher_kl = float(functional_meta["mean_teacher_kl"])
    probe_signal_valid = (
        mean_teacher_kl > PROBE_KL_FLOOR
        and functional_rows_finite
        and subspace.rank >= MIN_FUNCTIONAL_RANK
    )
    basis_orthonormal_pass = (
        subspace.orthonormal_error <= ORTHONORMAL_TOLERANCE
    )

    structural_gradient_finite = bool(
        torch.isfinite(structural_gradient).all()
    )
    structural_gradient_nonzero = (
        structural_gradient_finite
        and float(structural_gradient.norm()) > 0.0
    )
    tangent_available = (
        subspace.rank >= DAMAGE_RANK
        and structural_gradient_finite
        and structural_gradient_nonzero
    )
    tangent_failure_reason = None
    tangent = None
    retained_structural_energy_fraction = 0.0
    null_leakage_energy_fraction = float("inf")

    if tangent_available:
        try:
            tangent = functional_null_structural_tangent(
                structural_gradient,
                subspace,
                rank=DAMAGE_RANK,
            )
        except ValueError as exc:
            tangent_available = False
            tangent_failure_reason = str(exc)

    if tangent is not None:
        retained_structural_energy_fraction = float(
            tangent.retained_energy_fraction
        )
        null_leakage_energy_fraction = float(
            tangent.leakage_energy_fraction
        )

    null_leakage_pass = (
        tangent is not None
        and null_leakage_energy_fraction <= MAX_NULL_LEAKAGE_ENERGY
    )
    raw_null_predicted_benefit = (
        0.0
        if tangent is None
        else float(-torch.dot(structural_gradient, tangent.null_descent))
    )
    direction_validity = (
        probe_signal_valid
        and basis_orthonormal_pass
        and structural_gradient_finite
        and structural_gradient_nonzero
        and tangent is not None
        and null_leakage_pass
        and raw_null_predicted_benefit > 0.0
    )

    # Train-only finite trust tournament. No validation cache is touched here.
    trust_candidates: list[dict[str, object]] = []
    trust_candidate_states: dict[float, dict[str, torch.Tensor]] = {}

    if tangent is not None:
        for eta in ETAS:
            update = float(eta) * tangent.null_descent
            state = apply_relation_update(
                r15_state,
                update,
                alpha=1.0,
                layout=layout,
            )
            non_relation_ok = non_relation_bit_identical(
                r15_state,
                state,
                layout=layout,
            )
            model = HIRACore(d_model=256, dropout=0.0)
            model.load_state_dict(state, strict=True)
            logits = cached_logits(
                model,
                retention_train_cache,
                batch_size=args.eval_batch_size,
            )
            trust = functional_trust_metrics(
                logits,
                teacher_probabilities,
            )
            finite_trust_pass = bool(trust.finite)
            trust_region_pass = (
                direction_validity
                and non_relation_ok
                and trust_safe(
                    trust,
                    mean_kl_ceiling=TRUST_MEAN_KL_CEILING,
                    require_full_argmax_agreement=True,
                )
            )
            predicted_benefit = float(
                -torch.dot(structural_gradient, update)
            )
            selectable = (
                trust_region_pass
                and predicted_benefit > 0.0
            )
            trust_candidate_states[float(eta)] = state
            trust_candidates.append(
                {
                    "eta": float(eta),
                    "applied_update_l2": float(update.norm()),
                    "first_order_structural_predicted_benefit": (
                        predicted_benefit
                    ),
                    "mean_teacher_kl": trust.mean_teacher_kl,
                    "teacher_argmax_agreement_count": (
                        trust.teacher_argmax_agreement_count
                    ),
                    "teacher_argmax_agreement_fraction": (
                        trust.teacher_argmax_agreement_fraction
                    ),
                    "trust_n": trust.n,
                    "finite": finite_trust_pass,
                    "non_relation_bit_identical": non_relation_ok,
                    "trust_safe": trust_region_pass,
                    "selectable": selectable,
                }
            )

    safe_candidates = [
        row for row in trust_candidates if bool(row["selectable"])
    ]
    train_selected = bool(safe_candidates)
    if train_selected:
        train_selected_row = min(
            safe_candidates,
            key=lambda row: (
                -float(row["eta"]),
                float(row["mean_teacher_kl"]),
                float(row["applied_update_l2"]),
            ),
        )
        train_selected_eta = float(train_selected_row["eta"])
        experimental_state = trust_candidate_states[train_selected_eta]
    else:
        train_selected_row = None
        train_selected_eta = None
        experimental_state = {
            key: value.detach().cpu().clone()
            for key, value in r15_state.items()
        }

    # Validation begins only after train-only selection is frozen.
    baseline_matched, baseline_structural = evaluate_state(
        r15_state,
        matched_val_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )
    baseline_matched_accuracy = float(baseline_matched["accuracy"])
    baseline_structural_ne = float(
        baseline_structural["non_entailment_accuracy"]
    )
    baseline_correct = int(round(baseline_matched_accuracy * MATCHED_VAL_N))
    matched_label_counts = Counter(int(x) for x in matched_val["label"])
    matched_label_support_valid = all(
        matched_label_counts.get(label, 0) >= MIN_MATCHED_LABEL_COUNT
        for label in (0, 1, 2)
    )
    baseline_validity = (
        len(matched_indices) == MATCHED_VAL_N
        and matched_label_support_valid
        and baseline_matched_accuracy >= BASELINE_MATCHED_FLOOR
    )

    experimental_matched, experimental_structural = evaluate_state(
        experimental_state,
        matched_val_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )
    experimental_matched_accuracy = float(experimental_matched["accuracy"])
    experimental_correct = int(
        round(experimental_matched_accuracy * MATCHED_VAL_N)
    )
    zero_drop_retention_pass = (
        train_selected
        and finite_sample_relative_accuracy_pass(
            experimental_matched_accuracy,
            baseline_matched_accuracy,
            n=MATCHED_VAL_N,
            tolerance=MATCHED_RELATIVE_TOLERANCE,
        )
    )
    experimental_non_relation = non_relation_bit_identical(
        r15_state,
        experimental_state,
        layout=layout,
    )
    parameter_count_unchanged = count_parameters(HIRACore()) == 422_159

    final_eligible = (
        train_selected
        and baseline_validity
        and direction_validity
        and bool(train_selected_row["trust_safe"])
        and experimental_matched_accuracy >= BASELINE_MATCHED_FLOOR
        and zero_drop_retention_pass
        and experimental_non_relation
        and parameter_count_unchanged
    )

    experimental_structural_ne = float(
        experimental_structural["non_entailment_accuracy"]
    )
    structural_gain = (
        experimental_structural_ne - baseline_structural_ne
    )

    # Diagnosis-only controls are evaluated after train-only selection is frozen.
    controls: list[dict[str, object]] = []
    if tangent is not None:
        raw_eta = 1.0 / 64.0
        raw_update = raw_eta * tangent.raw_descent
        raw_state = apply_relation_update(
            r15_state,
            raw_update,
            alpha=1.0,
            layout=layout,
        )
        raw_matched, raw_structural = evaluate_state(
            raw_state,
            matched_val_cache,
            structural_val_cache,
            batch_size=args.eval_batch_size,
        )
        controls.append(
            {
                "family": "native_raw_structural_tangent",
                "eta": raw_eta,
                "applied_update_l2": float(raw_update.norm()),
                "matched": raw_matched,
                "ranked_structural": raw_structural,
                "eligible_for_selection": False,
            }
        )

        normalized_null = normalize_to_reference_l2(
            tangent.null_descent,
            delta,
        )
        r22_style_state = apply_relation_update(
            r15_state,
            normalized_null,
            alpha=0.125,
            layout=layout,
        )
        r22_matched, r22_structural = evaluate_state(
            r22_style_state,
            matched_val_cache,
            structural_val_cache,
            batch_size=args.eval_batch_size,
        )
        controls.append(
            {
                "family": "r22_style_normalized_null",
                "alpha": 0.125,
                "applied_update_l2": float(
                    0.125 * normalized_null.norm()
                ),
                "matched": r22_matched,
                "ranked_structural": r22_structural,
                "eligible_for_selection": False,
            }
        )

        protected_r16_update, _ = rotate_delta_away_from_retention(
            delta,
            subspace,
            rank=DAMAGE_RANK,
            removal_strength=0.5,
        )
        protected_state = apply_relation_update(
            r15_state,
            protected_r16_update,
            alpha=0.25,
            layout=layout,
        )
        protected_matched, protected_structural = evaluate_state(
            protected_state,
            matched_val_cache,
            structural_val_cache,
            batch_size=args.eval_batch_size,
        )
        controls.append(
            {
                "family": "protected_r16_rank32_removal0.5",
                "alpha": 0.25,
                "matched": protected_matched,
                "ranked_structural": protected_structural,
                "eligible_for_selection": False,
            }
        )

    # Persist candidate only when it passes final eligibility; otherwise exact R15.
    if final_eligible:
        persisted_state = experimental_state
        persisted_head_mode = "train_selected_candidate"
    else:
        persisted_state = {
            key: value.detach().cpu().clone()
            for key, value in r15_state.items()
        }
        persisted_head_mode = "exact_r15_fallback"

    head_path = args.out / "hira-head.pt"
    torch.save(persisted_state, head_path)
    selected_sha = file_sha256(head_path)

    gates = {
        "prior_train_exact_40000": len(prior_train) == 40000,
        "structural_train_exact_4000": len(structural_train_indices) == 4000,
        "retention_train_exact_6000": len(retention_indices) == 6000,
        "train_authorities_disjoint_from_each_other_and_prior": (
            train_sets_disjoint
        ),
        "probe_beta": PROBE_BETA,
        "probe_mean_teacher_kl": mean_teacher_kl,
        "probe_kl_floor": PROBE_KL_FLOOR,
        "probe_kl_pass": mean_teacher_kl > PROBE_KL_FLOOR,
        "functional_gradient_rows_finite": functional_rows_finite,
        "functional_damage_numerical_rank": subspace.rank,
        "functional_rank_floor": MIN_FUNCTIONAL_RANK,
        "functional_rank_pass": subspace.rank >= MIN_FUNCTIONAL_RANK,
        "probe_signal_valid": probe_signal_valid,
        "basis_orthonormal_error": subspace.orthonormal_error,
        "basis_orthonormal_pass": basis_orthonormal_pass,
        "structural_gradient_finite": structural_gradient_finite,
        "structural_gradient_nonzero": structural_gradient_nonzero,
        "tangent_available": tangent is not None,
        "retained_structural_gradient_energy_fraction": (
            retained_structural_energy_fraction
        ),
        "null_leakage_energy_fraction": null_leakage_energy_fraction,
        "null_leakage_energy_ceiling": MAX_NULL_LEAKAGE_ENERGY,
        "null_leakage_pass": null_leakage_pass,
        "raw_null_first_order_structural_predicted_benefit": (
            raw_null_predicted_benefit
        ),
        "direction_validity": direction_validity,
        "trust_mean_kl_ceiling": TRUST_MEAN_KL_CEILING,
        "train_selected": train_selected,
        "train_selected_eta": train_selected_eta,
        "train_selected_trust_safe": (
            False
            if train_selected_row is None
            else bool(train_selected_row["trust_safe"])
        ),
        "structural_validation_exact_500": len(structural_val_indices) == 500,
        "structural_validation_disjoint_from_r18_r19_r20_r21_r22": (
            structural_validation_disjoint
        ),
        "matched_validation_exact_1500": len(matched_indices) == 1500,
        "matched_validation_disjoint_from_prior": (
            matched_validation_disjoint
        ),
        "matched_label_counts": {
            str(label): int(matched_label_counts.get(label, 0))
            for label in (0, 1, 2)
        },
        "matched_label_support_valid": matched_label_support_valid,
        "baseline_matched_floor": BASELINE_MATCHED_FLOOR,
        "baseline_validity": baseline_validity,
        "baseline_matched_correct": baseline_correct,
        "experimental_matched_correct": experimental_correct,
        "matched_absolute_pass": (
            final_eligible
            and experimental_matched_accuracy >= BASELINE_MATCHED_FLOOR
        ),
        "zero_drop_matched_pass": (
            final_eligible and zero_drop_retention_pass
        ),
        "final_eligible": final_eligible,
        "ranked_structural_floor": STRUCTURAL_FLOOR,
        "ranked_structural_pass": (
            final_eligible
            and experimental_structural_ne >= STRUCTURAL_FLOOR
        ),
        "ranked_structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
        "ranked_structural_gain": structural_gain,
        "ranked_structural_gain_pass": (
            final_eligible
            and structural_gain >= STRUCTURAL_GAIN_FLOOR
        ),
        "non_relation_bit_identical": experimental_non_relation,
        "parameter_count_unchanged": parameter_count_unchanged,
    }
    gates["primary_pass"] = all(
        [
            gates["prior_train_exact_40000"],
            gates["structural_train_exact_4000"],
            gates["retention_train_exact_6000"],
            gates["train_authorities_disjoint_from_each_other_and_prior"],
            gates["probe_kl_pass"],
            gates["functional_gradient_rows_finite"],
            gates["functional_rank_pass"],
            gates["basis_orthonormal_pass"],
            gates["structural_gradient_finite"],
            gates["structural_gradient_nonzero"],
            gates["tangent_available"],
            gates["null_leakage_pass"],
            gates["direction_validity"],
            gates["train_selected"],
            gates["train_selected_trust_safe"],
            gates["structural_validation_exact_500"],
            gates["structural_validation_disjoint_from_r18_r19_r20_r21_r22"],
            gates["matched_validation_exact_1500"],
            gates["matched_validation_disjoint_from_prior"],
            gates["matched_label_support_valid"],
            gates["baseline_validity"],
            gates["final_eligible"],
            gates["matched_absolute_pass"],
            gates["zero_drop_matched_pass"],
            gates["ranked_structural_pass"],
            gates["ranked_structural_gain_pass"],
            gates["non_relation_bit_identical"],
            gates["parameter_count_unchanged"],
        ]
    )

    receipt = {
        "schema_version": "r23-finite-functional-trust-region-v1",
        "status": "PASS",
        "evidence_scope": (
            "train-only native functional-null structural tangent plus finite "
            "teacher-KL and exact teacher-decision trust selection; final "
            "evaluation on new disjoint MultiNLI development authorities; "
            "no HANS/Breaking/XNLI/MASSIVE/Banking77/Laya/Jev selection"
        ),
        "sources": {
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
            "mnli_revision": MNLI_REVISION,
            "r15_head_sha256": R15_HEAD_SHA256,
            "r16_failure_analysis_head_sha256": R16_HEAD_SHA256,
        },
        "protocol": {
            "issue_authority": 42,
            "probe_beta": PROBE_BETA,
            "damage_rank": DAMAGE_RANK,
            "etas": list(ETAS),
            "native_null_tangent": True,
            "r16_l2_normalization": False,
            "trust_mean_kl_ceiling": TRUST_MEAN_KL_CEILING,
            "trust_argmax_authority": "exact 6000/6000 teacher agreement",
            "r23_structural_train_rule": (
                "strongest 2000 neutral + 2000 contradiction after excluding "
                "exact R19/R20/R21/R22 scored-train source-index union"
            ),
            "r23_retention_shuffle_seed": R23_RETENTION_SEED,
            "r23_retention_n": R23_RETENTION_N,
            "structural_validation_window_per_label": [
                R23_STRUCTURAL_VAL_START,
                R23_STRUCTURAL_VAL_START + STRUCTURAL_VAL_PER_LABEL,
            ],
            "matched_validation_window": [
                R23_MATCHED_START,
                R23_MATCHED_START + MATCHED_VAL_N,
            ],
            "matched_shuffle_seed": MATCHED_SHUFFLE_SEED,
            "probe_kl_floor": PROBE_KL_FLOOR,
            "functional_rank_floor": MIN_FUNCTIONAL_RANK,
            "relative_eigenvalue_floor": RELATIVE_EIGENVALUE_FLOOR,
            "orthonormal_tolerance": ORTHONORMAL_TOLERANCE,
            "null_leakage_energy_ceiling": MAX_NULL_LEAKAGE_ENERGY,
            "baseline_matched_floor": BASELINE_MATCHED_FLOOR,
            "matched_relative_tolerance": MATCHED_RELATIVE_TOLERANCE,
            "matched_relative_authority": (
                "exact finite-sample correct-count comparison; zero-drop"
            ),
            "min_matched_label_count": MIN_MATCHED_LABEL_COUNT,
            "structural_floor": STRUCTURAL_FLOOR,
            "structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
            "train_selection_order": [
                "trust-safe candidates only",
                "largest eta",
                "smaller mean teacher KL",
                "smaller applied update L2",
            ],
            "artifact_output_rule": (
                "persist train-selected candidate only if final eligible; "
                "otherwise exact R15 fallback"
            ),
        },
        "data": {
            "prior_train_n": len(prior_train),
            "r19_structural_train_index_sha256": hash_indices(
                prior["r19_structural"]
            ),
            "r19_retention_train_index_sha256": hash_indices(
                prior["r19_retention"]
            ),
            "r20_structural_train_index_sha256": hash_indices(
                prior["r20_structural"]
            ),
            "r20_retention_train_index_sha256": hash_indices(
                prior["r20_retention"]
            ),
            "r21_structural_train_index_sha256": hash_indices(
                prior["r21_structural"]
            ),
            "r21_retention_train_index_sha256": hash_indices(
                prior["r21_retention"]
            ),
            "r22_structural_train_index_sha256": hash_indices(
                prior["r22_structural"]
            ),
            "r22_retention_train_index_sha256": hash_indices(
                prior["r22_retention"]
            ),
            "structural_train_n": len(structural_train_indices),
            "structural_train_index_sha256": hash_indices(
                structural_train_indices
            ),
            "structural_train_stats": structural_train_stats,
            "retention_train_n": len(retention_indices),
            "retention_train_index_sha256": hash_indices(retention_indices),
            "train_authorities_disjoint_from_each_other_and_prior": (
                train_sets_disjoint
            ),
            "prior_structural_validation_index_sha256": hash_indices(
                sorted(prior_structural_validation)
            ),
            "structural_validation_n": len(structural_val_indices),
            "structural_validation_index_sha256": hash_indices(
                structural_val_indices
            ),
            "structural_validation_stats": structural_val_stats,
            "structural_validation_disjoint_from_r18_r19_r20_r21_r22": (
                structural_validation_disjoint
            ),
            "prior_matched_index_sha256": hash_indices(prior_matched_indices),
            "matched_validation_n": len(matched_indices),
            "matched_validation_index_sha256": hash_indices(matched_indices),
            "matched_validation_disjoint_from_prior": (
                matched_validation_disjoint
            ),
        },
        "functional_damage": {
            "teacher_probabilities_shape": list(teacher_probabilities.shape),
            "probe_non_relation_bit_identical": (
                probe_non_relation_identical
            ),
            "probe_meta": functional_meta,
            "functional_subspace_rank": subspace.rank,
            "functional_subspace_orthonormal_error": (
                subspace.orthonormal_error
            ),
            "eigenvalue_energy_top32": [
                float(x)
                for x in subspace.cumulative_coverage[
                    :min(DAMAGE_RANK, subspace.rank)
                ]
            ],
            "original_relation_delta_l2": float(delta.norm()),
            "structural_gradient_l2": float(structural_gradient.norm()),
            "structural_gradient_meta": structural_grad_meta,
            "tangent_failure_reason": tangent_failure_reason,
            "native_null_tangent_l2": (
                None if tangent is None else float(tangent.null_descent.norm())
            ),
            "retained_structural_gradient_energy_fraction": (
                retained_structural_energy_fraction
            ),
            "null_leakage_energy_fraction": (
                null_leakage_energy_fraction
            ),
            "raw_null_first_order_structural_predicted_benefit": (
                raw_null_predicted_benefit
            ),
        },
        "trust_candidates": trust_candidates,
        "train_selected": train_selected_row,
        "baseline": {
            "matched": baseline_matched,
            "matched_correct": baseline_correct,
            "ranked_structural": baseline_structural,
            "matched_label_counts": {
                str(label): int(matched_label_counts.get(label, 0))
                for label in (0, 1, 2)
            },
            "validity": baseline_validity,
        },
        "experimental_final": {
            "matched": experimental_matched,
            "matched_correct": experimental_correct,
            "ranked_structural": experimental_structural,
            "zero_drop_retention_pass": zero_drop_retention_pass,
            "final_eligible": final_eligible,
        },
        "diagnosis_controls": controls,
        "persisted_head_mode": persisted_head_mode,
        "persisted_head_sha256": selected_sha,
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
                "baseline_validity": baseline_validity,
                "baseline_matched_accuracy": baseline_matched_accuracy,
                "baseline_matched_correct": baseline_correct,
                "baseline_ranked_structural": baseline_structural_ne,
                "probe_mean_teacher_kl": mean_teacher_kl,
                "functional_subspace_rank": subspace.rank,
                "native_null_tangent_l2": (
                    None if tangent is None else float(tangent.null_descent.norm())
                ),
                "retained_structural_gradient_energy_fraction": (
                    retained_structural_energy_fraction
                ),
                "null_leakage_energy_fraction": (
                    null_leakage_energy_fraction
                ),
                "train_selected": train_selected_row,
                "experimental_matched_accuracy": (
                    experimental_matched_accuracy
                ),
                "experimental_matched_correct": experimental_correct,
                "experimental_ranked_structural": (
                    experimental_structural_ne
                ),
                "ranked_structural_gain": structural_gain,
                "final_eligible": final_eligible,
                "persisted_head_mode": persisted_head_mode,
                "persisted_head_sha256": selected_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
