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
    teacher_kl_loss,
    weighted_batch_gradient_row,
)
from nmd.functional_trust import functional_trust_metrics
from nmd.hard_negative import anti_entailment_margin_loss, evaluate_repair_slice
from nmd.hira import HIRACore, count_parameters
from nmd.interpolation import multiset_recall, ordered_lcs_recall
from nmd.relation_cache import RelationCache, forward_cached, minibatches
from nmd.semantic import HFAutoSemanticEncoder
from nmd.sequential_trust import (
    SafeStepCandidate,
    choose_largest_safe_step,
    exact_zero_drop,
)
from nmd.structural import tokens
from nmd.structural_tangent import functional_null_structural_tangent
from nmd.subspace_projection import (
    apply_relation_update,
    build_retention_subspace,
    non_relation_bit_identical,
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

R24_STRUCTURAL_PER_LABEL = 2000
R24_STRUCTURAL_ROUNDS = 16
R24_STRUCTURAL_PER_LABEL_PER_ROUND = 125
R24_TRUST_N = 6000
R24_TRUST_SEED = 37
R24_HOLDOUT_N = 3000
R24_HOLDOUT_SEED = 43

STRUCTURAL_VAL_PER_LABEL = 250
R18_STRUCTURAL_VAL_START = 0
R19_STRUCTURAL_VAL_START = 250
R20_STRUCTURAL_VAL_START = 500
R21_STRUCTURAL_VAL_START = 750
R22_STRUCTURAL_VAL_START = 1000
R23_STRUCTURAL_VAL_START = 1250
R24_STRUCTURAL_VAL_START = 1500

MATCHED_SHUFFLE_SEED = 14
R24_MATCHED_TAIL_START = 9000
MIN_MATCHED_TAIL_N = 750
MIN_MATCHED_TAIL_LABEL_COUNT = 200
MIN_HOLDOUT_LABEL_COUNT = 800

MIN_HYPOTHESIS_TOKENS = 3
GRAD_BATCH_SIZE = 96
PROBE_BETA = 0.25
DAMAGE_RANK = 32
MICRO_ETAS = (
    1.0 / 64.0,
    1.0 / 128.0,
    1.0 / 256.0,
    1.0 / 512.0,
    1.0 / 1024.0,
    1.0 / 2048.0,
    1.0 / 4096.0,
)

PROBE_KL_FLOOR = 1e-4
MIN_FUNCTIONAL_RANK = 32
RELATIVE_EIGENVALUE_FLOOR = 1e-10
ORTHONORMAL_TOLERANCE = 1e-7
MAX_NULL_LEAKAGE_ENERGY = 1e-8
TRUST_MEAN_KL_CEILING = 1e-4
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



def exact_correct_count(
    state: dict[str, torch.Tensor],
    cache: RelationCache,
    *,
    batch_size: int,
) -> int:
    model = HIRACore(d_model=256, dropout=0.0)
    model.load_state_dict(state, strict=True)
    logits = cached_logits(model, cache, batch_size=batch_size)
    predictions = logits.argmax(dim=-1)
    return int((predictions == cache.labels.long()).sum().item())


def structural_rank_order_for_selected(
    train_full,
    selected_indices: list[int],
) -> dict[int, list[int]]:
    rows: dict[int, list[tuple[int, float, float]]] = {1: [], 2: []}
    for index in selected_indices:
        label = int(train_full[index]["label"])
        if label not in rows:
            raise RuntimeError("R24 selected structural row has invalid label")
        p = tokens(str(train_full[index]["premise"]))
        h = tokens(str(train_full[index]["hypothesis"]))
        multi = float(multiset_recall(p, h))
        lcs = float(ordered_lcs_recall(p, h))
        if lcs > multi + 1e-12:
            raise RuntimeError("ordered-LCS recall exceeded multiset recall")
        rows[label].append((int(index), multi, lcs))

    ordered: dict[int, list[int]] = {}
    for label in (1, 2):
        if len(rows[label]) != R24_STRUCTURAL_PER_LABEL:
            raise RuntimeError(
                f"R24 structural label {label} has {len(rows[label])} rows"
            )
        ranked = sorted(
            rows[label],
            key=lambda row: (-row[1], -row[2], row[0]),
        )
        ordered[label] = [row[0] for row in ranked]
    return ordered


def make_structural_round_indices(
    ordered_by_label: dict[int, list[int]],
) -> list[list[int]]:
    rounds: list[list[int]] = []
    seen: set[int] = set()
    for round_index in range(R24_STRUCTURAL_ROUNDS):
        start = round_index * R24_STRUCTURAL_PER_LABEL_PER_ROUND
        stop = start + R24_STRUCTURAL_PER_LABEL_PER_ROUND
        shard = (
            ordered_by_label[1][start:stop]
            + ordered_by_label[2][start:stop]
        )
        if len(shard) != 250:
            raise RuntimeError("R24 structural shard must contain 250 rows")
        if len(set(shard)) != 250:
            raise RuntimeError("R24 structural shard contains duplicates")
        if seen.intersection(shard):
            raise RuntimeError("R24 structural shards overlap")
        seen.update(shard)
        rounds.append(shard)
    if len(seen) != 4000:
        raise RuntimeError("R24 structural shards must cover exactly 4000 rows")
    return rounds


def changed_teacher_margin_stats(
    logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
) -> dict[str, object]:
    teacher = teacher_probabilities.to(logits.dtype)
    teacher_argmax = teacher.argmax(dim=-1)
    student_argmax = logits.argmax(dim=-1)
    changed = teacher_argmax != student_argmax
    sorted_probs = torch.sort(teacher, dim=-1, descending=True).values
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    changed_count = int(changed.sum().item())
    return {
        "changed_count": changed_count,
        "min_teacher_margin_all": float(margins.min().item()),
        "min_teacher_margin_changed": (
            None
            if changed_count == 0
            else float(margins[changed].min().item())
        ),
        "max_teacher_margin_changed": (
            None
            if changed_count == 0
            else float(margins[changed].max().item())
        ),
    }


def reconstruct_prior_train(train_full) -> dict[str, list[int]]:
    r19_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"], train_full["hypothesis"], train_full["label"],
        start_per_label=0, count_per_label=R19_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
    )
    r19_retention = random_exclusion_sample(
        train_full, seed=R19_RETENTION_SEED, n=R19_RETENTION_N,
        excluded=set(r19_structural),
    )
    union = set(r19_structural) | set(r19_retention)

    r20_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"], train_full["hypothesis"], train_full["label"],
        start_per_label=0, count_per_label=R20_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        exclude_indices=sorted(union),
    )
    r20_retention = random_exclusion_sample(
        train_full, seed=R20_RETENTION_SEED, n=R20_RETENTION_N,
        excluded=union | set(r20_structural),
    )
    union |= set(r20_structural) | set(r20_retention)

    r21_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"], train_full["hypothesis"], train_full["label"],
        start_per_label=0, count_per_label=R21_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        exclude_indices=sorted(union),
    )
    r21_retention = random_exclusion_sample(
        train_full, seed=R21_RETENTION_SEED, n=R21_RETENTION_N,
        excluded=union | set(r21_structural),
    )
    union |= set(r21_structural) | set(r21_retention)

    r22_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"], train_full["hypothesis"], train_full["label"],
        start_per_label=0, count_per_label=R22_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        exclude_indices=sorted(union),
    )
    r22_retention = random_exclusion_sample(
        train_full, seed=R22_RETENTION_SEED, n=R22_RETENTION_N,
        excluded=union | set(r22_structural),
    )
    union |= set(r22_structural) | set(r22_retention)

    r23_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"], train_full["hypothesis"], train_full["label"],
        start_per_label=0, count_per_label=R23_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        exclude_indices=sorted(union),
    )
    r23_retention = random_exclusion_sample(
        train_full, seed=R23_RETENTION_SEED, n=R23_RETENTION_N,
        excluded=union | set(r23_structural),
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
        "r23_structural": r23_structural,
        "r23_retention": r23_retention,
    }
    union_size = len(set().union(*(set(v) for v in result.values())))
    if union_size != 50000:
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
        str(snapshot), local_files_only=True
    )
    base = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    encoder = HFAutoSemanticEncoder(
        base, tokenizer, revision=A13_REVISION, max_length=128
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

    structural_indices, structural_stats = (
        balanced_ranked_structural_window_indices(
            train_full["premise"],
            train_full["hypothesis"],
            train_full["label"],
            start_per_label=0,
            count_per_label=R24_STRUCTURAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
            exclude_indices=sorted(prior_train),
        )
    )
    if len(structural_indices) != 4000:
        raise RuntimeError("R24 structural authority must contain 4000 rows")
    structural_set = set(structural_indices)
    if not structural_set.isdisjoint(prior_train):
        raise RuntimeError("R24 structural authority overlaps prior train")

    ordered_by_label = structural_rank_order_for_selected(
        train_full,
        structural_indices,
    )
    structural_round_indices = make_structural_round_indices(
        ordered_by_label
    )
    flattened_rounds = [
        index
        for shard in structural_round_indices
        for index in shard
    ]
    if set(flattened_rounds) != structural_set:
        raise RuntimeError("R24 structural shard coverage mismatch")

    trust_indices = random_exclusion_sample(
        train_full,
        seed=R24_TRUST_SEED,
        n=R24_TRUST_N,
        excluded=prior_train | structural_set,
    )
    trust_set = set(trust_indices)
    if not trust_set.isdisjoint(prior_train | structural_set):
        raise RuntimeError("R24 trust authority overlaps proposal/prior train")

    holdout_indices = random_exclusion_sample(
        train_full,
        seed=R24_HOLDOUT_SEED,
        n=R24_HOLDOUT_N,
        excluded=prior_train | structural_set | trust_set,
    )
    holdout_set = set(holdout_indices)
    if not holdout_set.isdisjoint(
        prior_train | structural_set | trust_set
    ):
        raise RuntimeError("R24 C1 holdout overlaps train authorities")

    holdout_label_counts = Counter(
        int(train_full[index]["label"]) for index in holdout_indices
    )
    holdout_label_support_valid = all(
        holdout_label_counts.get(label, 0) >= MIN_HOLDOUT_LABEL_COUNT
        for label in (0, 1, 2)
    )

    prior_structural_validation: set[int] = set()
    for start in (
        R18_STRUCTURAL_VAL_START,
        R19_STRUCTURAL_VAL_START,
        R20_STRUCTURAL_VAL_START,
        R21_STRUCTURAL_VAL_START,
        R22_STRUCTURAL_VAL_START,
        R23_STRUCTURAL_VAL_START,
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
            start_per_label=R24_STRUCTURAL_VAL_START,
            count_per_label=STRUCTURAL_VAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        )
    )
    structural_validation_disjoint = set(
        structural_val_indices
    ).isdisjoint(prior_structural_validation)
    if len(structural_val_indices) != 500:
        raise RuntimeError("R24 structural validation must contain 500 rows")
    if not structural_validation_disjoint:
        raise RuntimeError("R24 structural validation overlaps prior windows")

    indexed_matched = matched_full.add_column(
        "_source_index", list(range(len(matched_full)))
    )
    shuffled_matched = indexed_matched.shuffle(seed=MATCHED_SHUFFLE_SEED)
    if len(shuffled_matched) <= R24_MATCHED_TAIL_START:
        raise RuntimeError("R24 official matched tail is empty")
    prior_matched_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][:R24_MATCHED_TAIL_START]
    ]
    matched_tail_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][R24_MATCHED_TAIL_START:]
    ]
    if len(matched_tail_indices) < MIN_MATCHED_TAIL_N:
        raise RuntimeError(
            f"R24 matched tail too small: {len(matched_tail_indices)}"
        )
    matched_tail_disjoint = set(matched_tail_indices).isdisjoint(
        prior_matched_indices
    )
    matched_tail_label_counts = Counter(
        int(matched_full[index]["label"])
        for index in matched_tail_indices
    )
    matched_tail_label_support_valid = all(
        matched_tail_label_counts.get(label, 0)
        >= MIN_MATCHED_TAIL_LABEL_COUNT
        for label in (0, 1, 2)
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
        "laya_final_used": False,
        "jev_final_used": False,
    }

    structural_round_caches: list[RelationCache] = []
    for round_index, shard_indices in enumerate(structural_round_indices):
        shard = train_full.select(shard_indices)
        labels = Counter(int(x) for x in shard["label"])
        if labels.get(1, 0) != 125 or labels.get(2, 0) != 125:
            raise RuntimeError(
                f"R24 round {round_index + 1} is not 125/125 balanced"
            )
        structural_round_caches.append(
            encode_dataset(
                shard,
                encoder,
                option_embeddings,
                batch_size=args.encode_batch_size,
                n_segments=args.segments,
                metadata={
                    **common,
                    "role": f"r24-structural-round-{round_index + 1:02d}",
                    "n": len(shard),
                    "index_sha256": hash_indices(shard_indices),
                },
            )
        )

    trust_cache = encode_dataset(
        train_full.select(trust_indices),
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r24-train-exact-functional-trust",
            "n": len(trust_indices),
            "index_sha256": hash_indices(trust_indices),
        },
    )
    holdout_cache = encode_dataset(
        train_full.select(holdout_indices),
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r24-final-c1-train-heldout",
            "n": len(holdout_indices),
            "index_sha256": hash_indices(holdout_indices),
        },
    )
    matched_tail_cache = encode_dataset(
        matched_full.select(matched_tail_indices),
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r24-final-c2-matched-tail-9000-end",
            "n": len(matched_tail_indices),
            "index_sha256": hash_indices(matched_tail_indices),
        },
    )
    structural_val_cache = encode_dataset(
        mismatched_full.select(structural_val_indices),
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r24-ranked-structural-validation-ranks-1501-1750",
            "n": len(structural_val_indices),
            "index_sha256": hash_indices(structural_val_indices),
        },
    )

    r15_state = torch.load(
        args.r15_head, map_location="cpu", weights_only=True
    )
    r16_state = torch.load(
        args.r16_head, map_location="cpu", weights_only=True
    )
    if set(r15_state) != set(r16_state):
        raise RuntimeError("R15/R16 state keys differ")

    layout = relation_layout(r15_state)
    delta = relation_delta(r15_state, r16_state, layout=layout)

    teacher_model = HIRACore(d_model=256, dropout=0.0)
    teacher_model.load_state_dict(r15_state, strict=True)
    teacher_probabilities = frozen_teacher_probabilities(
        teacher_model,
        trust_cache,
        batch_size=args.eval_batch_size,
    )

    probe_state = apply_relation_update(
        r15_state,
        delta,
        alpha=PROBE_BETA,
        layout=layout,
    )
    if not non_relation_bit_identical(
        r15_state, probe_state, layout=layout
    ):
        raise RuntimeError("R24 probe changed non-relation state")

    probe_model = HIRACore(d_model=256, dropout=0.0)
    probe_model.load_state_dict(probe_state, strict=True)
    functional_rows, functional_meta = collect_functional_kl_gradient_rows(
        probe_model,
        trust_cache,
        teacher_probabilities,
        batch_size=GRAD_BATCH_SIZE,
        layout=layout,
    )
    functional_rows_finite = bool(torch.isfinite(functional_rows).all())
    subspace = build_retention_subspace(
        functional_rows,
        relative_eigenvalue_floor=RELATIVE_EIGENVALUE_FLOOR,
    )
    mean_probe_kl = float(functional_meta["mean_teacher_kl"])
    basis_valid = (
        mean_probe_kl > PROBE_KL_FLOOR
        and functional_rows_finite
        and subspace.rank >= MIN_FUNCTIONAL_RANK
        and subspace.orthonormal_error <= ORTHONORMAL_TOLERANCE
    )

    current_state = {
        key: value.detach().cpu().clone()
        for key, value in r15_state.items()
    }
    round_receipts: list[dict[str, object]] = []
    accepted_rounds = 0
    all_accepted_states_trust_safe = True

    for round_index, structural_cache in enumerate(structural_round_caches):
        current_model = HIRACore(d_model=256, dropout=0.0)
        current_model.load_state_dict(current_state, strict=True)
        grad_map, grad_meta = collect_structural_gradient(
            current_model,
            structural_cache,
            batch_size=GRAD_BATCH_SIZE,
            layout=layout,
        )
        structural_gradient = flatten_relation_named_tensors(
            grad_map,
            layout=layout,
        )
        gradient_finite = bool(torch.isfinite(structural_gradient).all())
        gradient_nonzero = (
            gradient_finite and float(structural_gradient.norm()) > 0.0
        )

        tangent = None
        tangent_error = None
        if basis_valid and gradient_nonzero:
            try:
                tangent = functional_null_structural_tangent(
                    structural_gradient,
                    subspace,
                    rank=DAMAGE_RANK,
                )
            except ValueError as exc:
                tangent_error = str(exc)

        direction_valid = False
        null_l2 = 0.0
        null_leakage = float("inf")
        raw_predicted_benefit = 0.0
        if tangent is not None:
            null_l2 = float(tangent.null_descent.norm())
            null_leakage = float(tangent.leakage_energy_fraction)
            raw_predicted_benefit = float(
                -torch.dot(
                    structural_gradient,
                    tangent.null_descent,
                )
            )
            direction_valid = (
                null_l2 > 0.0
                and null_leakage <= MAX_NULL_LEAKAGE_ENERGY
                and raw_predicted_benefit > 0.0
            )

        tested_rows: list[dict[str, object]] = []
        chosen_state = None
        chosen_candidate = None

        if direction_valid:
            for eta in MICRO_ETAS:
                update = float(eta) * tangent.null_descent
                candidate_state = apply_relation_update(
                    current_state,
                    update,
                    alpha=1.0,
                    layout=layout,
                )
                non_relation_ok = non_relation_bit_identical(
                    r15_state,
                    candidate_state,
                    layout=layout,
                )
                candidate_model = HIRACore(d_model=256, dropout=0.0)
                candidate_model.load_state_dict(candidate_state, strict=True)
                logits = cached_logits(
                    candidate_model,
                    trust_cache,
                    batch_size=args.eval_batch_size,
                )
                trust = functional_trust_metrics(
                    logits,
                    teacher_probabilities,
                )
                margin_stats = changed_teacher_margin_stats(
                    logits,
                    teacher_probabilities,
                )
                predicted_benefit = float(
                    -torch.dot(structural_gradient, update)
                )
                candidate = SafeStepCandidate(
                    eta=float(eta),
                    mean_teacher_kl=trust.mean_teacher_kl,
                    teacher_argmax_agreement_count=(
                        trust.teacher_argmax_agreement_count
                    ),
                    trust_n=trust.n,
                    finite=trust.finite,
                    non_relation_bit_identical=non_relation_ok,
                    first_order_structural_predicted_benefit=(
                        predicted_benefit
                    ),
                    applied_update_l2=float(update.norm()),
                )
                selected = choose_largest_safe_step(
                    [candidate],
                    mean_kl_ceiling=TRUST_MEAN_KL_CEILING,
                )
                row = {
                    "eta": float(eta),
                    "applied_update_l2": float(update.norm()),
                    "mean_teacher_kl": trust.mean_teacher_kl,
                    "teacher_argmax_agreement_count": (
                        trust.teacher_argmax_agreement_count
                    ),
                    "teacher_argmax_agreement_fraction": (
                        trust.teacher_argmax_agreement_fraction
                    ),
                    "trust_n": trust.n,
                    "finite": trust.finite,
                    "non_relation_bit_identical": non_relation_ok,
                    "first_order_structural_predicted_benefit": (
                        predicted_benefit
                    ),
                    "safe": selected is not None,
                    "teacher_margin_diagnosis": margin_stats,
                }
                tested_rows.append(row)
                if selected is not None:
                    chosen_candidate = row
                    chosen_state = candidate_state
                    break

        accepted = chosen_state is not None
        if accepted:
            current_state = chosen_state
            accepted_rounds += 1
            all_accepted_states_trust_safe = (
                all_accepted_states_trust_safe
                and bool(chosen_candidate["safe"])
                and int(
                    chosen_candidate["teacher_argmax_agreement_count"]
                ) == R24_TRUST_N
                and float(chosen_candidate["mean_teacher_kl"])
                <= TRUST_MEAN_KL_CEILING
            )

        round_receipts.append(
            {
                "round": round_index + 1,
                "structural_shard_index_sha256": hash_indices(
                    structural_round_indices[round_index]
                ),
                "structural_gradient_meta": grad_meta,
                "structural_gradient_l2": float(
                    structural_gradient.norm()
                ),
                "direction_valid": direction_valid,
                "tangent_error": tangent_error,
                "native_null_tangent_l2": null_l2,
                "null_leakage_energy_fraction": null_leakage,
                "raw_null_first_order_structural_predicted_benefit": (
                    raw_predicted_benefit
                ),
                "tested_candidates": tested_rows,
                "accepted": accepted,
                "chosen": chosen_candidate,
            }
        )

    train_selected = accepted_rounds > 0
    experimental_state = current_state

    # Final evaluation starts only after the 16-round train-only walk is frozen.
    baseline_holdout = evaluate_state(
        r15_state,
        holdout_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )[0]
    candidate_holdout = evaluate_state(
        experimental_state,
        holdout_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )[0]
    baseline_holdout_correct = exact_correct_count(
        r15_state,
        holdout_cache,
        batch_size=args.eval_batch_size,
    )
    candidate_holdout_correct = exact_correct_count(
        experimental_state,
        holdout_cache,
        batch_size=args.eval_batch_size,
    )
    c1_zero_drop = exact_zero_drop(
        candidate_correct=candidate_holdout_correct,
        baseline_correct=baseline_holdout_correct,
    )

    baseline_tail = evaluate_state(
        r15_state,
        matched_tail_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )[0]
    candidate_tail = evaluate_state(
        experimental_state,
        matched_tail_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )[0]
    baseline_tail_correct = exact_correct_count(
        r15_state,
        matched_tail_cache,
        batch_size=args.eval_batch_size,
    )
    candidate_tail_correct = exact_correct_count(
        experimental_state,
        matched_tail_cache,
        batch_size=args.eval_batch_size,
    )
    c2_zero_drop = exact_zero_drop(
        candidate_correct=candidate_tail_correct,
        baseline_correct=baseline_tail_correct,
    )

    _, baseline_structural = evaluate_state(
        r15_state,
        holdout_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )
    _, candidate_structural = evaluate_state(
        experimental_state,
        holdout_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )
    baseline_structural_ne = float(
        baseline_structural["non_entailment_accuracy"]
    )
    candidate_structural_ne = float(
        candidate_structural["non_entailment_accuracy"]
    )
    structural_gain = candidate_structural_ne - baseline_structural_ne

    final_non_relation = non_relation_bit_identical(
        r15_state,
        experimental_state,
        layout=layout,
    )
    parameter_count_unchanged = count_parameters(HIRACore()) == 422_159
    final_eligible = (
        len(prior_train) == 50000
        and basis_valid
        and train_selected
        and all_accepted_states_trust_safe
        and holdout_label_support_valid
        and len(matched_tail_indices) >= MIN_MATCHED_TAIL_N
        and matched_tail_disjoint
        and matched_tail_label_support_valid
        and structural_validation_disjoint
        and c1_zero_drop
        and c2_zero_drop
        and final_non_relation
        and parameter_count_unchanged
    )

    primary_pass = (
        final_eligible
        and candidate_structural_ne >= STRUCTURAL_FLOOR
        and structural_gain >= STRUCTURAL_GAIN_FLOOR
    )

    if final_eligible:
        persisted_state = experimental_state
        persisted_head_mode = "r24_train_selected_candidate"
    else:
        persisted_state = {
            key: value.detach().cpu().clone()
            for key, value in r15_state.items()
        }
        persisted_head_mode = "exact_r15_fallback"

    head_path = args.out / "hira-head.pt"
    torch.save(persisted_state, head_path)
    persisted_head_sha = file_sha256(head_path)

    cumulative_update = relation_delta(
        r15_state,
        experimental_state,
        layout=layout,
    )

    gates = {
        "prior_train_exact_50000": len(prior_train) == 50000,
        "r24_structural_exact_4000": len(structural_indices) == 4000,
        "r24_structural_shards_exact_16": (
            len(structural_round_indices) == 16
        ),
        "r24_trust_exact_6000": len(trust_indices) == 6000,
        "c1_holdout_exact_3000": len(holdout_indices) == 3000,
        "c1_holdout_label_support_valid": holdout_label_support_valid,
        "c2_matched_tail_n": len(matched_tail_indices),
        "c2_matched_tail_min_n_pass": (
            len(matched_tail_indices) >= MIN_MATCHED_TAIL_N
        ),
        "c2_matched_tail_disjoint": matched_tail_disjoint,
        "c2_matched_tail_label_support_valid": (
            matched_tail_label_support_valid
        ),
        "structural_validation_exact_500": (
            len(structural_val_indices) == 500
        ),
        "structural_validation_disjoint": structural_validation_disjoint,
        "probe_mean_teacher_kl": mean_probe_kl,
        "functional_rows_finite": functional_rows_finite,
        "functional_subspace_rank": subspace.rank,
        "functional_basis_orthonormal_error": (
            subspace.orthonormal_error
        ),
        "functional_basis_valid": basis_valid,
        "accepted_rounds": accepted_rounds,
        "rejected_rounds": R24_STRUCTURAL_ROUNDS - accepted_rounds,
        "train_selected": train_selected,
        "all_accepted_states_trust_safe": all_accepted_states_trust_safe,
        "c1_baseline_correct": baseline_holdout_correct,
        "c1_candidate_correct": candidate_holdout_correct,
        "c1_zero_drop": c1_zero_drop,
        "c2_baseline_correct": baseline_tail_correct,
        "c2_candidate_correct": candidate_tail_correct,
        "c2_zero_drop": c2_zero_drop,
        "final_non_relation_bit_identical": final_non_relation,
        "parameter_count_unchanged": parameter_count_unchanged,
        "final_eligible": final_eligible,
        "structural_floor": STRUCTURAL_FLOOR,
        "baseline_structural_non_entailment_accuracy": (
            baseline_structural_ne
        ),
        "candidate_structural_non_entailment_accuracy": (
            candidate_structural_ne
        ),
        "structural_gain": structural_gain,
        "structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
        "structural_floor_pass": (
            final_eligible and candidate_structural_ne >= STRUCTURAL_FLOOR
        ),
        "structural_gain_pass": (
            final_eligible and structural_gain >= STRUCTURAL_GAIN_FLOOR
        ),
        "primary_pass": primary_pass,
    }

    receipt = {
        "schema_version": "r24-sequential-relinearized-exact-trust-walk-v1",
        "status": "PASS",
        "sources": {
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
            "mnli_revision": MNLI_REVISION,
            "r15_head_sha256": R15_HEAD_SHA256,
            "r16_failure_analysis_head_sha256": R16_HEAD_SHA256,
        },
        "protocol": {
            "issue_authority": 44,
            "rounds": R24_STRUCTURAL_ROUNDS,
            "structural_examples_per_round": 250,
            "structural_examples_per_label_per_round": (
                R24_STRUCTURAL_PER_LABEL_PER_ROUND
            ),
            "micro_etas_descending": list(MICRO_ETAS),
            "trust_mean_kl_ceiling": TRUST_MEAN_KL_CEILING,
            "trust_exact_teacher_argmax": "6000/6000",
            "probe_beta": PROBE_BETA,
            "damage_rank": DAMAGE_RANK,
            "fixed_damage_basis": True,
            "relinearize_structural_gradient_each_round": True,
            "validation_used_for_walk_selection": False,
            "absolute_competence_floor": None,
            "final_competence_authorities": [
                "C1 fresh 3000-example train-heldout",
                "C2 untouched validation_matched shuffled tail 9000:end",
            ],
            "final_retention_authority": (
                "exact candidate correct count >= exact R15 correct count "
                "independently on C1 and C2"
            ),
            "structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
        },
        "data": {
            "prior_train_n": len(prior_train),
            "prior_index_hashes": {
                key: hash_indices(value) for key, value in prior.items()
            },
            "r24_structural_n": len(structural_indices),
            "r24_structural_index_sha256": hash_indices(structural_indices),
            "r24_structural_stats": structural_stats,
            "r24_structural_round_index_sha256": [
                hash_indices(shard) for shard in structural_round_indices
            ],
            "r24_trust_n": len(trust_indices),
            "r24_trust_index_sha256": hash_indices(trust_indices),
            "c1_holdout_n": len(holdout_indices),
            "c1_holdout_index_sha256": hash_indices(holdout_indices),
            "c1_holdout_label_counts": {
                str(k): int(v) for k, v in holdout_label_counts.items()
            },
            "c2_matched_tail_n": len(matched_tail_indices),
            "c2_matched_tail_index_sha256": hash_indices(
                matched_tail_indices
            ),
            "c2_matched_tail_label_counts": {
                str(k): int(v)
                for k, v in matched_tail_label_counts.items()
            },
            "prior_matched_index_sha256": hash_indices(
                prior_matched_indices
            ),
            "structural_validation_n": len(structural_val_indices),
            "structural_validation_index_sha256": hash_indices(
                structural_val_indices
            ),
            "structural_validation_stats": structural_val_stats,
        },
        "functional_damage": {
            "probe_meta": functional_meta,
            "functional_subspace_rank": subspace.rank,
            "functional_subspace_orthonormal_error": (
                subspace.orthonormal_error
            ),
            "original_relation_delta_l2": float(delta.norm()),
        },
        "rounds": round_receipts,
        "walk_summary": {
            "accepted_rounds": accepted_rounds,
            "rejected_rounds": R24_STRUCTURAL_ROUNDS - accepted_rounds,
            "cumulative_update_l2": float(cumulative_update.norm()),
        },
        "final": {
            "c1": {
                "baseline": baseline_holdout,
                "candidate": candidate_holdout,
                "baseline_correct": baseline_holdout_correct,
                "candidate_correct": candidate_holdout_correct,
                "zero_drop": c1_zero_drop,
            },
            "c2": {
                "baseline": baseline_tail,
                "candidate": candidate_tail,
                "baseline_correct": baseline_tail_correct,
                "candidate_correct": candidate_tail_correct,
                "zero_drop": c2_zero_drop,
            },
            "structural": {
                "baseline": baseline_structural,
                "candidate": candidate_structural,
                "gain": structural_gain,
            },
            "final_eligible": final_eligible,
        },
        "persisted_head_mode": persisted_head_mode,
        "persisted_head_sha256": persisted_head_sha,
        "gates": gates,
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "PASS",
        "primary_pass": primary_pass,
        "accepted_rounds": accepted_rounds,
        "cumulative_update_l2": float(cumulative_update.norm()),
        "c1_zero_drop": c1_zero_drop,
        "c2_zero_drop": c2_zero_drop,
        "baseline_structural": baseline_structural_ne,
        "candidate_structural": candidate_structural_ne,
        "structural_gain": structural_gain,
        "final_eligible": final_eligible,
        "persisted_head_mode": persisted_head_mode,
        "persisted_head_sha256": persisted_head_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
