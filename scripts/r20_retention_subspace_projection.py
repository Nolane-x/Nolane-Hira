from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
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
from nmd.hard_negative import anti_entailment_margin_loss, evaluate_repair_slice
from nmd.hira import HIRACore, count_parameters
from nmd.relation_cache import RelationCache, forward_cached, minibatches
from nmd.semantic import HFAutoSemanticEncoder
from nmd.subspace_projection import (
    apply_relation_update,
    build_retention_subspace,
    non_relation_bit_identical,
    projection_metrics,
    rank_for_coverage,
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
R20_STRUCTURAL_START = 2000
R20_STRUCTURAL_PER_LABEL = 2000
R20_RETENTION_N = 6000
R20_RETENTION_SHUFFLE_SEED = 17

R18_STRUCTURAL_START = 0
R19_STRUCTURAL_VAL_START = 250
R20_STRUCTURAL_VAL_START = 500
STRUCTURAL_VAL_PER_LABEL = 250

R20_MATCHED_START = 3000
MATCHED_VAL_N = 1500
MATCHED_SHUFFLE_SEED = 14

MIN_HYPOTHESIS_TOKENS = 3
GRAD_BATCH_SIZE = 96
COVERAGE_TARGETS = (0.50, 0.75, 0.90)
REMOVAL_STRENGTHS = (0.50, 1.00)
ALPHAS = (0.25, 0.50, 1.00)

BASELINE_MATCHED_FLOOR = 0.53
MATCHED_RELATIVE_TOLERANCE = 0.002
MIN_MATCHED_LABEL_COUNT = 400
STRUCTURAL_FLOOR = 0.60
STRUCTURAL_GAIN_FLOOR = 0.02
REMOVED_ENERGY_FLOOR = 0.05
ORTHONORMAL_TOLERANCE = 1e-7
RELATIVE_EIGENVALUE_FLOOR = 1e-10

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


def collect_retention_gradient_rows(
    model: HIRACore,
    cache: RelationCache,
    *,
    batch_size: int,
    layout,
) -> tuple[torch.Tensor, dict[str, float]]:
    params = relation_parameter_map(model)
    rows: list[torch.Tensor] = []
    seen = 0
    loss_weighted = 0.0
    total_n = len(cache)
    model.eval()

    for idx in minibatches(
        total_n, batch_size, seed=0, epoch=0, shuffle=False
    ):
        model.zero_grad(set_to_none=True)
        out = forward_cached(model, cache, idx)
        labels = cache.labels[idx].long()
        loss = F.cross_entropy(out.logits, labels)
        loss.backward()
        n = int(len(idx))
        seen += n
        loss_weighted += float(loss.detach().cpu()) * n

        grad_map: dict[str, torch.Tensor] = {}
        for name, param in params.items():
            if param.grad is None:
                raise RuntimeError(f"missing retention gradient for {name}")
            grad_map[name] = param.grad.detach().cpu()
        flat = flatten_relation_named_tensors(grad_map, layout=layout)
        rows.append(flat * math.sqrt(n / total_n))

    if seen != total_n:
        raise RuntimeError("retention gradient pass incomplete")
    matrix = torch.stack(rows).to(torch.float64)
    return matrix, {
        "n": float(seen),
        "batch_count": float(matrix.shape[0]),
        "mean_loss": loss_weighted / max(1, seen),
        "mode": "sqrt_sample_weighted_batch_mean_gradient_rows",
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


def exact_r19_retention_indices(train_full, r19_structural_indices: list[int]) -> list[int]:
    indexed = train_full.add_column("_source_index", list(range(len(train_full))))
    shuffled = indexed.shuffle(seed=13)
    excluded = set(r19_structural_indices)
    result: list[int] = []
    for raw in shuffled["_source_index"]:
        i = int(raw)
        if i in excluded:
            continue
        result.append(i)
        if len(result) == R19_RETENTION_N:
            break
    if len(result) != R19_RETENTION_N:
        raise RuntimeError("failed to reconstruct exact R19 retention set")
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

    snapshot = Path(snapshot_download(repo_id=A13_MODEL, revision=A13_REVISION))
    if file_sha256(snapshot / "model.safetensors") != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base_encoder = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
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

    # Reconstruct R19 train scoring authorities.
    r19_structural_indices, _ = balanced_ranked_structural_window_indices(
        train_full["premise"],
        train_full["hypothesis"],
        train_full["label"],
        start_per_label=0,
        count_per_label=R19_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
    )
    r19_retention_indices = exact_r19_retention_indices(
        train_full, r19_structural_indices
    )

    # R20 structural utility ranks 2001-4000 per label.
    structural_train_indices, structural_train_stats = (
        balanced_ranked_structural_window_indices(
            train_full["premise"],
            train_full["hypothesis"],
            train_full["label"],
            start_per_label=R20_STRUCTURAL_START,
            count_per_label=R20_STRUCTURAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        )
    )
    if len(structural_train_indices) != 4000:
        raise RuntimeError("R20 structural train must be exactly 4000")

    r19_scored_train = set(r19_structural_indices) | set(r19_retention_indices)
    r20_structural_set = set(structural_train_indices)
    structural_disjoint_from_r19 = r20_structural_set.isdisjoint(r19_scored_train)
    if not structural_disjoint_from_r19:
        raise RuntimeError("R20 structural train overlaps R19 scored train")

    # R20 retention set is disjoint from all R19 scored train and R20 structural.
    indexed_train = train_full.add_column(
        "_source_index", list(range(len(train_full)))
    )
    shuffled_r20 = indexed_train.shuffle(seed=R20_RETENTION_SHUFFLE_SEED)
    forbidden_train = r19_scored_train | r20_structural_set
    retention_indices: list[int] = []
    for raw in shuffled_r20["_source_index"]:
        i = int(raw)
        if i in forbidden_train:
            continue
        retention_indices.append(i)
        if len(retention_indices) == R20_RETENTION_N:
            break
    if len(retention_indices) != R20_RETENTION_N:
        raise RuntimeError("R20 retention train must be exactly 6000")
    retention_set = set(retention_indices)
    train_sets_disjoint = (
        retention_set.isdisjoint(r20_structural_set)
        and retention_set.isdisjoint(r19_scored_train)
        and structural_disjoint_from_r19
    )
    if not train_sets_disjoint:
        raise RuntimeError("R20 train disjointness invariant failed")

    # Validation structural windows: R18 [0:250], R19 [250:500], R20 [500:750].
    r18_structural_indices, _ = balanced_ranked_structural_window_indices(
        mismatched_full["premise"],
        mismatched_full["hypothesis"],
        mismatched_full["label"],
        start_per_label=R18_STRUCTURAL_START,
        count_per_label=STRUCTURAL_VAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
    )
    r19_structural_val_indices, _ = balanced_ranked_structural_window_indices(
        mismatched_full["premise"],
        mismatched_full["hypothesis"],
        mismatched_full["label"],
        start_per_label=R19_STRUCTURAL_VAL_START,
        count_per_label=STRUCTURAL_VAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
    )
    structural_val_indices, structural_val_stats = (
        balanced_ranked_structural_window_indices(
            mismatched_full["premise"],
            mismatched_full["hypothesis"],
            mismatched_full["label"],
            start_per_label=R20_STRUCTURAL_VAL_START,
            count_per_label=STRUCTURAL_VAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        )
    )
    prior_structural_val = set(r18_structural_indices) | set(r19_structural_val_indices)
    structural_validation_disjoint = set(structural_val_indices).isdisjoint(
        prior_structural_val
    )
    if len(structural_val_indices) != 500 or not structural_validation_disjoint:
        raise RuntimeError("R20 structural validation authority invalid")

    # Matched validation window positions 3000:4500 in exact seed-14 shuffle.
    indexed_matched = matched_full.add_column(
        "_source_index", list(range(len(matched_full)))
    )
    shuffled_matched = indexed_matched.shuffle(seed=MATCHED_SHUFFLE_SEED)
    stop = R20_MATCHED_START + MATCHED_VAL_N
    if len(shuffled_matched) < stop:
        raise RuntimeError("not enough matched validation rows for R20")
    prior_matched_indices = [
        int(x) for x in shuffled_matched["_source_index"][:R20_MATCHED_START]
    ]
    matched_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][
            R20_MATCHED_START:stop
        ]
    ]
    matched_validation_disjoint = set(prior_matched_indices).isdisjoint(
        matched_indices
    )
    if len(matched_indices) != MATCHED_VAL_N or not matched_validation_disjoint:
        raise RuntimeError("R20 matched validation authority invalid")

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
            "role": "r20-train-structural-ranks-2001-4000",
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
            "role": "r20-train-retention-subspace",
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
            "role": "r20-ranked-structural-validation-ranks-501-750",
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
            "role": "r20-matched-validation-window-3000-4500",
            "n": len(matched_val),
            "index_sha256": hash_indices(matched_indices),
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

    scoring_model = HIRACore(d_model=256, dropout=0.0)
    scoring_model.load_state_dict(r15_state, strict=True)
    structural_grad_map, structural_grad_meta = collect_structural_gradient(
        scoring_model,
        structural_train_cache,
        batch_size=GRAD_BATCH_SIZE,
    )
    structural_gradient = flatten_relation_named_tensors(
        structural_grad_map, layout=layout
    )

    scoring_model.load_state_dict(r15_state, strict=True)
    retention_rows, retention_rows_meta = collect_retention_gradient_rows(
        scoring_model,
        retention_train_cache,
        batch_size=GRAD_BATCH_SIZE,
        layout=layout,
    )
    subspace = build_retention_subspace(
        retention_rows,
        relative_eigenvalue_floor=RELATIVE_EIGENVALUE_FLOOR,
    )
    if subspace.orthonormal_error > ORTHONORMAL_TOLERANCE:
        raise RuntimeError(
            f"R20 retention basis orthonormal error too high: "
            f"{subspace.orthonormal_error}"
        )

    coverage_meta: dict[str, dict[str, float | int]] = {}
    coverage_ranks: dict[float, int] = {}
    coverage_achieved = True
    for target in COVERAGE_TARGETS:
        rank, realized = rank_for_coverage(subspace, target)
        coverage_ranks[target] = rank
        coverage_meta[str(target)] = {
            "rank": rank,
            "realized_coverage": realized,
        }
        coverage_achieved = coverage_achieved and realized + 1e-12 >= target

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
    label_counts = Counter(int(x) for x in matched_val["label"])
    label_support_valid = all(
        label_counts.get(label, 0) >= MIN_MATCHED_LABEL_COUNT
        for label in (0, 1, 2)
    )
    baseline_validity = (
        len(matched_indices) == MATCHED_VAL_N
        and label_support_valid
        and baseline_matched_accuracy >= BASELINE_MATCHED_FLOOR
    )
    relative_floor = baseline_matched_accuracy - MATCHED_RELATIVE_TOLERANCE

    candidates: list[dict[str, object]] = []
    candidate_states: dict[
        tuple[float, float, float], dict[str, torch.Tensor]
    ] = {}

    for coverage in COVERAGE_TARGETS:
        rank = coverage_ranks[coverage]
        for removal_strength in REMOVAL_STRENGTHS:
            rotated, projected = rotate_delta_away_from_retention(
                delta,
                subspace,
                rank=rank,
                removal_strength=removal_strength,
            )
            geometry = projection_metrics(
                delta,
                rotated,
                structural_gradient,
            )
            projected_fraction = float(
                projected.square().sum() / delta.square().sum()
            )
            for alpha in ALPHAS:
                state = apply_relation_update(
                    r15_state,
                    rotated,
                    alpha=alpha,
                    layout=layout,
                )
                non_relation_ok = non_relation_bit_identical(
                    r15_state, state, layout=layout
                )
                if not non_relation_ok:
                    raise RuntimeError(
                        f"non-relation state changed for "
                        f"{coverage}/{removal_strength}/{alpha}"
                    )
                matched_metrics, structural_metrics = evaluate_state(
                    state,
                    matched_val_cache,
                    structural_val_cache,
                    batch_size=args.eval_batch_size,
                )
                matched_accuracy = float(matched_metrics["accuracy"])
                eligible = (
                    baseline_validity
                    and matched_accuracy >= BASELINE_MATCHED_FLOOR
                    and matched_accuracy >= relative_floor
                    and float(
                        geometry["first_order_structural_predicted_benefit"]
                    ) > 0.0
                    and float(
                        geometry["removed_delta_energy_fraction"]
                    ) > 0.0
                )
                key = (coverage, removal_strength, alpha)
                candidate_states[key] = state
                row = {
                    "coverage_target": coverage,
                    "retention_rank": rank,
                    "realized_coverage": coverage_meta[str(coverage)][
                        "realized_coverage"
                    ],
                    "removal_strength": removal_strength,
                    "alpha": alpha,
                    "projected_component_energy_fraction": projected_fraction,
                    "applied_update_l2": (
                        alpha * float(geometry["retained_delta_l2"])
                    ),
                    "matched": matched_metrics,
                    "ranked_structural": structural_metrics,
                    "eligible": eligible,
                    "non_relation_bit_identical": non_relation_ok,
                    **geometry,
                }
                candidates.append(row)

    eligible_candidates = [row for row in candidates if bool(row["eligible"])]
    selected_eligible = bool(eligible_candidates)
    if selected_eligible:
        selected_row = min(
            eligible_candidates,
            key=lambda row: (
                -float(
                    row["ranked_structural"][
                        "non_entailment_accuracy"
                    ]
                ),
                -float(row["matched"]["accuracy"]),
                float(row["applied_update_l2"]),
                float(row["alpha"]),
                -float(row["removal_strength"]),
                float(row["coverage_target"]),
            ),
        )
        selected_key = (
            float(selected_row["coverage_target"]),
            float(selected_row["removal_strength"]),
            float(selected_row["alpha"]),
        )
        selected_state = candidate_states[selected_key]
    else:
        selected_row = None
        selected_state = {
            key: value.detach().cpu().clone()
            for key, value in r15_state.items()
        }

    head_path = args.out / "hira-head.pt"
    torch.save(selected_state, head_path)
    selected_sha = file_sha256(head_path)

    if selected_row is None:
        selected_matched = baseline_matched_accuracy
        selected_structural = baseline_structural_ne
        selected_removed_energy = 0.0
        selected_non_relation = True
    else:
        selected_matched = float(selected_row["matched"]["accuracy"])
        selected_structural = float(
            selected_row["ranked_structural"][
                "non_entailment_accuracy"
            ]
        )
        selected_removed_energy = float(
            selected_row["removed_delta_energy_fraction"]
        )
        selected_non_relation = bool(
            selected_row["non_relation_bit_identical"]
        )
    structural_gain = selected_structural - baseline_structural_ne

    gates = {
        "structural_train_exact_4000": len(structural_train_indices) == 4000,
        "retention_train_exact_6000": len(retention_indices) == 6000,
        "train_sets_disjoint_from_each_other_and_r19": train_sets_disjoint,
        "retention_subspace_numerical_rank": subspace.rank,
        "retention_basis_orthonormal_error": subspace.orthonormal_error,
        "retention_basis_orthonormal_pass": (
            subspace.orthonormal_error <= ORTHONORMAL_TOLERANCE
        ),
        "coverage_achieved": coverage_achieved,
        "structural_validation_exact_500": len(structural_val_indices) == 500,
        "structural_validation_disjoint_from_r18_r19": (
            structural_validation_disjoint
        ),
        "matched_validation_exact_1500": len(matched_indices) == 1500,
        "matched_validation_disjoint_from_prior": (
            matched_validation_disjoint
        ),
        "matched_label_counts": {
            str(label): int(label_counts.get(label, 0))
            for label in (0, 1, 2)
        },
        "matched_label_support_valid": label_support_valid,
        "baseline_matched_floor": BASELINE_MATCHED_FLOOR,
        "baseline_validity": baseline_validity,
        "selected_eligible": selected_eligible,
        "matched_relative_floor": relative_floor,
        "matched_absolute_pass": (
            selected_eligible and selected_matched >= BASELINE_MATCHED_FLOOR
        ),
        "matched_relative_pass": (
            selected_eligible and selected_matched >= relative_floor
        ),
        "ranked_structural_floor": STRUCTURAL_FLOOR,
        "ranked_structural_pass": (
            selected_eligible and selected_structural >= STRUCTURAL_FLOOR
        ),
        "ranked_structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
        "ranked_structural_gain": structural_gain,
        "ranked_structural_gain_pass": (
            selected_eligible and structural_gain >= STRUCTURAL_GAIN_FLOOR
        ),
        "removed_delta_energy_floor": REMOVED_ENERGY_FLOOR,
        "selected_removed_delta_energy_fraction": selected_removed_energy,
        "removed_delta_energy_pass": (
            selected_eligible
            and selected_removed_energy >= REMOVED_ENERGY_FLOOR
        ),
        "non_relation_bit_identical": selected_non_relation,
        "parameter_count_unchanged": count_parameters(HIRACore()) == 422_159,
    }
    gates["primary_pass"] = all(
        [
            gates["structural_train_exact_4000"],
            gates["retention_train_exact_6000"],
            gates["train_sets_disjoint_from_each_other_and_r19"],
            gates["retention_basis_orthonormal_pass"],
            gates["coverage_achieved"],
            gates["structural_validation_exact_500"],
            gates["structural_validation_disjoint_from_r18_r19"],
            gates["matched_validation_exact_1500"],
            gates["matched_validation_disjoint_from_prior"],
            gates["matched_label_support_valid"],
            gates["baseline_validity"],
            gates["selected_eligible"],
            gates["matched_absolute_pass"],
            gates["matched_relative_pass"],
            gates["ranked_structural_pass"],
            gates["ranked_structural_gain_pass"],
            gates["removed_delta_energy_pass"],
            gates["non_relation_bit_identical"],
            gates["parameter_count_unchanged"],
        ]
    )

    receipt = {
        "schema_version": "r20-retention-subspace-projection-v1",
        "status": "PASS",
        "evidence_scope": (
            "train-only retention-subspace projection plus new disjoint "
            "MultiNLI development validation; no HANS/Breaking/XNLI/MASSIVE/"
            "Banking77/Laya/Jev model selection"
        ),
        "sources": {
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
            "mnli_revision": MNLI_REVISION,
            "r15_head_sha256": R15_HEAD_SHA256,
            "r16_failure_analysis_head_sha256": R16_HEAD_SHA256,
        },
        "protocol": {
            "r20_structural_train_window_per_label": [
                R20_STRUCTURAL_START,
                R20_STRUCTURAL_START + R20_STRUCTURAL_PER_LABEL,
            ],
            "r20_retention_shuffle_seed": R20_RETENTION_SHUFFLE_SEED,
            "r20_retention_n": R20_RETENTION_N,
            "structural_validation_window_per_label": [
                R20_STRUCTURAL_VAL_START,
                R20_STRUCTURAL_VAL_START + STRUCTURAL_VAL_PER_LABEL,
            ],
            "matched_validation_window": [
                R20_MATCHED_START,
                R20_MATCHED_START + MATCHED_VAL_N,
            ],
            "matched_shuffle_seed": MATCHED_SHUFFLE_SEED,
            "coverage_targets": list(COVERAGE_TARGETS),
            "removal_strengths": list(REMOVAL_STRENGTHS),
            "alphas": list(ALPHAS),
            "baseline_matched_floor": BASELINE_MATCHED_FLOOR,
            "matched_relative_tolerance": MATCHED_RELATIVE_TOLERANCE,
            "min_matched_label_count": MIN_MATCHED_LABEL_COUNT,
            "structural_floor": STRUCTURAL_FLOOR,
            "structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
            "removed_energy_floor": REMOVED_ENERGY_FLOOR,
            "orthonormal_tolerance": ORTHONORMAL_TOLERANCE,
            "relative_eigenvalue_floor": RELATIVE_EIGENVALUE_FLOOR,
            "selection_order": [
                "eligible absolute and baseline-relative retention",
                "maximize ranked-structural non-entailment accuracy",
                "tie-break by matched accuracy",
                "prefer smaller applied update L2",
                "prefer smaller alpha",
                "prefer larger removal strength",
                "prefer lower coverage target",
            ],
        },
        "data": {
            "r19_structural_train_index_sha256": hash_indices(
                r19_structural_indices
            ),
            "r19_retention_train_index_sha256": hash_indices(
                r19_retention_indices
            ),
            "structural_train_n": len(structural_train_indices),
            "structural_train_index_sha256": hash_indices(
                structural_train_indices
            ),
            "structural_train_stats": structural_train_stats,
            "retention_train_n": len(retention_indices),
            "retention_train_index_sha256": hash_indices(retention_indices),
            "train_sets_disjoint_from_each_other_and_r19": train_sets_disjoint,
            "prior_structural_validation_index_sha256": hash_indices(
                sorted(prior_structural_val)
            ),
            "structural_validation_n": len(structural_val_indices),
            "structural_validation_index_sha256": hash_indices(
                structural_val_indices
            ),
            "structural_validation_stats": structural_val_stats,
            "structural_validation_disjoint_from_r18_r19": (
                structural_validation_disjoint
            ),
            "prior_matched_index_sha256": hash_indices(prior_matched_indices),
            "matched_validation_n": len(matched_indices),
            "matched_validation_index_sha256": hash_indices(matched_indices),
            "matched_validation_disjoint_from_prior": (
                matched_validation_disjoint
            ),
        },
        "gradient_evidence": {
            "structural_gradient_l2": float(structural_gradient.norm()),
            "structural_gradient_meta": structural_grad_meta,
            "retention_rows_meta": retention_rows_meta,
            "retention_subspace_rank": subspace.rank,
            "retention_subspace_orthonormal_error": (
                subspace.orthonormal_error
            ),
            "coverage": coverage_meta,
            "original_relation_delta_l2": float(delta.norm()),
            "original_first_order_structural_predicted_benefit": float(
                -torch.dot(structural_gradient, delta)
            ),
        },
        "baseline": {
            "matched": baseline_matched,
            "ranked_structural": baseline_structural,
            "matched_label_counts": {
                str(label): int(label_counts.get(label, 0))
                for label in (0, 1, 2)
            },
            "validity": baseline_validity,
        },
        "candidates": candidates,
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
                "baseline_validity": baseline_validity,
                "baseline_matched_accuracy": baseline_matched_accuracy,
                "baseline_ranked_structural": baseline_structural_ne,
                "retention_subspace_rank": subspace.rank,
                "selected": selected_row,
                "ranked_structural_gain": structural_gain,
                "selected_head_sha256": selected_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
