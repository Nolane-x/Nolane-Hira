from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

import r21_functional_kl_damage_subspace as r21

from nmd.delta_surgery import (
    balanced_ranked_structural_window_indices,
    flatten_relation_named_tensors,
    relation_delta,
    relation_layout,
)
from nmd.functional_damage import (
    finite_sample_relative_accuracy_pass,
    teacher_kl_loss,
)
from nmd.hira import HIRACore, count_parameters
from nmd.nullspace_augmentation import (
    combine_updates,
    first_order_structural_benefit,
    normalize_like,
    normalized_subspace_leakage,
    nullspace_component,
)
from nmd.relation_cache import RelationCache, forward_cached, minibatches
from nmd.semantic import HFAutoSemanticEncoder
from nmd.subspace_projection import (
    apply_relation_update,
    build_retention_subspace,
    non_relation_bit_identical,
    project_onto_retention_subspace,
)


A13_MODEL = r21.A13_MODEL
A13_REVISION = r21.A13_REVISION
A13_WEIGHT_SHA256 = r21.A13_WEIGHT_SHA256
MNLI_DATASET = r21.MNLI_DATASET
MNLI_REVISION = r21.MNLI_REVISION

R15_HEAD_SHA256 = r21.R15_HEAD_SHA256
R16_HEAD_SHA256 = r21.R16_HEAD_SHA256

R22_STRUCTURAL_PER_LABEL = 2000
R22_RETENTION_N = 6000
R22_RETENTION_SEED = 29

STRUCTURAL_VAL_PER_LABEL = 250
R22_STRUCTURAL_VAL_START = 1000
MATCHED_SHUFFLE_SEED = 14
R22_MATCHED_START = 6000
MATCHED_VAL_N = 1500

MIN_HYPOTHESIS_TOKENS = r21.MIN_HYPOTHESIS_TOKENS
GRAD_BATCH_SIZE = r21.GRAD_BATCH_SIZE

PROBE_BETA = 0.25
FUNCTIONAL_RANK = 32
PROTECTION_REMOVAL = 0.50
TRUST_ANCHOR_ALPHA = 0.25
ALPHAS = (0.0, 0.125, 0.25)
GAMMAS = (0.03125, 0.0625, 0.125, 0.25)
CONTROL_ALPHAS = (0.125, 0.25)

PROBE_KL_FLOOR = 1e-4
MIN_FUNCTIONAL_RANK = 32
RELATIVE_EIGENVALUE_FLOOR = 1e-10
ORTHONORMAL_TOLERANCE = 1e-7
MECHANISM_PROJECTED_ENERGY_FLOOR = 0.005
NULLSPACE_LEAKAGE_TOLERANCE = 1e-6

BASELINE_MATCHED_FLOOR = 0.53
MATCHED_RELATIVE_TOLERANCE = 0.002
MIN_MATCHED_LABEL_COUNT = 400
STRUCTURAL_FLOOR = 0.60
STRUCTURAL_GAIN_FLOOR = 0.02

OPTION_TEXTS = r21.OPTION_TEXTS


def collect_structural_ce_gradient(
    model: HIRACore,
    cache: RelationCache,
    *,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    params = r21.relation_parameter_map(model)
    sums = {
        name: torch.zeros_like(param, dtype=torch.float64, device="cpu")
        for name, param in params.items()
    }
    seen = 0
    loss_weighted = 0.0
    model.eval()

    for idx in minibatches(
        len(cache),
        batch_size,
        seed=0,
        epoch=0,
        shuffle=False,
    ):
        model.zero_grad(set_to_none=True)
        out = forward_cached(model, cache, idx)
        labels = cache.labels[idx].long()
        loss = F.cross_entropy(out.logits, labels)
        loss.backward()

        n = int(len(idx))
        seen += n
        loss_weighted += float(loss.detach().cpu()) * n
        for name, param in params.items():
            if param.grad is None:
                raise RuntimeError(f"missing R22 structural CE gradient for {name}")
            sums[name].add_(
                param.grad.detach().cpu().to(torch.float64),
                alpha=float(n),
            )

    if seen != len(cache):
        raise RuntimeError("R22 structural CE gradient pass incomplete")
    for name in sums:
        sums[name].div_(float(seen))

    return sums, {
        "n": float(seen),
        "mean_loss": loss_weighted / max(1, seen),
        "mode": "balanced_structural_mean_cross_entropy_gradient_at_exact_r15",
    }


@torch.inference_mode()
def mean_teacher_kl(
    model: HIRACore,
    cache: RelationCache,
    teacher_probabilities: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    model.eval()
    total = len(cache)
    seen = 0
    weighted = 0.0
    for idx in minibatches(
        total,
        batch_size,
        seed=0,
        epoch=0,
        shuffle=False,
    ):
        out = forward_cached(model, cache, idx)
        teacher = teacher_probabilities[idx].to(out.logits.dtype)
        loss = teacher_kl_loss(out.logits, teacher)
        n = int(len(idx))
        seen += n
        weighted += float(loss.detach().cpu()) * n
    if seen != total:
        raise RuntimeError("R22 teacher-KL evaluation incomplete")
    return weighted / max(1, seen)


def state_teacher_kl(
    state: dict[str, torch.Tensor],
    cache: RelationCache,
    teacher_probabilities: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    model = HIRACore(d_model=256, dropout=0.0)
    model.load_state_dict(state, strict=True)
    return mean_teacher_kl(
        model,
        cache,
        teacher_probabilities,
        batch_size=batch_size,
    )


def reconstruct_prior_train(train_full) -> dict[str, list[int]]:
    prior = r21.reconstruct_prior_train(train_full)
    prior20 = set().union(*(set(v) for v in prior.values()))
    if len(prior20) != 20000:
        raise RuntimeError("R19/R20 scored-train union must be exactly 20000")

    r21_structural, _ = balanced_ranked_structural_window_indices(
        train_full["premise"],
        train_full["hypothesis"],
        train_full["label"],
        start_per_label=0,
        count_per_label=r21.R21_STRUCTURAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        exclude_indices=sorted(prior20),
    )
    r21_structural_set = set(r21_structural)
    if not r21_structural_set.isdisjoint(prior20):
        raise RuntimeError("reconstructed R21 structural train overlaps prior")

    r21_retention = r21.random_exclusion_sample(
        train_full,
        seed=r21.R21_RETENTION_SEED,
        n=r21.R21_RETENTION_N,
        excluded=prior20 | r21_structural_set,
    )

    result = {
        **prior,
        "r21_structural": r21_structural,
        "r21_retention": r21_retention,
    }
    union = set().union(*(set(v) for v in result.values()))
    if len(union) != 30000:
        raise RuntimeError(
            f"R19/R20/R21 scored-train union must be exactly 30000, got {len(union)}"
        )
    return result


def correct_count(accuracy: float, n: int) -> int:
    return int(round(float(accuracy) * int(n)))


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

    if r21.file_sha256(args.r15_head) != R15_HEAD_SHA256:
        raise RuntimeError("R15 selected head SHA-256 mismatch")
    if r21.file_sha256(args.r16_head) != R16_HEAD_SHA256:
        raise RuntimeError("R16 failure-analysis head SHA-256 mismatch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=A13_MODEL,
            revision=A13_REVISION,
        )
    )
    if r21.file_sha256(snapshot / "model.safetensors") != A13_WEIGHT_SHA256:
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
    for parameter in base_encoder.parameters():
        parameter.requires_grad_(False)

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
    if len(prior_train) != 30000:
        raise RuntimeError("R22 prior scored-train authority must be 30000")

    structural_train_indices, structural_train_stats = (
        balanced_ranked_structural_window_indices(
            train_full["premise"],
            train_full["hypothesis"],
            train_full["label"],
            start_per_label=0,
            count_per_label=R22_STRUCTURAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
            exclude_indices=sorted(prior_train),
        )
    )
    if len(structural_train_indices) != 4000:
        raise RuntimeError("R22 structural train must be exactly 4000")

    structural_train_set = set(structural_train_indices)
    if not structural_train_set.isdisjoint(prior_train):
        raise RuntimeError("R22 structural train overlaps prior scored train")

    retention_indices = r21.random_exclusion_sample(
        train_full,
        seed=R22_RETENTION_SEED,
        n=R22_RETENTION_N,
        excluded=prior_train | structural_train_set,
    )
    retention_set = set(retention_indices)
    train_sets_disjoint = (
        retention_set.isdisjoint(prior_train)
        and retention_set.isdisjoint(structural_train_set)
        and structural_train_set.isdisjoint(prior_train)
    )
    if not train_sets_disjoint:
        raise RuntimeError("R22 train disjointness invariant failed")

    prior_structural_validation: set[int] = set()
    for start in (0, 250, 500, 750):
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
            start_per_label=R22_STRUCTURAL_VAL_START,
            count_per_label=STRUCTURAL_VAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        )
    )
    structural_validation_disjoint = set(structural_val_indices).isdisjoint(
        prior_structural_validation
    )
    if len(structural_val_indices) != 500:
        raise RuntimeError("R22 structural validation must be exactly 500")
    if not structural_validation_disjoint:
        raise RuntimeError("R22 structural validation overlaps prior windows")

    indexed_matched = matched_full.add_column(
        "_source_index",
        list(range(len(matched_full))),
    )
    shuffled_matched = indexed_matched.shuffle(seed=MATCHED_SHUFFLE_SEED)
    matched_stop = R22_MATCHED_START + MATCHED_VAL_N
    if len(shuffled_matched) < matched_stop:
        raise RuntimeError("not enough matched validation rows for R22")

    prior_matched_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][:R22_MATCHED_START]
    ]
    matched_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][
            R22_MATCHED_START:matched_stop
        ]
    ]
    matched_validation_disjoint = set(prior_matched_indices).isdisjoint(
        matched_indices
    )
    if len(matched_indices) != MATCHED_VAL_N:
        raise RuntimeError("R22 matched validation must be exactly 1500")
    if not matched_validation_disjoint:
        raise RuntimeError("R22 matched validation overlaps prior windows")

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

    structural_train_cache = r21.encode_dataset(
        structural_train,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r22-train-structural-exclusion-first-top-2000-per-label",
            "n": len(structural_train),
            "index_sha256": r21.hash_indices(structural_train_indices),
        },
    )
    retention_train_cache = r21.encode_dataset(
        retention_train,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r22-train-functional-teacher-kl-trust-region",
            "n": len(retention_train),
            "index_sha256": r21.hash_indices(retention_indices),
        },
    )
    structural_val_cache = r21.encode_dataset(
        structural_val,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r22-ranked-structural-validation-ranks-1001-1250",
            "n": len(structural_val),
            "index_sha256": r21.hash_indices(structural_val_indices),
        },
    )
    matched_val_cache = r21.encode_dataset(
        matched_val,
        encoder,
        option_embeddings,
        batch_size=args.encode_batch_size,
        n_segments=args.segments,
        metadata={
            **common,
            "role": "r22-matched-validation-window-6000-7500",
            "n": len(matched_val),
            "index_sha256": r21.hash_indices(matched_indices),
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
    structural_grad_map, structural_grad_meta = collect_structural_ce_gradient(
        structural_model,
        structural_train_cache,
        batch_size=GRAD_BATCH_SIZE,
    )
    structural_gradient = flatten_relation_named_tensors(
        structural_grad_map,
        layout=layout,
    ).to(torch.float64)

    teacher_model = HIRACore(d_model=256, dropout=0.0)
    teacher_model.load_state_dict(r15_state, strict=True)
    teacher_probabilities = r21.frozen_teacher_probabilities(
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
        raise RuntimeError("R22 probe changed non-relation state")

    probe_model = HIRACore(d_model=256, dropout=0.0)
    probe_model.load_state_dict(probe_state, strict=True)
    functional_rows, functional_meta = r21.collect_functional_kl_gradient_rows(
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

    mean_probe_kl = float(functional_meta["mean_teacher_kl"])
    basis_orthonormal_pass = (
        subspace.orthonormal_error <= ORTHONORMAL_TOLERANCE
    )

    rank32_projected_energy_fraction = 0.0
    if subspace.rank >= FUNCTIONAL_RANK:
        projected_delta = project_onto_retention_subspace(
            delta,
            subspace,
            rank=FUNCTIONAL_RANK,
        )
        rank32_projected_energy_fraction = float(
            projected_delta.square().sum() / delta.square().sum()
        )
    else:
        projected_delta = torch.zeros_like(delta, dtype=torch.float64)

    mechanism_validity = (
        mean_probe_kl > PROBE_KL_FLOOR
        and functional_rows_finite
        and subspace.rank >= MIN_FUNCTIONAL_RANK
        and basis_orthonormal_pass
        and rank32_projected_energy_fraction
        >= MECHANISM_PROJECTED_ENERGY_FLOOR
    )

    protected = (
        delta.detach().cpu().to(torch.float64)
        - PROTECTION_REMOVAL * projected_delta
    )

    if subspace.rank >= FUNCTIONAL_RANK:
        structural_null_gradient, structural_damage_component = (
            nullspace_component(
                structural_gradient,
                subspace,
                rank=FUNCTIONAL_RANK,
            )
        )
        null_raw = -structural_null_gradient
        null_direction = normalize_like(null_raw, protected)
        nullspace_leakage = normalized_subspace_leakage(
            null_raw,
            subspace,
            rank=FUNCTIONAL_RANK,
        )
    else:
        structural_damage_component = torch.zeros_like(structural_gradient)
        null_raw = torch.zeros_like(structural_gradient)
        null_direction = torch.zeros_like(structural_gradient)
        nullspace_leakage = float("inf")

    structural_gradient_finite = bool(torch.isfinite(structural_gradient).all())
    null_raw_finite = bool(torch.isfinite(null_raw).all())
    structural_gradient_l2 = float(structural_gradient.norm())
    null_raw_l2 = float(null_raw.norm())
    null_direction_l2 = float(null_direction.norm())
    protected_l2 = float(protected.norm())
    raw_null_predicted_benefit = (
        first_order_structural_benefit(structural_gradient, null_raw)
        if null_raw_l2 > 0.0
        else 0.0
    )
    direction_integrity = (
        structural_gradient_finite
        and null_raw_finite
        and structural_gradient_l2 > 0.0
        and null_raw_l2 > 0.0
        and nullspace_leakage <= NULLSPACE_LEAKAGE_TOLERANCE
        and raw_null_predicted_benefit > 0.0
    )

    anchor_state = apply_relation_update(
        r15_state,
        protected,
        alpha=TRUST_ANCHOR_ALPHA,
        layout=layout,
    )
    anchor_non_relation = non_relation_bit_identical(
        r15_state,
        anchor_state,
        layout=layout,
    )
    if not anchor_non_relation:
        raise RuntimeError("R22 trust anchor changed non-relation state")
    kl_budget = state_teacher_kl(
        anchor_state,
        retention_train_cache,
        teacher_probabilities,
        batch_size=args.eval_batch_size,
    )
    if not np.isfinite(kl_budget) or kl_budget < 0.0:
        raise RuntimeError("R22 KL budget is invalid")

    baseline_matched, baseline_structural = r21.evaluate_state(
        r15_state,
        matched_val_cache,
        structural_val_cache,
        batch_size=args.eval_batch_size,
    )
    baseline_matched_accuracy = float(baseline_matched["accuracy"])
    baseline_structural_ne = float(
        baseline_structural["non_entailment_accuracy"]
    )
    baseline_correct = correct_count(
        baseline_matched_accuracy,
        MATCHED_VAL_N,
    )
    matched_label_counts = Counter(int(x) for x in matched_val["label"])
    matched_label_support_valid = all(
        matched_label_counts.get(label, 0) >= MIN_MATCHED_LABEL_COUNT
        for label in (0, 1, 2)
    )
    baseline_validity = (
        matched_label_support_valid
        and baseline_matched_accuracy >= BASELINE_MATCHED_FLOOR
    )

    controls: list[dict[str, object]] = []
    for mode in ("protected", "unprojected"):
        base_update = protected if mode == "protected" else delta
        for alpha in CONTROL_ALPHAS:
            state = apply_relation_update(
                r15_state,
                base_update,
                alpha=alpha,
                layout=layout,
            )
            matched_metrics, structural_metrics = r21.evaluate_state(
                state,
                matched_val_cache,
                structural_val_cache,
                batch_size=args.eval_batch_size,
            )
            controls.append(
                {
                    "mode": mode,
                    "alpha": alpha,
                    "eligible_for_selection": False,
                    "train_teacher_kl": state_teacher_kl(
                        state,
                        retention_train_cache,
                        teacher_probabilities,
                        batch_size=args.eval_batch_size,
                    ),
                    "matched": matched_metrics,
                    "ranked_structural": structural_metrics,
                    "non_relation_bit_identical": non_relation_bit_identical(
                        r15_state,
                        state,
                        layout=layout,
                    ),
                }
            )

    candidates: list[dict[str, object]] = []
    candidate_states: dict[tuple[float, float], dict[str, torch.Tensor]] = {}

    for alpha in ALPHAS:
        for gamma in GAMMAS:
            update = combine_updates(
                protected,
                null_direction,
                alpha=alpha,
                gamma=gamma,
            )
            predicted_benefit = first_order_structural_benefit(
                structural_gradient,
                update,
            )
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
            if not non_relation_ok:
                raise RuntimeError(
                    f"R22 candidate changed non-relation state: {alpha}/{gamma}"
                )

            train_kl = state_teacher_kl(
                state,
                retention_train_cache,
                teacher_probabilities,
                batch_size=args.eval_batch_size,
            )
            trust_region_pass = train_kl <= kl_budget

            matched_metrics, structural_metrics = r21.evaluate_state(
                state,
                matched_val_cache,
                structural_val_cache,
                batch_size=args.eval_batch_size,
            )
            matched_accuracy = float(matched_metrics["accuracy"])
            candidate_correct = correct_count(
                matched_accuracy,
                MATCHED_VAL_N,
            )
            relative_retention_pass = finite_sample_relative_accuracy_pass(
                matched_accuracy,
                baseline_matched_accuracy,
                n=MATCHED_VAL_N,
                tolerance=MATCHED_RELATIVE_TOLERANCE,
            )

            eligible = (
                baseline_validity
                and mechanism_validity
                and direction_integrity
                and trust_region_pass
                and matched_accuracy >= BASELINE_MATCHED_FLOOR
                and relative_retention_pass
                and predicted_benefit > 0.0
                and non_relation_ok
                and count_parameters(HIRACore()) == 422_159
            )

            key = (float(alpha), float(gamma))
            candidate_states[key] = state
            candidates.append(
                {
                    "alpha": alpha,
                    "gamma": gamma,
                    "applied_update_l2": float(update.norm()),
                    "protected_component_l2": float((alpha * protected).norm()),
                    "null_component_l2": float((gamma * null_direction).norm()),
                    "first_order_structural_predicted_benefit": predicted_benefit,
                    "train_teacher_kl": train_kl,
                    "kl_budget": kl_budget,
                    "kl_budget_ratio": (
                        train_kl / kl_budget if kl_budget > 0.0 else float("inf")
                    ),
                    "trust_region_pass": trust_region_pass,
                    "matched": matched_metrics,
                    "matched_correct_count": candidate_correct,
                    "baseline_matched_correct_count": baseline_correct,
                    "relative_retention_pass": relative_retention_pass,
                    "ranked_structural": structural_metrics,
                    "eligible": eligible,
                    "non_relation_bit_identical": non_relation_ok,
                }
            )

    if len(candidates) != 12:
        raise RuntimeError(f"R22 candidate grid must contain 12 rows, got {len(candidates)}")

    eligible_candidates = [
        row for row in candidates if bool(row["eligible"])
    ]
    selected_eligible = bool(eligible_candidates)

    if selected_eligible:
        selected_row = min(
            eligible_candidates,
            key=lambda row: (
                -float(
                    row["ranked_structural"]["non_entailment_accuracy"]
                ),
                -float(row["matched"]["accuracy"]),
                float(row["kl_budget_ratio"]),
                float(row["applied_update_l2"]),
                float(row["gamma"]),
                float(row["alpha"]),
            ),
        )
        selected_key = (
            float(selected_row["alpha"]),
            float(selected_row["gamma"]),
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
    selected_sha = r21.file_sha256(head_path)

    if selected_row is None:
        selected_matched = baseline_matched_accuracy
        selected_structural = baseline_structural_ne
        selected_correct = baseline_correct
        selected_trust_pass = True
        selected_non_relation = True
    else:
        selected_matched = float(selected_row["matched"]["accuracy"])
        selected_structural = float(
            selected_row["ranked_structural"]["non_entailment_accuracy"]
        )
        selected_correct = int(selected_row["matched_correct_count"])
        selected_trust_pass = bool(selected_row["trust_region_pass"])
        selected_non_relation = bool(
            selected_row["non_relation_bit_identical"]
        )

    structural_gain = selected_structural - baseline_structural_ne
    selected_relative_pass = (
        selected_eligible
        and selected_correct >= baseline_correct - 3
    )

    gates = {
        "prior_train_exact_30000": len(prior_train) == 30000,
        "structural_train_exact_4000": len(structural_train_indices) == 4000,
        "retention_train_exact_6000": len(retention_indices) == 6000,
        "train_authorities_disjoint_from_each_other_and_prior": (
            train_sets_disjoint
        ),
        "structural_validation_exact_500": len(structural_val_indices) == 500,
        "structural_validation_disjoint_from_r18_r19_r20_r21": (
            structural_validation_disjoint
        ),
        "matched_validation_exact_1500": len(matched_indices) == 1500,
        "matched_validation_disjoint_from_prior": (
            matched_validation_disjoint
        ),
        "probe_beta": PROBE_BETA,
        "probe_mean_teacher_kl": mean_probe_kl,
        "probe_kl_floor": PROBE_KL_FLOOR,
        "probe_kl_pass": mean_probe_kl > PROBE_KL_FLOOR,
        "functional_gradient_rows_finite": functional_rows_finite,
        "functional_damage_numerical_rank": subspace.rank,
        "functional_rank_floor": MIN_FUNCTIONAL_RANK,
        "functional_rank_pass": subspace.rank >= MIN_FUNCTIONAL_RANK,
        "basis_orthonormal_error": subspace.orthonormal_error,
        "basis_orthonormal_pass": basis_orthonormal_pass,
        "rank32_projected_component_energy_fraction": (
            rank32_projected_energy_fraction
        ),
        "mechanism_projected_energy_floor": (
            MECHANISM_PROJECTED_ENERGY_FLOOR
        ),
        "mechanism_validity": mechanism_validity,
        "structural_gradient_finite": structural_gradient_finite,
        "structural_gradient_l2": structural_gradient_l2,
        "null_raw_finite": null_raw_finite,
        "null_raw_l2": null_raw_l2,
        "null_direction_l2": null_direction_l2,
        "protected_update_l2": protected_l2,
        "nullspace_leakage": nullspace_leakage,
        "nullspace_leakage_tolerance": NULLSPACE_LEAKAGE_TOLERANCE,
        "raw_null_first_order_structural_predicted_benefit": (
            raw_null_predicted_benefit
        ),
        "direction_integrity": direction_integrity,
        "kl_budget": kl_budget,
        "trust_anchor_alpha": TRUST_ANCHOR_ALPHA,
        "trust_anchor_non_relation_bit_identical": anchor_non_relation,
        "matched_label_counts": {
            str(label): int(matched_label_counts.get(label, 0))
            for label in (0, 1, 2)
        },
        "matched_label_support_valid": matched_label_support_valid,
        "baseline_matched_floor": BASELINE_MATCHED_FLOOR,
        "baseline_validity": baseline_validity,
        "baseline_matched_correct_count": baseline_correct,
        "selected_eligible": selected_eligible,
        "selected_trust_region_pass": (
            selected_eligible and selected_trust_pass
        ),
        "matched_absolute_pass": (
            selected_eligible
            and selected_matched >= BASELINE_MATCHED_FLOOR
        ),
        "matched_relative_tolerance_predictions": 3,
        "matched_relative_pass": selected_relative_pass,
        "selected_matched_correct_count": selected_correct,
        "ranked_structural_floor": STRUCTURAL_FLOOR,
        "ranked_structural_pass": (
            selected_eligible
            and selected_structural >= STRUCTURAL_FLOOR
        ),
        "ranked_structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
        "ranked_structural_gain": structural_gain,
        "ranked_structural_gain_pass": (
            selected_eligible
            and structural_gain >= STRUCTURAL_GAIN_FLOOR
        ),
        "non_relation_bit_identical": selected_non_relation,
        "parameter_count_unchanged": count_parameters(HIRACore()) == 422_159,
    }
    gates["primary_pass"] = all(
        [
            gates["prior_train_exact_30000"],
            gates["structural_train_exact_4000"],
            gates["retention_train_exact_6000"],
            gates["train_authorities_disjoint_from_each_other_and_prior"],
            gates["structural_validation_exact_500"],
            gates["structural_validation_disjoint_from_r18_r19_r20_r21"],
            gates["matched_validation_exact_1500"],
            gates["matched_validation_disjoint_from_prior"],
            gates["probe_kl_pass"],
            gates["functional_gradient_rows_finite"],
            gates["functional_rank_pass"],
            gates["basis_orthonormal_pass"],
            gates["mechanism_validity"],
            gates["direction_integrity"],
            gates["trust_anchor_non_relation_bit_identical"],
            gates["matched_label_support_valid"],
            gates["baseline_validity"],
            gates["selected_eligible"],
            gates["selected_trust_region_pass"],
            gates["matched_absolute_pass"],
            gates["matched_relative_pass"],
            gates["ranked_structural_pass"],
            gates["ranked_structural_gain_pass"],
            gates["non_relation_bit_identical"],
            gates["parameter_count_unchanged"],
        ]
    )

    receipt = {
        "schema_version": "r22-functional-nullspace-structural-v1",
        "status": "PASS",
        "primary_pass": gates["primary_pass"],
        "evidence_scope": (
            "new disjoint MultiNLI train/validation authority; exact-R15 "
            "teacher-KL functional damage subspace; structural CE descent "
            "projected into the rank-32 damage nullspace; train-only KL trust "
            "region; no HANS/Breaking/XNLI/MASSIVE/Banking77/Laya/Jev "
            "model selection"
        ),
        "sources": {
            "a13_revision": A13_REVISION,
            "a13_weight_sha256": A13_WEIGHT_SHA256,
            "mnli_revision": MNLI_REVISION,
            "r15_head_sha256": R15_HEAD_SHA256,
            "r16_failure_analysis_head_sha256": R16_HEAD_SHA256,
            "r21_authoritative_run_id": 35715962261,
        },
        "protocol": {
            "probe_beta": PROBE_BETA,
            "functional_rank": FUNCTIONAL_RANK,
            "protection_removal": PROTECTION_REMOVAL,
            "trust_anchor_alpha": TRUST_ANCHOR_ALPHA,
            "alphas": list(ALPHAS),
            "gammas": list(GAMMAS),
            "control_alphas": list(CONTROL_ALPHAS),
            "r22_structural_train_rule": (
                "strongest 2000 neutral + 2000 contradiction after excluding "
                "exact R19/R20/R21 scored-train source-index union"
            ),
            "r22_retention_shuffle_seed": R22_RETENTION_SEED,
            "r22_retention_n": R22_RETENTION_N,
            "structural_validation_window_per_label": [
                R22_STRUCTURAL_VAL_START,
                R22_STRUCTURAL_VAL_START + STRUCTURAL_VAL_PER_LABEL,
            ],
            "matched_validation_window": [
                R22_MATCHED_START,
                R22_MATCHED_START + MATCHED_VAL_N,
            ],
            "matched_shuffle_seed": MATCHED_SHUFFLE_SEED,
            "probe_kl_floor": PROBE_KL_FLOOR,
            "functional_rank_floor": MIN_FUNCTIONAL_RANK,
            "relative_eigenvalue_floor": RELATIVE_EIGENVALUE_FLOOR,
            "orthonormal_tolerance": ORTHONORMAL_TOLERANCE,
            "mechanism_projected_energy_floor": (
                MECHANISM_PROJECTED_ENERGY_FLOOR
            ),
            "nullspace_leakage_tolerance": (
                NULLSPACE_LEAKAGE_TOLERANCE
            ),
            "baseline_matched_floor": BASELINE_MATCHED_FLOOR,
            "matched_relative_tolerance": MATCHED_RELATIVE_TOLERANCE,
            "matched_relative_authority": (
                "exact finite-sample correct-count comparison; "
                "0.002 * 1500 = 3 predictions"
            ),
            "min_matched_label_count": MIN_MATCHED_LABEL_COUNT,
            "structural_floor": STRUCTURAL_FLOOR,
            "structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
            "selection_order": [
                "eligible candidates only",
                "maximize ranked-structural non-entailment accuracy",
                "tie-break by matched accuracy",
                "lower train KL / frozen KL budget",
                "smaller applied update L2",
                "smaller gamma",
                "smaller alpha",
            ],
        },
        "data": {
            "prior_train_n": len(prior_train),
            "prior_train_index_sha256": r21.hash_indices(
                sorted(prior_train)
            ),
            **{
                f"{name}_train_index_sha256": r21.hash_indices(indices)
                for name, indices in prior.items()
            },
            "structural_train_n": len(structural_train_indices),
            "structural_train_index_sha256": r21.hash_indices(
                structural_train_indices
            ),
            "structural_train_stats": structural_train_stats,
            "retention_train_n": len(retention_indices),
            "retention_train_index_sha256": r21.hash_indices(
                retention_indices
            ),
            "train_authorities_disjoint_from_each_other_and_prior": (
                train_sets_disjoint
            ),
            "prior_structural_validation_index_sha256": r21.hash_indices(
                sorted(prior_structural_validation)
            ),
            "structural_validation_n": len(structural_val_indices),
            "structural_validation_index_sha256": r21.hash_indices(
                structural_val_indices
            ),
            "structural_validation_stats": structural_val_stats,
            "structural_validation_disjoint_from_r18_r19_r20_r21": (
                structural_validation_disjoint
            ),
            "prior_matched_index_sha256": r21.hash_indices(
                prior_matched_indices
            ),
            "matched_validation_n": len(matched_indices),
            "matched_validation_index_sha256": r21.hash_indices(
                matched_indices
            ),
            "matched_validation_disjoint_from_prior": (
                matched_validation_disjoint
            ),
        },
        "functional_damage": {
            "teacher_probabilities_shape": list(
                teacher_probabilities.shape
            ),
            "probe_non_relation_bit_identical": (
                probe_non_relation_identical
            ),
            "probe_meta": functional_meta,
            "functional_subspace_rank": subspace.rank,
            "functional_subspace_orthonormal_error": (
                subspace.orthonormal_error
            ),
            "rank32_projected_component_energy_fraction": (
                rank32_projected_energy_fraction
            ),
            "original_relation_delta_l2": float(delta.norm()),
            "protected_update_l2": protected_l2,
            "kl_budget": kl_budget,
        },
        "structural_nullspace": {
            "structural_gradient_l2": structural_gradient_l2,
            "structural_gradient_meta": structural_grad_meta,
            "damage_component_l2": float(
                structural_damage_component.norm()
            ),
            "null_raw_l2": null_raw_l2,
            "normalized_null_direction_l2": null_direction_l2,
            "normalized_subspace_leakage": nullspace_leakage,
            "raw_first_order_structural_predicted_benefit": (
                raw_null_predicted_benefit
            ),
            "direction_integrity": direction_integrity,
        },
        "baseline": {
            "matched": baseline_matched,
            "matched_correct_count": baseline_correct,
            "ranked_structural": baseline_structural,
            "matched_label_counts": {
                str(label): int(matched_label_counts.get(label, 0))
                for label in (0, 1, 2)
            },
            "validity": baseline_validity,
        },
        "diagnosis_only_controls": controls,
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
                "probe_mean_teacher_kl": mean_probe_kl,
                "functional_subspace_rank": subspace.rank,
                "rank32_projected_component_energy_fraction": (
                    rank32_projected_energy_fraction
                ),
                "mechanism_validity": mechanism_validity,
                "direction_integrity": direction_integrity,
                "nullspace_leakage": nullspace_leakage,
                "kl_budget": kl_budget,
                "eligible_candidate_count": len(eligible_candidates),
                "selected": selected_row,
                "ranked_structural_gain": structural_gain,
                "selected_head_sha256": selected_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
