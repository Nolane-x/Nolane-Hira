from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from nmd.block_interpolation import (
    atomic_group_for_key,
    balanced_ranked_structural_non_entailment_indices,
)
from nmd.delta_surgery import (
    apply_relation_mask,
    balanced_ranked_structural_window_indices,
    coordinate_scores,
    flatten_relation_named_tensors,
    nested_masks,
    relation_delta,
    relation_layout,
    sparse_identity_invariants,
    top_fraction_mask,
)
from nmd.hard_negative import (
    anti_entailment_margin_loss,
    evaluate_repair_slice,
)
from nmd.hira import HIRACore, count_parameters
from nmd.relation_cache import RelationCache, forward_cached, minibatches
from nmd.semantic import HFAutoSemanticEncoder


A13_MODEL = "microsoft/xtremedistil-l6-h256-uncased"
A13_REVISION = "4226d9e4d2c08703e5cb0491b479bfc6a1607181"
A13_WEIGHT_SHA256 = "5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880"

MNLI_DATASET = "nyu-mll/multi_nli"
MNLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"

R15_HEAD_SHA256 = "007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f"
R16_HEAD_SHA256 = "dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75"

STRUCTURAL_TRAIN_PER_LABEL = 2000
RETENTION_TRAIN_N = 6000
STRUCTURAL_WINDOW_START = 250
STRUCTURAL_VAL_PER_LABEL = 250
MATCHED_WINDOW_START = 1500
MATCHED_VAL_N = 1500
MIN_HYPOTHESIS_TOKENS = 3
GRAD_BATCH_SIZE = 96
MASK_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
ALPHAS = (0.25, 0.50, 1.00)
POSITIVE_SUPPORT_FLOOR = 0.10
MATCHED_ABSOLUTE_FLOOR = 0.55
MATCHED_RELATIVE_TOLERANCE = 0.002
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
    expected = {
        span.key for span in relation_layout(model.state_dict())
    }
    if set(result) != expected:
        raise RuntimeError(
            f"relation parameter/state mismatch: params={sorted(result)}, "
            f"state={sorted(expected)}"
        )
    return result


def collect_relation_gradients(
    model: HIRACore,
    cache: RelationCache,
    *,
    batch_size: int,
    structural: bool,
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
        if structural:
            margin = anti_entailment_margin_loss(
                out.logits, labels, margin=0.5
            )
            loss = ce + 0.5 * margin
        else:
            margin = out.logits.sum() * 0.0
            loss = ce
        loss.backward()
        n = int(len(idx))
        seen += n
        loss_weighted += float(loss.detach().cpu()) * n
        for name, param in params.items():
            if param.grad is None:
                raise RuntimeError(f"missing relation gradient for {name}")
            grad = param.grad.detach().cpu().to(torch.float64)
            if structural:
                sums[name].add_(grad, alpha=float(n))
            else:
                sums[name].add_(grad.square(), alpha=float(n))

    if seen != len(cache):
        raise RuntimeError("gradient pass did not cover complete cache")
    for name in sums:
        sums[name].div_(float(seen))

    return sums, {
        "n": float(seen),
        "mean_loss": loss_weighted / max(1, seen),
        "mode": "structural_mean_gradient" if structural
        else "sample_weighted_batch_mean_gradient_square",
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
        evaluate_repair_slice(
            model, matched_cache, batch_size=batch_size
        ),
        evaluate_repair_slice(
            model, structural_cache, batch_size=batch_size
        ),
    )


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
        repo_id=A13_MODEL, revision=A13_REVISION
    ))
    if file_sha256(snapshot / "model.safetensors") != A13_WEIGHT_SHA256:
        raise RuntimeError("A13 weight SHA-256 mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True
    )
    base_encoder = AutoModel.from_pretrained(
        str(snapshot), local_files_only=True
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

    # Train-only structural utility set.
    structural_train_indices, structural_train_stats = (
        balanced_ranked_structural_non_entailment_indices(
            train_full["premise"],
            train_full["hypothesis"],
            train_full["label"],
            per_label=STRUCTURAL_TRAIN_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        )
    )
    if len(structural_train_indices) != 2 * STRUCTURAL_TRAIN_PER_LABEL:
        raise RuntimeError("R19 structural train set size mismatch")
    structural_train_set = set(structural_train_indices)

    # Train-only retention set, explicitly disjoint from structural utility.
    indexed_train = train_full.add_column(
        "_source_index", list(range(len(train_full)))
    )
    shuffled_train = indexed_train.shuffle(seed=args.seed)
    retention_indices: list[int] = []
    for source_index in shuffled_train["_source_index"]:
        i = int(source_index)
        if i in structural_train_set:
            continue
        retention_indices.append(i)
        if len(retention_indices) == RETENTION_TRAIN_N:
            break
    if len(retention_indices) != RETENTION_TRAIN_N:
        raise RuntimeError("R19 retention train set too small")
    train_sets_disjoint = structural_train_set.isdisjoint(
        retention_indices
    )
    if not train_sets_disjoint:
        raise RuntimeError("R19 train structural/retention sets overlap")

    # Structural validation: ranks 251-500 per label, disjoint from R18.
    r18_top_indices, _ = balanced_ranked_structural_window_indices(
        mismatched_full["premise"],
        mismatched_full["hypothesis"],
        mismatched_full["label"],
        start_per_label=0,
        count_per_label=STRUCTURAL_VAL_PER_LABEL,
        min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
    )
    structural_val_indices, structural_val_stats = (
        balanced_ranked_structural_window_indices(
            mismatched_full["premise"],
            mismatched_full["hypothesis"],
            mismatched_full["label"],
            start_per_label=STRUCTURAL_WINDOW_START,
            count_per_label=STRUCTURAL_VAL_PER_LABEL,
            min_hypothesis_tokens=MIN_HYPOTHESIS_TOKENS,
        )
    )
    structural_validation_disjoint = set(
        r18_top_indices
    ).isdisjoint(structural_val_indices)
    if len(structural_val_indices) != 500:
        raise RuntimeError("R19 structural validation must be 500")
    if not structural_validation_disjoint:
        raise RuntimeError("R19 structural validation overlaps R18")

    # Matched validation: shuffled positions 1500:3000, disjoint from prior.
    indexed_matched = matched_full.add_column(
        "_source_index", list(range(len(matched_full)))
    )
    shuffled_matched = indexed_matched.shuffle(seed=args.seed + 1)
    if len(shuffled_matched) < MATCHED_WINDOW_START + MATCHED_VAL_N:
        raise RuntimeError("not enough matched validation rows")
    prior_matched_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][
            :MATCHED_WINDOW_START
        ]
    ]
    matched_indices = [
        int(x)
        for x in shuffled_matched["_source_index"][
            MATCHED_WINDOW_START:MATCHED_WINDOW_START + MATCHED_VAL_N
        ]
    ]
    matched_validation_disjoint = set(
        prior_matched_indices
    ).isdisjoint(matched_indices)
    if len(matched_indices) != MATCHED_VAL_N:
        raise RuntimeError("R19 matched validation must be 1500")
    if not matched_validation_disjoint:
        raise RuntimeError("R19 matched validation overlaps prior slice")

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
            "role": "r19-train-structural-utility",
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
            "role": "r19-train-retention-energy",
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
            "role": "r19-ranked-structural-validation-window-251-500",
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
            "role": "r19-matched-validation-window-1500-3000",
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
    delta = relation_delta(
        r15_state, r16_state, layout=layout
    )

    scoring_model = HIRACore(d_model=256, dropout=0.0)
    scoring_model.load_state_dict(r15_state, strict=True)
    structural_grad_map, structural_grad_meta = collect_relation_gradients(
        scoring_model,
        structural_train_cache,
        batch_size=GRAD_BATCH_SIZE,
        structural=True,
    )
    structural_gradient = flatten_relation_named_tensors(
        structural_grad_map, layout=layout
    )

    scoring_model.load_state_dict(r15_state, strict=True)
    retention_energy_map, retention_energy_meta = collect_relation_gradients(
        scoring_model,
        retention_train_cache,
        batch_size=GRAD_BATCH_SIZE,
        structural=False,
    )
    retention_energy = flatten_relation_named_tensors(
        retention_energy_map, layout=layout
    )

    scores = coordinate_scores(
        structural_gradient, retention_energy, delta
    )
    positive_mask = scores["positive_mask"]
    relation_coordinate_count = int(delta.numel())
    positive_count = int(positive_mask.sum())
    positive_fraction = positive_count / max(1, relation_coordinate_count)
    positive_support_valid = positive_fraction >= POSITIVE_SUPPORT_FLOOR

    scorer_tensors = {
        "benefit": scores["benefit_score"],
        "benefit_over_cost": scores["benefit_over_cost_score"],
    }
    masks: dict[str, dict[float, torch.Tensor]] = {}
    mask_stats: dict[str, dict[str, object]] = {}
    masks_nested = True
    for scorer_name, scorer_score in scorer_tensors.items():
        scorer_masks: dict[float, torch.Tensor] = {}
        ordered: list[torch.Tensor] = []
        for fraction in MASK_FRACTIONS:
            mask = top_fraction_mask(
                scorer_score, positive_mask, fraction
            )
            scorer_masks[fraction] = mask
            ordered.append(mask)
        is_nested = nested_masks(ordered)
        masks_nested = masks_nested and is_nested
        masks[scorer_name] = scorer_masks
        mask_stats[scorer_name] = {
            "nested": is_nested,
            "changed_coordinate_counts": {
                str(fraction): int(scorer_masks[fraction].sum())
                for fraction in MASK_FRACTIONS
            },
            "predicted_raw_benefit_sum": {
                str(fraction): float(
                    scores["raw_benefit"][
                        scorer_masks[fraction]
                    ].sum()
                )
                for fraction in MASK_FRACTIONS
            },
            "movement_cost_sum": {
                str(fraction): float(
                    scores["cost"][scorer_masks[fraction]].sum()
                )
                for fraction in MASK_FRACTIONS
            },
        }
    if not masks_nested:
        raise RuntimeError("R19 masks are not nested")

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
    relative_floor = baseline_matched_accuracy - MATCHED_RELATIVE_TOLERANCE

    candidates: list[dict[str, object]] = []
    candidate_states: dict[
        tuple[str, float, float], dict[str, torch.Tensor]
    ] = {}

    for scorer_name in sorted(masks):
        for fraction in MASK_FRACTIONS:
            mask = masks[scorer_name][fraction]
            changed_count = int(mask.sum())
            for alpha in ALPHAS:
                state = apply_relation_mask(
                    r15_state,
                    r16_state,
                    mask,
                    alpha=alpha,
                    layout=layout,
                )
                invariants = sparse_identity_invariants(
                    r15_state,
                    state,
                    mask,
                    layout=layout,
                )
                if not all(invariants.values()):
                    raise RuntimeError(
                        f"identity invariant failed for "
                        f"{scorer_name}/{fraction}/{alpha}"
                    )
                matched_metrics, structural_metrics = evaluate_state(
                    state,
                    matched_val_cache,
                    structural_val_cache,
                    batch_size=args.eval_batch_size,
                )
                matched_accuracy = float(matched_metrics["accuracy"])
                eligible = (
                    matched_accuracy >= MATCHED_ABSOLUTE_FLOOR
                    and matched_accuracy >= relative_floor
                )
                key = (scorer_name, fraction, alpha)
                candidate_states[key] = state
                candidates.append(
                    {
                        "scorer": scorer_name,
                        "fraction": fraction,
                        "alpha": alpha,
                        "changed_relation_coordinates": changed_count,
                        "matched": matched_metrics,
                        "ranked_structural": structural_metrics,
                        "eligible": eligible,
                        **invariants,
                    }
                )

    eligible_candidates = [
        row for row in candidates if bool(row["eligible"])
    ]
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
                int(row["changed_relation_coordinates"]),
                float(row["alpha"]),
                str(row["scorer"]),
            ),
        )
        selected_key = (
            str(selected_row["scorer"]),
            float(selected_row["fraction"]),
            float(selected_row["alpha"]),
        )
        selected_state = candidate_states[selected_key]
        selected_mask = masks[
            str(selected_row["scorer"])
        ][float(selected_row["fraction"])]
    else:
        selected_row = None
        selected_state = {
            key: value.detach().cpu().clone()
            for key, value in r15_state.items()
        }
        selected_mask = torch.zeros(
            relation_coordinate_count, dtype=torch.bool
        )

    selected_invariants = sparse_identity_invariants(
        r15_state,
        selected_state,
        selected_mask,
        layout=layout,
    )
    head_path = args.out / "hira-head.pt"
    torch.save(selected_state, head_path)
    selected_sha = file_sha256(head_path)

    if selected_row is None:
        selected_matched_accuracy = baseline_matched_accuracy
        selected_structural_ne = baseline_structural_ne
    else:
        selected_matched_accuracy = float(
            selected_row["matched"]["accuracy"]
        )
        selected_structural_ne = float(
            selected_row["ranked_structural"][
                "non_entailment_accuracy"
            ]
        )
    structural_gain = selected_structural_ne - baseline_structural_ne

    gates = {
        "structural_train_exact_4000": (
            len(structural_train_indices) == 4000
        ),
        "retention_train_exact_6000": (
            len(retention_indices) == 6000
        ),
        "train_sets_disjoint": train_sets_disjoint,
        "positive_support_floor": POSITIVE_SUPPORT_FLOOR,
        "positive_benefit_fraction": positive_fraction,
        "positive_support_valid": positive_support_valid,
        "structural_validation_exact_500": (
            len(structural_val_indices) == 500
        ),
        "structural_validation_disjoint_from_r18": (
            structural_validation_disjoint
        ),
        "matched_validation_exact_1500": (
            len(matched_indices) == 1500
        ),
        "matched_validation_disjoint_from_prior": (
            matched_validation_disjoint
        ),
        "selected_eligible": selected_eligible,
        "matched_absolute_floor": MATCHED_ABSOLUTE_FLOOR,
        "matched_absolute_pass": (
            selected_eligible
            and selected_matched_accuracy >= MATCHED_ABSOLUTE_FLOOR
        ),
        "matched_relative_floor": relative_floor,
        "matched_relative_pass": (
            selected_eligible
            and selected_matched_accuracy >= relative_floor
        ),
        "ranked_structural_floor": STRUCTURAL_FLOOR,
        "ranked_structural_pass": (
            selected_eligible
            and selected_structural_ne >= STRUCTURAL_FLOOR
        ),
        "ranked_structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
        "ranked_structural_gain": structural_gain,
        "ranked_structural_gain_pass": (
            selected_eligible
            and structural_gain >= STRUCTURAL_GAIN_FLOOR
        ),
        "non_relation_bit_identical": (
            selected_invariants["non_relation_bit_identical"]
        ),
        "masked_off_relation_bit_identical": (
            selected_invariants["masked_off_relation_bit_identical"]
        ),
        "parameter_count_unchanged": (
            count_parameters(HIRACore()) == 422_159
        ),
        "masks_nested": masks_nested,
    }
    gates["primary_pass"] = all(
        [
            gates["structural_train_exact_4000"],
            gates["retention_train_exact_6000"],
            gates["train_sets_disjoint"],
            gates["positive_support_valid"],
            gates["structural_validation_exact_500"],
            gates["structural_validation_disjoint_from_r18"],
            gates["matched_validation_exact_1500"],
            gates["matched_validation_disjoint_from_prior"],
            gates["selected_eligible"],
            gates["matched_absolute_pass"],
            gates["matched_relative_pass"],
            gates["ranked_structural_pass"],
            gates["ranked_structural_gain_pass"],
            gates["non_relation_bit_identical"],
            gates["masked_off_relation_bit_identical"],
            gates["parameter_count_unchanged"],
            gates["masks_nested"],
        ]
    )

    receipt = {
        "schema_version": "r19-relation-delta-surgery-v1",
        "status": "PASS",
        "evidence_scope": (
            "train-only coordinate scoring plus disjoint adapted MultiNLI "
            "development validation; no HANS/Breaking/XNLI/MASSIVE/"
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
            "structural_train_per_label": STRUCTURAL_TRAIN_PER_LABEL,
            "retention_train_n": RETENTION_TRAIN_N,
            "gradient_batch_size": GRAD_BATCH_SIZE,
            "structural_validation_window": [
                STRUCTURAL_WINDOW_START,
                STRUCTURAL_WINDOW_START + STRUCTURAL_VAL_PER_LABEL,
            ],
            "matched_validation_window": [
                MATCHED_WINDOW_START,
                MATCHED_WINDOW_START + MATCHED_VAL_N,
            ],
            "mask_fractions": list(MASK_FRACTIONS),
            "alphas": list(ALPHAS),
            "positive_support_floor": POSITIVE_SUPPORT_FLOOR,
            "matched_absolute_floor": MATCHED_ABSOLUTE_FLOOR,
            "matched_relative_tolerance": MATCHED_RELATIVE_TOLERANCE,
            "ranked_structural_floor": STRUCTURAL_FLOOR,
            "ranked_structural_gain_floor": STRUCTURAL_GAIN_FLOOR,
            "coordinate_order": (
                "sorted relation state key then row-major flattened offset"
            ),
            "fraction_rounding": "ceil(fraction * positive_pool_size)",
            "selection_order": [
                "eligible absolute and relative matched retention",
                "maximize ranked-structural non-entailment accuracy",
                "tie-break by matched accuracy",
                "prefer fewer changed relation coordinates",
                "prefer smaller alpha",
                "lexical scorer name",
            ],
        },
        "data": {
            "structural_train_n": len(structural_train_indices),
            "structural_train_index_sha256": hash_indices(
                structural_train_indices
            ),
            "structural_train_stats": structural_train_stats,
            "retention_train_n": len(retention_indices),
            "retention_train_index_sha256": hash_indices(
                retention_indices
            ),
            "train_sets_disjoint": train_sets_disjoint,
            "r18_top_structural_index_sha256": hash_indices(
                r18_top_indices
            ),
            "structural_validation_n": len(structural_val_indices),
            "structural_validation_index_sha256": hash_indices(
                structural_val_indices
            ),
            "structural_validation_stats": structural_val_stats,
            "structural_validation_disjoint_from_r18": (
                structural_validation_disjoint
            ),
            "prior_matched_index_sha256": hash_indices(
                prior_matched_indices
            ),
            "matched_validation_n": len(matched_indices),
            "matched_validation_index_sha256": hash_indices(
                matched_indices
            ),
            "matched_validation_disjoint_from_prior": (
                matched_validation_disjoint
            ),
        },
        "coordinate_scoring": {
            "relation_coordinate_count": relation_coordinate_count,
            "nonzero_delta_count": int((delta.abs() > 1e-12).sum()),
            "positive_benefit_count": positive_count,
            "positive_benefit_fraction": positive_fraction,
            "structural_gradient_l2": float(
                structural_gradient.norm()
            ),
            "retention_energy_sum": float(
                retention_energy.sum()
            ),
            "delta_l2": float(delta.norm()),
            "structural_gradient_meta": structural_grad_meta,
            "retention_energy_meta": retention_energy_meta,
            "mask_stats": mask_stats,
        },
        "baseline": {
            "matched": baseline_matched,
            "ranked_structural": baseline_structural,
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
                "positive_benefit_fraction": positive_fraction,
                "baseline_matched_accuracy": baseline_matched_accuracy,
                "baseline_ranked_structural": baseline_structural_ne,
                "selected": selected_row,
                "ranked_structural_gain": structural_gain,
                "selected_head_sha256": selected_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
