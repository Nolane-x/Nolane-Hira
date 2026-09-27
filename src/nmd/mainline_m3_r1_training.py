from __future__ import annotations

from copy import deepcopy
import random
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .hira import HIRACore
from .mainline import W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256
from .mainline_m3_alignment import (
    AlignedSemanticEncoder,
    M3_ALIGNMENT_PARAMETER_COUNT,
    MultilingualAlignmentAdapter,
)
from .mainline_m3_eval import m3_multilingual_qualification
from .mainline_m3_r1_cache import (
    align_compiled_schema,
    align_state_memory,
    english_anchor_loss,
    paired_alignment_loss,
    validate_m3_r1_base_cache,
)
from .runtime import NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .w34_transfer_core import build_hira_v0_w34_core

M3_R1_EPOCHS = 8
M3_R1_LR = 2e-4
M3_R1_WEIGHT_DECAY = 0.01
M3_R1_GRAD_CLIP = 1.0
M3_R1_PAIR_COEFF = 0.35
M3_R1_ENGLISH_ANCHOR_COEFF = 0.10
M3_R1_PRIMARY_SEED = 23031
M3_R1_REPLICA_SEED = 23037

M3_R1_CHECKPOINT_SCHEMA = "hira-v0-mainline-m3-r1-alignment-v1"


def build_m3_r1_training_runtime(
    base_encoder: TextSemanticEncoder,
    t0_checkpoint_path,
    w34_checkpoint_path,
    *,
    seed: int,
    alignment_identity: str,
) -> tuple[NolaneHira, MultilingualAlignmentAdapter]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))

    adapter = MultilingualAlignmentAdapter()
    aligned = AlignedSemanticEncoder(
        base_encoder,
        adapter,
        alignment_identity=alignment_identity,
    )
    runtime = build_hira_v0_w34_core(
        aligned,
        t0_checkpoint_path,
        w34_checkpoint_path,
        expected_t0_sha256=W28_T0_CHECKPOINT_SHA256,
        expected_candidate_sha256=W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
        hira=HIRACore(d_model=256, dropout=0.0),
        include_unbridged_baseline=False,
    )

    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)

    runtime.eval()
    adapter.train(True)

    trainable = sum(
        parameter.numel()
        for parameter in runtime.parameters()
        if parameter.requires_grad
    )
    if trainable != M3_ALIGNMENT_PARAMETER_COUNT:
        raise RuntimeError("M3-R1 trainable parameter surface changed")
    if adapter.trainable_parameter_count != M3_ALIGNMENT_PARAMETER_COUNT:
        raise RuntimeError("M3-R1 adapter trainable count changed")

    if any(parameter.requires_grad for parameter in aligned.base.parameters()):
        raise RuntimeError("M3-R1 base semantic encoder must remain frozen")
    if any(parameter.requires_grad for parameter in runtime.hira.parameters()):
        raise RuntimeError("M3-R1 HIRACore must remain frozen")

    scorer = runtime.coevidence_symmetric_semantic_scorer
    if scorer is None:
        raise RuntimeError("M3-R1 requires W34 co-evidence scorer")
    if any(parameter.requires_grad for parameter in scorer.parameters()):
        raise RuntimeError("M3-R1 W34 scorer must remain frozen")
    return runtime, adapter


def _language_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("M3-R1 language summary requires rows")
    by_primitive: dict[str, list[bool]] = {}
    for row in rows:
        by_primitive.setdefault(str(row["primitive"]), []).append(
            bool(row["correct"])
        )
    return {
        "case_count": len(rows),
        "top1": sum(bool(row["correct"]) for row in rows) / len(rows),
        "mrr": sum(float(row["reciprocal_rank"]) for row in rows) / len(rows),
        "mean_gold_rank": sum(float(row["gold_rank"]) for row in rows) / len(rows),
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / len(rows),
        "per_primitive_top1": {
            primitive: sum(values) / len(values)
            for primitive, values in sorted(by_primitive.items())
        },
    }


def _gold_rank(probabilities: Tensor, gold_index: int) -> int:
    order = torch.argsort(probabilities, descending=True)
    match = (order == int(gold_index)).nonzero(as_tuple=False)
    if match.numel() != 1:
        raise RuntimeError("M3-R1 gold index missing from probability vector")
    return int(match.item()) + 1


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


@torch.no_grad()
def evaluate_m3_r1_cache(
    runtime: NolaneHira,
    adapter: MultilingualAlignmentAdapter,
    cache: dict[str, object],
    *,
    alignment_identity: str,
) -> dict[str, object]:
    validate_m3_r1_base_cache(cache)
    adapter.eval()

    language_rows: dict[str, list[dict[str, object]]] = {"en": [], "vi": []}
    pair_reports: list[dict[str, object]] = []
    probability_mass_max_error = 0.0
    relation_delta_max = 0.0
    full_k = True
    finite = True

    for pair in cache["pairs"]:
        outputs: dict[str, dict[str, object]] = {}
        for language in ("en", "vi"):
            payload = pair[language]
            memory = align_state_memory(
                payload["memory"],
                adapter,
                alignment_identity=alignment_identity,
            )
            schema = align_compiled_schema(
                payload["schema"],
                adapter,
                alignment_identity=alignment_identity,
            )
            output = runtime.forward_compiled(
                memory,
                schema,
                forced_budget=None,
                adaptive_budget=False,
                relation_mode="pooled",
                coarse_mode="coevidence_symmetric_semantic",
                relation_refinement=False,
            )
            probabilities = output.probabilities.detach().float()
            selected = int(probabilities.argmax())
            gold = int(pair["gold_index"])
            rank = _gold_rank(probabilities, gold)
            mass_error = abs(float(probabilities.sum()) - 1.0)
            relation_delta = float(output.hira.relation_delta.abs().max())
            case_full_k = (
                int(output.hira.candidate_budget.item()) == len(schema.options)
                and bool(output.hira.selected_mask.all())
            )
            case_finite = bool(
                torch.isfinite(output.logits).all()
                and torch.isfinite(output.probabilities).all()
            )
            probability_mass_max_error = max(
                probability_mass_max_error,
                mass_error,
            )
            relation_delta_max = max(relation_delta_max, relation_delta)
            full_k = full_k and case_full_k
            finite = finite and case_finite

            result = {
                "pair_id": pair["pair_id"],
                "primitive": pair["primitive"],
                "language": language,
                "selected_index": selected,
                "selected_option_id": schema.options[selected].option_id,
                "gold_option_id": schema.options[gold].option_id,
                "correct": selected == gold,
                "gold_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "confidence": float(probabilities.max()),
            }
            language_rows[language].append(result)
            outputs[language] = result

        en = outputs["en"]
        vi = outputs["vi"]
        if en["gold_option_id"] != vi["gold_option_id"]:
            raise RuntimeError("M3-R1 paired gold option identity changed")
        pair_reports.append(
            {
                "pair_id": pair["pair_id"],
                "primitive": pair["primitive"],
                "prediction_agreement": (
                    en["selected_option_id"] == vi["selected_option_id"]
                ),
                "both_correct": bool(en["correct"] and vi["correct"]),
            }
        )

    en_summary = _language_summary(language_rows["en"])
    vi_summary = _language_summary(language_rows["vi"])
    pair_count = len(cache["pairs"])

    report = {
        "pair_count": pair_count,
        "language_case_count": 2 * pair_count,
        "state_encode_count": int(cache["metadata"]["state_encode_count"]),
        "state_encodes_per_language_case": float(
            cache["metadata"]["state_encodes_per_language_case"]
        ),
        "en": en_summary,
        "vi": vi_summary,
        "vi_en_top1_ratio": _ratio(
            float(vi_summary["top1"]),
            float(en_summary["top1"]),
        ),
        "vi_en_mrr_ratio": _ratio(
            float(vi_summary["mrr"]),
            float(en_summary["mrr"]),
        ),
        "paired_prediction_agreement": (
            sum(bool(row["prediction_agreement"]) for row in pair_reports)
            / pair_count
        ),
        "paired_both_correct_rate": (
            sum(bool(row["both_correct"]) for row in pair_reports)
            / pair_count
        ),
        "probability_mass_max_error": probability_mass_max_error,
        "relation_delta_max": relation_delta_max,
        "finite": float(finite),
        "full_k": float(full_k),
        "gradient_updates_used": False,
        "candidate_pruning_used": False,
        "relation_refinement_used": False,
        "adaptive_budget_used": False,
    }
    report["qualification"] = m3_multilingual_qualification(report)
    return report


def _selection_key(report: Mapping[str, object], epoch: int) -> tuple[float, ...]:
    en = report["en"]
    vi = report["vi"]
    if not isinstance(en, Mapping) or not isinstance(vi, Mapping):
        raise ValueError("M3-R1 report missing language metrics")
    return (
        min(float(en["top1"]), float(vi["top1"])),
        float(report["paired_prediction_agreement"]),
        min(
            float(report["vi_en_top1_ratio"]),
            float(report["vi_en_mrr_ratio"]),
        ),
        min(float(en["mrr"]), float(vi["mrr"])),
        -float(epoch),
    )


def _aligned_pair(
    pair: Mapping[str, object],
    adapter: MultilingualAlignmentAdapter,
    *,
    alignment_identity: str,
):
    result = {}
    for language in ("en", "vi"):
        payload = pair[language]
        result[language] = (
            align_state_memory(
                payload["memory"],
                adapter,
                alignment_identity=alignment_identity,
            ),
            align_compiled_schema(
                payload["schema"],
                adapter,
                alignment_identity=alignment_identity,
            ),
        )
    return result


def train_m3_r1_candidate(
    runtime: NolaneHira,
    adapter: MultilingualAlignmentAdapter,
    train_cache: dict[str, object],
    dev_cache: dict[str, object],
    *,
    seed: int,
    role: str,
) -> dict[str, object]:
    validate_m3_r1_base_cache(train_cache, expected_partition="train")
    validate_m3_r1_base_cache(dev_cache, expected_partition="dev")
    if role not in {"primary", "replica"}:
        raise ValueError("M3-R1 role must be primary or replica")

    torch.manual_seed(int(seed))
    random.seed(int(seed))
    alignment_identity = f"m3-r1-{role}-seed-{seed}"

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=M3_R1_LR,
        weight_decay=M3_R1_WEIGHT_DECAY,
    )

    best_key = None
    best_epoch = None
    best_state = None
    best_report = None
    history = []

    train_pairs = list(train_cache["pairs"])

    for epoch in range(1, M3_R1_EPOCHS + 1):
        adapter.train(True)
        order = list(range(len(train_pairs)))
        random.Random(int(seed) * 1000 + epoch).shuffle(order)

        total_loss_sum = 0.0
        task_loss_sum = 0.0
        pair_loss_sum = 0.0
        anchor_loss_sum = 0.0

        for index in order:
            pair = train_pairs[index]
            optimizer.zero_grad(set_to_none=True)
            aligned = _aligned_pair(
                pair,
                adapter,
                alignment_identity=alignment_identity,
            )

            task_losses = []
            for language in ("en", "vi"):
                memory, schema = aligned[language]
                output = runtime.forward_compiled(
                    memory,
                    schema,
                    forced_budget=None,
                    adaptive_budget=False,
                    relation_mode="pooled",
                    coarse_mode="coevidence_symmetric_semantic",
                    relation_refinement=False,
                )
                target = torch.tensor(
                    [int(pair["gold_index"])],
                    dtype=torch.long,
                    device=output.logits.device,
                )
                task_losses.append(
                    F.cross_entropy(output.logits.unsqueeze(0), target)
                )

            task_loss = torch.stack(task_losses).mean()
            pair_loss = paired_alignment_loss(
                aligned["en"][0],
                aligned["en"][1],
                aligned["vi"][0],
                aligned["vi"][1],
            )
            anchor_loss = english_anchor_loss(
                pair["en"]["memory"],
                pair["en"]["schema"],
                aligned["en"][0],
                aligned["en"][1],
            )
            loss = (
                task_loss
                + M3_R1_PAIR_COEFF * pair_loss
                + M3_R1_ENGLISH_ANCHOR_COEFF * anchor_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                tuple(adapter.parameters()),
                M3_R1_GRAD_CLIP,
            )
            optimizer.step()

            total_loss_sum += float(loss.detach())
            task_loss_sum += float(task_loss.detach())
            pair_loss_sum += float(pair_loss.detach())
            anchor_loss_sum += float(anchor_loss.detach())

        report = evaluate_m3_r1_cache(
            runtime,
            adapter,
            dev_cache,
            alignment_identity=alignment_identity,
        )
        key = _selection_key(report, epoch)
        row = {
            "epoch": epoch,
            "mean_total_loss": total_loss_sum / len(train_pairs),
            "mean_task_loss": task_loss_sum / len(train_pairs),
            "mean_pair_alignment_loss": pair_loss_sum / len(train_pairs),
            "mean_english_anchor_loss": anchor_loss_sum / len(train_pairs),
            "dev": deepcopy(report),
            "selection_key": list(key),
        }
        history.append(row)

        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            }
            best_report = deepcopy(report)

    if best_state is None or best_report is None or best_epoch is None:
        raise RuntimeError("M3-R1 training produced no selected checkpoint")

    adapter.load_candidate_state_dict(best_state, freeze=True)
    selected_report = evaluate_m3_r1_cache(
        runtime,
        adapter,
        dev_cache,
        alignment_identity=alignment_identity,
    )
    if selected_report != best_report:
        # Tensor-free report should replay exactly from the frozen checkpoint.
        raise RuntimeError("M3-R1 selected checkpoint did not replay DEV metrics")

    return {
        "schema_version": M3_R1_CHECKPOINT_SCHEMA,
        "role": role,
        "seed": int(seed),
        "selected_epoch": int(best_epoch),
        "parameter_count": M3_ALIGNMENT_PARAMETER_COUNT,
        "candidate_state_dict": best_state,
        "selected_dev": selected_report,
        "dev_qualification": selected_report["qualification"],
        "history": history,
        "alignment_identity": alignment_identity,
    }


def m3_r1_joint_qualification(
    primary: Mapping[str, object],
    replica: Mapping[str, object],
) -> dict[str, object]:
    primary_gate = primary["dev_qualification"]
    replica_gate = replica["dev_qualification"]
    if not isinstance(primary_gate, Mapping) or not isinstance(replica_gate, Mapping):
        raise ValueError("M3-R1 candidate qualification missing")
    return {
        "pass": bool(primary_gate["pass"] and replica_gate["pass"]),
        "primary": dict(primary_gate),
        "replica": dict(replica_gate),
    }


__all__ = [
    "M3_R1_CHECKPOINT_SCHEMA",
    "M3_R1_ENGLISH_ANCHOR_COEFF",
    "M3_R1_EPOCHS",
    "M3_R1_GRAD_CLIP",
    "M3_R1_LR",
    "M3_R1_PAIR_COEFF",
    "M3_R1_PRIMARY_SEED",
    "M3_R1_REPLICA_SEED",
    "M3_R1_WEIGHT_DECAY",
    "build_m3_r1_training_runtime",
    "evaluate_m3_r1_cache",
    "m3_r1_joint_qualification",
    "train_m3_r1_candidate",
]
