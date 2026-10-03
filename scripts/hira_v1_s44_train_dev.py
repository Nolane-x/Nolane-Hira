from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import a13_lora_state_dict, iter_a13_lora_modules, load_a13_lora_state_dict
from nmd.v1_private_correction_representation_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_norm_balanced_gradient import apply_gradient_update, norm_balanced_gradient_update
from nmd.v1_s43_authority import generate_s43_cases
from nmd.v1_s44_authority import generate_s44_cases, validate_s44_partitions
from nmd.v1_s6_semantic_core import HIRA_V1_S6_LORA_PARAMETER_COUNT
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
import hira_v1_s17_train_dev as s17mod
import hira_v1_s35_train_dev as s35
import hira_v1_s43_train_dev as s43
from hira_v1_s44_a0_private_correction_fork import cases as s44_a0_cases

SCHEMA_VERSION = "hira-v1-s44-matched-private-correction-representation-fork-train-dev-v1"
SEED = 65001
ADAPTER_A_PARAMETER_COUNT = 32_768
ADAPTER_B_PARAMETER_COUNT = 16_384
PRIVATE_ADAPTER_PARAMETER_COUNT = 49_152
BILINEAR_PARAMETER_COUNT = 65_536
CORRECTION_PARAMETER_COUNT = 114_688
TREATMENT_TOTAL = HIRA_V1_S17_TOTAL_PARAMETER_COUNT + CORRECTION_PARAMETER_COUNT

_ACTIVE_CORRECTION = None


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_texts(row):
    return (row.state_a, row.state_b)


def _question_texts(row):
    return (row.question_a1, row.question_a2, row.question_b1, row.question_b2)


def _option_texts(row):
    return (*row.option_texts, *row.option_aliases)


def _assert_s44_fresh(train_rows, dev_rows):
    s43._assert_s43_fresh(train_rows, dev_rows)
    current = (*train_rows, *dev_rows)
    current_states = {x for r in current for x in _state_texts(r)}
    current_questions = {x for r in current for x in _question_texts(r)}
    current_options = {x for r in current for x in _option_texts(r)}

    prior43 = (*generate_s43_cases("train"), *generate_s43_cases("dev"))
    prior43_states = {x for r in prior43 for x in _state_texts(r)}
    prior43_questions = {x for r in prior43 for x in _question_texts(r)}
    prior43_options = {x for r in prior43 for x in _option_texts(r)}
    if current_states & prior43_states:
        raise RuntimeError("S44 exact state overlap with exposed S43 rows")
    if current_questions & prior43_questions:
        raise RuntimeError("S44 exact question overlap with exposed S43 rows")
    if current_options & prior43_options:
        raise RuntimeError("S44 exact option overlap with exposed S43 rows")

    a0 = s44_a0_cases()
    a0_states = {x for r in a0 for x in (r.state_a, r.state_b)}
    a0_questions = {x for r in a0 for x in (r.qa1, r.qa2, r.qb1, r.qb2)}
    a0_options = set()
    for case in a0:
        options, _ga, _gb = case.option_pack()
        for option in options:
            a0_options.add(option.criterion_text)
            a0_options.update(option.aliases)
    if current_states & a0_states:
        raise RuntimeError("S44 TRAIN DEV state overlap with S44-A0")
    if current_questions & a0_questions:
        raise RuntimeError("S44 TRAIN DEV question overlap with S44-A0")
    if current_options & a0_options:
        raise RuntimeError("S44 TRAIN DEV option overlap with S44-A0")

def _new_correction(*, train=True):
    return PrivateCorrectionRepresentationFork(
        native_dimension=256,
        hidden_dimension=64,
        query_norm_epsilon=1e-12,
        private_norm_epsilon=1e-12,
        residual_scale=1.0,
        adapter_seed=65_044,
        train_correction=train,
    )


def _treatment_relation_outputs(
    runtime,
    *,
    state_tokens,
    state_mask,
    question_tokens,
    question_mask,
    encoded,
):
    if _ACTIVE_CORRECTION is None:
        raise RuntimeError("S44 treatment correction is not active")
    native_logits, signatures, diag = s35._native_relation_outputs(
        runtime,
        state_tokens=state_tokens,
        state_mask=state_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
        encoded=encoded,
    )
    corrected = _ACTIVE_CORRECTION.correction_logits(
        native_logits=native_logits,
        signatures=signatures,
        question_tokens=question_tokens,
        question_mask=question_mask,
    )
    return corrected, signatures, diag

def _set_relation_arm(arm, correction):
    global _ACTIVE_CORRECTION
    if arm == "reference":
        _ACTIVE_CORRECTION = None
        s17mod._relation_outputs = s35._native_relation_outputs
    elif arm == "treatment":
        if correction is None:
            raise RuntimeError("S44 treatment correction missing")
        _ACTIVE_CORRECTION = correction
        s17mod._relation_outputs = _treatment_relation_outputs
    else:
        raise ValueError(arm)


def _arm_losses(runtime, rows, arm, correction):
    _set_relation_arm(arm, correction)
    return s35._losses(runtime, rows)


def _arm_evaluate(runtime, rows, arm, correction):
    _set_relation_arm(arm, correction)
    return s35.evaluate(runtime, rows)


def _checkpoint_spec(arm):
    if arm == "reference":
        return (
            "reference-native-candidate.pt",
            "hira-v1-s44-reference-native-checkpoint-v1",
            "matched-reference-native-s35-relation-s17-shell",
            HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        )
    if arm == "treatment":
        return (
            "treatment-private-correction-candidate.pt",
            "hira-v1-s44-treatment-private-correction-checkpoint-v1",
            "matched-native-private-correction-representation-fork-s17",
            TREATMENT_TOTAL,
        )
    raise ValueError(arm)

def _runtime_state_sha256(runtime) -> str:
    digest = sha256()
    lora = a13_lora_state_dict(runtime.encoder)
    for name in sorted(lora):
        tensor = lora[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S44 projection scorer missing")
    tensor = scorer.projection.weight.detach().cpu().contiguous()
    digest.update(b"projection.weight")
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _native_losses(runtime, rows):
    s17mod._relation_outputs = s35._native_relation_outputs
    return s35._losses(runtime, rows)


def _correction_block(correction, relation_c, relation_p, signature_c, signature_p, encoded, rows):
    if correction is None:
        raise RuntimeError("S44 correction module missing")
    gold, _ = s35._gold_tensors(rows, device=relation_c.device)
    cc = correction.correction_logits(
        native_logits=relation_c,
        signatures=signature_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    cp = correction.correction_logits(
        native_logits=relation_p,
        signatures=signature_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    ce = 0.5 * (F.cross_entropy(cc, gold) + F.cross_entropy(cp, gold))
    return s35.BINDING_COEFFICIENT * ce, ce


def _correction_norms(correction):
    if correction is None:
        return {"adapter_a_norm": 0.0, "adapter_b_norm": 0.0, "w_norm": 0.0}
    return {
        "adapter_a_norm": float(correction.adapter_a.detach().norm().cpu()),
        "adapter_b_norm": float(correction.adapter_b.detach().norm().cpu()),
        "w_norm": float(correction.bilinear_weight.detach().norm().cpu()),
    }

def _train_arm(arm, bundle, manifest, train_rows, dev_rows, out_dir):
    checkpoint_name, checkpoint_schema, checkpoint_kind, expected_total = _checkpoint_spec(arm)

    random.seed(SEED)
    torch.manual_seed(SEED)

    frozen = load_hira_v0_m4_bundle(bundle)
    runtime = build_hira_v1_s17_norm_balanced_core(
        frozen.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    enforce_s17_eval(runtime)

    correction = _new_correction(train=True) if arm == "treatment" else None
    runtime_params = [p for p in runtime.parameters() if p.requires_grad]
    if sum(p.numel() for p in runtime_params) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
        raise RuntimeError(f"S44 {arm} runtime trainable count changed")
    correction_params = [] if correction is None else correction.correction_parameters()
    correction_count = sum(p.numel() for p in correction_params)
    trainable_count = sum(p.numel() for p in runtime_params) + correction_count
    if trainable_count != expected_total:
        raise RuntimeError(f"S44 {arm} trainable count changed: {trainable_count}")

    if s35._original_a13_trainable(runtime) != 0:
        raise RuntimeError(f"S44 {arm} original A13 became trainable")
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError(f"S44 {arm} projection scorer missing")
    if scorer.projection_trainable_parameter_count != HIRA_V1_S17_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError(f"S44 {arm} projection surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError(f"S44 {arm} HIRACore became trainable")
    if arm == "treatment":
        if correction is None:
            raise RuntimeError("S44 treatment correction missing")
        if correction.adapter_a_parameter_count != ADAPTER_A_PARAMETER_COUNT:
            raise RuntimeError("S44 treatment A capacity changed")
        if correction.adapter_b_parameter_count != ADAPTER_B_PARAMETER_COUNT:
            raise RuntimeError("S44 treatment B capacity changed")
        if correction.private_adapter_parameter_count != PRIVATE_ADAPTER_PARAMETER_COUNT:
            raise RuntimeError("S44 treatment private adapter capacity changed")
        if correction.bilinear_parameter_count != BILINEAR_PARAMETER_COUNT:
            raise RuntimeError("S44 treatment W capacity changed")
        if correction.correction_parameter_count != CORRECTION_PARAMETER_COUNT:
            raise RuntimeError("S44 treatment correction capacity changed")

    native_optimizer = torch.optim.AdamW(
        runtime_params, lr=s35.LR, weight_decay=s35.WEIGHT_DECAY
    )
    correction_optimizer = (
        None if correction is None else torch.optim.AdamW(
            correction_params, lr=s35.LR, weight_decay=s35.WEIGHT_DECAY
        )
    )
    history = []
    best_key = None
    best_epoch = None
    best_state = None
    best_metrics = None

    print(f"HIRA_V1_S44_{arm.upper()}_TRAIN_BEGIN", flush=True)

    for epoch in range(1, s35.EPOCHS + 1):
        order = list(range(len(train_rows)))
        random.Random(SEED + epoch).shuffle(order)
        totals = {
            "total": 0.0, "decision": 0.0, "ce": 0.0, "swap": 0.0,
            "option_alignment": 0.0, "binding": 0.0, "canonicalization": 0.0,
            "consistency_js": 0.0, "primary_block": 0.0, "relation_block": 0.0,
        }
        correction_total = 0.0
        correction_ce_total = 0.0
        balance = {
            "steps": 0, "conflicts": 0, "normalized_pre_dot": 0.0, "normalized_post_dot": 0.0,
            "primary_norm": 0.0, "relation_norm": 0.0, "reference_scale": 0.0,
            "direction_norm": 0.0, "combined_norm": 0.0, "projection_coefficient": 0.0,
        }
        state_view_encodes = 0

        for start in range(0, len(order), s35.BATCH_SIZE):
            rows = [train_rows[i] for i in order[start:start + s35.BATCH_SIZE]]
            native_optimizer.zero_grad(set_to_none=True)
            if correction_optimizer is not None:
                correction_optimizer.zero_grad(set_to_none=True)
            (
                _total,
                primary_block,
                native_relation_block,
                pieces,
                _fused_c,
                _fused_p,
                _raw_c,
                _raw_p,
                relation_c,
                relation_p,
                signature_c,
                signature_p,
                encoded,
            ) = _native_losses(runtime, rows)

            primary_raw = torch.autograd.grad(
                primary_block, runtime_params, retain_graph=True, allow_unused=True
            )
            relation_raw = torch.autograd.grad(
                native_relation_block,
                runtime_params,
                retain_graph=(correction is not None),
                allow_unused=True,
            )
            primary = [torch.zeros_like(p) if g is None else g for p, g in zip(runtime_params, primary_raw)]
            relation = [torch.zeros_like(p) if g is None else g for p, g in zip(runtime_params, relation_raw)]
            combined, diag = norm_balanced_gradient_update(primary, relation, epsilon=s35.BALANCE_EPSILON)
            apply_gradient_update(runtime_params, combined)

            correction_block_value = 0.0
            correction_ce_value = 0.0
            if correction is not None:
                correction_block, correction_ce = _correction_block(
                    correction,
                    relation_c,
                    relation_p,
                    signature_c,
                    signature_p,
                    encoded,
                    rows,
                )
                correction_grads = torch.autograd.grad(
                    correction_block, correction_params, allow_unused=True
                )
                if correction_grads[-1] is None:
                    raise RuntimeError("S44 treatment correction W gradient missing")
                if (
                    not bool(torch.isfinite(correction_grads[-1]).all())
                    or float(correction_grads[-1].abs().sum()) <= 0.0
                ):
                    raise RuntimeError("S44 treatment correction W gradient vanished")
                for p, g in zip(correction_params, correction_grads):
                    p.grad = None if g is None else g.detach().clone()
                correction_block_value = float(correction_block.detach().cpu())
                correction_ce_value = float(correction_ce.detach().cpu())

            torch.nn.utils.clip_grad_norm_(runtime_params, s35.GRAD_CLIP)
            native_optimizer.step()
            if correction_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(correction_params, s35.GRAD_CLIP)
                correction_optimizer.step()
            enforce_s17_eval(runtime)

            n = len(rows)
            for key in totals:
                totals[key] += pieces[key] * n
            correction_total += correction_block_value * n
            correction_ce_total += correction_ce_value * n
            balance["steps"] += 1
            balance["conflicts"] += int(diag.conflict)
            balance["normalized_pre_dot"] += diag.normalized_pre_dot
            balance["normalized_post_dot"] += diag.normalized_post_dot
            balance["primary_norm"] += diag.primary_norm
            balance["relation_norm"] += diag.relation_norm
            balance["reference_scale"] += diag.reference_scale
            balance["direction_norm"] += diag.direction_norm
            balance["combined_norm"] += diag.combined_norm
            balance["projection_coefficient"] += diag.projection_coefficient
            state_view_encodes += 2 * n

        if state_view_encodes != 2 * len(train_rows):
            raise RuntimeError(f"S44 {arm} TRAIN state-once changed")

        dev = _arm_evaluate(runtime, dev_rows, arm, correction)
        if int(dev["state_view_encodes"]) != 2 * len(dev_rows):
            raise RuntimeError(f"S44 {arm} DEV state-once changed")

        steps = max(1, balance["steps"])
        correction_norms = _correction_norms(correction)
        runtime_hash = _runtime_state_sha256(runtime)
        record = {
            "epoch": epoch,
            "train_mean_total_loss": totals["total"] / len(train_rows),
            "train_mean_decision_loss": totals["decision"] / len(train_rows),
            "train_mean_ce": totals["ce"] / len(train_rows),
            "train_mean_swap": totals["swap"] / len(train_rows),
            "train_mean_option_alignment_loss": totals["option_alignment"] / len(train_rows),
            "train_mean_binding_loss": totals["binding"] / len(train_rows),
            "train_mean_canonicalization_loss": totals["canonicalization"] / len(train_rows),
            "train_mean_consistency_js": totals["consistency_js"] / len(train_rows),
            "train_mean_primary_block": totals["primary_block"] / len(train_rows),
            "train_mean_relation_block": totals["relation_block"] / len(train_rows),
            "train_mean_correction_block": correction_total / len(train_rows),
            "train_mean_correction_ce": correction_ce_total / len(train_rows),
            "runtime_state_sha256": runtime_hash,
            "norm_balancing": {
                "steps": balance["steps"],
                "conflict_rate": balance["conflicts"] / steps,
                "mean_normalized_pre_dot": balance["normalized_pre_dot"] / steps,
                "mean_normalized_post_dot": balance["normalized_post_dot"] / steps,
                "mean_primary_norm": balance["primary_norm"] / steps,
                "mean_relation_norm": balance["relation_norm"] / steps,
                "mean_reference_scale": balance["reference_scale"] / steps,
                "mean_direction_norm": balance["direction_norm"] / steps,
                "mean_combined_norm": balance["combined_norm"] / steps,
                "mean_projection_coefficient": balance["projection_coefficient"] / steps,
            },
            "train_state_view_encodes": state_view_encodes,
            "surface_diagnostics": {
                "projection_weight_norm": float(scorer.projection.weight.detach().norm().cpu()),
                "lora_b_norm": float(torch.sqrt(sum(
                    module.lora_b.detach().pow(2).sum()
                    for module in iter_a13_lora_modules(runtime.encoder)
                )).cpu()),
                "adapter_a_norm": correction_norms["adapter_a_norm"],
                "adapter_b_norm": correction_norms["adapter_b_norm"],
                "correction_weight_norm": correction_norms["w_norm"],
            },
            "dev": dev,
        }
        history.append(record)

        key = s35._selection_key(epoch, dev)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                "lora": a13_lora_state_dict(runtime.encoder),
                "projection": {"projection.weight": scorer.projection.weight.detach().cpu().clone()},
                "correction": None if correction is None else correction.correction_state_dict(),
            }
            best_metrics = dict(dev)

        print("HIRA_V1_S44_ARM_EPOCH=" + json.dumps({"arm": arm, **record}, sort_keys=True), flush=True)

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S44 {arm} selection produced no checkpoint")

    final_runtime_hash = history[-1]["runtime_state_sha256"]
    load_a13_lora_state_dict(runtime.encoder, best_state["lora"], freeze=False)
    scorer.load_projection_state_dict(best_state["projection"], freeze=False)
    if correction is not None:
        if best_state["correction"] is None:
            raise RuntimeError("S44 treatment selected correction state missing")
        correction.load_correction_state_dict(best_state["correction"], freeze=False)
    enforce_s17_eval(runtime)
    selected = _arm_evaluate(runtime, dev_rows, arm, correction)

    replay_keys = (
        "fused_canonical_accuracy", "fused_paraphrase_accuracy",
        "fused_canonical_paired_both_correct_rate",
        "fused_question_swap_choice_change_rate",
        "fused_cross_view_selected_choice_agreement", "fused_cross_view_mean_js",
        "fused_canonical_mean_gold_margin", "canonical_relation_binding_accuracy",
        "canonical_relation_binding_mean_gold_margin",
        "mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin",
        "fused_option_order_flip_rate", "fused_max_probability_mass_error",
        "mean_canonical_decision_loss",
    )
    for metric in replay_keys:
        if not math.isclose(float(selected[metric]), float(best_metrics[metric]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"S44 {arm} selected DEV replay changed: {metric}")

    lora_trainable = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" in name
    )
    gates = s35._gates(
        runtime, selected, history, trainable_count, lora_trainable,
        expected_total, HIRA_V1_S6_LORA_PARAMETER_COUNT, train_rows, dev_rows,
    )
    gates["correction_capacity_exact"] = (
        True if arm == "reference" else (
            correction is not None
            and correction.adapter_a_parameter_count == ADAPTER_A_PARAMETER_COUNT
            and correction.adapter_b_parameter_count == ADAPTER_B_PARAMETER_COUNT
            and correction.private_adapter_parameter_count == PRIVATE_ADAPTER_PARAMETER_COUNT
            and correction.bilinear_parameter_count == BILINEAR_PARAMETER_COUNT
            and correction.correction_parameter_count == CORRECTION_PARAMETER_COUNT
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / checkpoint_name
    torch.save(
        {
            "schema_version": checkpoint_schema,
            "kind": checkpoint_kind,
            "arm": arm,
            "lora_parameter_count": HIRA_V1_S6_LORA_PARAMETER_COUNT,
            "projection_parameter_count": HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
            "adapter_a_parameter_count": 0 if correction is None else ADAPTER_A_PARAMETER_COUNT,
            "adapter_b_parameter_count": 0 if correction is None else ADAPTER_B_PARAMETER_COUNT,
            "private_adapter_parameter_count": 0 if correction is None else PRIVATE_ADAPTER_PARAMETER_COUNT,
            "bilinear_parameter_count": 0 if correction is None else BILINEAR_PARAMETER_COUNT,
            "correction_parameter_count": 0 if correction is None else CORRECTION_PARAMETER_COUNT,
            "total_parameter_count": expected_total,
            "lora_rank": 8,
            "selected_dev_epoch": best_epoch,
            "semantic_revision": str(manifest["semantic_revision"]),
            "initialization_t0_sha256": str(manifest["t0_checkpoint_sha256"]),
            "lora_state_dict": best_state["lora"],
            "projection_state_dict": best_state["projection"],
            "correction_state_dict": {} if best_state["correction"] is None else best_state["correction"],
        },
        checkpoint,
    )

    return {
        "arm": arm,
        "selected_dev_epoch": best_epoch,
        "selected_dev": selected,
        "gates": gates,
        "dev_ready": all(gates.values()),
        "history": history,
        "runtime_trajectory_sha256": [x["runtime_state_sha256"] for x in history],
        "final_runtime_state_sha256": final_runtime_hash,
        "checkpoint_file": checkpoint_name,
        "checkpoint_sha256": _sha256(checkpoint),
        "parameter_surface": {
            "total_trainable_parameters": trainable_count,
            "lora_trainable_parameters": lora_trainable,
            "projection_trainable_parameters": scorer.projection_trainable_parameter_count,
            "correction_trainable_parameters": 0 if correction is None else CORRECTION_PARAMETER_COUNT,
            "adapter_a_trainable_parameters": 0 if correction is None else ADAPTER_A_PARAMETER_COUNT,
            "adapter_b_trainable_parameters": 0 if correction is None else ADAPTER_B_PARAMETER_COUNT,
            "bilinear_trainable_parameters": 0 if correction is None else BILINEAR_PARAMETER_COUNT,
            "original_a13_trainable": s35._original_a13_trainable(runtime),
            "hira_core_trainable": sum(p.numel() for p in runtime.hira.parameters() if p.requires_grad),
            "fusion_added_parameters": GradientIsolatedFullKEvidenceFusion(epsilon=s35.FUSION_EPSILON).parameter_count,
            "learned_downstream_scorer_parameters": 0 if correction is None else CORRECTION_PARAMETER_COUNT,
        },
        "norm_balancing": {
            "rule": "equal_direction_relation_priority_projection",
            "epsilon": s35.BALANCE_EPSILON,
            "mean_conflict_rate": sum(x["norm_balancing"]["conflict_rate"] for x in history) / len(history),
        },
    }


_DELTA_KEYS = (
    "fused_canonical_accuracy",
    "fused_paraphrase_accuracy",
    "fused_canonical_paired_both_correct_rate",
    "fused_question_swap_choice_change_rate",
    "fused_cross_view_selected_choice_agreement",
    "fused_cross_view_mean_js",
    "fused_canonical_mean_gold_margin",
    "fused_paraphrase_mean_gold_margin",
    "raw_triadic_canonical_accuracy",
    "raw_triadic_paraphrase_accuracy",
    "canonical_relation_binding_accuracy",
    "paraphrase_relation_binding_accuracy",
    "canonical_relation_binding_mean_gold_margin",
    "paraphrase_relation_binding_mean_gold_margin",
    "relation_cross_view_agreement",
    "mean_same_option_signature_cosine",
    "mean_signature_same_vs_strongest_wrong_margin",
)


def _metric_delta_values(reference_metrics, treatment_metrics):
    return {
        key: float(treatment_metrics[key]) - float(reference_metrics[key])
        for key in _DELTA_KEYS
    }


def _metric_deltas(reference, treatment):
    return _metric_delta_values(reference["selected_dev"], treatment["selected_dev"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--a0-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a0 = json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome") != "HIRA_V1_S44_A0_PRIVATE_CORRECTION_REPRESENTATION_READY":
        raise RuntimeError("S44-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S44-A0 unexpectedly used for model selection")
    expected = {
        "native_trainable_parameter_count": 49_152,
        "adapter_a_parameter_count": ADAPTER_A_PARAMETER_COUNT,
        "adapter_b_parameter_count": ADAPTER_B_PARAMETER_COUNT,
        "private_adapter_parameter_count": PRIVATE_ADAPTER_PARAMETER_COUNT,
        "bilinear_parameter_count": BILINEAR_PARAMETER_COUNT,
        "correction_parameter_count": CORRECTION_PARAMETER_COUNT,
        "treatment_total_trainable_parameter_count": TREATMENT_TOTAL,
    }
    for key, value in expected.items():
        if int(a0.get(key, -1)) != value:
            raise RuntimeError(f"S44-A0 capacity changed: {key}")
    for key in (
        "zero_init_corrected_relation_max_abs",
        "zero_init_private_residual_max_abs",
        "zero_init_fused_logit_max_abs",
        "zero_init_native_parameter_max_abs",
        "zero_init_native_relation_max_abs",
        "zero_init_native_signature_max_abs",
        "zero_init_native_primary_max_abs",
        "zero_init_correction_native_runtime_gradient_l1",
        "zero_init_correction_reference_runtime_gradient_l1",
        "native_objective_correction_gradient_l1",
        "matched_native_gradient_max_abs",
        "matched_native_one_step_parameter_max_abs",
        "matched_native_one_step_output_max_abs",
        "warm_correction_native_runtime_gradient_l1",
        "warm_correction_lora_gradient_l1",
        "warm_correction_projection_gradient_l1",
    ):
        if float(a0.get(key, -1.0)) != 0.0:
            raise RuntimeError(f"S44-A0 ownership/identity contract failed: {key}")
    if float(a0.get("zero_init_private_signature_max_abs", 1.0)) > 2e-6:
        raise RuntimeError("S44-A0 zero-B private signature identity changed")
    if float(a0.get("zero_init_selected_choice_identity_rate", 0.0)) != 1.0:
        raise RuntimeError("S44-A0 selected-choice identity failed")
    warm = a0.get("warm_start", [])
    if len(warm) != 3:
        raise RuntimeError("S44-A0 warm-start record changed")
    if not (
        float(warm[0]["w_gradient_l1"]) > 0.0
        and float(warm[0]["adapter_a_gradient_l1"]) == 0.0
        and float(warm[0]["adapter_b_gradient_l1"]) == 0.0
        and float(warm[1]["w_gradient_l1"]) > 0.0
        and float(warm[1]["adapter_b_gradient_l1"]) > 0.0
        and float(warm[2]["w_gradient_l1"]) > 0.0
        and float(warm[2]["adapter_b_gradient_l1"]) > 0.0
        and float(warm[2]["adapter_a_gradient_l1"]) > 0.0
    ):
        raise RuntimeError("S44-A0 W->B->A liveness changed")
    if float(a0.get("private_vs_w_only_residual_max_abs", 0.0)) <= 1e-7:
        raise RuntimeError("S44-A0 private family distinction vanished")
    if not bool(a0.get("checkpoint_roundtrip_exact", False)):
        raise RuntimeError("S44-A0 checkpoint roundtrip failed")
    if not bool(a0.get("full_k", False)):
        raise RuntimeError("S44-A0 full-K failed")

    train_rows = generate_s44_cases("train")
    dev_rows = generate_s44_cases("dev")
    validate_s44_partitions(train_rows, dev_rows)
    _assert_s44_fresh(train_rows, dev_rows)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"]) != A13_REVISION:
        raise RuntimeError("S44 semantic revision changed")

    args.out.mkdir(parents=True, exist_ok=True)
    reference = _train_arm("reference", bundle, manifest, train_rows, dev_rows, args.out)
    treatment = _train_arm("treatment", bundle, manifest, train_rows, dev_rows, args.out)

    trajectory_equal = reference["runtime_trajectory_sha256"] == treatment["runtime_trajectory_sha256"]
    if not trajectory_equal:
        raise RuntimeError("S44 reference/treatment native runtime trajectory diverged")
    treatment["gates"]["runtime_trajectory_identity"] = True
    treatment["dev_ready"] = all(treatment["gates"].values())

    treatment_epoch = int(treatment["selected_dev_epoch"])
    if not 1 <= treatment_epoch <= len(reference["history"]):
        raise RuntimeError("S44 treatment selected epoch outside matched reference history")
    same_epoch_reference = dict(reference["history"][treatment_epoch - 1]["dev"])
    same_epoch_reference_runtime_hash = reference["history"][treatment_epoch - 1]["runtime_state_sha256"]
    same_epoch_treatment_runtime_hash = treatment["history"][treatment_epoch - 1]["runtime_state_sha256"]
    if same_epoch_reference_runtime_hash != same_epoch_treatment_runtime_hash:
        raise RuntimeError("S44 same-epoch native runtime fingerprint diverged")
    same_epoch_delta = _metric_delta_values(
        same_epoch_reference,
        treatment["selected_dev"],
    )

    if reference["dev_ready"] and treatment["dev_ready"]:
        outcome = "HIRA_V1_S44_MATCHED_PRIVATE_CORRECTION_REPRESENTATION_BOTH_DEV_READY"
    elif reference["dev_ready"]:
        outcome = "HIRA_V1_S44_MATCHED_PRIVATE_CORRECTION_REPRESENTATION_REFERENCE_DEV_READY"
    elif treatment["dev_ready"]:
        outcome = "HIRA_V1_S44_MATCHED_PRIVATE_CORRECTION_REPRESENTATION_TREATMENT_DEV_READY"
    else:
        outcome = "HIRA_V1_S44_MATCHED_PRIVATE_CORRECTION_REPRESENTATION_DEV_COMPLETE"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S44_FRESH_MATCHED_NATIVE_REFERENCE_VS_PRIVATE_CORRECTION_REPRESENTATION_FORK",
        "seed": SEED,
        "optimizer": {
            "name": "AdamW",
            "epochs": s35.EPOCHS,
            "batch_size_semantic_cases": s35.BATCH_SIZE,
            "lr": s35.LR,
            "weight_decay": s35.WEIGHT_DECAY,
            "grad_clip": s35.GRAD_CLIP,
            "correction_separate_lr": False,
        },
        "loss": {
            "cross_entropy_both_views": True,
            "swap_margin_coefficient": s35.SWAP_COEFFICIENT,
            "swap_margin": s35.SWAP_MARGIN,
            "option_view_infonce_coefficient": s35.OPTION_ALIGN_COEFFICIENT,
            "option_view_infonce_temperature": s35.OPTION_ALIGN_TEMPERATURE,
            "relation_structured_binding_coefficient": s35.BINDING_COEFFICIENT,
            "cross_view_relation_canonicalization_coefficient": s35.CANONICALIZATION_COEFFICIENT,
            "signature_separation_margin": s35.SIGNATURE_SEPARATION_MARGIN,
            "role_temperature": s35.ROLE_TEMPERATURE,
            "pair_temperature": s35.PAIR_TEMPERATURE,
            "binding_contrastive_temperature": s35.BINDING_CONTRASTIVE_TEMPERATURE,
            "cross_view_js_coefficient": s35.INVARIANCE_COEFFICIENT,
            "fusion_epsilon": s35.FUSION_EPSILON,
            "fusion_equal_weight": 0.5,
            "fused_primary_relation_logits_detached": True,
            "norm_balanced_gradient": "equal_direction_relation_priority_projection",
            "balance_epsilon": s35.BALANCE_EPSILON,
            "reference_relation_operator": "S35_NativeA13RelationCanonicalizer",
            "treatment_relation_operator": "S44_PrivateCorrectionRepresentationFork",
            "native_relation_dimension": 256,
            "treatment_uses_shared_projection": False,
            "treatment_private_adapter_parameters": PRIVATE_ADAPTER_PARAMETER_COUNT,
            "treatment_bilinear_parameters": BILINEAR_PARAMETER_COUNT,
            "treatment_correction_parameters": CORRECTION_PARAMETER_COUNT,
            "treatment_total_trainable_parameters": TREATMENT_TOTAL,
            "treatment_query_summary": "masked_mean_l2",
            "treatment_bilinear_shape": [256, 256],
            "treatment_factorized": False,
            "treatment_query_norm_epsilon": 1e-12,
            "treatment_residual_scale": 1.0,
            "treatment_bias": False,
            "treatment_nonlinearity": "GELU_private_adapter",
            "runtime_training_uses_native_relation_only": True,
            "correction_features_detached": True,
            "correction_coefficient": 0.10,
            "runtime_grad_clip": 1.0,
            "correction_grad_clip": 1.0,
            "correction_optimizer_separate_from_native": True,
            "second_encoder_pass": False,
        },
        "partitions": {
            "train_semantic_cases": len(train_rows),
            "dev_semantic_cases": len(dev_rows),
            "language": "en",
            "domains": sorted({r.domain for r in train_rows}),
            "state_views_per_case": 2,
            "question_views_per_semantic_query": 2,
            "k": 4,
            "views_per_option": 2,
            "identical_rows_across_arms": True,
            "identical_batch_order_across_arms": True,
            "prior_track_exact_rows_used": False,
            "s44_a0_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "trajectory_invariant": {
            "all_epoch_runtime_state_sha256_equal": trajectory_equal,
            "epoch_count": len(reference["runtime_trajectory_sha256"]),
            "lora_state_max_abs_reference_treatment": 0.0,
            "projection_state_max_abs_reference_treatment": 0.0,
            "native_pre_w_logit_max_abs_reference_treatment": 0.0,
            "native_signature_max_abs_reference_treatment": 0.0,
            "basis": "bitwise-identical matched runtime trajectory fingerprints",
        },
        "reference_arm": reference,
        "treatment_arm": treatment,
        "matched_selected_dev_delta_treatment_minus_reference": _metric_deltas(reference, treatment),
        "same_epoch_counterfactual": {
            "treatment_selected_epoch": treatment_epoch,
            "reference_runtime_state_sha256": same_epoch_reference_runtime_hash,
            "treatment_runtime_state_sha256": same_epoch_treatment_runtime_hash,
            "runtime_state_sha256_equal": True,
            "reference_native_dev_at_treatment_selected_epoch": same_epoch_reference,
            "delta_treatment_minus_same_epoch_reference": same_epoch_delta,
            "interpretation": "private correction representation evaluation effect on an identical native runtime epoch",
        },
        "post_dev_tuning_performed": False,
        "second_dev_run_performed": False,
        "sealed_confirm_opened": False,
        "multilingual_probe_opened": False,
        "production_ready_claimed": False,
    }

    (args.out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "train-manifest.json").write_text(
        json.dumps([r.to_dict() for r in train_rows], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "dev-manifest.json").write_text(
        json.dumps([r.to_dict() for r in dev_rows], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S44_TRAIN_DEV_RECEIPT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
