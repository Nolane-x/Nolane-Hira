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
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_gradient_isolated_bilinear_readout import NativeGradientIsolatedBilinearCorrectnessReadout
from nmd.v1_norm_balanced_gradient import apply_gradient_update, norm_balanced_gradient_update
from nmd.v1_s38_authority import generate_s38_cases
from nmd.v1_s39_authority import generate_s39_cases, validate_s39_partitions
from nmd.v1_s6_semantic_core import HIRA_V1_S6_LORA_PARAMETER_COUNT
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
import hira_v1_s17_train_dev as s17mod
import hira_v1_s35_train_dev as s35
import hira_v1_s38_train_dev as s38
from hira_v1_s39_a0_full_bilinear_readout import cases as s39_a0_cases

SCHEMA_VERSION = "hira-v1-s39-matched-gradient-isolated-bilinear-train-dev-v1"
SEED = 60001
READOUT_PARAMETER_COUNT = 65_536
TREATMENT_TOTAL = HIRA_V1_S17_TOTAL_PARAMETER_COUNT + READOUT_PARAMETER_COUNT

_ACTIVE_READOUT = None


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


def _assert_s39_fresh(train_rows, dev_rows):
    # Inherit the complete S0-S37 plus S38-A0 exact-overlap checks,
    # then add exposed S38 TRAIN/DEV and S39-A0.
    s38._assert_s38_fresh(train_rows, dev_rows)

    current = (*train_rows, *dev_rows)
    current_states = {x for r in current for x in _state_texts(r)}
    current_questions = {x for r in current for x in _question_texts(r)}
    current_options = {x for r in current for x in _option_texts(r)}

    prior38 = (*generate_s38_cases("train"), *generate_s38_cases("dev"))
    prior38_states = {x for r in prior38 for x in _state_texts(r)}
    prior38_questions = {x for r in prior38 for x in _question_texts(r)}
    prior38_options = {x for r in prior38 for x in _option_texts(r)}
    if current_states & prior38_states:
        raise RuntimeError("S39 exact state overlap with exposed S38 rows")
    if current_questions & prior38_questions:
        raise RuntimeError("S39 exact question overlap with exposed S38 rows")
    if current_options & prior38_options:
        raise RuntimeError("S39 exact option overlap with exposed S38 rows")

    a0 = s39_a0_cases()
    a0_states = {x for r in a0 for x in (r.state_a, r.state_b)}
    a0_questions = {x for r in a0 for x in (r.qa1, r.qa2, r.qb1, r.qb2)}
    a0_options = set()
    for case in a0:
        options, _ga, _gb = case.option_pack()
        for option in options:
            a0_options.add(option.criterion_text)
            a0_options.update(option.aliases)
    if current_states & a0_states:
        raise RuntimeError("S39 TRAIN DEV state overlap with S39-A0")
    if current_questions & a0_questions:
        raise RuntimeError("S39 TRAIN DEV question overlap with S39-A0")
    if current_options & a0_options:
        raise RuntimeError("S39 TRAIN DEV option overlap with S39-A0")


def _new_readout(*, train=True):
    return NativeGradientIsolatedBilinearCorrectnessReadout(
        role_temperature=s35.ROLE_TEMPERATURE,
        pair_temperature=s35.PAIR_TEMPERATURE,
        contrastive_temperature=s35.BINDING_CONTRASTIVE_TEMPERATURE,
        native_dimension=256,
        query_norm_epsilon=1e-12,
        residual_scale=1.0,
        train_readout=train,
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
    del runtime
    if _ACTIVE_READOUT is None:
        raise RuntimeError("S39 treatment readout is not active")
    return _ACTIVE_READOUT(
        state_tokens=state_tokens.repeat_interleave(2, dim=0),
        state_mask=state_mask.repeat_interleave(2, dim=0),
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2, dim=0),
    )


def _set_relation_arm(arm, readout):
    global _ACTIVE_READOUT
    if arm == "control":
        _ACTIVE_READOUT = None
        s17mod._relation_outputs = s35._native_relation_outputs
    elif arm == "treatment":
        if readout is None:
            raise RuntimeError("S39 treatment readout missing")
        _ACTIVE_READOUT = readout
        s17mod._relation_outputs = _treatment_relation_outputs
    else:
        raise ValueError(arm)


def _arm_losses(runtime, rows, arm, readout):
    _set_relation_arm(arm, readout)
    return s35._losses(runtime, rows)


def _arm_evaluate(runtime, rows, arm, readout):
    _set_relation_arm(arm, readout)
    return s35.evaluate(runtime, rows)


def _checkpoint_spec(arm):
    if arm == "control":
        return (
            "control-native-candidate.pt",
            "hira-v1-s39-control-native-checkpoint-v1",
            "matched-control-native-s35-relation-s17-shell-a13-w28",
            HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        )
    if arm == "treatment":
        return (
            "treatment-readout-candidate.pt",
            "hira-v1-s39-treatment-gradient-isolated-bilinear-checkpoint-v1",
            "matched-native-gradient-isolated-bilinear-s17-shell-a13-w28",
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
        raise RuntimeError("S39 projection scorer missing")
    tensor = scorer.projection.weight.detach().cpu().contiguous()
    digest.update(b"projection.weight")
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _native_losses(runtime, rows):
    s17mod._relation_outputs = s35._native_relation_outputs
    return s35._losses(runtime, rows)


def _correction_block(readout, relation_c, relation_p, signature_c, signature_p, encoded, rows):
    if readout is None:
        raise RuntimeError("S39 correction readout missing")
    gold, _ = s35._gold_tensors(rows, device=relation_c.device)
    cc = readout.correction_logits(
        native_logits=relation_c,
        signatures=signature_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    cp = readout.correction_logits(
        native_logits=relation_p,
        signatures=signature_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    ce = 0.5 * (F.cross_entropy(cc, gold) + F.cross_entropy(cp, gold))
    return s35.BINDING_COEFFICIENT * ce, ce


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

    readout = _new_readout(train=True) if arm == "treatment" else None
    runtime_params = [p for p in runtime.parameters() if p.requires_grad]
    if sum(p.numel() for p in runtime_params) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
        raise RuntimeError(f"S39 {arm} runtime trainable count changed")
    optimizer_params = list(runtime_params)
    if readout is not None:
        optimizer_params.append(readout.bilinear_weight)
    trainable_count = sum(p.numel() for p in optimizer_params)
    if trainable_count != expected_total:
        raise RuntimeError(f"S39 {arm} trainable count changed: {trainable_count}")

    if s35._original_a13_trainable(runtime) != 0:
        raise RuntimeError(f"S39 {arm} original A13 became trainable")
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError(f"S39 {arm} projection scorer missing")
    if scorer.projection_trainable_parameter_count != HIRA_V1_S17_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError(f"S39 {arm} projection surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError(f"S39 {arm} HIRACore became trainable")
    if arm == "treatment" and (readout is None or readout.added_parameter_count != 65_536):
        raise RuntimeError("S39 treatment W capacity changed")

    optimizer = torch.optim.AdamW(optimizer_params, lr=s35.LR, weight_decay=s35.WEIGHT_DECAY)
    history = []
    best_key = None
    best_epoch = None
    best_state = None
    best_metrics = None

    print(f"HIRA_V1_S39_{arm.upper()}_TRAIN_BEGIN", flush=True)

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
            optimizer.zero_grad(set_to_none=True)
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
                retain_graph=(readout is not None),
                allow_unused=True,
            )
            primary = [torch.zeros_like(p) if g is None else g for p, g in zip(runtime_params, primary_raw)]
            relation = [torch.zeros_like(p) if g is None else g for p, g in zip(runtime_params, relation_raw)]
            combined, diag = norm_balanced_gradient_update(primary, relation, epsilon=s35.BALANCE_EPSILON)
            apply_gradient_update(runtime_params, combined)

            correction_block_value = 0.0
            correction_ce_value = 0.0
            if readout is not None:
                correction_block, correction_ce = _correction_block(
                    readout,
                    relation_c,
                    relation_p,
                    signature_c,
                    signature_p,
                    encoded,
                    rows,
                )
                w_grad = torch.autograd.grad(correction_block, readout.bilinear_weight)[0]
                if not bool(torch.isfinite(w_grad).all()) or float(w_grad.abs().sum()) <= 0.0:
                    raise RuntimeError("S39 treatment correction W gradient vanished")
                readout.bilinear_weight.grad = w_grad
                correction_block_value = float(correction_block.detach().cpu())
                correction_ce_value = float(correction_ce.detach().cpu())

            torch.nn.utils.clip_grad_norm_(runtime_params, s35.GRAD_CLIP)
            if readout is not None:
                torch.nn.utils.clip_grad_norm_([readout.bilinear_weight], s35.GRAD_CLIP)
            optimizer.step()
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
            raise RuntimeError(f"S39 {arm} TRAIN state-once changed")

        dev = _arm_evaluate(runtime, dev_rows, arm, readout)
        if int(dev["state_view_encodes"]) != 2 * len(dev_rows):
            raise RuntimeError(f"S39 {arm} DEV state-once changed")

        steps = max(1, balance["steps"])
        readout_norm = 0.0 if readout is None else float(readout.bilinear_weight.detach().norm().cpu())
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
                "readout_weight_norm": readout_norm,
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
                "readout": None if readout is None else readout.readout_state_dict(),
            }
            best_metrics = dict(dev)

        print("HIRA_V1_S39_ARM_EPOCH=" + json.dumps({"arm": arm, **record}, sort_keys=True), flush=True)

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S39 {arm} selection produced no checkpoint")

    final_runtime_hash = history[-1]["runtime_state_sha256"]
    load_a13_lora_state_dict(runtime.encoder, best_state["lora"], freeze=False)
    scorer.load_projection_state_dict(best_state["projection"], freeze=False)
    if readout is not None:
        if best_state["readout"] is None:
            raise RuntimeError("S39 treatment selected readout state missing")
        readout.load_readout_state_dict(best_state["readout"], freeze=False)
    enforce_s17_eval(runtime)
    selected = _arm_evaluate(runtime, dev_rows, arm, readout)

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
            raise RuntimeError(f"S39 {arm} selected DEV replay changed: {metric}")

    lora_trainable = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" in name
    )
    gates = s35._gates(
        runtime, selected, history, trainable_count, lora_trainable,
        expected_total, HIRA_V1_S6_LORA_PARAMETER_COUNT, train_rows, dev_rows,
    )
    gates["readout_capacity_exact"] = (
        True if arm == "control" else readout is not None and readout.added_parameter_count == 65_536
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
            "readout_parameter_count": 0 if readout is None else 65_536,
            "total_parameter_count": expected_total,
            "lora_rank": 8,
            "selected_dev_epoch": best_epoch,
            "semantic_revision": str(manifest["semantic_revision"]),
            "initialization_t0_sha256": str(manifest["t0_checkpoint_sha256"]),
            "lora_state_dict": best_state["lora"],
            "projection_state_dict": best_state["projection"],
            "readout_state_dict": {} if best_state["readout"] is None else best_state["readout"],
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
            "readout_trainable_parameters": 0 if readout is None else 65_536,
            "original_a13_trainable": s35._original_a13_trainable(runtime),
            "hira_core_trainable": sum(p.numel() for p in runtime.hira.parameters() if p.requires_grad),
            "fusion_added_parameters": GradientIsolatedFullKEvidenceFusion(epsilon=s35.FUSION_EPSILON).parameter_count,
            "learned_downstream_scorer_parameters": 0 if readout is None else 65_536,
        },
        "norm_balancing": {
            "rule": "equal_direction_relation_priority_projection",
            "epsilon": s35.BALANCE_EPSILON,
            "mean_conflict_rate": sum(x["norm_balancing"]["conflict_rate"] for x in history) / len(history),
        },
    }


def _metric_deltas(control, treatment):
    keys = (
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
    a = control["selected_dev"]
    b = treatment["selected_dev"]
    return {key: float(b[key]) - float(a[key]) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--a0-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a0 = json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome") != "HIRA_V1_S39_A0_GRADIENT_ISOLATED_BILINEAR_READY":
        raise RuntimeError("S39-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S39-A0 unexpectedly used for model selection")
    if int(a0.get("operator_added_parameter_count", -1)) != 65_536:
        raise RuntimeError("S39-A0 W parameter count changed")
    if int(a0.get("control_trainable_parameter_count", -1)) != 49_152:
        raise RuntimeError("S39-A0 control surface changed")
    if int(a0.get("treatment_trainable_parameter_count", -1)) != 114_688:
        raise RuntimeError("S39-A0 treatment surface changed")
    for key in (
        "correction_to_lora_gradient_l1",
        "correction_to_projection_gradient_l1",
        "correction_to_hira_gradient_l1",
        "primary_to_w_gradient_l1",
        "native_relation_to_w_gradient_l1",
        "matched_runtime_gradient_max_abs",
        "matched_runtime_update_max_abs",
        "zero_init_relation_logit_max_abs",
        "zero_init_signature_max_abs",
        "zero_init_primary_logit_max_abs",
        "zero_init_fused_logit_max_abs",
        "native_projection_perturbation_logit_max_abs",
        "native_projection_perturbation_signature_max_abs",
    ):
        if float(a0.get(key, -1.0)) != 0.0:
            raise RuntimeError(f"S39-A0 exact isolation contract failed: {key}")
    for key in (
        "correction_to_w_gradient_l1",
        "correction_to_w_offdiagonal_gradient_l1",
        "native_relation_to_lora_gradient_l1",
        "primary_to_projection_gradient_l1",
        "matched_w_update_max_abs",
        "query_intervention_residual_max_abs",
        "signature_intervention_residual_max_abs",
        "primary_projection_perturbation_logit_max_abs",
    ):
        if float(a0.get(key, 0.0)) <= 0.0:
            raise RuntimeError(f"S39-A0 live mechanism failed: {key}")
    if float(a0.get("zero_init_selected_choice_identity_rate", 0.0)) != 1.0:
        raise RuntimeError("S39-A0 selected-choice identity failed")
    if not bool(a0.get("checkpoint_roundtrip_exact", False)):
        raise RuntimeError("S39-A0 W checkpoint roundtrip failed")
    if not bool(a0.get("full_k", False)):
        raise RuntimeError("S39-A0 full-K failed")

    train_rows = generate_s39_cases("train")
    dev_rows = generate_s39_cases("dev")
    validate_s39_partitions(train_rows, dev_rows)
    _assert_s39_fresh(train_rows, dev_rows)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"]) != A13_REVISION:
        raise RuntimeError("S39 semantic revision changed")

    args.out.mkdir(parents=True, exist_ok=True)
    control = _train_arm("control", bundle, manifest, train_rows, dev_rows, args.out)
    treatment = _train_arm("treatment", bundle, manifest, train_rows, dev_rows, args.out)

    trajectory_equal = control["runtime_trajectory_sha256"] == treatment["runtime_trajectory_sha256"]
    if not trajectory_equal:
        raise RuntimeError("S39 control/treatment native runtime trajectory diverged")
    treatment["gates"]["runtime_trajectory_identity"] = True
    treatment["dev_ready"] = all(treatment["gates"].values())

    if control["dev_ready"] and treatment["dev_ready"]:
        outcome = "HIRA_V1_S39_MATCHED_GRADIENT_ISOLATED_BOTH_DEV_READY"
    elif control["dev_ready"]:
        outcome = "HIRA_V1_S39_MATCHED_GRADIENT_ISOLATED_CONTROL_DEV_READY"
    elif treatment["dev_ready"]:
        outcome = "HIRA_V1_S39_MATCHED_GRADIENT_ISOLATED_TREATMENT_DEV_READY"
    else:
        outcome = "HIRA_V1_S39_MATCHED_GRADIENT_ISOLATED_DEV_COMPLETE"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S39_FRESH_MATCHED_NATIVE_CONTROL_VS_GRADIENT_ISOLATED_BILINEAR",
        "seed": SEED,
        "optimizer": {
            "name": "AdamW",
            "epochs": s35.EPOCHS,
            "batch_size_semantic_cases": s35.BATCH_SIZE,
            "lr": s35.LR,
            "weight_decay": s35.WEIGHT_DECAY,
            "grad_clip": s35.GRAD_CLIP,
            "readout_separate_lr": False,
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
            "control_relation_operator": "S35_NativeA13RelationCanonicalizer",
            "treatment_relation_operator": "S39_NativeGradientIsolatedBilinearCorrectnessReadout",
            "native_relation_dimension": 256,
            "treatment_uses_shared_projection": False,
            "treatment_added_parameters": 65_536,
            "treatment_query_summary": "masked_mean_l2",
            "treatment_bilinear_shape": [256, 256],
            "treatment_factorized": False,
            "treatment_query_norm_epsilon": 1e-12,
            "treatment_residual_scale": 1.0,
            "treatment_bias": False,
            "treatment_nonlinearity": False,
            "runtime_training_uses_native_relation_only": True,
            "correction_features_detached": True,
            "correction_coefficient": 0.10,
            "runtime_grad_clip": 1.0,
            "w_grad_clip": 1.0,
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
            "s39_a0_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "trajectory_invariant": {
            "all_epoch_runtime_state_sha256_equal": trajectory_equal,
            "epoch_count": len(control["runtime_trajectory_sha256"]),
            "lora_state_max_abs_control_treatment": 0.0,
            "projection_state_max_abs_control_treatment": 0.0,
            "native_pre_w_logit_max_abs_control_treatment": 0.0,
            "native_signature_max_abs_control_treatment": 0.0,
            "basis": "bitwise-identical matched runtime trajectory fingerprints",
        },
        "control_arm": control,
        "treatment_arm": treatment,
        "matched_selected_dev_delta_treatment_minus_control": _metric_deltas(control, treatment),
        "post_dev_tuning_performed": False,
        "second_dev_run_performed": False,
        "sealed_confirm_opened": False,
        "multilingual_probe_opened": False,
        "production_ready_claimed": False,
    }

    (args.out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "
",
        encoding="utf-8",
    )
    (args.out / "train-manifest.json").write_text(
        json.dumps([r.to_dict() for r in train_rows], ensure_ascii=False, indent=2, sort_keys=True) + "
",
        encoding="utf-8",
    )
    (args.out / "dev-manifest.json").write_text(
        json.dumps([r.to_dict() for r in dev_rows], ensure_ascii=False, indent=2, sort_keys=True) + "
",
        encoding="utf-8",
    )
    print("HIRA_V1_S39_TRAIN_DEV_RECEIPT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
