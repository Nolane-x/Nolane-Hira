from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import torch

from nmd.local_runtime import A13_REVISION, load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import a13_lora_state_dict, iter_a13_lora_modules, load_a13_lora_state_dict
from nmd.v1_bilinear_signature_readout import NativeBilinearSignatureCorrectnessReadout
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_norm_balanced_gradient import apply_gradient_update, norm_balanced_gradient_update
from nmd.v1_reference_anchor_projection import (
    project_runtime_gradient_against_reference_anchor,
    signature_anchor_loss,
)
from nmd.v1_s39_authority import generate_s39_cases
from nmd.v1_s40_authority import generate_s40_cases, validate_s40_partitions
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
import hira_v1_s39_train_dev as s39
from hira_v1_s40_a0_reference_anchored_coadaptation import cases as s40_a0_cases

SCHEMA_VERSION = "hira-v1-s40-matched-reference-anchored-coadaptation-train-dev-v1"
SEED = 61001
READOUT_PARAMETER_COUNT = 65_536
REFERENCE_TOTAL = HIRA_V1_S17_TOTAL_PARAMETER_COUNT
TREATMENT_TOTAL = REFERENCE_TOTAL + READOUT_PARAMETER_COUNT


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


def _assert_s40_fresh(train_rows, dev_rows):
    s39._assert_s39_fresh(train_rows, dev_rows)

    current = (*train_rows, *dev_rows)
    current_states = {x for r in current for x in _state_texts(r)}
    current_questions = {x for r in current for x in _question_texts(r)}
    current_options = {x for r in current for x in _option_texts(r)}

    prior39 = (*generate_s39_cases("train"), *generate_s39_cases("dev"))
    prior_states = {x for r in prior39 for x in _state_texts(r)}
    prior_questions = {x for r in prior39 for x in _question_texts(r)}
    prior_options = {x for r in prior39 for x in _option_texts(r)}
    if current_states & prior_states:
        raise RuntimeError("S40 exact state overlap with exposed S39 rows")
    if current_questions & prior_questions:
        raise RuntimeError("S40 exact question overlap with exposed S39 rows")
    if current_options & prior_options:
        raise RuntimeError("S40 exact option overlap with exposed S39 rows")

    a0 = s40_a0_cases()
    a0_states = {x for r in a0 for x in (r.state_a, r.state_b)}
    a0_questions = {x for r in a0 for x in (r.qa1, r.qa2, r.qb1, r.qb2)}
    a0_options = set()
    for case in a0:
        options, _ga, _gb = case.option_pack()
        for option in options:
            a0_options.add(option.criterion_text)
            a0_options.update(option.aliases)
    if current_states & a0_states:
        raise RuntimeError("S40 TRAIN DEV state overlap with S40-A0")
    if current_questions & a0_questions:
        raise RuntimeError("S40 TRAIN DEV question overlap with S40-A0")
    if current_options & a0_options:
        raise RuntimeError("S40 TRAIN DEV option overlap with S40-A0")


def _runtime(bundle, manifest, seed):
    torch.manual_seed(seed)
    fresh = load_hira_v0_m4_bundle(bundle)
    runtime = build_hira_v1_s17_norm_balanced_core(
        fresh.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del fresh
    runtime.clear_schema_cache()
    enforce_s17_eval(runtime)
    return runtime


def _readout():
    return NativeBilinearSignatureCorrectnessReadout(
        role_temperature=s35.ROLE_TEMPERATURE,
        pair_temperature=s35.PAIR_TEMPERATURE,
        contrastive_temperature=s35.BINDING_CONTRASTIVE_TEMPERATURE,
        native_dimension=256,
        query_norm_epsilon=1e-12,
        residual_scale=1.0,
        train_readout=True,
    )


def _native_losses(runtime, rows):
    s17mod._relation_outputs = s35._native_relation_outputs
    return s35._losses(runtime, rows)


def _treatment_losses(runtime, rows, readout):
    s38._ACTIVE_READOUT = readout
    s17mod._relation_outputs = s38._treatment_relation_outputs
    return s35._losses(runtime, rows)


def _evaluate_reference(runtime, rows):
    return s38._arm_evaluate(runtime, rows, "control", None)


def _evaluate_treatment(runtime, rows, readout):
    return s38._arm_evaluate(runtime, rows, "treatment", readout)


def _zeros(params, grads):
    return [
        torch.zeros_like(p) if g is None else g
        for p, g in zip(params, grads)
    ]


def _lora_count(runtime):
    return sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" in name
    )


def _state(runtime, readout=None):
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S40 projection scorer missing")
    return {
        "lora": a13_lora_state_dict(runtime.encoder),
        "projection": {"projection.weight": scorer.projection.weight.detach().cpu().clone()},
        "readout": None if readout is None else readout.readout_state_dict(),
    }


def _load_state(runtime, state, readout=None):
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S40 projection scorer missing")
    load_a13_lora_state_dict(runtime.encoder, state["lora"], freeze=False)
    scorer.load_projection_state_dict(state["projection"], freeze=False)
    if readout is not None:
        if state["readout"] is None:
            raise RuntimeError("S40 treatment readout state missing")
        readout.load_readout_state_dict(state["readout"], freeze=False)
    enforce_s17_eval(runtime)


def _surface(runtime, readout=None):
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S40 projection scorer missing")
    return {
        "total_trainable_parameters": sum(p.numel() for p in runtime.parameters() if p.requires_grad)
        + (0 if readout is None else readout.added_parameter_count),
        "lora_trainable_parameters": _lora_count(runtime),
        "projection_trainable_parameters": scorer.projection_trainable_parameter_count,
        "readout_trainable_parameters": 0 if readout is None else readout.added_parameter_count,
        "original_a13_trainable": s35._original_a13_trainable(runtime),
        "hira_core_trainable": sum(p.numel() for p in runtime.hira.parameters() if p.requires_grad),
        "fusion_added_parameters": GradientIsolatedFullKEvidenceFusion(epsilon=s35.FUSION_EPSILON).parameter_count,
        "learned_downstream_scorer_parameters": 0 if readout is None else readout.added_parameter_count,
    }


def _record_base(epoch, totals, state_view_encodes, dev, runtime, readout=None):
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S40 projection scorer missing")
    return {
        "epoch": epoch,
        "train_mean_total_loss": totals["total"] / totals["cases"],
        "train_mean_primary_block": totals["primary_block"] / totals["cases"],
        "train_mean_relation_block": totals["relation_block"] / totals["cases"],
        "train_state_view_encodes": state_view_encodes,
        "surface_diagnostics": {
            "projection_weight_norm": float(scorer.projection.weight.detach().norm().cpu()),
            "lora_b_norm": float(torch.sqrt(sum(
                module.lora_b.detach().pow(2).sum()
                for module in iter_a13_lora_modules(runtime.encoder)
            )).cpu()),
            "readout_weight_norm": 0.0 if readout is None else float(readout.bilinear_weight.detach().norm().cpu()),
        },
        "dev": dev,
    }


def _checkpoint(
    out_dir,
    filename,
    schema,
    kind,
    arm,
    runtime,
    state,
    selected_epoch,
    manifest,
    readout=None,
):
    total = REFERENCE_TOTAL if readout is None else TREATMENT_TOTAL
    path = out_dir / filename
    torch.save(
        {
            "schema_version": schema,
            "kind": kind,
            "arm": arm,
            "lora_parameter_count": HIRA_V1_S6_LORA_PARAMETER_COUNT,
            "projection_parameter_count": HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
            "readout_parameter_count": 0 if readout is None else READOUT_PARAMETER_COUNT,
            "total_parameter_count": total,
            "lora_rank": 8,
            "selected_dev_epoch": selected_epoch,
            "semantic_revision": str(manifest["semantic_revision"]),
            "initialization_t0_sha256": str(manifest["t0_checkpoint_sha256"]),
            "lora_state_dict": state["lora"],
            "projection_state_dict": state["projection"],
            "readout_state_dict": {} if state["readout"] is None else state["readout"],
        },
        path,
    )
    return path


def _metric_deltas(reference, treatment):
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
    return {
        key: float(treatment["selected_dev"][key]) - float(reference["selected_dev"][key])
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--a0-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a0 = json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome") != "HIRA_V1_S40_A0_REFERENCE_ANCHORED_PROJECTED_COADAPTATION_READY":
        raise RuntimeError("S40-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S40-A0 unexpectedly used for model selection")
    if int(a0.get("operator_added_parameter_count", -1)) != READOUT_PARAMETER_COUNT:
        raise RuntimeError("S40-A0 W capacity changed")
    if int(a0.get("reference_trainable_parameter_count", -1)) != REFERENCE_TOTAL:
        raise RuntimeError("S40-A0 reference surface changed")
    if int(a0.get("treatment_trainable_parameter_count", -1)) != TREATMENT_TOTAL:
        raise RuntimeError("S40-A0 treatment surface changed")
    if a0.get("anchor_coefficient", "unexpected") is not None:
        raise RuntimeError("S40-A0 anchor coefficient appeared")
    if float(a0.get("projection_epsilon", -1.0)) != 1e-12:
        raise RuntimeError("S40-A0 projection epsilon changed")
    for key in (
        "anchor_reference_runtime_gradient_l1",
        "anchor_w_gradient_l1",
        "conflict_post_dot",
        "nonconflict_identity_max_abs",
        "w_projection_max_abs",
        "zero_init_relation_logit_max_abs",
        "zero_init_signature_max_abs",
        "zero_init_fused_logit_max_abs",
        "zero_init_reference_treatment_relation_max_abs",
        "zero_init_reference_treatment_signature_max_abs",
    ):
        if float(a0.get(key, -1.0)) != 0.0:
            raise RuntimeError(f"S40-A0 exact-zero contract failed: {key}")
    for key in (
        "synthetic_signature_drift_max_abs",
        "synthetic_signature_anchor",
        "anchor_treatment_lora_gradient_l1",
        "joint_correctness_w_gradient_l1",
        "joint_correctness_w_offdiagonal_gradient_l1",
        "joint_correctness_lora_gradient_l1",
    ):
        if float(a0.get(key, 0.0)) <= 0.0:
            raise RuntimeError(f"S40-A0 live mechanism failed: {key}")
    if a0.get("conflict_projected") is not True or float(a0.get("conflict_pre_dot", 0.0)) >= 0.0:
        raise RuntimeError("S40-A0 conflict projection not qualified")
    if a0.get("nonconflict_projected") is not False:
        raise RuntimeError("S40-A0 nonconflict identity not qualified")
    if not bool(a0.get("checkpoint_roundtrip_exact", False)) or not bool(a0.get("full_k", False)):
        raise RuntimeError("S40-A0 checkpoint/full-K failed")

    train_rows = generate_s40_cases("train")
    dev_rows = generate_s40_cases("dev")
    validate_s40_partitions(train_rows, dev_rows)
    _assert_s40_fresh(train_rows, dev_rows)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"]) != A13_REVISION:
        raise RuntimeError("S40 semantic revision changed")

    random.seed(SEED)
    reference = _runtime(bundle, manifest, SEED)
    treatment = _runtime(bundle, manifest, SEED)
    readout = _readout()

    reference_params = [p for p in reference.parameters() if p.requires_grad]
    treatment_runtime_params = [p for p in treatment.parameters() if p.requires_grad]
    treatment_all = [*treatment_runtime_params, readout.bilinear_weight]
    if sum(p.numel() for p in reference_params) != REFERENCE_TOTAL:
        raise RuntimeError("S40 reference trainable surface changed")
    if sum(p.numel() for p in treatment_all) != TREATMENT_TOTAL:
        raise RuntimeError("S40 treatment trainable surface changed")

    ref_opt = torch.optim.AdamW(reference_params, lr=s35.LR, weight_decay=s35.WEIGHT_DECAY)
    tr_opt = torch.optim.AdamW(treatment_all, lr=s35.LR, weight_decay=s35.WEIGHT_DECAY)

    ref_history = []
    tr_history = []
    ref_best_key = tr_best_key = None
    ref_best_epoch = tr_best_epoch = None
    ref_best_state = tr_best_state = None
    ref_best_metrics = tr_best_metrics = None

    print("HIRA_V1_S40_MATCHED_TRAIN_BEGIN", flush=True)

    for epoch in range(1, s35.EPOCHS + 1):
        order = list(range(len(train_rows)))
        random.Random(SEED + epoch).shuffle(order)
        ref_totals = {"total":0.0,"primary_block":0.0,"relation_block":0.0,"cases":0}
        tr_totals = {"total":0.0,"primary_block":0.0,"relation_block":0.0,"cases":0}
        anchor_sum = pre_dot_sum = post_dot_sum = 0.0
        anchor_steps = anchor_conflicts = 0
        ref_encodes = tr_encodes = 0

        for start in range(0, len(order), s35.BATCH_SIZE):
            rows = [train_rows[i] for i in order[start:start+s35.BATCH_SIZE]]
            n = len(rows)
            ref_opt.zero_grad(set_to_none=True)
            tr_opt.zero_grad(set_to_none=True)

            (
                ref_total, ref_primary, ref_relation, ref_pieces,
                _rfc, _rfp, _rrc, _rrp, _rrelc, _rrelp, ref_sig_c, ref_sig_p, _ref_encoded,
            ) = _native_losses(reference, rows)
            ref_sig_c_det = ref_sig_c.detach()
            ref_sig_p_det = ref_sig_p.detach()

            ref_pg_raw = torch.autograd.grad(ref_primary, reference_params, retain_graph=True, allow_unused=True)
            ref_rg_raw = torch.autograd.grad(ref_relation, reference_params, allow_unused=True)
            ref_pg = _zeros(reference_params, ref_pg_raw)
            ref_rg = _zeros(reference_params, ref_rg_raw)
            ref_combined, _ref_diag = norm_balanced_gradient_update(
                ref_pg, ref_rg, epsilon=s35.BALANCE_EPSILON
            )

            (
                tr_total, tr_primary, tr_relation, tr_pieces,
                _tfc, _tfp, _trc, _trp, _trelc, _trelp, tr_sig_c, tr_sig_p, _tr_encoded,
            ) = _treatment_losses(treatment, rows, readout)

            tr_pg_raw = torch.autograd.grad(tr_primary, treatment_all, retain_graph=True, allow_unused=True)
            tr_rg_raw = torch.autograd.grad(tr_relation, treatment_all, retain_graph=True, allow_unused=True)
            tr_pg = _zeros(treatment_all, tr_pg_raw)
            tr_rg = _zeros(treatment_all, tr_rg_raw)
            tr_joint, _tr_diag = norm_balanced_gradient_update(
                tr_pg, tr_rg, epsilon=s35.BALANCE_EPSILON
            )

            anchor = 0.5 * (
                signature_anchor_loss(tr_sig_c, ref_sig_c_det)
                + signature_anchor_loss(tr_sig_p, ref_sig_p_det)
            )
            anchor_raw = torch.autograd.grad(
                anchor, treatment_runtime_params, allow_unused=True
            )
            anchor_grad = _zeros(treatment_runtime_params, anchor_raw)
            projected_runtime, pdiag = project_runtime_gradient_against_reference_anchor(
                tr_joint[:-1], anchor_grad, epsilon=1e-12
            )
            projected_all = [*projected_runtime, tr_joint[-1]]

            apply_gradient_update(reference_params, ref_combined)
            apply_gradient_update(treatment_all, projected_all)
            torch.nn.utils.clip_grad_norm_(reference_params, s35.GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(treatment_all, s35.GRAD_CLIP)
            ref_opt.step()
            tr_opt.step()
            enforce_s17_eval(reference)
            enforce_s17_eval(treatment)

            ref_totals["total"] += float(ref_total.detach().cpu()) * n
            ref_totals["primary_block"] += float(ref_primary.detach().cpu()) * n
            ref_totals["relation_block"] += float(ref_relation.detach().cpu()) * n
            ref_totals["cases"] += n
            tr_totals["total"] += float(tr_total.detach().cpu()) * n
            tr_totals["primary_block"] += float(tr_primary.detach().cpu()) * n
            tr_totals["relation_block"] += float(tr_relation.detach().cpu()) * n
            tr_totals["cases"] += n

            anchor_sum += float(anchor.detach().cpu())
            pre_dot_sum += pdiag.pre_dot
            post_dot_sum += pdiag.post_dot
            anchor_steps += 1
            anchor_conflicts += int(pdiag.projected)
            ref_encodes += 2*n
            tr_encodes += 2*n

        if ref_encodes != 2*len(train_rows) or tr_encodes != 2*len(train_rows):
            raise RuntimeError("S40 TRAIN state-once changed")

        ref_dev = _evaluate_reference(reference, dev_rows)
        tr_dev = _evaluate_treatment(treatment, dev_rows, readout)
        if int(ref_dev["state_view_encodes"]) != 2*len(dev_rows):
            raise RuntimeError("S40 reference DEV state-once changed")
        if int(tr_dev["state_view_encodes"]) != 2*len(dev_rows):
            raise RuntimeError("S40 treatment DEV state-once changed")

        ref_record = _record_base(epoch, ref_totals, ref_encodes, ref_dev, reference)
        tr_record = _record_base(epoch, tr_totals, tr_encodes, tr_dev, treatment, readout)
        tr_record["reference_anchor"] = {
            "steps": anchor_steps,
            "conflict_rate": anchor_conflicts / max(1, anchor_steps),
            "mean_anchor": anchor_sum / max(1, anchor_steps),
            "mean_pre_dot": pre_dot_sum / max(1, anchor_steps),
            "mean_post_dot": post_dot_sum / max(1, anchor_steps),
            "mean_reference_signature_cosine_proxy": 1.0 - anchor_sum / max(1, anchor_steps),
        }
        ref_history.append(ref_record)
        tr_history.append(tr_record)

        rk = s35._selection_key(epoch, ref_dev)
        if ref_best_key is None or rk > ref_best_key:
            ref_best_key = rk
            ref_best_epoch = epoch
            ref_best_state = _state(reference)
            ref_best_metrics = dict(ref_dev)

        tk = s35._selection_key(epoch, tr_dev)
        if tr_best_key is None or tk > tr_best_key:
            tr_best_key = tk
            tr_best_epoch = epoch
            tr_best_state = _state(treatment, readout)
            tr_best_metrics = dict(tr_dev)

        print("HIRA_V1_S40_EPOCH=" + json.dumps({
            "epoch": epoch,
            "reference_dev": ref_dev,
            "treatment_dev": tr_dev,
            "reference_anchor": tr_record["reference_anchor"],
            "w_norm": tr_record["surface_diagnostics"]["readout_weight_norm"],
        }, sort_keys=True), flush=True)

    if None in (ref_best_epoch, tr_best_epoch) or ref_best_state is None or tr_best_state is None:
        raise RuntimeError("S40 selection produced no checkpoint")

    _load_state(reference, ref_best_state)
    _load_state(treatment, tr_best_state, readout)
    ref_selected = _evaluate_reference(reference, dev_rows)
    tr_selected = _evaluate_treatment(treatment, dev_rows, readout)

    replay_keys = (
        "fused_canonical_accuracy","fused_paraphrase_accuracy",
        "fused_canonical_paired_both_correct_rate","fused_question_swap_choice_change_rate",
        "fused_cross_view_selected_choice_agreement","fused_cross_view_mean_js",
        "fused_canonical_mean_gold_margin","canonical_relation_binding_accuracy",
        "canonical_relation_binding_mean_gold_margin","mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin","fused_option_order_flip_rate",
        "fused_max_probability_mass_error","mean_canonical_decision_loss",
    )
    for key in replay_keys:
        if not math.isclose(float(ref_selected[key]), float(ref_best_metrics[key]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"S40 reference selected DEV replay changed: {key}")
        if not math.isclose(float(tr_selected[key]), float(tr_best_metrics[key]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"S40 treatment selected DEV replay changed: {key}")

    ref_gates = s35._gates(
        reference, ref_selected, ref_history, REFERENCE_TOTAL, _lora_count(reference),
        REFERENCE_TOTAL, HIRA_V1_S6_LORA_PARAMETER_COUNT, train_rows, dev_rows,
    )
    tr_gates = s35._gates(
        treatment, tr_selected, tr_history, TREATMENT_TOTAL, _lora_count(treatment),
        TREATMENT_TOTAL, HIRA_V1_S6_LORA_PARAMETER_COUNT, train_rows, dev_rows,
    )
    tr_gates["readout_capacity_exact"] = readout.added_parameter_count == READOUT_PARAMETER_COUNT
    tr_gates["reference_anchor_parameter_free"] = True

    args.out.mkdir(parents=True, exist_ok=True)
    ref_ckpt = _checkpoint(
        args.out, "reference-native-candidate.pt",
        "hira-v1-s40-reference-native-checkpoint-v1",
        "matched-reference-native-s35-s17",
        "reference", reference, ref_best_state, ref_best_epoch, manifest,
    )
    tr_ckpt = _checkpoint(
        args.out, "treatment-anchored-candidate.pt",
        "hira-v1-s40-treatment-reference-anchored-checkpoint-v1",
        "matched-reference-anchored-full-bilinear-s17",
        "treatment", treatment, tr_best_state, tr_best_epoch, manifest, readout,
    )

    reference_out = {
        "arm": "reference",
        "selected_dev_epoch": ref_best_epoch,
        "selected_dev": ref_selected,
        "gates": ref_gates,
        "dev_ready": all(ref_gates.values()),
        "history": ref_history,
        "checkpoint_file": ref_ckpt.name,
        "checkpoint_sha256": _sha256(ref_ckpt),
        "parameter_surface": _surface(reference),
    }
    treatment_out = {
        "arm": "treatment",
        "selected_dev_epoch": tr_best_epoch,
        "selected_dev": tr_selected,
        "gates": tr_gates,
        "dev_ready": all(tr_gates.values()),
        "history": tr_history,
        "checkpoint_file": tr_ckpt.name,
        "checkpoint_sha256": _sha256(tr_ckpt),
        "parameter_surface": _surface(treatment, readout),
    }

    if reference_out["dev_ready"] and treatment_out["dev_ready"]:
        outcome = "HIRA_V1_S40_MATCHED_REFERENCE_ANCHORED_BOTH_DEV_READY"
    elif reference_out["dev_ready"]:
        outcome = "HIRA_V1_S40_MATCHED_REFERENCE_ANCHORED_REFERENCE_DEV_READY"
    elif treatment_out["dev_ready"]:
        outcome = "HIRA_V1_S40_MATCHED_REFERENCE_ANCHORED_TREATMENT_DEV_READY"
    else:
        outcome = "HIRA_V1_S40_MATCHED_REFERENCE_ANCHORED_DEV_COMPLETE"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S40_FRESH_MATCHED_NATIVE_REFERENCE_VS_REFERENCE_ANCHORED_FULL_BILINEAR",
        "seed": SEED,
        "optimizer": {
            "name": "AdamW",
            "epochs": s35.EPOCHS,
            "batch_size_semantic_cases": s35.BATCH_SIZE,
            "lr": s35.LR,
            "weight_decay": s35.WEIGHT_DECAY,
            "grad_clip": s35.GRAD_CLIP,
            "w_separate_lr": False,
        },
        "anchor": {
            "kind": "mean_one_minus_cosine_native_signature_to_detached_matched_reference",
            "coefficient": None,
            "projection_epsilon": 1e-12,
            "w_gradient_projected": False,
            "reference_gradient_detached": True,
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
            "s40_a0_rows_used": False,
            "m5_final_rows_used": False,
            "w29_w34_sealed_rows_used": False,
        },
        "reference_arm": reference_out,
        "treatment_arm": treatment_out,
        "matched_selected_dev_delta_treatment_minus_reference": _metric_deltas(reference_out, treatment_out),
        "anchor_diagnostics": {
            "mean_conflict_rate": sum(x["reference_anchor"]["conflict_rate"] for x in tr_history) / len(tr_history),
            "mean_anchor": sum(x["reference_anchor"]["mean_anchor"] for x in tr_history) / len(tr_history),
            "mean_pre_dot": sum(x["reference_anchor"]["mean_pre_dot"] for x in tr_history) / len(tr_history),
            "mean_post_dot": sum(x["reference_anchor"]["mean_post_dot"] for x in tr_history) / len(tr_history),
            "selected_epoch_w_norm": tr_history[tr_best_epoch-1]["surface_diagnostics"]["readout_weight_norm"],
        },
        "post_dev_tuning_performed": False,
        "second_dev_run_performed": False,
        "sealed_confirm_opened": False,
        "multilingual_probe_opened": False,
        "production_ready_claimed": False,
    }

    (args.out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (args.out/"train-manifest.json").write_text(
        json.dumps([r.to_dict() for r in train_rows],indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    (args.out/"dev-manifest.json").write_text(
        json.dumps([r.to_dict() for r in dev_rows],indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S40_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
