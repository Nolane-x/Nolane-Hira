from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import load_a13_lora_state_dict
from nmd.v1_evidence_fusion import (
    GradientIsolatedFullKEvidenceFusion,
    fused_gold_vs_max_wrong_margin,
)
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_robust_three_expert_consensus import RobustThreeExpertMedianFusion
from nmd.v1_s17_semantic_core import build_hira_v1_s17_norm_balanced_core, enforce_s17_eval
from nmd.v1_s45_authority import generate_s45_cases
from nmd.v1_s46_authority import generate_s46_cases, validate_s46_partitions
import hira_v1_s17_train_dev as s17mod
import hira_v1_s35_train_dev as s35
import hira_v1_s45_train_dev as s45
from hira_v1_s46_a0_robust_three_expert_consensus import cases as s46_a0_cases

SCHEMA_VERSION = "hira-v1-s46-matched-robust-three-expert-consensus-train-dev-v1"
SEED = 67_001

_SHELL_KEYS = (
    "fused_canonical_accuracy",
    "fused_paraphrase_accuracy",
    "fused_canonical_paired_both_correct_rate",
    "fused_question_swap_choice_change_rate",
    "fused_cross_view_selected_choice_agreement",
    "fused_cross_view_mean_js",
    "fused_canonical_mean_gold_margin",
    "fused_paraphrase_mean_gold_margin",
    "fused_option_order_flip_rate",
    "fused_max_probability_mass_error",
)


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


def _assert_s46_fresh(train_rows, dev_rows):
    # Reuse all prior freshness guards through S45's parent surfaces.
    s45._assert_s45_fresh(train_rows, dev_rows)

    current = (*train_rows, *dev_rows)
    current_states = {x for r in current for x in _state_texts(r)}
    current_questions = {x for r in current for x in _question_texts(r)}
    current_options = {x for r in current for x in _option_texts(r)}

    prior45 = (*generate_s45_cases("train"), *generate_s45_cases("dev"))
    prior45_states = {x for r in prior45 for x in _state_texts(r)}
    prior45_questions = {x for r in prior45 for x in _question_texts(r)}
    prior45_options = {x for r in prior45 for x in _option_texts(r)}
    if current_states & prior45_states:
        raise RuntimeError("S46 exact state overlap with exposed S45 rows")
    if current_questions & prior45_questions:
        raise RuntimeError("S46 exact question overlap with exposed S45 rows")
    if current_options & prior45_options:
        raise RuntimeError("S46 exact option overlap with exposed S45 rows")

    a0 = s46_a0_cases()
    a0_states = {x for r in a0 for x in (r.state_a, r.state_b)}
    a0_questions = {x for r in a0 for x in (r.qa1, r.qa2, r.qb1, r.qb2)}
    a0_options = set()
    for case in a0:
        options, _ga, _gb = case.option_pack()
        for option in options:
            a0_options.add(option.criterion_text)
            a0_options.update(option.aliases)
    if current_states & a0_states:
        raise RuntimeError("S46 TRAIN DEV state overlap with S46-A0")
    if current_questions & a0_questions:
        raise RuntimeError("S46 TRAIN DEV question overlap with S46-A0")
    if current_options & a0_options:
        raise RuntimeError("S46 TRAIN DEV option overlap with S46-A0")


def _load_treatment_checkpoint(bundle, manifest, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["arm"] != "treatment":
        raise RuntimeError("S46 selected checkpoint is not treatment")
    if int(payload["total_parameter_count"]) != 163_840:
        raise RuntimeError("S46 treatment total changed")
    if int(payload["correction_parameter_count"]) != 114_688:
        raise RuntimeError("S46 correction capacity changed")

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
    load_a13_lora_state_dict(runtime.encoder, payload["lora_state_dict"], freeze=True)
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S46 projection scorer missing")
    scorer.load_projection_state_dict(payload["projection_state_dict"], freeze=True)

    correction = s45._new_correction(train=False)
    correction.load_correction_state_dict(payload["correction_state_dict"], freeze=True)
    enforce_s17_eval(runtime)
    correction.eval()
    return runtime, correction, payload


def _gold_margin_sum(logits, gold):
    return float(fused_gold_vs_max_wrong_margin(logits, gold).sum().cpu())


@torch.inference_mode()
def _same_checkpoint_shells(runtime, correction, rows):
    legacy_op = GradientIsolatedFullKEvidenceFusion(epsilon=s35.FUSION_EPSILON)
    robust_op = RobustThreeExpertMedianFusion(epsilon=s35.FUSION_EPSILON)

    names = ("legacy_s45", "robust_s46")
    acc = {
        name: {
            "fused_canonical_correct": 0,
            "fused_paraphrase_correct": 0,
            "paired_both": 0,
            "question_swap": 0,
            "agreement_sum": 0.0,
            "js_sum": 0.0,
            "canonical_margin_sum": 0.0,
            "paraphrase_margin_sum": 0.0,
            "order_flips": 0,
            "max_mass_error": 0.0,
        }
        for name in names
    }

    expert = {
        "raw_c_correct": 0,
        "raw_p_correct": 0,
        "native_c_correct": 0,
        "native_p_correct": 0,
        "corrected_c_correct": 0,
        "corrected_p_correct": 0,
        "corrected_agreement_sum": 0.0,
        "corrected_js_sum": 0.0,
        "native_agreement_sum": 0.0,
        "native_js_sum": 0.0,
        "signature_cosine_sum": 0.0,
        "signature_margin_sum": 0.0,
        "signature_count": 0,
    }
    semantic_cases = 0
    state_view_encodes = 0
    full_k = True

    # Force the single encoded batch to expose native relation evidence.
    s45._set_relation_arm("reference", None)

    for start in range(0, len(rows), s35.BATCH_SIZE):
        batch_rows = list(rows[start:start + s35.BATCH_SIZE])
        (
            _total, _primary, _relation, _pieces,
            _native_fused_c, _native_fused_p,
            raw_c, raw_p,
            native_c, native_p,
            signature_c, signature_p,
            encoded,
        ) = s45._native_losses(runtime, batch_rows)

        corrected_c = correction.correction_logits(
            native_logits=native_c,
            signatures=signature_c,
            question_tokens=encoded["question_canonical_tokens"],
            question_mask=encoded["question_canonical_mask"],
        )
        corrected_p = correction.correction_logits(
            native_logits=native_p,
            signatures=signature_p,
            question_tokens=encoded["question_paraphrase_tokens"],
            question_mask=encoded["question_paraphrase_mask"],
        )

        legacy_c, _ = legacy_op(raw_c, corrected_c)
        legacy_p, _ = legacy_op(raw_p, corrected_p)
        robust_c, _ = robust_op(raw_c, native_c, corrected_c)
        robust_p, _ = robust_op(raw_p, native_p, corrected_p)

        n = len(batch_rows)
        queries = 2 * n
        gold, _other = s35._gold_tensors(batch_rows, device=raw_c.device)

        shell_values = {
            "legacy_s45": (legacy_c, legacy_p),
            "robust_s46": (robust_c, robust_p),
        }
        perm = torch.tensor([3, 2, 1, 0], device=raw_c.device)

        for name, (fc_logits, fp_logits) in shell_values.items():
            fc = fc_logits.argmax(-1)
            fp = fp_logits.argmax(-1)
            acc[name]["fused_canonical_correct"] += int((fc == gold).sum().item())
            acc[name]["fused_paraphrase_correct"] += int((fp == gold).sum().item())

            pairs = fc.reshape(n, 2)
            gpairs = gold.reshape(n, 2)
            acc[name]["paired_both"] += int(((pairs == gpairs).all(-1)).sum().item())
            acc[name]["question_swap"] += int((pairs[:, 0] != pairs[:, 1]).sum().item())
            acc[name]["agreement_sum"] += float(selected_choice_agreement(fc_logits, fp_logits)) * queries
            acc[name]["js_sum"] += float(symmetric_js_divergence(fc_logits, fp_logits)) * queries
            acc[name]["canonical_margin_sum"] += _gold_margin_sum(fc_logits, gold)
            acc[name]["paraphrase_margin_sum"] += _gold_margin_sum(fp_logits, gold)

            if name == "legacy_s45":
                permuted, _ = legacy_op(raw_c[:, perm], corrected_c[:, perm])
            else:
                permuted, _ = robust_op(
                    raw_c[:, perm], native_c[:, perm], corrected_c[:, perm]
                )
            mapped = perm[permuted.argmax(-1)]
            acc[name]["order_flips"] += int((mapped != fc).sum().item())

            for logits in (fc_logits, fp_logits):
                probs = torch.softmax(logits, dim=-1)
                acc[name]["max_mass_error"] = max(
                    acc[name]["max_mass_error"],
                    float((probs.sum(-1) - 1.0).abs().max().cpu()),
                )

        expert["raw_c_correct"] += int((raw_c.argmax(-1) == gold).sum().item())
        expert["raw_p_correct"] += int((raw_p.argmax(-1) == gold).sum().item())
        expert["native_c_correct"] += int((native_c.argmax(-1) == gold).sum().item())
        expert["native_p_correct"] += int((native_p.argmax(-1) == gold).sum().item())
        expert["corrected_c_correct"] += int((corrected_c.argmax(-1) == gold).sum().item())
        expert["corrected_p_correct"] += int((corrected_p.argmax(-1) == gold).sum().item())
        expert["corrected_agreement_sum"] += float(
            selected_choice_agreement(corrected_c, corrected_p)
        ) * queries
        expert["corrected_js_sum"] += float(
            symmetric_js_divergence(corrected_c, corrected_p)
        ) * queries
        expert["native_agreement_sum"] += float(
            selected_choice_agreement(native_c, native_p)
        ) * queries
        expert["native_js_sum"] += float(
            symmetric_js_divergence(native_c, native_p)
        ) * queries

        same = s17mod.relation_signature_same_option_cosine(signature_c, signature_p)
        c_norm = F.normalize(signature_c, dim=-1)
        p_norm = F.normalize(signature_p, dim=-1)
        cross = torch.einsum("nkd,njd->nkj", c_norm, p_norm)
        k = cross.shape[-1]
        eye = torch.eye(k, dtype=torch.bool, device=cross.device)[None]
        wrong = cross.masked_fill(eye, float("-inf")).amax(-1)
        expert["signature_cosine_sum"] += float(same.sum().cpu())
        expert["signature_margin_sum"] += float((same - wrong).sum().cpu())
        expert["signature_count"] += int(same.numel())

        semantic_cases += n
        state_view_encodes += 2 * n
        full_k = full_k and all(
            x.shape[-1] == 4
            for x in (
                raw_c, raw_p, native_c, native_p, corrected_c, corrected_p,
                legacy_c, legacy_p, robust_c, robust_p,
            )
        )

    queries_total = len(rows) * 2
    shells = {}
    for name in names:
        a = acc[name]
        shells[name] = {
            "semantic_cases": semantic_cases,
            "canonical_queries": queries_total,
            "paraphrase_queries": queries_total,
            "fused_canonical_accuracy": a["fused_canonical_correct"] / queries_total,
            "fused_paraphrase_accuracy": a["fused_paraphrase_correct"] / queries_total,
            "fused_canonical_paired_both_correct_rate": a["paired_both"] / len(rows),
            "fused_question_swap_choice_change_rate": a["question_swap"] / len(rows),
            "fused_cross_view_selected_choice_agreement": a["agreement_sum"] / queries_total,
            "fused_cross_view_mean_js": a["js_sum"] / queries_total,
            "fused_canonical_mean_gold_margin": a["canonical_margin_sum"] / queries_total,
            "fused_paraphrase_mean_gold_margin": a["paraphrase_margin_sum"] / queries_total,
            "fused_option_order_flip_rate": a["order_flips"] / queries_total,
            "fused_max_probability_mass_error": a["max_mass_error"],
            "full_k": bool(full_k),
            "state_view_encodes": state_view_encodes,
        }

    expert_metrics = {
        "raw_triadic_canonical_accuracy": expert["raw_c_correct"] / queries_total,
        "raw_triadic_paraphrase_accuracy": expert["raw_p_correct"] / queries_total,
        "native_relation_canonical_accuracy": expert["native_c_correct"] / queries_total,
        "native_relation_paraphrase_accuracy": expert["native_p_correct"] / queries_total,
        "native_relation_cross_view_agreement": expert["native_agreement_sum"] / queries_total,
        "native_relation_cross_view_mean_js": expert["native_js_sum"] / queries_total,
        "corrected_relation_canonical_accuracy": expert["corrected_c_correct"] / queries_total,
        "corrected_relation_paraphrase_accuracy": expert["corrected_p_correct"] / queries_total,
        "corrected_relation_cross_view_agreement": expert["corrected_agreement_sum"] / queries_total,
        "corrected_relation_cross_view_mean_js": expert["corrected_js_sum"] / queries_total,
        "mean_same_option_signature_cosine": expert["signature_cosine_sum"] / expert["signature_count"],
        "mean_signature_same_vs_strongest_wrong_margin": expert["signature_margin_sum"] / expert["signature_count"],
    }
    return shells, expert_metrics


def _shell_delta(reference, treatment):
    return {key: float(treatment[key]) - float(reference[key]) for key in _SHELL_KEYS}


def _robust_gates(s45_treatment, robust):
    gates = dict(s45_treatment["gates"])
    replacements = {
        "fused_canonical_accuracy_gte_0_85": robust["fused_canonical_accuracy"] >= 0.85,
        "fused_paired_both_correct_gte_0_75": robust["fused_canonical_paired_both_correct_rate"] >= 0.75,
        "fused_question_swap_change_gte_0_80": robust["fused_question_swap_choice_change_rate"] >= 0.80,
        "fused_cross_view_choice_agreement_gte_0_95": robust["fused_cross_view_selected_choice_agreement"] >= 0.95,
        "fused_cross_view_mean_js_lte_0_05": robust["fused_cross_view_mean_js"] <= 0.05,
        "fused_canonical_margin_gte_0_15": robust["fused_canonical_mean_gold_margin"] >= 0.15,
        "fused_option_order_flip_lte_0_02": robust["fused_option_order_flip_rate"] <= 0.02,
        "fused_probability_mass_error_lte_1e_6": robust["fused_max_probability_mass_error"] <= 1e-6,
        "full_k": bool(robust["full_k"]),
        "fusion_added_parameters_zero": RobustThreeExpertMedianFusion(epsilon=s35.FUSION_EPSILON).parameter_count == 0,
    }
    gates.update(replacements)
    return gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--a0-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a0 = json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome") != "HIRA_V1_S46_A0_ROBUST_THREE_EXPERT_CONSENSUS_READY":
        raise RuntimeError("S46-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S46-A0 unexpectedly used for model selection")
    expected = {
        "native_trainable_parameter_count": 49_152,
        "private_adapter_parameter_count": 49_152,
        "bilinear_parameter_count": 65_536,
        "correction_parameter_count": 114_688,
        "treatment_total_trainable_parameter_count": 163_840,
        "fusion_parameter_count": 0,
    }
    for key, value in expected.items():
        if int(a0.get(key, -1)) != value:
            raise RuntimeError(f"S46-A0 capacity changed: {key}")
    if bool(a0.get("second_encoder_pass", True)):
        raise RuntimeError("S46-A0 opened a second encoder pass")
    for key in (
        "arbitrary_k3_pass", "arbitrary_k7_pass", "arbitrary_k255_pass",
        "flat_expert_finite", "actual_shell_one_encoder_batch",
    ):
        if not bool(a0.get(key, False)):
            raise RuntimeError(f"S46-A0 mechanics failed: {key}")
    if float(a0.get("logical_option_permutation_max_abs_error", 1.0)) > 3e-6:
        raise RuntimeError("S46-A0 permutation court failed")
    if float(a0.get("one_extreme_outlier_max_abs_error", 1.0)) > 3e-6:
        raise RuntimeError("S46-A0 outlier containment failed")
    ownership = a0.get("ownership", {})
    for key in (
        "matched_native_one_step_parameter_max_abs",
        "matched_native_one_step_output_max_abs",
        "zero_init_correction_native_runtime_gradient_l1",
        "native_objective_correction_gradient_l1",
        "js_only_native_runtime_gradient_l1",
    ):
        if float(ownership.get(key, -1.0)) != 0.0:
            raise RuntimeError(f"S46-A0 ownership failed: {key}")

    train_rows = generate_s46_cases("train")
    dev_rows = generate_s46_cases("dev")
    validate_s46_partitions(train_rows, dev_rows)
    _assert_s46_fresh(train_rows, dev_rows)

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"]) != A13_REVISION:
        raise RuntimeError("S46 semantic revision changed")

    args.out.mkdir(parents=True, exist_ok=True)

    # Exact S45 training mechanics, with only a fresh authority seed/rows.
    s45.SEED = SEED
    reference = s45._train_arm("reference", bundle, manifest, train_rows, dev_rows, args.out)
    treatment = s45._train_arm("treatment", bundle, manifest, train_rows, dev_rows, args.out)

    if reference["runtime_trajectory_sha256"] != treatment["runtime_trajectory_sha256"]:
        raise RuntimeError("S46 matched native runtime trajectory diverged")
    treatment["gates"]["runtime_trajectory_identity"] = True
    treatment["dev_ready"] = all(treatment["gates"].values())

    runtime, correction, checkpoint_payload = _load_treatment_checkpoint(
        bundle,
        manifest,
        args.out / "treatment-private-correction-candidate.pt",
    )
    shells, experts = _same_checkpoint_shells(runtime, correction, dev_rows)
    legacy = shells["legacy_s45"]
    robust = shells["robust_s46"]

    if int(legacy["state_view_encodes"]) != 2 * len(dev_rows):
        raise RuntimeError("S46 legacy shell state-once changed")
    if int(robust["state_view_encodes"]) != 2 * len(dev_rows):
        raise RuntimeError("S46 robust shell state-once changed")

    robust_gates = _robust_gates(treatment, robust)
    robust_dev_ready = all(robust_gates.values())

    if robust_dev_ready:
        outcome = "HIRA_V1_S46_ROBUST_THREE_EXPERT_CONSENSUS_DEV_READY"
    else:
        outcome = "HIRA_V1_S46_ROBUST_THREE_EXPERT_CONSENSUS_DEV_COMPLETE"

    train_manifest = args.out / "train-manifest.json"
    dev_manifest = args.out / "dev-manifest.json"
    train_manifest.write_text(
        json.dumps([r.to_dict() for r in train_rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dev_manifest.write_text(
        json.dumps([r.to_dict() for r in dev_rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S46_FRESH_SAME_CHECKPOINT_LEGACY_VS_ROBUST_THREE_EXPERT_CONSENSUS",
        "seed": SEED,
        "training_mechanics": {
            "family": "exact_s45_cross_view_consistent_private_correction",
            "checkpoint_selection": "exact_s45_selection_rule_before_robust_shell_evaluation",
            "epochs": s35.EPOCHS,
            "batch_size_semantic_cases": s35.BATCH_SIZE,
            "lr": s35.LR,
            "weight_decay": s35.WEIGHT_DECAY,
            "grad_clip": s35.GRAD_CLIP,
            "correction_ce_coefficient": s35.BINDING_COEFFICIENT,
            "correction_cross_view_js_coefficient": s35.INVARIANCE_COEFFICIENT,
            "second_encoder_pass": False,
        },
        "decision_shell": {
            "baseline": "S45_equal_mean_standardized_primary_plus_corrected_relation",
            "treatment": "S46_coordinatewise_median_standardized_primary_native_relation_corrected_relation",
            "fusion_epsilon": s35.FUSION_EPSILON,
            "treatment_trainable_parameters": 0,
            "learned_gate": False,
            "temperature": None,
            "threshold": None,
        },
        "partitions": {
            "train_semantic_cases": len(train_rows),
            "dev_semantic_cases": len(dev_rows),
            "domains": sorted({r.domain for r in train_rows}),
            "k": 4,
            "views_per_option": 2,
            "identical_rows_across_training_arms": True,
            "prior_track_exact_rows_used": False,
            "s45_train_dev_rows_used": False,
            "s46_a0_rows_used": False,
        },
        "reference_arm": reference,
        "treatment_training_arm": treatment,
        "selected_treatment_checkpoint": {
            "epoch": int(checkpoint_payload["selected_dev_epoch"]),
            "total_parameter_count": int(checkpoint_payload["total_parameter_count"]),
            "correction_parameter_count": int(checkpoint_payload["correction_parameter_count"]),
        },
        "same_checkpoint": {
            "legacy_s45_shell": legacy,
            "robust_s46_shell": robust,
            "delta_robust_minus_legacy": _shell_delta(legacy, robust),
            "expert_metrics": experts,
            "robust_gates": robust_gates,
            "robust_dev_ready": robust_dev_ready,
        },
        "trajectory_invariant": {
            "all_epoch_runtime_state_sha256_equal": True,
            "reference_runtime_trajectory_sha256": reference["runtime_trajectory_sha256"],
            "treatment_runtime_trajectory_sha256": treatment["runtime_trajectory_sha256"],
            "epoch_count": len(reference["history"]),
        },
        "post_dev_tuning_performed": False,
        "second_dev_run_performed": False,
        "sealed_confirm_opened": False,
        "external_laya_jev_evaluation_opened": False,
        "production_ready_claimed": False,
        "train_manifest_sha256": _sha256(train_manifest),
        "dev_manifest_sha256": _sha256(dev_manifest),
    }

    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S46_TRAIN_DEV_RECEIPT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
