from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import torch

from nmd.local_runtime import (
    A13_REVISION,
    load_hira_v0_m4_bundle,
    read_runtime_bundle_manifest,
)
from nmd.v1_a13_ffn_lora import (
    a13_ffn_only_lora_state_dict,
    iter_a13_ffn_only_lora_modules,
    load_a13_ffn_only_lora_state_dict,
)
from nmd.v1_a13_lora import (
    a13_lora_state_dict,
    iter_a13_lora_modules,
    load_a13_lora_state_dict,
)
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_norm_balanced_gradient import (
    apply_gradient_update,
    norm_balanced_gradient_update,
)
from nmd.v1_s29_authority import generate_s29_cases
from nmd.v1_s30_authority import generate_s30_cases, validate_s30_partitions
from nmd.v1_s30_semantic_core import (
    S30_ATTENTION_LORA_PARAMETER_COUNT,
    S30_ATTENTION_TOTAL_PARAMETER_COUNT,
    S30_FFN_LORA_PARAMETER_COUNT,
    S30_FFN_TOTAL_PARAMETER_COUNT,
    S30_PROJECTION_PARAMETER_COUNT,
    build_s30_attention_arm,
    build_s30_ffn_arm,
    enforce_s30_attention_eval,
    enforce_s30_ffn_eval,
)
from hira_v1_s17_train_dev import (
    BALANCE_EPSILON,
    BATCH_SIZE,
    BINDING_COEFFICIENT,
    BINDING_CONTRASTIVE_TEMPERATURE,
    CANONICALIZATION_COEFFICIENT,
    EPOCHS,
    FUSION_EPSILON,
    GRAD_CLIP,
    INVARIANCE_COEFFICIENT,
    LR,
    OPTION_ALIGN_COEFFICIENT,
    OPTION_ALIGN_TEMPERATURE,
    PAIR_TEMPERATURE,
    ROLE_TEMPERATURE,
    SIGNATURE_SEPARATION_MARGIN,
    SWAP_COEFFICIENT,
    SWAP_MARGIN,
    WEIGHT_DECAY,
    _losses,
    _original_a13_trainable,
    _selection_key,
    evaluate,
)
from hira_v1_s29_train_dev import _assert_fresh_against_prior
from hira_v1_s30_a0_matched_adaptation import cases as s30_a0_cases


SCHEMA_VERSION="hira-v1-s30-matched-adaptation-train-dev-v1"
SEED=51001


def _sha256(path: Path) -> str:
    digest=sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_texts(row):
    return (row.state_a,row.state_b)


def _question_texts(row):
    return (row.question_a1,row.question_a2,row.question_b1,row.question_b2)


def _assert_s30_fresh(train_rows,dev_rows):
    # S29 helper already rejects exact state/question overlap with S0-S28.
    _assert_fresh_against_prior(train_rows,dev_rows)

    current=(*train_rows,*dev_rows)
    current_states={x for r in current for x in _state_texts(r)}
    current_questions={x for r in current for x in _question_texts(r)}

    prior29=(*generate_s29_cases("train"),*generate_s29_cases("dev"))
    prior29_states={x for r in prior29 for x in _state_texts(r)}
    prior29_questions={x for r in prior29 for x in _question_texts(r)}
    if current_states & prior29_states:
        raise RuntimeError("S30 exact state overlap with exposed S29 rows")
    if current_questions & prior29_questions:
        raise RuntimeError("S30 exact question overlap with exposed S29 rows")

    a0=s30_a0_cases()
    a0_states={x for r in a0 for x in (r.state_a,r.state_b)}
    a0_questions={x for r in a0 for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    if current_states & a0_states:
        raise RuntimeError("S30 TRAIN DEV state overlap with S30-A0")
    if current_questions & a0_questions:
        raise RuntimeError("S30 TRAIN DEV question overlap with S30-A0")


def _arm_config(arm: str):
    if arm=="attention":
        return {
            "builder":build_s30_attention_arm,
            "enforce":enforce_s30_attention_eval,
            "iter_modules":iter_a13_lora_modules,
            "state_dict":a13_lora_state_dict,
            "load_state":load_a13_lora_state_dict,
            "lora_count":S30_ATTENTION_LORA_PARAMETER_COUNT,
            "total_count":S30_ATTENTION_TOTAL_PARAMETER_COUNT,
            "checkpoint_name":"attention-only-candidate.pt",
            "checkpoint_schema":"hira-v1-s30-attention-only-checkpoint-v1",
            "checkpoint_kind":"matched-attention-only-s17-shell-a13-w28",
        }
    if arm=="ffn":
        return {
            "builder":build_s30_ffn_arm,
            "enforce":enforce_s30_ffn_eval,
            "iter_modules":iter_a13_ffn_only_lora_modules,
            "state_dict":a13_ffn_only_lora_state_dict,
            "load_state":load_a13_ffn_only_lora_state_dict,
            "lora_count":S30_FFN_LORA_PARAMETER_COUNT,
            "total_count":S30_FFN_TOTAL_PARAMETER_COUNT,
            "checkpoint_name":"ffn-only-candidate.pt",
            "checkpoint_schema":"hira-v1-s30-ffn-only-checkpoint-v1",
            "checkpoint_kind":"matched-ffn-only-s17-shell-a13-w28",
        }
    raise ValueError(arm)


def _gates(runtime,selected,history,trainable_count,lora_trainable,expected_total,expected_lora,train_rows,dev_rows):
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S30 projection scorer missing")
    return {
        "fused_canonical_accuracy_gte_0_85":selected["fused_canonical_accuracy"]>=0.85,
        "fused_paired_both_correct_gte_0_75":selected["fused_canonical_paired_both_correct_rate"]>=0.75,
        "fused_question_swap_change_gte_0_80":selected["fused_question_swap_choice_change_rate"]>=0.80,
        "fused_cross_view_choice_agreement_gte_0_95":selected["fused_cross_view_selected_choice_agreement"]>=0.95,
        "fused_cross_view_mean_js_lte_0_05":selected["fused_cross_view_mean_js"]<=0.05,
        "fused_canonical_margin_gte_0_15":selected["fused_canonical_mean_gold_margin"]>=0.15,
        "canonical_relation_accuracy_gte_0_80":selected["canonical_relation_binding_accuracy"]>=0.80,
        "canonical_relation_margin_gte_0_15":selected["canonical_relation_binding_mean_gold_margin"]>=0.15,
        "same_option_signature_cosine_gte_0_90":selected["mean_same_option_signature_cosine"]>=0.90,
        "signature_margin_gte_0_15":selected["mean_signature_same_vs_strongest_wrong_margin"]>=0.15,
        "fused_option_order_flip_lte_0_02":selected["fused_option_order_flip_rate"]<=0.02,
        "fused_probability_mass_error_lte_1e_6":selected["fused_max_probability_mass_error"]<=1e-6,
        "full_k":bool(selected["full_k"]),
        "relation_delta_zero":float(selected["relation_delta_max_abs"])==0.0,
        "state_once_train_both_views":all(int(x["train_state_view_encodes"])==2*len(train_rows) for x in history),
        "state_once_dev_both_views":all(int(x["dev"]["state_view_encodes"])==2*len(dev_rows) for x in history),
        "trainable_total_exact":trainable_count==expected_total,
        "trainable_lora_exact":lora_trainable==expected_lora,
        "trainable_projection_exact_32768":scorer.projection_trainable_parameter_count==S30_PROJECTION_PARAMETER_COUNT,
        "original_a13_frozen":_original_a13_trainable(runtime)==0,
        "hira_core_frozen":not any(p.requires_grad for p in runtime.hira.parameters()),
        "fusion_added_parameters_zero":GradientIsolatedFullKEvidenceFusion(epsilon=FUSION_EPSILON).parameter_count==0,
    }


def _train_arm(arm,bundle,manifest,train_rows,dev_rows,out_dir):
    cfg=_arm_config(arm)

    # Reset construction RNG so each arm begins from its own deterministic
    # zero-B LoRA initialization rather than inheriting RNG state from the
    # other arm.
    random.seed(SEED)
    torch.manual_seed(SEED)

    frozen=load_hira_v0_m4_bundle(bundle)
    runtime=cfg["builder"](
        frozen.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    cfg["enforce"](runtime)

    trainable=[p for p in runtime.parameters() if p.requires_grad]
    trainable_count=sum(p.numel() for p in trainable)
    if trainable_count!=cfg["total_count"]:
        raise RuntimeError(f"S30 {arm} trainable count changed: {trainable_count}")
    if _original_a13_trainable(runtime)!=0:
        raise RuntimeError(f"S30 {arm} original A13 became trainable")
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError(f"S30 {arm} projection scorer missing")
    if scorer.projection_trainable_parameter_count!=S30_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError(f"S30 {arm} projection surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError(f"S30 {arm} HIRACore became trainable")

    optimizer=torch.optim.AdamW(trainable,lr=LR,weight_decay=WEIGHT_DECAY)
    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    print(f"HIRA_V1_S30_{arm.upper()}_TRAIN_BEGIN",flush=True)

    for epoch in range(1,EPOCHS+1):
        order=list(range(len(train_rows)))
        random.Random(SEED+epoch).shuffle(order)
        totals={
            "total":0.0,"decision":0.0,"ce":0.0,"swap":0.0,
            "option_alignment":0.0,"binding":0.0,"canonicalization":0.0,
            "consistency_js":0.0,"primary_block":0.0,"relation_block":0.0,
        }
        balance={
            "steps":0,"conflicts":0,"normalized_pre_dot":0.0,"normalized_post_dot":0.0,
            "primary_norm":0.0,"relation_norm":0.0,"reference_scale":0.0,
            "direction_norm":0.0,"combined_norm":0.0,"projection_coefficient":0.0,
        }
        state_view_encodes=0

        for start in range(0,len(order),BATCH_SIZE):
            rows=[train_rows[i] for i in order[start:start+BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True)
            _total,primary_block,relation_block,pieces,*_rest=_losses(runtime,rows)

            primary_raw=torch.autograd.grad(primary_block,trainable,retain_graph=True,allow_unused=True)
            relation_raw=torch.autograd.grad(relation_block,trainable,allow_unused=True)
            primary=[
                torch.zeros_like(p) if g is None else g
                for p,g in zip(trainable,primary_raw)
            ]
            relation=[
                torch.zeros_like(p) if g is None else g
                for p,g in zip(trainable,relation_raw)
            ]
            combined,diag=norm_balanced_gradient_update(primary,relation,epsilon=BALANCE_EPSILON)
            apply_gradient_update(trainable,combined)
            torch.nn.utils.clip_grad_norm_(trainable,GRAD_CLIP)
            optimizer.step()
            cfg["enforce"](runtime)

            n=len(rows)
            for key in totals:
                totals[key]+=pieces[key]*n
            balance["steps"]+=1
            balance["conflicts"]+=int(diag.conflict)
            balance["normalized_pre_dot"]+=diag.normalized_pre_dot
            balance["normalized_post_dot"]+=diag.normalized_post_dot
            balance["primary_norm"]+=diag.primary_norm
            balance["relation_norm"]+=diag.relation_norm
            balance["reference_scale"]+=diag.reference_scale
            balance["direction_norm"]+=diag.direction_norm
            balance["combined_norm"]+=diag.combined_norm
            balance["projection_coefficient"]+=diag.projection_coefficient
            state_view_encodes+=2*n

        if state_view_encodes!=2*len(train_rows):
            raise RuntimeError(f"S30 {arm} TRAIN state-once changed")

        dev=evaluate(runtime,dev_rows)
        if int(dev["state_view_encodes"])!=2*len(dev_rows):
            raise RuntimeError(f"S30 {arm} DEV state-once changed")

        steps=max(1,balance["steps"])
        record={
            "epoch":epoch,
            "train_mean_total_loss":totals["total"]/len(train_rows),
            "train_mean_decision_loss":totals["decision"]/len(train_rows),
            "train_mean_ce":totals["ce"]/len(train_rows),
            "train_mean_swap":totals["swap"]/len(train_rows),
            "train_mean_option_alignment_loss":totals["option_alignment"]/len(train_rows),
            "train_mean_binding_loss":totals["binding"]/len(train_rows),
            "train_mean_canonicalization_loss":totals["canonicalization"]/len(train_rows),
            "train_mean_consistency_js":totals["consistency_js"]/len(train_rows),
            "train_mean_primary_block":totals["primary_block"]/len(train_rows),
            "train_mean_relation_block":totals["relation_block"]/len(train_rows),
            "norm_balancing":{
                "steps":balance["steps"],
                "conflict_rate":balance["conflicts"]/steps,
                "mean_normalized_pre_dot":balance["normalized_pre_dot"]/steps,
                "mean_normalized_post_dot":balance["normalized_post_dot"]/steps,
                "mean_primary_norm":balance["primary_norm"]/steps,
                "mean_relation_norm":balance["relation_norm"]/steps,
                "mean_reference_scale":balance["reference_scale"]/steps,
                "mean_direction_norm":balance["direction_norm"]/steps,
                "mean_combined_norm":balance["combined_norm"]/steps,
                "mean_projection_coefficient":balance["projection_coefficient"]/steps,
            },
            "train_state_view_encodes":state_view_encodes,
            "surface_diagnostics":{
                "projection_weight_norm":float(scorer.projection.weight.detach().norm().cpu()),
                "lora_b_norm":float(torch.sqrt(sum(
                    module.lora_b.detach().pow(2).sum()
                    for module in cfg["iter_modules"](runtime.encoder)
                )).cpu()),
            },
            "dev":dev,
        }
        history.append(record)

        key=_selection_key(epoch,dev)
        if best_key is None or key>best_key:
            best_key=key
            best_epoch=epoch
            best_state={
                "lora":cfg["state_dict"](runtime.encoder),
                "projection":{"projection.weight":scorer.projection.weight.detach().cpu().clone()},
            }
            best_metrics=dict(dev)

        print(
            "HIRA_V1_S30_ARM_EPOCH="+json.dumps(
                {"arm":arm,**record},sort_keys=True
            ),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S30 {arm} selection produced no checkpoint")

    cfg["load_state"](runtime.encoder,best_state["lora"],freeze=False)
    scorer.load_projection_state_dict(best_state["projection"],freeze=False)
    cfg["enforce"](runtime)
    selected=evaluate(runtime,dev_rows)

    replay_keys=(
        "fused_canonical_accuracy","fused_paraphrase_accuracy",
        "fused_canonical_paired_both_correct_rate",
        "fused_question_swap_choice_change_rate",
        "fused_cross_view_selected_choice_agreement","fused_cross_view_mean_js",
        "fused_canonical_mean_gold_margin","canonical_relation_binding_accuracy",
        "canonical_relation_binding_mean_gold_margin",
        "mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin",
        "fused_option_order_flip_rate","fused_max_probability_mass_error",
        "mean_canonical_decision_loss",
    )
    for key in replay_keys:
        if not math.isclose(
            float(selected[key]),
            float(best_metrics[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"S30 {arm} selected DEV replay changed: {key}")

    lora_trainable=sum(
        p.numel()
        for name,p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" in name
    )
    gates=_gates(
        runtime,selected,history,trainable_count,lora_trainable,
        cfg["total_count"],cfg["lora_count"],train_rows,dev_rows,
    )

    out_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=out_dir/cfg["checkpoint_name"]
    torch.save({
        "schema_version":cfg["checkpoint_schema"],
        "kind":cfg["checkpoint_kind"],
        "arm":arm,
        "lora_parameter_count":cfg["lora_count"],
        "projection_parameter_count":S30_PROJECTION_PARAMETER_COUNT,
        "total_parameter_count":cfg["total_count"],
        "lora_rank":8,
        "selected_dev_epoch":best_epoch,
        "semantic_revision":str(manifest["semantic_revision"]),
        "initialization_t0_sha256":str(manifest["t0_checkpoint_sha256"]),
        "lora_state_dict":best_state["lora"],
        "projection_state_dict":best_state["projection"],
    },checkpoint)

    return {
        "arm":arm,
        "selected_dev_epoch":best_epoch,
        "selected_dev":selected,
        "gates":gates,
        "dev_ready":all(gates.values()),
        "history":history,
        "checkpoint_file":cfg["checkpoint_name"],
        "checkpoint_sha256":_sha256(checkpoint),
        "parameter_surface":{
            "total_trainable_parameters":trainable_count,
            "lora_trainable_parameters":lora_trainable,
            "projection_trainable_parameters":scorer.projection_trainable_parameter_count,
            "original_a13_trainable":_original_a13_trainable(runtime),
            "hira_core_trainable":sum(p.numel() for p in runtime.hira.parameters() if p.requires_grad),
            "fusion_added_parameters":0,
            "learned_downstream_scorer_parameters":0,
        },
        "norm_balancing":{
            "rule":"equal_direction_relation_priority_projection",
            "epsilon":BALANCE_EPSILON,
            "mean_conflict_rate":sum(x["norm_balancing"]["conflict_rate"] for x in history)/len(history),
        },
    }


def _metric_deltas(attention,ffn):
    keys=(
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
    a=attention["selected_dev"]
    f=ffn["selected_dev"]
    return {key:float(f[key])-float(a[key]) for key in keys}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S30_A0_MATCHED_ADAPTATION_READY":
        raise RuntimeError("S30-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S30-A0 unexpectedly used for model selection")
    if float(a0.get("arm_token_output_identity_max_abs",-1.0))!=0.0:
        raise RuntimeError("S30-A0 token identity failed")
    if float(a0.get("arm_pooled_output_identity_max_abs",-1.0))!=0.0:
        raise RuntimeError("S30-A0 pooled identity failed")
    if float(a0.get("exact_logit_identity_rate",-1.0))!=1.0:
        raise RuntimeError("S30-A0 logit identity failed")

    train_rows=generate_s30_cases("train")
    dev_rows=generate_s30_cases("dev")
    validate_s30_partitions(train_rows,dev_rows)
    _assert_s30_fresh(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S30 semantic revision changed")

    args.out.mkdir(parents=True,exist_ok=True)
    attention=_train_arm("attention",bundle,manifest,train_rows,dev_rows,args.out)
    ffn=_train_arm("ffn",bundle,manifest,train_rows,dev_rows,args.out)

    if attention["dev_ready"] and ffn["dev_ready"]:
        outcome="HIRA_V1_S30_MATCHED_ADAPTATION_BOTH_DEV_READY"
    elif attention["dev_ready"]:
        outcome="HIRA_V1_S30_MATCHED_ADAPTATION_ATTENTION_DEV_READY"
    elif ffn["dev_ready"]:
        outcome="HIRA_V1_S30_MATCHED_ADAPTATION_FFN_DEV_READY"
    else:
        outcome="HIRA_V1_S30_MATCHED_ADAPTATION_DEV_COMPLETE"

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S30_FRESH_MATCHED_ATTENTION_VS_FFN_ONLY",
        "seed":SEED,
        "optimizer":{
            "name":"AdamW","epochs":EPOCHS,"batch_size_semantic_cases":BATCH_SIZE,
            "lr":LR,"weight_decay":WEIGHT_DECAY,"grad_clip":GRAD_CLIP,
        },
        "loss":{
            "cross_entropy_both_views":True,
            "swap_margin_coefficient":SWAP_COEFFICIENT,
            "swap_margin":SWAP_MARGIN,
            "option_view_infonce_coefficient":OPTION_ALIGN_COEFFICIENT,
            "option_view_infonce_temperature":OPTION_ALIGN_TEMPERATURE,
            "relation_structured_binding_coefficient":BINDING_COEFFICIENT,
            "cross_view_relation_canonicalization_coefficient":CANONICALIZATION_COEFFICIENT,
            "signature_separation_margin":SIGNATURE_SEPARATION_MARGIN,
            "role_temperature":ROLE_TEMPERATURE,
            "pair_temperature":PAIR_TEMPERATURE,
            "binding_contrastive_temperature":BINDING_CONTRASTIVE_TEMPERATURE,
            "cross_view_js_coefficient":INVARIANCE_COEFFICIENT,
            "fusion_epsilon":FUSION_EPSILON,
            "fusion_equal_weight":0.5,
            "fused_primary_relation_logits_detached":True,
            "norm_balanced_gradient":"equal_direction_relation_priority_projection",
            "balance_epsilon":BALANCE_EPSILON,
        },
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "language":"en",
            "domains":sorted({r.domain for r in train_rows}),
            "state_views_per_case":2,
            "question_views_per_semantic_query":2,
            "k":4,
            "views_per_option":2,
            "identical_rows_across_arms":True,
            "identical_batch_order_across_arms":True,
            "prior_track_exact_rows_used":False,
            "s30_a0_rows_used":False,
            "m5_final_rows_used":False,
            "w29_w34_sealed_rows_used":False,
        },
        "attention_arm":attention,
        "ffn_arm":ffn,
        "matched_selected_dev_delta_ffn_minus_attention":_metric_deltas(attention,ffn),
        "post_dev_tuning_performed":False,
        "second_dev_run_performed":False,
        "sealed_confirm_opened":False,
        "multilingual_probe_opened":False,
        "production_ready_claimed":False,
    }

    (args.out/"result.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    (args.out/"train-manifest.json").write_text(
        json.dumps([r.to_dict() for r in train_rows],ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    (args.out/"dev-manifest.json").write_text(
        json.dumps([r.to_dict() for r in dev_rows],ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S30_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
