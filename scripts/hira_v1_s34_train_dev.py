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
from nmd.v1_s33_authority import generate_s33_cases
from nmd.v1_s34_authority import generate_s34_cases, validate_s34_partitions
from nmd.v1_s6_semantic_core import HIRA_V1_S6_LORA_PARAMETER_COUNT
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
from nmd.v1_entropic_relation_transport import QueryConditionedEntropicRelationTransport
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
    _gold_tensors,
    _losses,
    _original_a13_trainable,
    _selection_key,
    evaluate,
)
import hira_v1_s17_train_dev as s17mod
from hira_v1_s33_train_dev import _assert_s33_fresh
from hira_v1_s34_a0_entropic_relation_transport import cases as s34_a0_cases


SCHEMA_VERSION="hira-v1-s34-matched-entropic-relation-transport-train-dev-v1"
SEED=55001


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


def _assert_s34_fresh(train_rows,dev_rows):
    # S34 helper rejects exact S0-S32 and S33-A0 overlaps.
    _assert_s34_fresh(train_rows,dev_rows)

    current=(*train_rows,*dev_rows)
    current_states={x for r in current for x in _state_texts(r)}
    current_questions={x for r in current for x in _question_texts(r)}

    prior33=(*generate_s34_cases("train"),*generate_s34_cases("dev"))
    prior33_states={x for r in prior33 for x in _state_texts(r)}
    prior33_questions={x for r in prior33 for x in _question_texts(r)}
    if current_states & prior33_states:
        raise RuntimeError("S34 exact state overlap with exposed S34 rows")
    if current_questions & prior33_questions:
        raise RuntimeError("S34 exact question overlap with exposed S34 rows")

    a0=s34_a0_cases()
    a0_states={x for r in a0 for x in (r.state_a,r.state_b)}
    a0_questions={x for r in a0 for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    if current_states & a0_states:
        raise RuntimeError("S34 TRAIN DEV state overlap with S34-A0")
    if current_questions & a0_questions:
        raise RuntimeError("S34 TRAIN DEV question overlap with S34-A0")


def _arm_config(arm: str):
    if arm not in {"control","transport"}:
        raise ValueError(arm)
    return {
        "builder":build_hira_v1_s17_norm_balanced_core,
        "enforce":enforce_s17_eval,
        "iter_modules":iter_a13_lora_modules,
        "state_dict":a13_lora_state_dict,
        "load_state":load_a13_lora_state_dict,
        "lora_count":HIRA_V1_S6_LORA_PARAMETER_COUNT,
        "total_count":HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        "checkpoint_name":f"{arm}-relation-candidate.pt",
        "checkpoint_schema":(
            "hira-v1-s34-control-relation-checkpoint-v1"
            if arm=="control"
            else "hira-v1-s34-transport-relation-checkpoint-v1"
        ),
        "checkpoint_kind":(
            "matched-control-s13-relation-s17-shell-a13-w28"
            if arm=="control"
            else "matched-entropic-transport-relation-s17-shell-a13-w28"
        ),
    }


def _transport_relation_outputs(
    runtime,
    *,
    state_tokens,
    state_mask,
    question_tokens,
    question_mask,
    encoded,
):
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S34 transport projection scorer missing")
    operator=QueryConditionedEntropicRelationTransport(
        state_relevance_temperature=0.10,
        option_relevance_temperature=0.10,
        kernel_temperature=0.10,
        logit_temperature=0.10,
        sinkhorn_iterations=12,
        epsilon=1e-12,
    )
    logits,signatures,diag=operator(
        projection=scorer.projection,
        state_tokens=state_tokens.repeat_interleave(2,dim=0),
        state_mask=state_mask.repeat_interleave(2,dim=0),
        question_tokens=question_tokens,
        question_mask=question_mask,
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2,dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2,dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2,dim=0),
    )
    return logits,signatures,diag.base


_CONTROL_RELATION_OUTPUTS=s17mod._relation_outputs


def _set_relation_arm(arm):
    if arm=="control":
        s17mod._relation_outputs=_CONTROL_RELATION_OUTPUTS
    elif arm=="transport":
        s17mod._relation_outputs=_transport_relation_outputs
    else:
        raise ValueError(arm)


def _arm_losses(runtime,rows,arm):
    _set_relation_arm(arm)
    return _losses(runtime,rows)


def _arm_evaluate(runtime,rows,arm):
    _set_relation_arm(arm)
    return evaluate(runtime,rows)


def _gates(runtime,selected,history,trainable_count,lora_trainable,expected_total,expected_lora,train_rows,dev_rows):
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S34 projection scorer missing")
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
        "trainable_projection_exact_32768":scorer.projection_trainable_parameter_count==HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
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
        raise RuntimeError(f"S34 {arm} trainable count changed: {trainable_count}")
    if _original_a13_trainable(runtime)!=0:
        raise RuntimeError(f"S34 {arm} original A13 became trainable")
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError(f"S34 {arm} projection scorer missing")
    if scorer.projection_trainable_parameter_count!=HIRA_V1_S17_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError(f"S34 {arm} projection surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError(f"S34 {arm} HIRACore became trainable")

    optimizer=torch.optim.AdamW(trainable,lr=LR,weight_decay=WEIGHT_DECAY)
    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    print(f"HIRA_V1_S34_{arm.upper()}_TRAIN_BEGIN",flush=True)

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
            _total,primary_block,relation_block,pieces,*_rest=_arm_losses(runtime,rows,arm)

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
            raise RuntimeError(f"S34 {arm} TRAIN state-once changed")

        dev=_arm_evaluate(runtime,dev_rows,arm)
        if int(dev["state_view_encodes"])!=2*len(dev_rows):
            raise RuntimeError(f"S34 {arm} DEV state-once changed")

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
            "HIRA_V1_S34_ARM_EPOCH="+json.dumps(
                {"arm":arm,**record},sort_keys=True
            ),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S34 {arm} selection produced no checkpoint")

    cfg["load_state"](runtime.encoder,best_state["lora"],freeze=False)
    scorer.load_projection_state_dict(best_state["projection"],freeze=False)
    cfg["enforce"](runtime)
    selected=_arm_evaluate(runtime,dev_rows,arm)

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
            raise RuntimeError(f"S34 {arm} selected DEV replay changed: {key}")

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
        "projection_parameter_count":HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
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


def _metric_deltas(local,global_arm):
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
    a=local["selected_dev"]
    b=global_arm["selected_dev"]
    return {key:float(b[key])-float(a[key]) for key in keys}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S34_A0_ENTROPIC_RELATION_TRANSPORT_READY":
        raise RuntimeError("S34-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S34-A0 unexpectedly used for model selection")
    if int(a0.get("operator_added_parameter_count",-1))!=0:
        raise RuntimeError("S34-A0 operator parameter count changed")
    if int(a0.get("physical_trainable_parameter_count",-1))!=49152:
        raise RuntimeError("S34-A0 physical surface changed")
    if int(a0.get("sinkhorn_iterations",-1))!=12:
        raise RuntimeError("S34-A0 Sinkhorn iterations changed")
    if float(a0.get("real_max_row_marginal_residual",1.0))>1e-4:
        raise RuntimeError("S34-A0 row marginal residual failed")
    if float(a0.get("real_max_column_marginal_residual",1.0))>1e-4:
        raise RuntimeError("S34-A0 column marginal residual failed")
    if int(a0.get("controlled_q_a_selected_option",-1))!=0:
        raise RuntimeError("S34-A0 controlled qA failed")
    if int(a0.get("controlled_q_b_selected_option",-1))!=1:
        raise RuntimeError("S34-A0 controlled qB failed")
    if float(a0.get("controlled_transport_plan_intervention_max_abs",0.0))<=0.0:
        raise RuntimeError("S34-A0 transport intervention inactive")
    if not bool(a0.get("controlled_degenerate_finite",False)):
        raise RuntimeError("S34-A0 degenerate geometry failed")
    if not bool(a0.get("checkpoint_roundtrip_exact",False)):
        raise RuntimeError("S34-A0 checkpoint roundtrip failed")

    train_rows=generate_s34_cases("train")
    dev_rows=generate_s34_cases("dev")
    validate_s34_partitions(train_rows,dev_rows)
    _assert_s34_fresh(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S34 semantic revision changed")

    args.out.mkdir(parents=True,exist_ok=True)
    control=_train_arm("control",bundle,manifest,train_rows,dev_rows,args.out)
    transport=_train_arm("transport",bundle,manifest,train_rows,dev_rows,args.out)

    if control["dev_ready"] and transport["dev_ready"]:
        outcome="HIRA_V1_S34_MATCHED_TRANSPORT_BOTH_DEV_READY"
    elif control["dev_ready"]:
        outcome="HIRA_V1_S34_MATCHED_TRANSPORT_CONTROL_DEV_READY"
    elif transport["dev_ready"]:
        outcome="HIRA_V1_S34_MATCHED_TRANSPORT_TREATMENT_DEV_READY"
    else:
        outcome="HIRA_V1_S34_MATCHED_TRANSPORT_DEV_COMPLETE"

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S34_FRESH_MATCHED_CONTROL_VS_ENTROPIC_TRANSPORT",
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
            "control_relation_operator":"S13_CrossViewRelationCanonicalizer",
            "treatment_relation_operator":"S34_QueryConditionedEntropicRelationTransport",
            "state_relevance_temperature":0.10,
            "option_relevance_temperature":0.10,
            "transport_kernel_temperature":0.10,
            "transport_logit_temperature":0.10,
            "sinkhorn_iterations":12,
            "transport_epsilon":1e-12,
            "treatment_added_parameters":0,
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
            "s34_a0_rows_used":False,
            "m5_final_rows_used":False,
            "w29_w34_sealed_rows_used":False,
        },
        "control_arm":control,
        "transport_arm":transport,
        "matched_selected_dev_delta_transport_minus_control":_metric_deltas(control,transport),
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
    print("HIRA_V1_S34_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
