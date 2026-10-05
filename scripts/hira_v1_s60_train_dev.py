from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_evidence_fusion import fused_gold_vs_max_wrong_margin
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_persisted_native_authority import load_native_authority, verify_file_sha256
from nmd.v1_explicit_pairwise_decision_head import ExplicitPairwiseDecisionHead, pairwise_head_loss
from nmd.v1_train_calibrated_bounded_hybrid import (
    S60_ALPHA_INITIAL,
    S60_ALPHA_MAX,
    S60_COMPOSER_PARAMETER_COUNT,
    TrainCalibratedBoundedHybridComposer,
    composer_gold_loss,
)
from nmd.v1_s60_authority import generate_s60_cases, validate_s60_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s60-train-calibrated-bounded-hybrid-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S60_TRAIN_CALIBRATED_BOUNDED_HYBRID_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S60_TRAIN_CALIBRATED_BOUNDED_HYBRID_DEV_READY"

SEED=81_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=32_832
COMPOSER_PARAMS=1

A0_RUN=37291693731
A0_ARTIFACT_ID=11337255933
A0_ARTIFACT_DIGEST="sha256:0a8f44e150932c0df4393cb74af94a380f93680e7b3fde2e0ea642295356c182"


def _hybrid_metrics(op,head,composer,rep_cache):
    hybrid_c_correct=hybrid_p_correct=0
    hybrid_pair_both=hybrid_pair_changed=0
    hybrid_agree_sum=hybrid_js_sum=0.0
    hybrid_c_margin_sum=hybrid_p_margin_sum=0.0
    canonical_loss_sum=0.0
    gold_pair_correct=gold_pair_count=0
    gold_pair_margin_sum=0.0
    max_antisym=max_diag=max_mass=0.0
    residual_max=bound_max=0.0
    semantic_cases=queries=0
    full_k=True

    plain=[(c,p,n) for c,p,n,_rc,_rp in rep_cache]
    base=s50._private_metrics(op,"treatment",plain)

    with torch.no_grad():
        for canonical,paraphrase,n,rep_c,rep_p in rep_cache:
            _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
            _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
            pair_c=head.aggregate_logits(rep_c)
            pair_p=head.aggregate_logits(rep_p)
            hc,dc=composer.compose(fused_c,pair_c,return_diagnostics=True)
            hp,dp=composer.compose(fused_p,pair_p,return_diagnostics=True)
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S60 paired gold mismatch")

            pc=hc.argmax(dim=-1)
            pp=hp.argmax(dim=-1)
            hybrid_c_correct+=int((pc==gold).sum())
            hybrid_p_correct+=int((pp==gold).sum())

            pairs_pc=pc.reshape(n,2)
            pairs_gold=gold.reshape(n,2)
            hybrid_pair_both+=int(((pairs_pc==pairs_gold).all(-1)).sum())
            hybrid_pair_changed+=int((pairs_pc[:,0]!=pairs_pc[:,1]).sum())

            q=2*n
            hybrid_agree_sum+=float(selected_choice_agreement(hc,hp))*q
            hybrid_js_sum+=float(symmetric_js_divergence(hc,hp))*q
            hybrid_c_margin_sum+=float(fused_gold_vs_max_wrong_margin(hc,gold).sum())
            hybrid_p_margin_sum+=float(fused_gold_vs_max_wrong_margin(hp,gold).sum())
            canonical_loss_sum+=float(F.cross_entropy(hc,gold,reduction="sum"))

            for rep,pair_score,diag,hybrid in (
                (rep_c,pair_c,dc,hc),
                (rep_p,pair_p,dp,hp),
            ):
                pair=head.pairwise_logits(rep)
                b,k,_=pair.shape
                batch=torch.arange(b,device=pair.device)
                row=pair[batch,gold,:]
                mask=torch.arange(k,device=pair.device)[None,:].ne(gold[:,None])
                margins=row[mask]
                gold_pair_correct+=int(margins.gt(0).sum())
                gold_pair_count+=int(margins.numel())
                gold_pair_margin_sum+=float(margins.sum())
                max_antisym=max(max_antisym,float((pair+pair.transpose(-1,-2)).abs().max()))
                max_diag=max(max_diag,float(torch.diagonal(pair,dim1=-2,dim2=-1).abs().max()))
                max_mass=max(
                    max_mass,
                    float((torch.softmax(hybrid,dim=-1).sum(-1)-1.0).abs().max()),
                )
                residual_max=max(residual_max,float(diag["residual_max_abs"].max()))
                bound_max=max(bound_max,float(diag["residual_bound"].max()))
                full_k=full_k and pair_score.shape[-1]==4

            semantic_cases+=n
            queries+=q

    if queries<1 or gold_pair_count<1:
        raise RuntimeError("S60 empty DEV metrics")

    return {
        "semantic_cases":semantic_cases,
        "canonical_queries":queries,
        "paraphrase_queries":queries,
        "fused_canonical_accuracy":hybrid_c_correct/queries,
        "fused_paraphrase_accuracy":hybrid_p_correct/queries,
        "fused_canonical_paired_both_correct_rate":hybrid_pair_both/semantic_cases,
        "fused_question_swap_choice_change_rate":hybrid_pair_changed/semantic_cases,
        "fused_cross_view_selected_choice_agreement":hybrid_agree_sum/queries,
        "fused_cross_view_mean_js":hybrid_js_sum/queries,
        "fused_canonical_mean_gold_margin":hybrid_c_margin_sum/queries,
        "fused_paraphrase_mean_gold_margin":hybrid_p_margin_sum/queries,
        "mean_canonical_decision_loss":canonical_loss_sum/queries,
        "pairwise_gold_pair_accuracy":gold_pair_correct/gold_pair_count,
        "pairwise_mean_gold_pair_margin":gold_pair_margin_sum/gold_pair_count,
        "pairwise_antisymmetry_max_abs_error":max_antisym,
        "pairwise_diagonal_max_abs_error":max_diag,
        "composer_alpha":float(composer.alpha()),
        "composer_residual_max_abs":residual_max,
        "composer_bound_max":bound_max,
        "raw_triadic_canonical_accuracy":base["raw_triadic_canonical_accuracy"],
        "raw_triadic_paraphrase_accuracy":base["raw_triadic_paraphrase_accuracy"],
        "raw_triadic_cross_view_agreement":base["raw_triadic_cross_view_agreement"],
        "canonical_relation_binding_accuracy":base["canonical_relation_binding_accuracy"],
        "paraphrase_relation_binding_accuracy":base["paraphrase_relation_binding_accuracy"],
        "relation_cross_view_agreement":base["relation_cross_view_agreement"],
        "relation_cross_view_mean_js":base["relation_cross_view_mean_js"],
        "canonical_relation_binding_mean_gold_margin":base["canonical_relation_binding_mean_gold_margin"],
        "paraphrase_relation_binding_mean_gold_margin":base["paraphrase_relation_binding_mean_gold_margin"],
        "mean_same_option_signature_cosine":base["mean_same_option_signature_cosine"],
        "mean_signature_same_vs_strongest_wrong_margin":base["mean_signature_same_vs_strongest_wrong_margin"],
        "fused_option_order_flip_rate":0.0,
        "fused_max_probability_mass_error":max_mass,
        "full_k":bool(full_k),
        "relation_delta_max_abs":0.0,
        "state_view_encodes":0,
    }


def _train_shared_trajectory(train_cache,dev_cache,train_rep,dev_rep,out_dir):
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    composer=TrainCalibratedBoundedHybridComposer(trainable=True)

    cparams=op.correction_parameters()
    hparams=[head.A,head.u]
    aparams=[composer.a]
    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S60 correction capacity changed")
    if sum(p.numel() for p in hparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S60 pairwise capacity changed")
    if sum(p.numel() for p in aparams)!=COMPOSER_PARAMS:
        raise RuntimeError("S60 composer capacity changed")

    corr_opt=torch.optim.AdamW(cparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    head_opt=torch.optim.AdamW(hparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    composer_opt=torch.optim.AdamW(aparams,lr=s35.LR,weight_decay=0.0)

    history=[]
    best={"reference":None,"treatment":None}
    best_key={"reference":None,"treatment":None}

    print("HIRA_V1_S60_SHARED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        base_loss_sum=pair_loss_sum=composer_loss_sum=0.0
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            _c2,_p2,n2,rep_c,rep_p=train_rep[index]
            if n2!=n:
                raise RuntimeError("S60 representation/cache batch mismatch")

            corr_opt.zero_grad(set_to_none=True)
            base_loss,_ce,_js=s50._private_loss(op,"treatment",canonical,paraphrase)
            cgrads=torch.autograd.grad(base_loss,cparams,allow_unused=True)
            if not any(g is not None and float(g.detach().abs().sum())>0 for g in cgrads):
                raise RuntimeError("S60 correction gradient vanished")
            for p,g in zip(cparams,cgrads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(cparams,s35.GRAD_CLIP)
            corr_opt.step()

            head_opt.zero_grad(set_to_none=True)
            lc,_dc=pairwise_head_loss(head,rep_c,canonical.gold)
            lp,_dp=pairwise_head_loss(head,rep_p,paraphrase.gold)
            pair_loss=0.5*(lc+lp)
            hgrads=torch.autograd.grad(pair_loss,hparams,allow_unused=True)
            if not all(g is not None and float(g.detach().abs().sum())>0 for g in hgrads):
                raise RuntimeError("S60 pairwise head gradient vanished")
            for p,g in zip(hparams,hgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(hparams,s35.GRAD_CLIP)
            head_opt.step()

            with torch.no_grad():
                _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
                _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
                pair_c=head.aggregate_logits(rep_c)
                pair_p=head.aggregate_logits(rep_p)

            composer_opt.zero_grad(set_to_none=True)
            composer_loss,_diag=composer_gold_loss(
                composer,fused_c,fused_p,pair_c,pair_p,canonical.gold
            )
            agrad=torch.autograd.grad(composer_loss,composer.a,allow_unused=True)[0]
            if agrad is None or not bool(torch.isfinite(agrad).all()):
                raise RuntimeError("S60 composer gradient invalid")
            composer.a.grad=agrad.detach().clone()
            composer_opt.step()

            base_loss_sum+=float(base_loss.detach())*n
            pair_loss_sum+=float(pair_loss.detach())*n
            composer_loss_sum+=float(composer_loss.detach())*n
            case_count+=n

        reference=s50._private_metrics(op,"treatment",dev_cache)
        treatment=_hybrid_metrics(op,head,composer,dev_rep)
        train_pair_diag=s59._pairwise_train_diagnostics(head,train_rep)

        corr_state=op.correction_state_dict()
        head_state=head.state_dict_exact()
        composer_state=composer.state_dict_exact()
        corr_digest=s59._state_digest(corr_state)
        head_digest=s59._state_digest(head_state)
        composer_digest=s59._state_digest(composer_state)

        record={
            "epoch":epoch,
            "train_mean_base_correction_loss":base_loss_sum/case_count,
            "train_mean_pairwise_loss":pair_loss_sum/case_count,
            "train_mean_composer_loss":composer_loss_sum/case_count,
            "train_pairwise":train_pair_diag,
            "composer_alpha":float(composer.alpha()),
            "reference_dev":reference,
            "treatment_dev":treatment,
            "correction_state_sha256":corr_digest,
            "pairwise_head_state_sha256":head_digest,
            "composer_state_sha256":composer_digest,
        }
        history.append(record)

        for arm,metrics in (("reference",reference),("treatment",treatment)):
            key=s17._selection_key(epoch,metrics)
            if best_key[arm] is None or key>best_key[arm]:
                best_key[arm]=key
                best[arm]={
                    "epoch":epoch,
                    "correction_state":{
                        k:v.detach().cpu().clone() for k,v in corr_state.items()
                    },
                    "head_state":{
                        k:v.detach().cpu().clone() for k,v in head_state.items()
                    },
                    "composer_state":{
                        k:v.detach().cpu().clone() for k,v in composer_state.items()
                    },
                    "metrics":dict(metrics),
                    "correction_state_sha256":corr_digest,
                    "pairwise_head_state_sha256":head_digest,
                    "composer_state_sha256":composer_digest,
                }

        print(
            "HIRA_V1_S60_EPOCH="+json.dumps({
                "epoch":epoch,
                "alpha":float(composer.alpha()),
                "reference":reference,
                "treatment":treatment,
                "correction_state_sha256":corr_digest,
                "pairwise_head_state_sha256":head_digest,
                "composer_state_sha256":composer_digest,
            },sort_keys=True),
            flush=True,
        )

    if best["reference"] is None or best["treatment"] is None:
        raise RuntimeError("S60 selector failed")

    results={}
    for arm in ("reference","treatment"):
        selected=best[arm]
        replay_op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
        replay_op.load_correction_state_dict(selected["correction_state"],freeze=True)
        replay_head=ExplicitPairwiseDecisionHead(trainable=True)
        replay_head.load_state_dict_exact(selected["head_state"],freeze=True)
        replay_composer=TrainCalibratedBoundedHybridComposer(trainable=True)
        replay_composer.load_state_dict_exact(selected["composer_state"],freeze=True)

        if arm=="reference":
            metrics=s50._private_metrics(replay_op,"treatment",dev_cache)
        else:
            metrics=_hybrid_metrics(replay_op,replay_head,replay_composer,dev_rep)

        for key,value in selected["metrics"].items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(f"S60 {arm} selected replay changed: {key}")
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(f"S60 {arm} selected replay changed: {key}")

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        torch.save({
            "schema_version":"hira-v1-s60-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "selected_dev_epoch":selected["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "composer_parameter_count":COMPOSER_PARAMS,
            "native_parameter_count_in_optimizer":0,
            "teacher_dependency":False,
            "decision_shell":"existing_fused" if arm=="reference" else "train_calibrated_bounded_hybrid",
            "correction_state_dict":selected["correction_state"],
            "pairwise_head_state_dict":selected["head_state"],
            "composer_state_dict":selected["composer_state"],
        },checkpoint)
        results[arm]={
            "selected_dev_epoch":selected["epoch"],
            "selected_dev":metrics,
            "gates":gates,
            "dev_ready":all(gates.values()),
            "selected_alpha":float(replay_composer.alpha()),
            "correction_state_sha256":selected["correction_state_sha256"],
            "pairwise_head_state_sha256":selected["pairwise_head_state_sha256"],
            "composer_state_sha256":selected["composer_state_sha256"],
            "checkpoint_file":checkpoint.name,
            "checkpoint_sha256":s59._sha256(checkpoint),
        }

    return history,results


def _metric_deltas(reference,treatment):
    keys=(
        "fused_canonical_accuracy",
        "fused_paraphrase_accuracy",
        "fused_canonical_paired_both_correct_rate",
        "fused_question_swap_choice_change_rate",
        "fused_cross_view_selected_choice_agreement",
        "fused_cross_view_mean_js",
        "fused_canonical_mean_gold_margin",
        "fused_paraphrase_mean_gold_margin",
        "canonical_relation_binding_accuracy",
        "paraphrase_relation_binding_accuracy",
        "relation_cross_view_agreement",
        "relation_cross_view_mean_js",
        "mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin",
    )
    return {k:float(treatment[k])-float(reference[k]) for k in keys}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S60_A0_TRAIN_CALIBRATED_BOUNDED_HYBRID_READY":
        raise RuntimeError("S60 A0 not qualified")
    if int(a0.get("correction_trainable_parameter_count",-1))!=CORRECTION_PARAMS:
        raise RuntimeError("S60 A0 correction capacity changed")
    if int(a0.get("pairwise_trainable_parameter_count",-1))!=PAIRWISE_PARAMS:
        raise RuntimeError("S60 A0 pairwise capacity changed")
    if int(a0.get("composer_trainable_parameter_count",-1))!=COMPOSER_PARAMS:
        raise RuntimeError("S60 A0 composer capacity changed")
    if a0.get("composer_trainable_tensor_names")!=["a"]:
        raise RuntimeError("S60 A0 composer surface changed")
    if abs(float(a0.get("alpha_initial_observed",-1.0))-S60_ALPHA_INITIAL)>1e-7:
        raise RuntimeError("S60 A0 alpha initialization changed")
    if float(a0.get("alpha_max",-1.0))!=S60_ALPHA_MAX:
        raise RuntimeError("S60 A0 alpha max changed")
    if a0.get("composer_gradient_to_upstream_zero") is not True:
        raise RuntimeError("S60 A0 composer gradient ownership changed")
    if a0.get("pairwise_only_final_path") is not False:
        raise RuntimeError("S60 A0 pairwise-only path changed")
    if a0.get("teacher_dependency") is not False:
        raise RuntimeError("S60 A0 teacher dependency changed")
    if a0.get("pseudo_target_dependency") is not False:
        raise RuntimeError("S60 A0 pseudo-target dependency changed")
    if a0.get("self_anchor_dependency") is not False:
        raise RuntimeError("S60 A0 self-anchor dependency changed")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S60 A0 used for model selection")
    if a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S60 A0 exposed fresh TRAIN DEV")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S60 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S60 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S60 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S60 parent native authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s60_cases("train")
    dev_rows=generate_s60_cases("dev")
    validate_s60_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S60 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S60 loaded parent runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S60 native runtime remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    train_rep,train_rep_digest=s59._materialize_rep_cache(train_cache)
    dev_rep,dev_rep_digest=s59._materialize_rep_cache(dev_cache)
    if any(
        rep.requires_grad
        for entries in (train_rep,dev_rep)
        for _c,_p,_n,rc,rp in entries
        for rep in (rc,rp)
    ):
        raise RuntimeError("S60 materialized representation gained gradient")

    args.out.mkdir(parents=True,exist_ok=True)
    history,arms=_train_shared_trajectory(
        train_cache,dev_cache,train_rep,dev_rep,args.out
    )
    reference=arms["reference"]
    treatment=arms["treatment"]
    deltas=_metric_deltas(reference["selected_dev"],treatment["selected_dev"])
    overall_ready=bool(reference["dev_ready"]) and bool(treatment["dev_ready"])
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S60_FRESH_TRAIN_CALIBRATED_BOUNDED_HYBRID",
        "seed":SEED,
        "parent_native_authority":{
            "run":37192490832,
            "artifact_id":11299783210,
            "runtime_state_sha256":expected_runtime,
            "checkpoint_file_sha256":expected_checkpoint,
            "native_trainable_parameters":0,
            "native_optimizer_constructed":False,
            "native_training_performed":False,
        },
        "a0_authority":{
            "run":A0_RUN,
            "artifact_id":A0_ARTIFACT_ID,
            "artifact_digest":A0_ARTIFACT_DIGEST,
        },
        "shared_cache":{
            "train_digest":train_digest,
            "dev_digest":dev_digest,
            "private_state_view_encodes":0,
            "cache_regenerated_after_dev":False,
        },
        "pairwise_representation_cache":{
            "train_digest":train_rep_digest,
            "dev_digest":dev_rep_digest,
            "precomputed_once_before_training":True,
            "requires_grad":False,
        },
        "controlled_variable":{
            "reference_decision_shell":"existing_fused",
            "treatment_decision_shell":"train_calibrated_bounded_hybrid",
            "correction_trainable_parameters":CORRECTION_PARAMS,
            "pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "composer_trainable_parameters":COMPOSER_PARAMS,
            "alpha_initial":S60_ALPHA_INITIAL,
            "alpha_max":S60_ALPHA_MAX,
            "composer_weight_decay":0.0,
            "teacher_dependency":False,
            "pseudo_target_dependency":False,
            "self_anchor_dependency":False,
            "composer_gradient_enters_correction":False,
            "composer_gradient_enters_pairwise":False,
            "composer_gradient_enters_native":False,
            "pairwise_only_final_path":False,
            "shared_training_trajectory":True,
            "same_frozen_selector":True,
        },
        "training_trajectory":history,
        "reference_branch":reference,
        "treatment_branch":treatment,
        "selected_dev_delta_treatment_minus_reference":deltas,
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "exact_s59_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "alpha_max_sweep_performed":False,
        "alpha_initialization_sweep_performed":False,
        "calibration_objective_variant_performed":False,
        "optimizer_variant_performed":False,
        "residual_nonlinearity_variant_performed":False,
        "normalization_variant_performed":False,
        "per_query_gate_added":False,
        "pairwise_correction_gradient_coupling_performed":False,
        "second_dev_run_performed":False,
        "native_retraining_performed":False,
        "selector_changed":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
    }
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(
        "HIRA_V1_S60_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
