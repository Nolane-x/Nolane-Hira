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
from nmd.v1_explicit_pairwise_decision_head import (
    ExplicitPairwiseDecisionHead,
    S59_PAIRWISE_PARAMETER_COUNT,
    pairwise_head_loss,
)
from nmd.v1_direct_set_arbitration_core import (
    S74_PARAMETER_COUNT,
    DirectSetArbitrationCore,
    direct_set_arbitration_features,
    direct_set_arbitration_loss,
)
from nmd.v1_s74_authority import generate_s74_cases, validate_s74_partitions

import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59
import hira_v1_s72_train_dev as s72


SCHEMA_VERSION="hira-v1-s74-direct-set-arbitration-core-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S74_DIRECT_SET_ARBITRATION_CORE_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S74_DIRECT_SET_ARBITRATION_CORE_DEV_READY"

SEED=95_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=S59_PAIRWISE_PARAMETER_COUNT
CORE_PARAMS=S74_PARAMETER_COUNT

A0_RUN=37484291103
A0_ARTIFACT_ID=11422648744
A0_ARTIFACT_DIGEST="sha256:68620a7c563e5a22325b4415b53020720e37ba05e8f2eb675fef5d83d4089093"


def _dsac_metrics(op,head,core,rep_cache,*,use_relational:bool):
    c_correct=p_correct=0
    pair_both=pair_changed=0
    agree_sum=js_sum=0.0
    c_margin_sum=p_margin_sum=0.0
    canonical_loss_sum=0.0
    gold_pair_correct=gold_pair_count=0
    gold_pair_margin_sum=0.0
    max_antisym=max_diag=max_mass=0.0
    relational_max_abs=0.0
    cases=queries=0

    plain=[(c,p,n) for c,p,n,_rc,_rp in rep_cache]
    base=s50._private_metrics(op,"treatment",plain)

    with torch.no_grad():
        for canonical,paraphrase,n,rc,rp in rep_cache:
            _a,_b,fused_c=s50._branch_logits(op,"treatment",canonical)
            _a,_b,fused_p=s50._branch_logits(op,"treatment",paraphrase)
            pair_c=head.pairwise_logits(rc)
            pair_p=head.pairwise_logits(rp)

            feat_c=direct_set_arbitration_features(
                fused_c,pair_c,use_relational=use_relational
            )
            feat_p=direct_set_arbitration_features(
                fused_p,pair_p,use_relational=use_relational
            )
            logits_c=core.logits_from_features(feat_c)
            logits_p=core.logits_from_features(feat_p)

            relational_max_abs=max(
                relational_max_abs,
                float(feat_c[...,4:].abs().max()),
                float(feat_p[...,4:].abs().max()),
            )

            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S74 paired gold mismatch")

            pred_c=logits_c.argmax(-1)
            pred_p=logits_p.argmax(-1)
            c_correct+=int((pred_c==gold).sum())
            p_correct+=int((pred_p==gold).sum())

            pairs_pred=pred_c.reshape(n,2)
            pairs_gold=gold.reshape(n,2)
            pair_both+=int(((pairs_pred==pairs_gold).all(-1)).sum())
            pair_changed+=int((pairs_pred[:,0]!=pairs_pred[:,1]).sum())

            q=2*n
            agree_sum+=float(selected_choice_agreement(logits_c,logits_p))*q
            js_sum+=float(symmetric_js_divergence(logits_c,logits_p))*q
            c_margin_sum+=float(fused_gold_vs_max_wrong_margin(logits_c,gold).sum())
            p_margin_sum+=float(fused_gold_vs_max_wrong_margin(logits_p,gold).sum())
            canonical_loss_sum+=float(F.cross_entropy(logits_c,gold,reduction="sum"))

            for matrix,logits in ((pair_c,logits_c),(pair_p,logits_p)):
                b,k,_=matrix.shape
                batch=torch.arange(b,device=matrix.device)
                row=matrix[batch,gold,:]
                mask=torch.arange(k,device=matrix.device)[None,:].ne(gold[:,None])
                margins=row[mask]
                gold_pair_correct+=int(margins.gt(0).sum())
                gold_pair_count+=int(margins.numel())
                gold_pair_margin_sum+=float(margins.sum())
                max_antisym=max(
                    max_antisym,
                    float((matrix+matrix.transpose(-1,-2)).abs().max()),
                )
                max_diag=max(
                    max_diag,
                    float(torch.diagonal(matrix,dim1=-2,dim2=-1).abs().max()),
                )
                max_mass=max(
                    max_mass,
                    float((torch.softmax(logits,-1).sum(-1)-1).abs().max()),
                )

            cases+=n
            queries+=q

    if cases<1 or queries<1 or gold_pair_count<1:
        raise RuntimeError("S74 empty DEV metrics")

    return {
        "semantic_cases":cases,
        "canonical_queries":queries,
        "paraphrase_queries":queries,
        "fused_canonical_accuracy":c_correct/queries,
        "fused_paraphrase_accuracy":p_correct/queries,
        "fused_canonical_paired_both_correct_rate":pair_both/cases,
        "fused_question_swap_choice_change_rate":pair_changed/cases,
        "fused_cross_view_selected_choice_agreement":agree_sum/queries,
        "fused_cross_view_mean_js":js_sum/queries,
        "fused_canonical_mean_gold_margin":c_margin_sum/queries,
        "fused_paraphrase_mean_gold_margin":p_margin_sum/queries,
        "mean_canonical_decision_loss":canonical_loss_sum/queries,
        "pairwise_gold_pair_accuracy":gold_pair_correct/gold_pair_count,
        "pairwise_mean_gold_pair_margin":gold_pair_margin_sum/gold_pair_count,
        "pairwise_antisymmetry_max_abs_error":max_antisym,
        "pairwise_diagonal_max_abs_error":max_diag,
        "dsac_relational_feature_max_abs":relational_max_abs,
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
        "full_k":True,
        "relation_delta_max_abs":0.0,
        "state_view_encodes":0,
    }


def _same_state(a,b):
    sa=a.state_dict_exact()
    sb=b.state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _train_matched(train_cache,dev_cache,train_rep,dev_rep,out_dir):
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    reference=DirectSetArbitrationCore(trainable=True)
    treatment=DirectSetArbitrationCore(trainable=True)

    if not _same_state(reference,treatment):
        raise RuntimeError("S74 core initialization changed")

    cparams=op.correction_parameters()
    hparams=[head.A,head.u]
    rparams=list(reference.parameters())
    tparams=list(treatment.parameters())

    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S74 correction capacity changed")
    if sum(p.numel() for p in hparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S74 pairwise capacity changed")
    if sum(p.numel() for p in rparams)!=CORE_PARAMS:
        raise RuntimeError("S74 reference core capacity changed")
    if sum(p.numel() for p in tparams)!=CORE_PARAMS:
        raise RuntimeError("S74 treatment core capacity changed")

    corr_opt=torch.optim.AdamW(cparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    head_opt=torch.optim.AdamW(hparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    ref_opt=torch.optim.AdamW(rparams,lr=s35.LR,weight_decay=0.0)
    trt_opt=torch.optim.AdamW(tparams,lr=s35.LR,weight_decay=0.0)

    history=[]
    best=None
    best_key=None
    ref_grad_live=trt_grad_live=False

    print("HIRA_V1_S74_MATCHED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        base_sum=pair_sum=ref_sum=trt_sum=0.0
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            pc,pp,pn,rep_c,rep_p=train_rep[index]
            if n!=pn or canonical is not pc or paraphrase is not pp:
                raise RuntimeError("S74 representation/cache mismatch")

            corr_opt.zero_grad(set_to_none=True)
            base_loss,_ce,_js=s50._private_loss(op,"treatment",canonical,paraphrase)
            cgrads=torch.autograd.grad(base_loss,cparams,allow_unused=True)
            if not any(g is not None and float(g.detach().abs().sum())>0 for g in cgrads):
                raise RuntimeError("S74 correction gradient vanished")
            for p,g in zip(cparams,cgrads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(cparams,s35.GRAD_CLIP)
            corr_opt.step()

            head_opt.zero_grad(set_to_none=True)
            lc,_=pairwise_head_loss(head,rep_c,canonical.gold)
            lp,_=pairwise_head_loss(head,rep_p,paraphrase.gold)
            pair_loss=0.5*(lc+lp)
            hgrads=torch.autograd.grad(pair_loss,hparams,allow_unused=True)
            if not all(g is not None and float(g.detach().abs().sum())>0 for g in hgrads):
                raise RuntimeError("S74 pairwise gradient vanished")
            for p,g in zip(hparams,hgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(hparams,s35.GRAD_CLIP)
            head_opt.step()

            with torch.no_grad():
                _a,_b,fused_c=s50._branch_logits(op,"treatment",canonical)
                _a,_b,fused_p=s50._branch_logits(op,"treatment",paraphrase)
                pair_c=head.pairwise_logits(rep_c)
                pair_p=head.pairwise_logits(rep_p)

            ref_opt.zero_grad(set_to_none=True)
            ref_loss,_=direct_set_arbitration_loss(
                reference,fused_c,fused_p,pair_c,pair_p,canonical.gold,
                use_relational=False,
            )
            rgrads=torch.autograd.grad(ref_loss,rparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in rgrads):
                raise RuntimeError("S74 reference core gradient invalid")
            if any(float(g.detach().abs().sum())>0 for g in rgrads):
                ref_grad_live=True
            for p,g in zip(rparams,rgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(rparams,s35.GRAD_CLIP)
            ref_opt.step()

            trt_opt.zero_grad(set_to_none=True)
            trt_loss,_=direct_set_arbitration_loss(
                treatment,fused_c,fused_p,pair_c,pair_p,canonical.gold,
                use_relational=True,
            )
            tgrads=torch.autograd.grad(trt_loss,tparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in tgrads):
                raise RuntimeError("S74 treatment core gradient invalid")
            if any(float(g.detach().abs().sum())>0 for g in tgrads):
                trt_grad_live=True
            for p,g in zip(tparams,tgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(tparams,s35.GRAD_CLIP)
            trt_opt.step()

            base_sum+=float(base_loss.detach())*n
            pair_sum+=float(pair_loss.detach())*n
            ref_sum+=float(ref_loss.detach())*n
            trt_sum+=float(trt_loss.detach())*n
            case_count+=n

        if not ref_grad_live or not trt_grad_live:
            raise RuntimeError("S74 core gradient never activated")

        fused_shadow=s50._private_metrics(op,"treatment",dev_cache)
        shared_pair_train=s59._pairwise_train_diagnostics(head,train_rep)
        ref_metrics=_dsac_metrics(
            op,head,reference,dev_rep,use_relational=False
        )
        trt_metrics=_dsac_metrics(
            op,head,treatment,dev_rep,use_relational=True
        )

        for key in ("pairwise_gold_pair_accuracy","pairwise_mean_gold_pair_margin"):
            if abs(float(ref_metrics[key])-float(trt_metrics[key]))>1e-12:
                raise RuntimeError(f"S74 pairwise evidence diverged: {key}")
        if ref_metrics["dsac_relational_feature_max_abs"]!=0.0:
            raise RuntimeError("S74 reference relational channel leaked")
        if trt_metrics["dsac_relational_feature_max_abs"]<=1e-5:
            raise RuntimeError("S74 treatment relational channels degenerate")

        corr_state=op.correction_state_dict()
        head_state=head.state_dict_exact()
        ref_state=reference.state_dict_exact()
        trt_state=treatment.state_dict_exact()
        corr_digest=s59._state_digest(corr_state)
        head_digest=s59._state_digest(head_state)
        ref_digest=s59._state_digest(ref_state)
        trt_digest=s59._state_digest(trt_state)

        record={
            "epoch":epoch,
            "train_mean_base_correction_loss":base_sum/case_count,
            "train_mean_shared_pairwise_loss":pair_sum/case_count,
            "train_mean_reference_dsac_loss":ref_sum/case_count,
            "train_mean_treatment_dsac_loss":trt_sum/case_count,
            "reference_core_gradient_ever_live":ref_grad_live,
            "treatment_core_gradient_ever_live":trt_grad_live,
            "shared_train_pairwise":shared_pair_train,
            "fused_shadow_dev":fused_shadow,
            "reference_dev":ref_metrics,
            "treatment_dev":trt_metrics,
            "correction_state_sha256":corr_digest,
            "shared_pairwise_head_state_sha256":head_digest,
            "reference_dsac_state_sha256":ref_digest,
            "treatment_dsac_state_sha256":trt_digest,
        }
        history.append(record)

        key=s17._selection_key(epoch,fused_shadow)
        if best_key is None or key>best_key:
            best_key=key
            best={
                "epoch":epoch,
                "correction_state":{k:v.detach().cpu().clone() for k,v in corr_state.items()},
                "head_state":{k:v.detach().cpu().clone() for k,v in head_state.items()},
                "reference_state":{k:v.detach().cpu().clone() for k,v in ref_state.items()},
                "treatment_state":{k:v.detach().cpu().clone() for k,v in trt_state.items()},
                "reference_metrics":dict(ref_metrics),
                "treatment_metrics":dict(trt_metrics),
                "fused_shadow":dict(fused_shadow),
                "correction_state_sha256":corr_digest,
                "head_state_sha256":head_digest,
                "reference_state_sha256":ref_digest,
                "treatment_state_sha256":trt_digest,
            }

        print("HIRA_V1_S74_EPOCH="+json.dumps({
            "epoch":epoch,
            "shared_pairwise_gold_pair_accuracy":shared_pair_train["gold_pair_accuracy"],
            "reference":ref_metrics,
            "treatment":trt_metrics,
        },sort_keys=True),flush=True)

    if best is None:
        raise RuntimeError("S74 shared selector failed")

    replay_op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    replay_op.load_correction_state_dict(best["correction_state"],freeze=True)
    replay_head=ExplicitPairwiseDecisionHead(trainable=True)
    replay_head.load_state_dict_exact(best["head_state"],freeze=True)

    results={}
    for arm,use_relational,state,selected_metrics in (
        ("reference",False,best["reference_state"],best["reference_metrics"]),
        ("treatment",True,best["treatment_state"],best["treatment_metrics"]),
    ):
        replay=DirectSetArbitrationCore(trainable=True)
        replay.load_state_dict_exact(state,freeze=True)
        metrics=_dsac_metrics(
            replay_op,replay_head,replay,dev_rep,use_relational=use_relational
        )
        for key,value in selected_metrics.items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(f"S74 {arm} replay changed: {key}")
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(f"S74 {arm} replay changed: {key}")

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        torch.save({
            "schema_version":"hira-v1-s74-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "shared_selected_dev_epoch":best["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "representation_family":"s69_query_gated_identity_interaction",
            "representation_parameter_count":0,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "pairwise_family":"s59_explicit_pairwise",
            "decision_core_family":"s74_direct_set_arbitration_core",
            "decision_core_parameter_count":CORE_PARAMS,
            "use_relational_channels":use_relational,
            "final_logit_semantics":"direct_dsac_scores",
            "objective":"0.5_ce_c_plus_0.5_ce_p_plus_0.10_js",
            "native_parameter_count_in_optimizer":0,
            "teacher_dependency":False,
            "correction_state_dict":best["correction_state"],
            "pairwise_head_state_dict":best["head_state"],
            "dsac_state_dict":state,
        },checkpoint)
        results[arm]={
            "selected_dev_epoch":best["epoch"],
            "selected_dev":metrics,
            "selected_fused_shadow_dev":best["fused_shadow"],
            "gates":gates,
            "dev_ready":all(gates.values()),
            "correction_state_sha256":best["correction_state_sha256"],
            "pairwise_head_state_sha256":best["head_state_sha256"],
            "dsac_state_sha256":
                best["reference_state_sha256"] if arm=="reference"
                else best["treatment_state_sha256"],
            "checkpoint_file":checkpoint.name,
            "checkpoint_sha256":s59._sha256(checkpoint),
        }
    return history,results,best["epoch"]


def _delta(reference,treatment,keys):
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
    if a0.get("outcome")!="HIRA_V1_S74_A0_DIRECT_SET_ARBITRATION_CORE_READY":
        raise RuntimeError("S74 A0 not qualified")
    if int(a0.get("reference_parameter_count",-1))!=CORE_PARAMS:
        raise RuntimeError("S74 A0 reference capacity changed")
    if int(a0.get("treatment_parameter_count",-1))!=CORE_PARAMS:
        raise RuntimeError("S74 A0 treatment capacity changed")
    if a0.get("parameter_initialization_bit_identical") is not True:
        raise RuntimeError("S74 A0 initialization changed")
    if float(a0.get("reference_relational_channels_max_abs",1.0))!=0.0:
        raise RuntimeError("S74 A0 reference relational isolation changed")
    if float(a0.get("treatment_relational_channels_max_abs",0.0))<=1e-5:
        raise RuntimeError("S74 A0 treatment relational evidence degenerate")
    if a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S74 A0 exposed fresh data")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S74 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S74 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S74 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S74 parent authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s74_cases("train")
    dev_rows=generate_s74_cases("dev")
    validate_s74_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S74 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S74 loaded parent runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S74 native remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    (
        train_rep,_train_context,
        train_rep_digest,_train_context_digest,train_rep_difference,
    )=s72._materialize_rep_caches(train_cache)
    (
        dev_rep,_dev_context,
        dev_rep_digest,_dev_context_digest,dev_rep_difference,
    )=s72._materialize_rep_caches(dev_cache)

    args.out.mkdir(parents=True,exist_ok=True)
    history,arms,shared_epoch=_train_matched(
        train_cache,dev_cache,train_rep,dev_rep,args.out
    )

    reference=arms["reference"]
    treatment=arms["treatment"]
    for key in ("pairwise_gold_pair_accuracy","pairwise_mean_gold_pair_margin"):
        if abs(float(reference["selected_dev"][key])-float(treatment["selected_dev"][key]))>1e-12:
            raise RuntimeError(f"S74 selected pairwise evidence not matched: {key}")
    if reference["pairwise_head_state_sha256"]!=treatment["pairwise_head_state_sha256"]:
        raise RuntimeError("S74 selected pairwise states diverged")
    if reference["correction_state_sha256"]!=treatment["correction_state_sha256"]:
        raise RuntimeError("S74 selected correction states diverged")
    if reference["selected_dev_epoch"]!=treatment["selected_dev_epoch"]:
        raise RuntimeError("S74 selected epochs diverged")
    if reference["selected_dev"]["dsac_relational_feature_max_abs"]!=0.0:
        raise RuntimeError("S74 selected reference relational leakage")
    if treatment["selected_dev"]["dsac_relational_feature_max_abs"]<=1e-5:
        raise RuntimeError("S74 selected treatment relational evidence degenerate")

    keys=(
        "fused_canonical_accuracy",
        "fused_paraphrase_accuracy",
        "fused_canonical_paired_both_correct_rate",
        "fused_question_swap_choice_change_rate",
        "fused_cross_view_selected_choice_agreement",
        "fused_cross_view_mean_js",
        "fused_canonical_mean_gold_margin",
        "fused_paraphrase_mean_gold_margin",
        "pairwise_gold_pair_accuracy",
        "pairwise_mean_gold_pair_margin",
        "dsac_relational_feature_max_abs",
    )
    deltas=_delta(reference["selected_dev"],treatment["selected_dev"],keys)
    outcome=OUTCOME_READY if bool(treatment["dev_ready"]) else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S74_FRESH_DIRECT_SET_ARBITRATION_CORE",
        "seed":SEED,
        "parent_s73":{
            "merged_main":"45ff1bc301972fc8aa04c1456200283f65616937",
            "scientific_run":37480212536,
            "artifact_id":11420533826,
            "artifact_digest":"sha256:0f08d05bea219333495529c6ae2c54d65fd9a276dd5f783adff0b0ac3925982f",
            "verdict":"CASE_C",
        },
        "parent_native_authority":{
            "run":37192490832,
            "artifact_id":11299783210,
            "runtime_state_sha256":expected_runtime,
            "checkpoint_file_sha256":expected_checkpoint,
            "native_trainable_parameters":0,
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
        "representation_cache":{
            "pair_family":"s69_query_gated_identity_interaction",
            "pair_train_digest":train_rep_digest,
            "pair_dev_digest":dev_rep_digest,
            "train_pair_context_max_abs_difference":train_rep_difference,
            "dev_pair_context_max_abs_difference":dev_rep_difference,
            "precomputed_once_before_training":True,
            "requires_grad":False,
        },
        "controlled_variable":{
            "reference_input":"fused_native_plus_zero_relational",
            "treatment_input":"fused_native_plus_live_s69_relational",
            "dsac_trainable_parameters_per_arm":CORE_PARAMS,
            "parameter_initialization_bit_identical":True,
            "input_dim":12,
            "hidden_dim":16,
            "final_logit_semantics":"direct_dsac_scores",
            "shared_pairwise_head":True,
            "shared_pairwise_training_trajectory":True,
            "shared_correction_trajectory":True,
            "shared_selection_epoch":True,
            "shared_selected_dev_epoch":shared_epoch,
            "same_train_rows_and_order":True,
            "same_optimizer_lr_weight_decay":True,
            "same_frozen_selector":True,
            "objective_both_arms":"0.5_ce_c_plus_0.5_ce_p_plus_0.10_js",
            "teacher_dependency":False,
            "pseudo_target_dependency":False,
            "dev_target_dependency":False,
        },
        "training_trajectory":history,
        "reference_branch":reference,
        "treatment_branch":treatment,
        "selected_dev_delta_treatment_minus_reference":deltas,
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "exact_s73_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "hidden_width_sweep_performed":False,
        "channel_sweep_performed":False,
        "loss_weight_sweep_performed":False,
        "activation_sweep_performed":False,
        "fused_residual_fallback_added":False,
        "target_changed":False,
        "optimizer_lr_weight_decay_changed":False,
        "native_retraining_performed":False,
        "selector_changed":False,
        "second_dev_run_performed":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
    }

    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S74_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
