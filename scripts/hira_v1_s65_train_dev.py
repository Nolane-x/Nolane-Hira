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
from nmd.v1_confidence_adaptive_bounded_hybrid import S61_ALPHA_INITIAL, S61_ALPHA_MAX
from nmd.v1_reliability_supervised_adaptive_gate import S62_ALPHA_PROBE, S62_TARGET_TOLERANCE
from nmd.v1_contextual_reliability_gate import (
    S64_CONTEXT_DIM,
    S64_CONTEXT_PROJECTION_SEED,
    S64_ENCODER_INIT_SEED,
    S64_GATE_PARAMETER_COUNT,
    S64_OPTION_HIDDEN_DIM,
    S64_OPTION_INPUT_DIM,
    S64_POOLED_DIM,
    S64_REPRESENTATION_DIM,
    ContextInjectedReliabilityGate,
    contextual_reliability_loss,
)
from nmd.v1_cross_view_soft_and_reliability import cross_view_soft_and_reliability_loss
from nmd.v1_s65_authority import generate_s65_cases, validate_s65_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s65-cross-view-soft-and-reliability-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S65_CROSS_VIEW_SOFT_AND_RELIABILITY_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S65_CROSS_VIEW_SOFT_AND_RELIABILITY_DEV_READY"

SEED=86_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=32_832
REFERENCE_GATE_PARAMS=60
TREATMENT_GATE_PARAMS=60

A0_RUN=37322871071
A0_ARTIFACT_ID=11349983389
A0_ARTIFACT_DIGEST="sha256:1abecf58d75be991cd89dd3bb6732a859b1159426023a209758b1d5a7d459dea"

def _hybrid_metrics(op,head,gate,rep_cache):
    hybrid_c_correct=hybrid_p_correct=0
    hybrid_pair_both=hybrid_pair_changed=0
    hybrid_agree_sum=hybrid_js_sum=0.0
    hybrid_c_margin_sum=hybrid_p_margin_sum=0.0
    canonical_loss_sum=0.0
    gold_pair_correct=gold_pair_count=0
    gold_pair_margin_sum=0.0
    max_antisym=max_diag=max_mass=0.0
    residual_max=bound_max=0.0
    alpha_sum=alpha_sq_sum=0.0
    alpha_min=float("inf")
    alpha_max=0.0
    alpha_count=0
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
            hc,dc=gate.compose(fused_c,pair_c,rep_c,return_diagnostics=True)
            hp,dp=gate.compose(fused_p,pair_p,rep_p,return_diagnostics=True)
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S65 paired gold mismatch")

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
                a=diag["alpha"].reshape(-1)
                alpha_sum+=float(a.sum())
                alpha_sq_sum+=float((a*a).sum())
                alpha_min=min(alpha_min,float(a.min()))
                alpha_max=max(alpha_max,float(a.max()))
                alpha_count+=int(a.numel())
                full_k=full_k and pair_score.shape[-1]==4

            semantic_cases+=n
            queries+=q

    if queries<1 or gold_pair_count<1 or alpha_count<1:
        raise RuntimeError("S65 empty DEV metrics")

    alpha_mean=alpha_sum/alpha_count
    alpha_var=max(0.0,alpha_sq_sum/alpha_count-alpha_mean*alpha_mean)

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
        "gate_mean_alpha":alpha_mean,
        "gate_min_alpha":alpha_min,
        "gate_max_alpha":alpha_max,
        "gate_std_alpha":alpha_var**0.5,
        "gate_residual_max_abs":residual_max,
        "gate_bound_max":bound_max,
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
    reference_gate=ContextInjectedReliabilityGate(use_context=True,trainable=True)
    treatment_gate=ContextInjectedReliabilityGate(use_context=True,trainable=True)

    cparams=op.correction_parameters()
    hparams=[head.A,head.u]
    order_names=("W_phi","b_phi","w_out","b_out")
    rnamed=dict(reference_gate.named_parameters())
    tnamed=dict(treatment_gate.named_parameters())
    rparams=[rnamed[k] for k in order_names]
    tparams=[tnamed[k] for k in order_names]

    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S65 correction capacity changed")
    if sum(p.numel() for p in hparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S65 pairwise capacity changed")
    if sum(p.numel() for p in rparams)!=REFERENCE_GATE_PARAMS:
        raise RuntimeError("S65 reference gate capacity changed")
    if sum(p.numel() for p in tparams)!=TREATMENT_GATE_PARAMS:
        raise RuntimeError("S65 treatment gate capacity changed")
    if any(not torch.equal(rnamed[k],tnamed[k]) for k in order_names):
        raise RuntimeError("S65 matched gate initialization changed")
    if not torch.equal(reference_gate.context_projection,treatment_gate.context_projection):
        raise RuntimeError("S65 matched context path changed")

    corr_opt=torch.optim.AdamW(cparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    head_opt=torch.optim.AdamW(hparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    ref_opt=torch.optim.AdamW(rparams,lr=s35.LR,weight_decay=0.0)
    trt_opt=torch.optim.AdamW(tparams,lr=s35.LR,weight_decay=0.0)

    history=[]
    best={"reference":None,"treatment":None}
    best_key={"reference":None,"treatment":None}
    reference_phi_gradient_ever_live=False
    treatment_phi_gradient_ever_live=False

    print("HIRA_V1_S65_SHARED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        base_loss_sum=pair_loss_sum=ref_loss_sum=trt_loss_sum=0.0
        reliability_positive=0.0
        reliability_count=0
        canonical_lower=paraphrase_lower=tie_count=0
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            _c2,_p2,n2,rep_c,rep_p=train_rep[index]
            if n2!=n:
                raise RuntimeError("S65 representation/cache batch mismatch")

            corr_opt.zero_grad(set_to_none=True)
            base_loss,_ce,_js=s50._private_loss(op,"treatment",canonical,paraphrase)
            cgrads=torch.autograd.grad(base_loss,cparams,allow_unused=True)
            if not any(g is not None and float(g.detach().abs().sum())>0 for g in cgrads):
                raise RuntimeError("S65 correction gradient vanished")
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
                raise RuntimeError("S65 pairwise head gradient vanished")
            for p,g in zip(hparams,hgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(hparams,s35.GRAD_CLIP)
            head_opt.step()

            with torch.no_grad():
                _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
                _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
                pair_c=head.aggregate_logits(rep_c)
                pair_p=head.aggregate_logits(rep_p)

            ref_opt.zero_grad(set_to_none=True)
            ref_loss,ref_diag=contextual_reliability_loss(
                reference_gate,
                fused_c,fused_p,pair_c,pair_p,
                rep_c,rep_p,canonical.gold,
            )
            rgrads=torch.autograd.grad(ref_loss,rparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in rgrads):
                raise RuntimeError("S65 reference gate gradient invalid")
            if float(rgrads[2].detach().abs().sum())<=0.0 or float(rgrads[3].detach().abs().sum())<=0.0:
                raise RuntimeError("S65 reference output gradient vanished")
            if float(rgrads[0].detach().abs().sum())>0.0 or float(rgrads[1].detach().abs().sum())>0.0:
                reference_phi_gradient_ever_live=True
            for p,g in zip(rparams,rgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(rparams,s35.GRAD_CLIP)
            ref_opt.step()

            trt_opt.zero_grad(set_to_none=True)
            trt_loss,trt_diag=cross_view_soft_and_reliability_loss(
                treatment_gate,
                fused_c,fused_p,pair_c,pair_p,
                rep_c,rep_p,canonical.gold,
            )
            tgrads=torch.autograd.grad(trt_loss,tparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in tgrads):
                raise RuntimeError("S65 treatment gate gradient invalid")
            if float(tgrads[2].detach().abs().sum())<=0.0 or float(tgrads[3].detach().abs().sum())<=0.0:
                raise RuntimeError("S65 treatment output gradient vanished")
            if float(tgrads[0].detach().abs().sum())>0.0 or float(tgrads[1].detach().abs().sum())>0.0:
                treatment_phi_gradient_ever_live=True
            for p,g in zip(tparams,tgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(tparams,s35.GRAD_CLIP)
            trt_opt.step()

            if int(ref_diag["target_count"])!=int(trt_diag["target_count"]):
                raise RuntimeError("S65 matched reliability target count changed")
            if int(ref_diag["target_positive_count"])!=int(trt_diag["target_positive_count"]):
                raise RuntimeError("S65 matched reliability target labels changed")

            base_loss_sum+=float(base_loss.detach())*n
            pair_loss_sum+=float(pair_loss.detach())*n
            ref_loss_sum+=float(ref_loss.detach())*n
            trt_loss_sum+=float(trt_loss.detach())*n
            reliability_positive+=float(trt_diag["target_positive_count"])
            reliability_count+=int(trt_diag["target_count"])
            canonical_lower+=int(trt_diag["canonical_lower_count"])
            paraphrase_lower+=int(trt_diag["paraphrase_lower_count"])
            tie_count+=int(trt_diag["tie_count"])
            case_count+=n

        if not reference_phi_gradient_ever_live:
            raise RuntimeError("S65 reference encoder never received gradient")
        if not treatment_phi_gradient_ever_live:
            raise RuntimeError("S65 treatment encoder never received gradient")

        fused_shadow=s50._private_metrics(op,"treatment",dev_cache)
        reference=_hybrid_metrics(op,head,reference_gate,dev_rep)
        treatment=_hybrid_metrics(op,head,treatment_gate,dev_rep)
        train_pair_diag=s59._pairwise_train_diagnostics(head,train_rep)

        corr_state=op.correction_state_dict()
        head_state=head.state_dict_exact()
        ref_state=reference_gate.state_dict_exact()
        trt_state=treatment_gate.state_dict_exact()
        corr_digest=s59._state_digest(corr_state)
        head_digest=s59._state_digest(head_state)
        ref_digest=s59._state_digest(ref_state)
        trt_digest=s59._state_digest(trt_state)

        record={
            "epoch":epoch,
            "train_mean_base_correction_loss":base_loss_sum/case_count,
            "train_mean_pairwise_loss":pair_loss_sum/case_count,
            "train_mean_reference_independent_bce_loss":ref_loss_sum/case_count,
            "train_mean_treatment_soft_and_bce_loss":trt_loss_sum/case_count,
            "train_reliability_positive_fraction":reliability_positive/max(1,reliability_count),
            "train_soft_and_canonical_lower_count":canonical_lower,
            "train_soft_and_paraphrase_lower_count":paraphrase_lower,
            "train_soft_and_tie_count":tie_count,
            "reference_phi_gradient_ever_live":reference_phi_gradient_ever_live,
            "treatment_phi_gradient_ever_live":treatment_phi_gradient_ever_live,
            "train_pairwise":train_pair_diag,
            "fused_shadow_dev":fused_shadow,
            "reference_dev":reference,
            "treatment_dev":treatment,
            "correction_state_sha256":corr_digest,
            "pairwise_head_state_sha256":head_digest,
            "reference_gate_state_sha256":ref_digest,
            "treatment_gate_state_sha256":trt_digest,
        }
        history.append(record)

        for arm,metrics,gate_state,gate_digest in (
            ("reference",reference,ref_state,ref_digest),
            ("treatment",treatment,trt_state,trt_digest),
        ):
            key=s17._selection_key(epoch,metrics)
            if best_key[arm] is None or key>best_key[arm]:
                best_key[arm]=key
                best[arm]={
                    "epoch":epoch,
                    "correction_state":{k:v.detach().cpu().clone() for k,v in corr_state.items()},
                    "head_state":{k:v.detach().cpu().clone() for k,v in head_state.items()},
                    "gate_state":{k:v.detach().cpu().clone() for k,v in gate_state.items()},
                    "metrics":dict(metrics),
                    "fused_shadow":dict(fused_shadow),
                    "correction_state_sha256":corr_digest,
                    "pairwise_head_state_sha256":head_digest,
                    "gate_state_sha256":gate_digest,
                }

        print(
            "HIRA_V1_S65_EPOCH="+json.dumps({
                "epoch":epoch,
                "train_reliability_positive_fraction":reliability_positive/max(1,reliability_count),
                "soft_and_route":{
                    "canonical_lower":canonical_lower,
                    "paraphrase_lower":paraphrase_lower,
                    "ties":tie_count,
                },
                "reference":reference,
                "treatment":treatment,
                "correction_state_sha256":corr_digest,
                "pairwise_head_state_sha256":head_digest,
                "reference_gate_state_sha256":ref_digest,
                "treatment_gate_state_sha256":trt_digest,
            },sort_keys=True),
            flush=True,
        )

    if best["reference"] is None or best["treatment"] is None:
        raise RuntimeError("S65 selector failed")

    results={}
    for arm in ("reference","treatment"):
        selected=best[arm]
        replay_op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
        replay_op.load_correction_state_dict(selected["correction_state"],freeze=True)
        replay_head=ExplicitPairwiseDecisionHead(trainable=True)
        replay_head.load_state_dict_exact(selected["head_state"],freeze=True)
        replay_gate=ContextInjectedReliabilityGate(use_context=True,trainable=True)
        replay_gate.load_state_dict_exact(selected["gate_state"],freeze=True)

        metrics=_hybrid_metrics(replay_op,replay_head,replay_gate,dev_rep)
        shadow=s50._private_metrics(replay_op,"treatment",dev_cache)
        for key,value in selected["metrics"].items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(f"S65 {arm} selected replay changed: {key}")
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(f"S65 {arm} selected replay changed: {key}")

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        objective="independent_view_bce" if arm=="reference" else "cross_view_exact_min_soft_and_bce"
        torch.save({
            "schema_version":"hira-v1-s65-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "selected_dev_epoch":selected["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "gate_parameter_count":S64_GATE_PARAMETER_COUNT,
            "gate_family":"s64_context_injected",
            "gate_objective":objective,
            "use_context":True,
            "single_view_inference":True,
            "context_projection_seed":S64_CONTEXT_PROJECTION_SEED,
            "alpha_probe":S62_ALPHA_PROBE,
            "target_tolerance":S62_TARGET_TOLERANCE,
            "native_parameter_count_in_optimizer":0,
            "teacher_dependency":False,
            "correction_state_dict":selected["correction_state"],
            "pairwise_head_state_dict":selected["head_state"],
            "gate_state_dict":selected["gate_state"],
        },checkpoint)
        results[arm]={
            "selected_dev_epoch":selected["epoch"],
            "selected_dev":metrics,
            "selected_fused_shadow_dev":shadow,
            "gates":gates,
            "dev_ready":all(gates.values()),
            "selected_gate_mean_alpha":metrics["gate_mean_alpha"],
            "selected_gate_min_alpha":metrics["gate_min_alpha"],
            "selected_gate_max_alpha":metrics["gate_max_alpha"],
            "selected_gate_std_alpha":metrics["gate_std_alpha"],
            "correction_state_sha256":selected["correction_state_sha256"],
            "pairwise_head_state_sha256":selected["pairwise_head_state_sha256"],
            "gate_state_sha256":selected["gate_state_sha256"],
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
    if a0.get("outcome")!="HIRA_V1_S65_A0_CROSS_VIEW_SOFT_AND_RELIABILITY_READY":
        raise RuntimeError("S65 A0 not qualified")
    if int(a0.get("correction_trainable_parameter_count",-1))!=CORRECTION_PARAMS:
        raise RuntimeError("S65 A0 correction capacity changed")
    if int(a0.get("pairwise_trainable_parameter_count",-1))!=PAIRWISE_PARAMS:
        raise RuntimeError("S65 A0 pairwise capacity changed")
    if int(a0.get("reference_gate_parameter_count",-1))!=REFERENCE_GATE_PARAMS:
        raise RuntimeError("S65 A0 reference gate capacity changed")
    if int(a0.get("treatment_gate_parameter_count",-1))!=TREATMENT_GATE_PARAMS:
        raise RuntimeError("S65 A0 treatment gate capacity changed")
    if int(a0.get("added_treatment_parameter_count",-1))!=0:
        raise RuntimeError("S65 A0 parameter advantage changed")
    if a0.get("parameter_initialization_bit_identical") is not True:
        raise RuntimeError("S65 A0 matched initialization changed")
    if a0.get("context_projection_bit_identical") is not True:
        raise RuntimeError("S65 A0 context path changed")
    if a0.get("reference_objective")!="independent_view_bce":
        raise RuntimeError("S65 A0 reference objective changed")
    if a0.get("treatment_objective")!="cross_view_exact_min_soft_and_bce":
        raise RuntimeError("S65 A0 treatment objective changed")
    if a0.get("same_single_view_inference_path") is not True:
        raise RuntimeError("S65 A0 inference path changed")
    if a0.get("same_context_path") is not True or a0.get("same_reliability_target") is not True:
        raise RuntimeError("S65 A0 controlled variable changed")
    if float(a0.get("soft_and_exact_min_max_abs_error",1.0))>1e-7:
        raise RuntimeError("S65 A0 soft-AND exact-min contract changed")
    if float(a0.get("soft_and_pair_symmetry_max_abs_error",1.0))!=0.0:
        raise RuntimeError("S65 A0 pair symmetry changed")
    if float(a0.get("lower_view_gradient_route_max_abs_error",1.0))!=0.0:
        raise RuntimeError("S65 A0 gradient routing changed")
    if int(a0.get("treatment_extra_trainable_parameter_count",-1))!=0:
        raise RuntimeError("S65 A0 objective gained parameters")
    if a0.get("reference_gradient_to_upstream_zero") is not True or a0.get("treatment_gradient_to_upstream_zero") is not True:
        raise RuntimeError("S65 A0 gradient ownership changed")
    if float(a0.get("alpha_probe",-1.0))!=S62_ALPHA_PROBE or float(a0.get("target_tolerance",-1.0))!=S62_TARGET_TOLERANCE:
        raise RuntimeError("S65 A0 target authority changed")
    if a0.get("inherited_target_values")!=[1,0]:
        raise RuntimeError("S65 A0 target semantics changed")
    if a0.get("teacher_dependency") is not False or a0.get("dev_target_dependency") is not False or a0.get("self_anchor_dependency") is not False:
        raise RuntimeError("S65 A0 dependency contract changed")
    if a0.get("pairwise_only_final_path") is not False:
        raise RuntimeError("S65 A0 pairwise-only path changed")
    if a0.get("used_for_model_selection") is not False or a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S65 A0 exposed scientific DEV")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S65 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime or authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S65 parent native authority digest changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S65 parent native authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s65_cases("train")
    dev_rows=generate_s65_cases("dev")
    validate_s65_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S65 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S65 loaded parent runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S65 native runtime remained trainable")

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
        raise RuntimeError("S65 context representation gained gradient")

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
        "scientific_authority":"V1_S65_FRESH_CROSS_VIEW_SOFT_AND_RELIABILITY",
        "seed":SEED,
        "parent_s64":{
            "merged_main":"c5063da0365d49dc4f9dc2c64f94bc309caef0ba",
            "scientific_run":37319541951,
            "artifact_id":11348594503,
            "artifact_digest":"sha256:f3d04d452a51b7358a23a9391579bca6ce1ab2c16fa107dc9f7f79740dba90ea",
            "verdict":"CASE_B",
        },
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
        "context_representation_cache":{
            "train_digest":train_rep_digest,
            "dev_digest":dev_rep_digest,
            "precomputed_once_before_training":True,
            "requires_grad":False,
            "source":"s59_detached_identity_plus_joint_state_query_option_context",
        },
        "controlled_variable":{
            "reference_gate_family":"s64_context_injected",
            "treatment_gate_family":"s64_context_injected",
            "reference_gate_objective":"independent_view_bce",
            "treatment_gate_objective":"cross_view_exact_min_soft_and_bce",
            "cross_view_interaction_train_only":True,
            "single_view_inference_both_arms":True,
            "correction_trainable_parameters":CORRECTION_PARAMS,
            "pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "reference_gate_trainable_parameters":REFERENCE_GATE_PARAMS,
            "treatment_gate_trainable_parameters":TREATMENT_GATE_PARAMS,
            "added_treatment_parameters":0,
            "parameter_initialization_bit_identical":True,
            "same_context_path":True,
            "context_projection_seed":S64_CONTEXT_PROJECTION_SEED,
            "context_projection_trainable_parameters":0,
            "encoder_init_seed":S64_ENCODER_INIT_SEED,
            "alpha_initial":S61_ALPHA_INITIAL,
            "alpha_max":S61_ALPHA_MAX,
            "alpha_probe":S62_ALPHA_PROBE,
            "target_tolerance":S62_TARGET_TOLERANCE,
            "reference_gate_weight_decay":0.0,
            "treatment_gate_weight_decay":0.0,
            "teacher_dependency":False,
            "dev_target_dependency":False,
            "self_anchor_dependency":False,
            "gate_gradient_enters_correction":False,
            "gate_gradient_enters_pairwise":False,
            "gate_gradient_enters_native":False,
            "pairwise_only_final_path":False,
            "shared_correction_head_trajectory":True,
            "same_reliability_target":True,
            "same_frozen_selector":True,
            "fused_shadow_selection_arm":False,
        },
        "training_trajectory":history,
        "reference_branch":reference,
        "treatment_branch":treatment,
        "selected_dev_delta_treatment_minus_reference":deltas,
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "exact_s64_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "soft_min_temperature_performed":False,
        "min_mean_max_sweep_performed":False,
        "objective_mixing_performed":False,
        "bce_weighting_performed":False,
        "target_change_performed":False,
        "context_architecture_change_performed":False,
        "optimizer_lr_weight_decay_changed":False,
        "gate_regularizer_added":False,
        "gradient_coupling_performed":False,
        "native_retraining_performed":False,
        "selector_changed":False,
        "second_dev_run_performed":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
    }
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(
        "HIRA_V1_S65_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
