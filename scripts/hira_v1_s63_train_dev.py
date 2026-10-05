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
from nmd.v1_confidence_adaptive_bounded_hybrid import (
    S61_ALPHA_INITIAL,
    S61_ALPHA_MAX,
    ConfidenceAdaptiveBoundedHybridGate,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    reliability_gate_loss,
)
from nmd.v1_learned_set_reliability_gate import (
    S63_OPTION_INPUT_DIM,
    S63_OPTION_HIDDEN_DIM,
    S63_POOLED_DIM,
    S63_REPRESENTATION_DIM,
    S63_GATE_PARAMETER_COUNT,
    S63_INIT_SEED,
    LearnedSetReliabilityGate,
    learned_set_reliability_loss,
)
from nmd.v1_s63_authority import generate_s63_cases, validate_s63_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s63-learned-set-reliability-gate-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S63_LEARNED_SET_RELIABILITY_GATE_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S63_LEARNED_SET_RELIABILITY_GATE_DEV_READY"

SEED=84_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=32_832
REFERENCE_GATE_PARAMS=5
TREATMENT_GATE_PARAMS=61

A0_RUN=37312740736
A0_ARTIFACT_ID=11346084572
A0_ARTIFACT_DIGEST="sha256:39c9c84a93e5da0075b13018b7a5a4151dda4563476834478524e042b1cfac9f"

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
            hc,dc=gate.compose(fused_c,pair_c,return_diagnostics=True)
            hp,dp=gate.compose(fused_p,pair_p,return_diagnostics=True)
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S63 paired gold mismatch")

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
        raise RuntimeError("S63 empty DEV metrics")

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
    reference_gate=ConfidenceAdaptiveBoundedHybridGate(trainable=True)
    treatment_gate=LearnedSetReliabilityGate(trainable=True)

    cparams=op.correction_parameters()
    hparams=[head.A,head.u]
    rparams=[reference_gate.w,reference_gate.b]
    tnamed=dict(treatment_gate.named_parameters())
    tparams=[tnamed[k] for k in ("W_phi","b_phi","w_out","b_out")]

    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S63 correction capacity changed")
    if sum(p.numel() for p in hparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S63 pairwise capacity changed")
    if sum(p.numel() for p in rparams)!=REFERENCE_GATE_PARAMS:
        raise RuntimeError("S63 reference gate capacity changed")
    if sum(p.numel() for p in tparams)!=TREATMENT_GATE_PARAMS:
        raise RuntimeError("S63 treatment gate capacity changed")

    corr_opt=torch.optim.AdamW(cparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    head_opt=torch.optim.AdamW(hparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    ref_opt=torch.optim.AdamW(rparams,lr=s35.LR,weight_decay=0.0)
    trt_opt=torch.optim.AdamW(tparams,lr=s35.LR,weight_decay=0.0)

    history=[]
    best={"reference":None,"treatment":None}
    best_key={"reference":None,"treatment":None}
    treatment_phi_gradient_ever_live=False

    print("HIRA_V1_S63_SHARED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        base_loss_sum=pair_loss_sum=ref_loss_sum=trt_loss_sum=0.0
        reliability_positive=0.0
        reliability_count=0
        case_count=0

        for step_index,index in enumerate(order):
            canonical,paraphrase,n=train_cache[index]
            _c2,_p2,n2,rep_c,rep_p=train_rep[index]
            if n2!=n:
                raise RuntimeError("S63 representation/cache batch mismatch")

            corr_opt.zero_grad(set_to_none=True)
            base_loss,_ce,_js=s50._private_loss(op,"treatment",canonical,paraphrase)
            cgrads=torch.autograd.grad(base_loss,cparams,allow_unused=True)
            if not any(g is not None and float(g.detach().abs().sum())>0 for g in cgrads):
                raise RuntimeError("S63 correction gradient vanished")
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
                raise RuntimeError("S63 pairwise head gradient vanished")
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
            ref_loss,ref_diag=reliability_gate_loss(
                reference_gate,fused_c,fused_p,pair_c,pair_p,canonical.gold
            )
            rgrads=torch.autograd.grad(ref_loss,rparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in rgrads):
                raise RuntimeError("S63 reference gate gradient invalid")
            if not any(float(g.detach().abs().sum())>0 for g in rgrads):
                raise RuntimeError("S63 reference gate gradient vanished")
            for p,g in zip(rparams,rgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(rparams,s35.GRAD_CLIP)
            ref_opt.step()

            trt_opt.zero_grad(set_to_none=True)
            trt_loss,trt_diag=learned_set_reliability_loss(
                treatment_gate,fused_c,fused_p,pair_c,pair_p,canonical.gold
            )
            tgrads=torch.autograd.grad(trt_loss,tparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in tgrads):
                raise RuntimeError("S63 treatment gate gradient invalid")
            if float(tgrads[2].detach().abs().sum())<=0.0 or float(tgrads[3].detach().abs().sum())<=0.0:
                raise RuntimeError("S63 treatment output gradient vanished")
            if float(tgrads[0].detach().abs().sum())>0.0 or float(tgrads[1].detach().abs().sum())>0.0:
                treatment_phi_gradient_ever_live=True
            for p,g in zip(tparams,tgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(tparams,s35.GRAD_CLIP)
            trt_opt.step()

            if int(ref_diag["target_count"])!=int(trt_diag["target_count"]):
                raise RuntimeError("S63 matched reliability target count changed")
            if int(ref_diag["target_positive_count"])!=int(trt_diag["target_positive_count"]):
                raise RuntimeError("S63 matched reliability target labels changed")

            base_loss_sum+=float(base_loss.detach())*n
            pair_loss_sum+=float(pair_loss.detach())*n
            ref_loss_sum+=float(ref_loss.detach())*n
            trt_loss_sum+=float(trt_loss.detach())*n
            reliability_positive+=float(trt_diag["target_positive_count"])
            reliability_count+=int(trt_diag["target_count"])
            case_count+=n

        if not treatment_phi_gradient_ever_live:
            raise RuntimeError("S63 learned set encoder never received gradient")

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
            "train_mean_reference_reliability_bce_loss":ref_loss_sum/case_count,
            "train_mean_treatment_reliability_bce_loss":trt_loss_sum/case_count,
            "train_reliability_positive_fraction":reliability_positive/max(1,reliability_count),
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
            "HIRA_V1_S63_EPOCH="+json.dumps({
                "epoch":epoch,
                "train_reliability_positive_fraction":reliability_positive/max(1,reliability_count),
                "reference_alpha":{
                    "mean":reference["gate_mean_alpha"],
                    "min":reference["gate_min_alpha"],
                    "max":reference["gate_max_alpha"],
                    "std":reference["gate_std_alpha"],
                },
                "treatment_alpha":{
                    "mean":treatment["gate_mean_alpha"],
                    "min":treatment["gate_min_alpha"],
                    "max":treatment["gate_max_alpha"],
                    "std":treatment["gate_std_alpha"],
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
        raise RuntimeError("S63 selector failed")

    results={}
    for arm in ("reference","treatment"):
        selected=best[arm]
        replay_op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
        replay_op.load_correction_state_dict(selected["correction_state"],freeze=True)
        replay_head=ExplicitPairwiseDecisionHead(trainable=True)
        replay_head.load_state_dict_exact(selected["head_state"],freeze=True)
        if arm=="reference":
            replay_gate=ConfidenceAdaptiveBoundedHybridGate(trainable=True)
        else:
            replay_gate=LearnedSetReliabilityGate(trainable=True)
        replay_gate.load_state_dict_exact(selected["gate_state"],freeze=True)

        metrics=_hybrid_metrics(replay_op,replay_head,replay_gate,dev_rep)
        shadow=s50._private_metrics(replay_op,"treatment",dev_cache)
        for key,value in selected["metrics"].items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(f"S63 {arm} selected replay changed: {key}")
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(f"S63 {arm} selected replay changed: {key}")

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        torch.save({
            "schema_version":"hira-v1-s63-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "selected_dev_epoch":selected["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "gate_parameter_count":REFERENCE_GATE_PARAMS if arm=="reference" else TREATMENT_GATE_PARAMS,
            "gate_family":"s62_scalar_reliability" if arm=="reference" else "s63_learned_set_reliability",
            "gate_objective":"train_only_reliability_bce",
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
    if a0.get("outcome")!="HIRA_V1_S63_A0_LEARNED_SET_RELIABILITY_GATE_READY":
        raise RuntimeError("S63 A0 not qualified")
    if int(a0.get("correction_trainable_parameter_count",-1))!=CORRECTION_PARAMS:
        raise RuntimeError("S63 A0 correction capacity changed")
    if int(a0.get("pairwise_trainable_parameter_count",-1))!=PAIRWISE_PARAMS:
        raise RuntimeError("S63 A0 pairwise capacity changed")
    if int(a0.get("reference_gate_parameter_count",-1))!=REFERENCE_GATE_PARAMS:
        raise RuntimeError("S63 A0 reference gate capacity changed")
    if int(a0.get("treatment_gate_parameter_count",-1))!=TREATMENT_GATE_PARAMS:
        raise RuntimeError("S63 A0 treatment gate capacity changed")
    if int(a0.get("added_treatment_parameter_count",-1))!=56:
        raise RuntimeError("S63 A0 treatment capacity delta changed")
    if a0.get("treatment_trainable_tensor_names")!=["W_phi","b_out","b_phi","w_out"]:
        raise RuntimeError("S63 A0 treatment tensor surface changed")
    if a0.get("treatment_parameter_shapes")!={"W_phi":[8,4],"b_out":[],"b_phi":[8],"w_out":[20]}:
        raise RuntimeError("S63 A0 treatment tensor shapes changed")
    if int(a0.get("option_input_dimension",-1))!=S63_OPTION_INPUT_DIM:
        raise RuntimeError("S63 A0 option input dimension changed")
    if int(a0.get("option_hidden_dimension",-1))!=S63_OPTION_HIDDEN_DIM:
        raise RuntimeError("S63 A0 option hidden dimension changed")
    if int(a0.get("pooled_dimension",-1))!=S63_POOLED_DIM:
        raise RuntimeError("S63 A0 pooled dimension changed")
    if int(a0.get("representation_dimension",-1))!=S63_REPRESENTATION_DIM:
        raise RuntimeError("S63 A0 representation dimension changed")
    if int(a0.get("representation_init_seed",-1))!=S63_INIT_SEED:
        raise RuntimeError("S63 A0 init seed changed")
    if a0.get("initial_w_out_exact_zero") is not True:
        raise RuntimeError("S63 A0 output initialization changed")
    if float(a0.get("initial_alpha_max_abs_error",1.0))>1e-7:
        raise RuntimeError("S63 A0 initial alpha changed")
    if float(a0.get("option_permutation_alpha_max_abs_error",1.0))>1e-6:
        raise RuntimeError("S63 A0 permutation invariance changed")
    if float(a0.get("surface_affine_alpha_max_abs_error",1.0))>2e-5:
        raise RuntimeError("S63 A0 affine invariance changed")
    if float(a0.get("initial_w_out_gradient_l1",0.0))<=0.0 or float(a0.get("initial_b_out_gradient_l1",0.0))<=0.0:
        raise RuntimeError("S63 A0 output gradient vanished")
    if float(a0.get("post_output_warm_W_phi_gradient_l1",0.0))<=0.0 or float(a0.get("post_output_warm_b_phi_gradient_l1",0.0))<=0.0:
        raise RuntimeError("S63 A0 set encoder gradient path vanished")
    if a0.get("production_initial_state_unchanged") is not True:
        raise RuntimeError("S63 A0 production init changed")
    if a0.get("reference_gradient_to_upstream_zero") is not True or a0.get("treatment_gradient_to_upstream_zero") is not True:
        raise RuntimeError("S63 A0 gradient ownership changed")
    if float(a0.get("alpha_probe",-1.0))!=S62_ALPHA_PROBE or float(a0.get("target_tolerance",-1.0))!=S62_TARGET_TOLERANCE:
        raise RuntimeError("S63 A0 inherited reliability target changed")
    if a0.get("inherited_target_values")!=[1,0]:
        raise RuntimeError("S63 A0 inherited target semantics changed")
    if a0.get("teacher_dependency") is not False or a0.get("dev_target_dependency") is not False or a0.get("self_anchor_dependency") is not False:
        raise RuntimeError("S63 A0 dependency contract changed")
    if a0.get("pairwise_only_final_path") is not False:
        raise RuntimeError("S63 A0 pairwise-only path changed")
    if a0.get("used_for_model_selection") is not False or a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S63 A0 exposed scientific DEV")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S63 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime or authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S63 parent native authority digest changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S63 parent native authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s63_cases("train")
    dev_rows=generate_s63_cases("dev")
    validate_s63_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S63 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S63 loaded parent runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S63 native runtime remained trainable")

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
        raise RuntimeError("S63 materialized representation gained gradient")

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
        "scientific_authority":"V1_S63_FRESH_LEARNED_SET_RELIABILITY_GATE",
        "seed":SEED,
        "parent_s62":{
            "merged_main":"798f5c4d933635b36f4e0eaf76503f8213afedd9",
            "scientific_run":37307658289,
            "artifact_id":11344855774,
            "artifact_digest":"sha256:7812c348d707ea8675fc9370af573a87d9e5b1a0b10e55a1f335bf890ddb3305",
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
        "pairwise_representation_cache":{
            "train_digest":train_rep_digest,
            "dev_digest":dev_rep_digest,
            "precomputed_once_before_training":True,
            "requires_grad":False,
        },
        "controlled_variable":{
            "reference_gate_family":"s62_four_scalar_reliability_gate",
            "treatment_gate_family":"s63_learned_permutation_invariant_set_reliability_gate",
            "reference_gate_objective":"train_only_reliability_bce",
            "treatment_gate_objective":"train_only_reliability_bce",
            "correction_trainable_parameters":CORRECTION_PARAMS,
            "pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "reference_gate_trainable_parameters":REFERENCE_GATE_PARAMS,
            "treatment_gate_trainable_parameters":TREATMENT_GATE_PARAMS,
            "added_treatment_parameters":56,
            "option_input_dimension":S63_OPTION_INPUT_DIM,
            "option_hidden_dimension":S63_OPTION_HIDDEN_DIM,
            "pooled_dimension":S63_POOLED_DIM,
            "representation_dimension":S63_REPRESENTATION_DIM,
            "representation_init_seed":S63_INIT_SEED,
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
            "exact_s62_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "option_feature_change_performed":False,
        "hidden_width_sweep_performed":False,
        "pooling_change_performed":False,
        "initialization_sweep_performed":False,
        "alpha_probe_target_change_performed":False,
        "bce_weighting_performed":False,
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
        "HIRA_V1_S63_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
