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
from nmd.v1_multistat_pairwise_row_composer import (
    S71_PARAMETER_COUNT,
    MultiStatPairwiseRowComposer,
    multistat_responsibility_loss,
)
from nmd.v1_opponent_profile_vector_residual import compose_with_residual_direction
from nmd.v1_counterfactual_safety_veto import (
    S73_ACCEPT_THRESHOLD,
    S73_PARAMETER_COUNT,
    CounterfactualSafetyVeto,
    counterfactual_safety_loss,
    counterfactual_safety_targets,
    safety_features,
)
from nmd.v1_s73_authority import generate_s73_cases, validate_s73_partitions

import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59
import hira_v1_s72_train_dev as s72


SCHEMA_VERSION="hira-v1-s73-counterfactual-safety-veto-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S73_COUNTERFACTUAL_SAFETY_VETO_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S73_COUNTERFACTUAL_SAFETY_VETO_DEV_READY"

SEED=94_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=S59_PAIRWISE_PARAMETER_COUNT
COMPOSER_PARAMS=S71_PARAMETER_COUNT
PREDICTOR_PARAMS=S73_PARAMETER_COUNT

A0_RUN=37476327384
A0_ARTIFACT_ID=11418978543
A0_ARTIFACT_DIGEST="sha256:bab338c55ae37bd322fcc20d838ed3b4250da959e2d7135525ec611da8f9c083"


def _veto_metrics(op,head,composer,predictor,pair_rep_cache,context_rep_cache,*,use_veto:bool):
    c_correct=p_correct=0
    pair_both=pair_changed=0
    agree_sum=js_sum=0.0
    c_margin_sum=p_margin_sum=0.0
    canonical_loss_sum=0.0
    gold_pair_correct=gold_pair_count=0
    gold_pair_margin_sum=0.0
    max_antisym=max_diag=max_mass=0.0
    residual_max=bound_max=0.0
    alpha_sum=alpha_sq_sum=0.0
    alpha_min=float("inf")
    alpha_max=0.0
    alpha_count=0
    accept_sum=prob_sum=prob_sq_sum=0.0
    accept_count=0
    cases=queries=0

    if len(pair_rep_cache)!=len(context_rep_cache):
        raise RuntimeError("S73 pair/context cache length mismatch")
    plain=[(c,p,n) for c,p,n,_rc,_rp in context_rep_cache]
    base=s50._private_metrics(op,"treatment",plain)

    with torch.no_grad():
        for pair_entry,context_entry in zip(pair_rep_cache,context_rep_cache):
            canonical,paraphrase,n,pair_rc,pair_rp=pair_entry
            cc,cp,cn,context_rc,context_rp=context_entry
            if n!=cn or canonical is not cc or paraphrase is not cp:
                raise RuntimeError("S73 matched cache identity changed")

            _a,_b,fused_c=s50._branch_logits(op,"treatment",canonical)
            _a,_b,fused_p=s50._branch_logits(op,"treatment",paraphrase)
            matrix_c=head.pairwise_logits(pair_rc)
            matrix_p=head.pairwise_logits(pair_rp)

            cand_c,diag_c=compose_with_residual_direction(
                composer,fused_c,matrix_c,context_rc,
                use_opponent_profile_direction=True,
                return_diagnostics=True,
            )
            cand_p,diag_p=compose_with_residual_direction(
                composer,fused_p,matrix_p,context_rp,
                use_opponent_profile_direction=True,
                return_diagnostics=True,
            )

            feat_c=safety_features(
                fused_c,cand_c,matrix_c,diag_c["direction"],diag_c["alpha"]
            )
            feat_p=safety_features(
                fused_p,cand_p,matrix_p,diag_p["direction"],diag_p["alpha"]
            )

            if use_veto:
                hc,vd_c=predictor.apply(
                    fused_c,cand_c,feat_c,return_diagnostics=True
                )
                hp,vd_p=predictor.apply(
                    fused_p,cand_p,feat_p,return_diagnostics=True
                )
            else:
                hc=cand_c
                hp=cand_p
                pc=predictor.probability(feat_c)
                pp=predictor.probability(feat_p)
                vd_c={
                    "accept":torch.ones_like(pc,dtype=torch.bool),
                    "accept_probability":pc,
                }
                vd_p={
                    "accept":torch.ones_like(pp,dtype=torch.bool),
                    "accept_probability":pp,
                }

            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S73 paired gold mismatch")

            pred_c=hc.argmax(-1)
            pred_p=hp.argmax(-1)
            c_correct+=int((pred_c==gold).sum())
            p_correct+=int((pred_p==gold).sum())

            pairs_pred=pred_c.reshape(n,2)
            pairs_gold=gold.reshape(n,2)
            pair_both+=int(((pairs_pred==pairs_gold).all(-1)).sum())
            pair_changed+=int((pairs_pred[:,0]!=pairs_pred[:,1]).sum())

            q=2*n
            agree_sum+=float(selected_choice_agreement(hc,hp))*q
            js_sum+=float(symmetric_js_divergence(hc,hp))*q
            c_margin_sum+=float(fused_gold_vs_max_wrong_margin(hc,gold).sum())
            p_margin_sum+=float(fused_gold_vs_max_wrong_margin(hp,gold).sum())
            canonical_loss_sum+=float(F.cross_entropy(hc,gold,reduction="sum"))

            for matrix,diag,hybrid,vdiag in (
                (matrix_c,diag_c,hc,vd_c),
                (matrix_p,diag_p,hp,vd_p),
            ):
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
                    float((torch.softmax(hybrid,-1).sum(-1)-1).abs().max()),
                )
                residual_max=max(residual_max,float(diag["residual_max_abs"].max()))
                bound_max=max(bound_max,float(diag["residual_bound"].max()))

                a=diag["alpha"].reshape(-1)
                alpha_sum+=float(a.sum())
                alpha_sq_sum+=float((a*a).sum())
                alpha_min=min(alpha_min,float(a.min()))
                alpha_max=max(alpha_max,float(a.max()))
                alpha_count+=int(a.numel())

                ac=vdiag["accept"].reshape(-1)
                pr=vdiag["accept_probability"].reshape(-1)
                accept_sum+=float(ac.to(pr.dtype).sum())
                prob_sum+=float(pr.sum())
                prob_sq_sum+=float((pr*pr).sum())
                accept_count+=int(pr.numel())

            cases+=n
            queries+=q

    if queries<1 or gold_pair_count<1 or alpha_count<1 or accept_count<1:
        raise RuntimeError("S73 empty DEV metrics")

    alpha_mean=alpha_sum/alpha_count
    alpha_var=max(0.0,alpha_sq_sum/alpha_count-alpha_mean*alpha_mean)
    prob_mean=prob_sum/accept_count
    prob_var=max(0.0,prob_sq_sum/accept_count-prob_mean*prob_mean)
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
        "composer_mean_alpha":alpha_mean,
        "composer_min_alpha":alpha_min,
        "composer_max_alpha":alpha_max,
        "composer_std_alpha":alpha_var**0.5,
        "composer_residual_max_abs":residual_max,
        "composer_bound_max":bound_max,
        "veto_accept_rate":accept_sum/accept_count if use_veto else 1.0,
        "veto_mean_accept_probability":prob_mean,
        "veto_std_accept_probability":prob_var**0.5,
        "veto_threshold":S73_ACCEPT_THRESHOLD,
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


def _train_matched(train_cache,dev_cache,train_pair_rep,train_context_rep,dev_pair_rep,dev_context_rep,out_dir):
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    composer=MultiStatPairwiseRowComposer(use_multistat=False,trainable=True)
    predictor=CounterfactualSafetyVeto(trainable=True)

    cparams=op.correction_parameters()
    hparams=[head.A,head.u]
    cnamed=dict(composer.named_parameters())
    cparams_comp=[cnamed[k] for k in ("W_phi","b_phi","w_out","b_out")]
    pparams=[predictor.weight,predictor.bias]

    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S73 correction capacity changed")
    if sum(p.numel() for p in hparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S73 pairwise capacity changed")
    if sum(p.numel() for p in cparams_comp)!=COMPOSER_PARAMS:
        raise RuntimeError("S73 composer capacity changed")
    if sum(p.numel() for p in pparams)!=PREDICTOR_PARAMS:
        raise RuntimeError("S73 predictor capacity changed")

    corr_opt=torch.optim.AdamW(cparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    head_opt=torch.optim.AdamW(hparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    comp_opt=torch.optim.AdamW(cparams_comp,lr=s35.LR,weight_decay=0.0)
    pred_opt=torch.optim.AdamW(pparams,lr=s35.LR,weight_decay=0.0)

    history=[]
    best=None
    best_key=None
    comp_phi_live=False
    pred_weight_live=False

    print("HIRA_V1_S73_MATCHED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        base_sum=pair_sum=comp_sum=pred_sum=0.0
        safe_c=safe_p=pred_c=pred_p=0.0
        case_count=query_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            pc,pp,pn,pair_c_rep,pair_p_rep=train_pair_rep[index]
            cc,cp,cn,context_c_rep,context_p_rep=train_context_rep[index]
            if (
                n!=pn or n!=cn
                or canonical is not pc or canonical is not cc
                or paraphrase is not pp or paraphrase is not cp
            ):
                raise RuntimeError("S73 representation/cache mismatch")

            corr_opt.zero_grad(set_to_none=True)
            base_loss,_ce,_js=s50._private_loss(op,"treatment",canonical,paraphrase)
            cgrads=torch.autograd.grad(base_loss,cparams,allow_unused=True)
            if not any(g is not None and float(g.detach().abs().sum())>0 for g in cgrads):
                raise RuntimeError("S73 correction gradient vanished")
            for p,g in zip(cparams,cgrads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(cparams,s35.GRAD_CLIP)
            corr_opt.step()

            head_opt.zero_grad(set_to_none=True)
            lc,_=pairwise_head_loss(head,pair_c_rep,canonical.gold)
            lp,_=pairwise_head_loss(head,pair_p_rep,paraphrase.gold)
            pair_loss=0.5*(lc+lp)
            hgrads=torch.autograd.grad(pair_loss,hparams,allow_unused=True)
            if not all(g is not None and float(g.detach().abs().sum())>0 for g in hgrads):
                raise RuntimeError("S73 pairwise gradient vanished")
            for p,g in zip(hparams,hgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(hparams,s35.GRAD_CLIP)
            head_opt.step()

            with torch.no_grad():
                _a,_b,fused_c=s50._branch_logits(op,"treatment",canonical)
                _a,_b,fused_p=s50._branch_logits(op,"treatment",paraphrase)
                matrix_c=head.pairwise_logits(pair_c_rep)
                matrix_p=head.pairwise_logits(pair_p_rep)

            comp_opt.zero_grad(set_to_none=True)
            comp_loss,comp_diag=multistat_responsibility_loss(
                composer,fused_c,fused_p,matrix_c,matrix_p,
                context_c_rep,context_p_rep,canonical.gold,
            )
            comp_grads=torch.autograd.grad(comp_loss,cparams_comp,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in comp_grads):
                raise RuntimeError("S73 composer gradient invalid")
            if float(comp_grads[2].abs().sum())<=0 or float(comp_grads[3].abs().sum())<=0:
                raise RuntimeError("S73 composer output gradient vanished")
            if float(comp_grads[0].abs().sum())>0 or float(comp_grads[1].abs().sum())>0:
                comp_phi_live=True
            for p,g in zip(cparams_comp,comp_grads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(cparams_comp,s35.GRAD_CLIP)
            comp_opt.step()

            with torch.no_grad():
                cand_c,diag_c=compose_with_residual_direction(
                    composer,fused_c,matrix_c,context_c_rep,
                    use_opponent_profile_direction=True,
                    return_diagnostics=True,
                )
                cand_p,diag_p=compose_with_residual_direction(
                    composer,fused_p,matrix_p,context_p_rep,
                    use_opponent_profile_direction=True,
                    return_diagnostics=True,
                )
                yc,yp,target_diag=counterfactual_safety_targets(
                    fused_c,fused_p,cand_c,cand_p,canonical.gold
                )
                feat_c=safety_features(
                    fused_c,cand_c,matrix_c,diag_c["direction"],diag_c["alpha"]
                )
                feat_p=safety_features(
                    fused_p,cand_p,matrix_p,diag_p["direction"],diag_p["alpha"]
                )

            pred_opt.zero_grad(set_to_none=True)
            pred_loss,pred_diag=counterfactual_safety_loss(
                predictor,feat_c,feat_p,yc,yp
            )
            pgrads=torch.autograd.grad(pred_loss,pparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in pgrads):
                raise RuntimeError("S73 predictor gradient invalid")
            if float(pgrads[0].abs().sum())>0:
                pred_weight_live=True
            for p,g in zip(pparams,pgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(pparams,s35.GRAD_CLIP)
            pred_opt.step()

            q=int(canonical.gold.numel())
            base_sum+=float(base_loss.detach())*n
            pair_sum+=float(pair_loss.detach())*n
            comp_sum+=float(comp_loss.detach())*n
            pred_sum+=float(pred_loss.detach())*n
            safe_c+=float(target_diag["canonical_safe_count"])
            safe_p+=float(target_diag["paraphrase_safe_count"])
            pred_c+=float(pred_diag["canonical_predicted_accept_count"])
            pred_p+=float(pred_diag["paraphrase_predicted_accept_count"])
            query_count+=q
            case_count+=n

        if not comp_phi_live:
            raise RuntimeError("S73 composer phi gradient never activated")
        if not pred_weight_live:
            raise RuntimeError("S73 predictor weight gradient never activated")

        fused_shadow=s50._private_metrics(op,"treatment",dev_cache)
        shared_pair_train=s59._pairwise_train_diagnostics(head,train_pair_rep)
        shared_aggregate=s72._aggregate_metrics(op,head,dev_pair_rep)
        ref_metrics=_veto_metrics(
            op,head,composer,predictor,dev_pair_rep,dev_context_rep,use_veto=False
        )
        trt_metrics=_veto_metrics(
            op,head,composer,predictor,dev_pair_rep,dev_context_rep,use_veto=True
        )

        for key in ("pairwise_gold_pair_accuracy","pairwise_mean_gold_pair_margin"):
            if abs(float(ref_metrics[key])-float(trt_metrics[key]))>1e-12:
                raise RuntimeError(f"S73 pairwise evidence diverged: {key}")
        for key in (
            "composer_mean_alpha","composer_min_alpha",
            "composer_max_alpha","composer_std_alpha",
        ):
            if abs(float(ref_metrics[key])-float(trt_metrics[key]))>1e-12:
                raise RuntimeError(f"S73 alpha policy diverged: {key}")

        corr_state=op.correction_state_dict()
        head_state=head.state_dict_exact()
        comp_state=composer.state_dict_exact()
        pred_state=predictor.state_dict_exact()
        corr_digest=s59._state_digest(corr_state)
        head_digest=s59._state_digest(head_state)
        comp_digest=s59._state_digest(comp_state)
        pred_digest=s59._state_digest(pred_state)

        record={
            "epoch":epoch,
            "train_mean_base_correction_loss":base_sum/case_count,
            "train_mean_shared_pairwise_loss":pair_sum/case_count,
            "train_mean_shared_composer_loss":comp_sum/case_count,
            "train_mean_shared_veto_loss":pred_sum/case_count,
            "train_canonical_safe_fraction":safe_c/max(1,query_count),
            "train_paraphrase_safe_fraction":safe_p/max(1,query_count),
            "train_canonical_predicted_accept_fraction":pred_c/max(1,query_count),
            "train_paraphrase_predicted_accept_fraction":pred_p/max(1,query_count),
            "shared_train_pairwise":shared_pair_train,
            "shared_pairwise_aggregate_dev":shared_aggregate,
            "fused_shadow_dev":fused_shadow,
            "reference_dev":ref_metrics,
            "treatment_dev":trt_metrics,
            "correction_state_sha256":corr_digest,
            "shared_pairwise_head_state_sha256":head_digest,
            "shared_composer_state_sha256":comp_digest,
            "shared_veto_predictor_state_sha256":pred_digest,
        }
        history.append(record)

        key=s17._selection_key(epoch,fused_shadow)
        if best_key is None or key>best_key:
            best_key=key
            best={
                "epoch":epoch,
                "correction_state":{k:v.detach().cpu().clone() for k,v in corr_state.items()},
                "head_state":{k:v.detach().cpu().clone() for k,v in head_state.items()},
                "composer_state":{k:v.detach().cpu().clone() for k,v in comp_state.items()},
                "predictor_state":{k:v.detach().cpu().clone() for k,v in pred_state.items()},
                "reference_metrics":dict(ref_metrics),
                "treatment_metrics":dict(trt_metrics),
                "shared_aggregate":dict(shared_aggregate),
                "fused_shadow":dict(fused_shadow),
                "correction_state_sha256":corr_digest,
                "head_state_sha256":head_digest,
                "composer_state_sha256":comp_digest,
                "predictor_state_sha256":pred_digest,
            }

        print("HIRA_V1_S73_EPOCH="+json.dumps({
            "epoch":epoch,
            "shared_pairwise_gold_pair_accuracy":shared_pair_train["gold_pair_accuracy"],
            "reference":ref_metrics,
            "treatment":trt_metrics,
        },sort_keys=True),flush=True)

    if best is None:
        raise RuntimeError("S73 shared selector failed")

    replay_op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    replay_op.load_correction_state_dict(best["correction_state"],freeze=True)
    replay_head=ExplicitPairwiseDecisionHead(trainable=True)
    replay_head.load_state_dict_exact(best["head_state"],freeze=True)
    replay_comp=MultiStatPairwiseRowComposer(use_multistat=False,trainable=True)
    replay_comp.load_state_dict_exact(best["composer_state"],freeze=True)
    replay_pred=CounterfactualSafetyVeto(trainable=True)
    replay_pred.load_state_dict_exact(best["predictor_state"],freeze=True)

    results={}
    for arm,use_veto,selected_metrics in (
        ("reference",False,best["reference_metrics"]),
        ("treatment",True,best["treatment_metrics"]),
    ):
        metrics=_veto_metrics(
            replay_op,replay_head,replay_comp,replay_pred,
            dev_pair_rep,dev_context_rep,use_veto=use_veto,
        )
        for key,value in selected_metrics.items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(f"S73 {arm} replay changed: {key}")
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(f"S73 {arm} replay changed: {key}")

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        torch.save({
            "schema_version":"hira-v1-s73-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "shared_selected_dev_epoch":best["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "representation_family":"s69_query_gated_identity_interaction",
            "representation_parameter_count":0,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "pairwise_family":"s59_explicit_pairwise",
            "composer_family":"s71_mean_only_pairwise_row",
            "composer_parameter_count":COMPOSER_PARAMS,
            "residual_direction":"s72_fused_opponent_profile_confidence",
            "veto_predictor_family":"s73_counterfactual_safety_linear",
            "veto_predictor_parameter_count":PREDICTOR_PARAMS,
            "veto_threshold":S73_ACCEPT_THRESHOLD,
            "use_hard_veto":use_veto,
            "veto_output_mode":"candidate_or_fused_endpoint_only",
            "native_parameter_count_in_optimizer":0,
            "teacher_dependency":False,
            "correction_state_dict":best["correction_state"],
            "pairwise_head_state_dict":best["head_state"],
            "composer_state_dict":best["composer_state"],
            "veto_predictor_state_dict":best["predictor_state"],
        },checkpoint)
        results[arm]={
            "selected_dev_epoch":best["epoch"],
            "selected_dev":metrics,
            "selected_pairwise_aggregate_dev":best["shared_aggregate"],
            "selected_fused_shadow_dev":best["fused_shadow"],
            "gates":gates,
            "dev_ready":all(gates.values()),
            "correction_state_sha256":best["correction_state_sha256"],
            "pairwise_head_state_sha256":best["head_state_sha256"],
            "composer_state_sha256":best["composer_state_sha256"],
            "veto_predictor_state_sha256":best["predictor_state_sha256"],
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
    if a0.get("outcome")!="HIRA_V1_S73_A0_COUNTERFACTUAL_SAFETY_VETO_READY":
        raise RuntimeError("S73 A0 not qualified")
    if int(a0.get("reference_predictor_parameter_count",-1))!=PREDICTOR_PARAMS:
        raise RuntimeError("S73 A0 reference predictor capacity changed")
    if int(a0.get("treatment_predictor_parameter_count",-1))!=PREDICTOR_PARAMS:
        raise RuntimeError("S73 A0 treatment predictor capacity changed")
    if a0.get("predictor_initialization_bit_identical") is not True:
        raise RuntimeError("S73 A0 predictor initialization changed")
    if float(a0.get("hard_accept_threshold",-1.0))!=S73_ACCEPT_THRESHOLD:
        raise RuntimeError("S73 A0 threshold changed")
    if float(a0.get("endpoint_only_max_abs_error",1.0))!=0.0:
        raise RuntimeError("S73 A0 endpoint-only contract changed")
    if a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S73 A0 exposed fresh data")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S73 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S73 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S73 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S73 parent authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s73_cases("train")
    dev_rows=generate_s73_cases("dev")
    validate_s73_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S73 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S73 loaded parent runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S73 native remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    (
        train_pair_rep,train_context_rep,
        train_pair_digest,train_context_digest,train_rep_difference,
    )=s72._materialize_rep_caches(train_cache)
    (
        dev_pair_rep,dev_context_rep,
        dev_pair_digest,dev_context_digest,dev_rep_difference,
    )=s72._materialize_rep_caches(dev_cache)

    args.out.mkdir(parents=True,exist_ok=True)
    history,arms,shared_epoch=_train_matched(
        train_cache,dev_cache,
        train_pair_rep,train_context_rep,
        dev_pair_rep,dev_context_rep,
        args.out,
    )

    reference=arms["reference"]
    treatment=arms["treatment"]
    for key in ("pairwise_gold_pair_accuracy","pairwise_mean_gold_pair_margin"):
        if abs(float(reference["selected_dev"][key])-float(treatment["selected_dev"][key]))>1e-12:
            raise RuntimeError(f"S73 selected pairwise evidence not matched: {key}")
    for key in (
        "composer_mean_alpha","composer_min_alpha",
        "composer_max_alpha","composer_std_alpha",
    ):
        if abs(float(reference["selected_dev"][key])-float(treatment["selected_dev"][key]))>1e-12:
            raise RuntimeError(f"S73 selected alpha policy diverged: {key}")
    for key in (
        "correction_state_sha256","pairwise_head_state_sha256",
        "composer_state_sha256","veto_predictor_state_sha256",
        "selected_dev_epoch",
    ):
        if reference[key]!=treatment[key]:
            raise RuntimeError(f"S73 selected matched state diverged: {key}")

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
        "composer_mean_alpha",
        "composer_min_alpha",
        "composer_max_alpha",
        "composer_std_alpha",
        "veto_accept_rate",
        "veto_mean_accept_probability",
        "veto_std_accept_probability",
    )
    deltas=_delta(reference["selected_dev"],treatment["selected_dev"],keys)
    outcome=OUTCOME_READY if bool(treatment["dev_ready"]) else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S73_FRESH_COUNTERFACTUAL_SAFETY_VETO",
        "seed":SEED,
        "parent_s72":{
            "merged_main":"16eb7a583376cd824421d5e7ae9df948979b3953",
            "scientific_run":37470998139,
            "artifact_id":11416962628,
            "artifact_digest":"sha256:9a653a681e0cb3182819634fe24eb345ce82a48f2d078d15e510d1d3ebfb0834",
            "verdict":"CASE_B",
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
            "context_family":"s59_concat",
            "pair_train_digest":train_pair_digest,
            "context_train_digest":train_context_digest,
            "pair_dev_digest":dev_pair_digest,
            "context_dev_digest":dev_context_digest,
            "train_pair_context_max_abs_difference":train_rep_difference,
            "dev_pair_context_max_abs_difference":dev_rep_difference,
            "precomputed_once_before_training":True,
            "requires_grad":False,
        },
        "controlled_variable":{
            "representation_family_both_arms":"s69_query_gated_identity_interaction",
            "pairwise_family_both_arms":"s59_explicit_pairwise",
            "pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "shared_pairwise_head":True,
            "shared_pairwise_training_trajectory":True,
            "shared_correction_trajectory":True,
            "composer_family_both_arms":"s71_mean_only_pairwise_row",
            "composer_trainable_parameters_per_arm":COMPOSER_PARAMS,
            "residual_direction_both_arms":"s72_fused_opponent_profile_confidence",
            "veto_predictor_family_both_arms":"s73_counterfactual_safety_linear",
            "veto_predictor_trainable_parameters_per_arm":PREDICTOR_PARAMS,
            "veto_predictor_shared_state":True,
            "reference_veto_usage":"ignored_always_candidate",
            "treatment_veto_usage":"hard_candidate_or_fused",
            "hard_accept_threshold":S73_ACCEPT_THRESHOLD,
            "veto_target":"train_only_own_ce_and_cross_view_js_nonworse",
            "veto_output_mode":"endpoint_only",
            "shared_selection_epoch":True,
            "shared_selected_dev_epoch":shared_epoch,
            "same_train_rows_and_order":True,
            "same_optimizer_lr_weight_decay":True,
            "same_frozen_selector":True,
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
            "exact_s72_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "threshold_sweep_performed":False,
        "feature_sweep_performed":False,
        "class_weight_or_smoothing_changed":False,
        "safety_target_changed":False,
        "residual_or_representation_or_head_changed":False,
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
    print("HIRA_V1_S73_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
