from __future__ import annotations

import argparse
import json
from hashlib import sha256
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
from nmd.v1_query_gated_identity_interaction import (
    build_reference_pairwise_representation,
    build_query_gated_pairwise_representation,
)
from nmd.v1_multistat_pairwise_row_composer import (
    S71_PARAMETER_COUNT,
    MultiStatPairwiseRowComposer,
    multistat_responsibility_loss,
    pairwise_row_mean,
)
from nmd.v1_opponent_profile_vector_residual import (
    compose_with_residual_direction,
    opponent_profile_vector_direction,
    reference_pairwise_mean_direction,
)
from nmd.v1_s72_authority import generate_s72_cases, validate_s72_partitions

import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s72-opponent-profile-vector-residual-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S72_OPPONENT_PROFILE_VECTOR_RESIDUAL_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S72_OPPONENT_PROFILE_VECTOR_RESIDUAL_DEV_READY"

SEED=93_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=S59_PAIRWISE_PARAMETER_COUNT
COMPOSER_PARAMS=S71_PARAMETER_COUNT

A0_RUN=37463175304
A0_ARTIFACT_ID=11413616350
A0_ARTIFACT_DIGEST="sha256:a4d8052a0e76160310d4a99e86ef71f07810a96e77681cb34d747b85c98ef52a"


def _sources(op,evidence):
    identity=op.identity_signatures(
        state_tokens=evidence.state_tokens,
        state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
    )
    context=op.joint_query_context(
        state_tokens=evidence.state_tokens,
        state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
        question_tokens=evidence.question_tokens,
        question_mask=evidence.question_mask,
    )
    return identity,context


def _rep_digest(entries):
    d=sha256()
    for _c,_p,_n,rc,rp in entries:
        for tag,tensor in (("c",rc),("p",rp)):
            t=tensor.detach().cpu().contiguous()
            d.update(tag.encode("ascii"))
            d.update(str(tuple(t.shape)).encode("ascii"))
            d.update(str(t.dtype).encode("ascii"))
            d.update(t.numpy().tobytes())
    return d.hexdigest()


def _materialize_rep_caches(cache):
    geometry=JointStateQueryOptionPrivateCorrectionFork(train_correction=False)
    pair_entries=[]
    context_entries=[]
    max_pair_context_difference=0.0
    for canonical,paraphrase,n in cache:
        ic,qc=_sources(geometry,canonical)
        ip,qp=_sources(geometry,paraphrase)

        pair_c=build_query_gated_pairwise_representation(ic,qc)
        pair_p=build_query_gated_pairwise_representation(ip,qp)
        context_c=build_reference_pairwise_representation(ic,qc)
        context_p=build_reference_pairwise_representation(ip,qp)

        max_pair_context_difference=max(
            max_pair_context_difference,
            float((pair_c-context_c).abs().max()),
            float((pair_p-context_p).abs().max()),
        )
        pair_entries.append((canonical,paraphrase,n,pair_c,pair_p))
        context_entries.append((canonical,paraphrase,n,context_c,context_p))

    if max_pair_context_difference<=1e-7:
        raise RuntimeError("S72 S69 pair representation degenerate")

    return (
        pair_entries,
        context_entries,
        _rep_digest(pair_entries),
        _rep_digest(context_entries),
        max_pair_context_difference,
    )


def _aggregate_metrics(op,head,rep_cache):
    c_correct=p_correct=0
    pair_both=pair_changed=0
    agree_sum=js_sum=0.0
    c_margin_sum=p_margin_sum=0.0
    max_mass=max_antisym=max_diag=0.0
    cases=queries=0

    with torch.no_grad():
        for canonical,paraphrase,n,rc,rp in rep_cache:
            pair_c=head.pairwise_logits(rc)
            pair_p=head.pairwise_logits(rp)
            score_c=pairwise_row_mean(pair_c)
            score_p=pairwise_row_mean(pair_p)
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S72 paired gold mismatch")

            pc=score_c.argmax(-1)
            pp=score_p.argmax(-1)
            c_correct+=int((pc==gold).sum())
            p_correct+=int((pp==gold).sum())

            pairs_pc=pc.reshape(n,2)
            pairs_gold=gold.reshape(n,2)
            pair_both+=int(((pairs_pc==pairs_gold).all(-1)).sum())
            pair_changed+=int((pairs_pc[:,0]!=pairs_pc[:,1]).sum())

            q=2*n
            agree_sum+=float(selected_choice_agreement(score_c,score_p))*q
            js_sum+=float(symmetric_js_divergence(score_c,score_p))*q
            c_margin_sum+=float(fused_gold_vs_max_wrong_margin(score_c,gold).sum())
            p_margin_sum+=float(fused_gold_vs_max_wrong_margin(score_p,gold).sum())

            max_antisym=max(
                max_antisym,
                float((pair_c+pair_c.transpose(-1,-2)).abs().max()),
                float((pair_p+pair_p.transpose(-1,-2)).abs().max()),
            )
            max_diag=max(
                max_diag,
                float(torch.diagonal(pair_c,dim1=-2,dim2=-1).abs().max()),
                float(torch.diagonal(pair_p,dim1=-2,dim2=-1).abs().max()),
            )
            max_mass=max(
                max_mass,
                float((torch.softmax(score_c,-1).sum(-1)-1).abs().max()),
                float((torch.softmax(score_p,-1).sum(-1)-1).abs().max()),
            )
            cases+=n
            queries+=q

    if cases<1 or queries<1:
        raise RuntimeError("S72 empty aggregate metrics")
    return {
        "semantic_cases":cases,
        "canonical_queries":queries,
        "paraphrase_queries":queries,
        "canonical_accuracy":c_correct/queries,
        "paraphrase_accuracy":p_correct/queries,
        "canonical_paired_both_correct_rate":pair_both/cases,
        "question_swap_choice_change_rate":pair_changed/cases,
        "cross_view_selected_choice_agreement":agree_sum/queries,
        "cross_view_mean_js":js_sum/queries,
        "canonical_mean_gold_margin":c_margin_sum/queries,
        "paraphrase_mean_gold_margin":p_margin_sum/queries,
        "antisymmetry_max_abs_error":max_antisym,
        "diagonal_max_abs_error":max_diag,
        "max_probability_mass_error":max_mass,
    }


def _hybrid_metrics(op,head,composer,pair_rep_cache,context_rep_cache,*,use_opponent_profile_direction:bool):
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
    direction_cos_sum=0.0
    direction_count=0
    direction_max_abs_difference=0.0
    cases=queries=0

    if len(pair_rep_cache)!=len(context_rep_cache):
        raise RuntimeError("S72 pair/context cache length mismatch")
    plain=[(c,p,n) for c,p,n,_rc,_rp in context_rep_cache]
    base=s50._private_metrics(op,"treatment",plain)

    with torch.no_grad():
        for pair_entry,context_entry in zip(pair_rep_cache,context_rep_cache):
            canonical,paraphrase,n,pair_rc,pair_rp=pair_entry
            cc,cp,cn,context_rc,context_rp=context_entry
            if n!=cn or canonical is not cc or paraphrase is not cp:
                raise RuntimeError("S72 matched cache identity changed")

            _a,_b,fused_c=s50._branch_logits(op,"treatment",canonical)
            _a,_b,fused_p=s50._branch_logits(op,"treatment",paraphrase)
            matrix_c=head.pairwise_logits(pair_rc)
            matrix_p=head.pairwise_logits(pair_rp)

            hc,dc=compose_with_residual_direction(
                composer,fused_c,matrix_c,context_rc,
                use_opponent_profile_direction=use_opponent_profile_direction,
                return_diagnostics=True,
            )
            hp,dp=compose_with_residual_direction(
                composer,fused_p,matrix_p,context_rp,
                use_opponent_profile_direction=use_opponent_profile_direction,
                return_diagnostics=True,
            )
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S72 paired gold mismatch")

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

            for matrix,diag,hybrid,fused_view in (
                (matrix_c,dc,hc,fused_c),
                (matrix_p,dp,hp,fused_p),
            ):
                b,k,_=matrix.shape
                batch=torch.arange(b,device=matrix.device)
                row=matrix[batch,gold,:]
                mask=torch.arange(k,device=matrix.device)[None,:].ne(gold[:,None])
                margins=row[mask]
                gold_pair_correct+=int(margins.gt(0).sum())
                gold_pair_count+=int(margins.numel())
                gold_pair_margin_sum+=float(margins.sum())
                max_antisym=max(max_antisym,float((matrix+matrix.transpose(-1,-2)).abs().max()))
                max_diag=max(max_diag,float(torch.diagonal(matrix,dim1=-2,dim2=-1).abs().max()))
                max_mass=max(max_mass,float((torch.softmax(hybrid,-1).sum(-1)-1).abs().max()))
                residual_max=max(residual_max,float(diag["residual_max_abs"].max()))
                bound_max=max(bound_max,float(diag["residual_bound"].max()))
                ref_direction=reference_pairwise_mean_direction(matrix)
                trt_direction,_=opponent_profile_vector_direction(fused_view,matrix)
                direction_cos_sum+=float(
                    F.cosine_similarity(ref_direction,trt_direction,dim=-1).sum()
                )
                direction_count+=int(ref_direction.shape[0])
                direction_max_abs_difference=max(
                    direction_max_abs_difference,
                    float((ref_direction-trt_direction).abs().max()),
                )

                a=diag["alpha"].reshape(-1)
                alpha_sum+=float(a.sum())
                alpha_sq_sum+=float((a*a).sum())
                alpha_min=min(alpha_min,float(a.min()))
                alpha_max=max(alpha_max,float(a.max()))
                alpha_count+=int(a.numel())

            cases+=n
            queries+=q

    if queries<1 or gold_pair_count<1 or alpha_count<1:
        raise RuntimeError("S72 empty DEV metrics")

    alpha_mean=alpha_sum/alpha_count
    alpha_var=max(0.0,alpha_sq_sum/alpha_count-alpha_mean*alpha_mean)
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
        "direction_mean_cosine":direction_cos_sum/max(1,direction_count),
        "direction_max_abs_difference":direction_max_abs_difference,
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


def _matched_state(a,b):
    sa=a.parameter_state_dict_exact()
    sb=b.parameter_state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _train_matched(train_cache,dev_cache,train_pair_rep,train_context_rep,dev_pair_rep,dev_context_rep,out_dir):
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    reference=MultiStatPairwiseRowComposer(use_multistat=False,trainable=True)
    treatment=MultiStatPairwiseRowComposer(use_multistat=False,trainable=True)

    if not _matched_state(reference,treatment):
        raise RuntimeError("S72 composer initialization changed")
    if not torch.equal(reference.context_projection,treatment.context_projection):
        raise RuntimeError("S72 context projection changed")

    cparams=op.correction_parameters()
    hparams=[head.A,head.u]
    names=("W_phi","b_phi","w_out","b_out")
    rnamed=dict(reference.named_parameters())
    tnamed=dict(treatment.named_parameters())
    rparams=[rnamed[k] for k in names]
    tparams=[tnamed[k] for k in names]

    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S72 correction capacity changed")
    if sum(p.numel() for p in hparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S72 pairwise capacity changed")
    if sum(p.numel() for p in rparams)!=COMPOSER_PARAMS:
        raise RuntimeError("S72 reference composer capacity changed")
    if sum(p.numel() for p in tparams)!=COMPOSER_PARAMS:
        raise RuntimeError("S72 treatment composer capacity changed")

    corr_opt=torch.optim.AdamW(cparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    head_opt=torch.optim.AdamW(hparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    ref_opt=torch.optim.AdamW(rparams,lr=s35.LR,weight_decay=0.0)
    trt_opt=torch.optim.AdamW(tparams,lr=s35.LR,weight_decay=0.0)

    history=[]
    best=None
    best_key=None
    ref_phi_live=trt_phi_live=False

    print("HIRA_V1_S72_MATCHED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        base_sum=pair_sum=ref_sum=trt_sum=0.0
        ref_disagree=trt_disagree=0.0
        ref_count=trt_count=case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            pc,pp,pn,pair_c_rep,pair_p_rep=train_pair_rep[index]
            cc,cp,cn,context_c_rep,context_p_rep=train_context_rep[index]
            if (
                n!=pn or n!=cn
                or canonical is not pc or canonical is not cc
                or paraphrase is not pp or paraphrase is not cp
            ):
                raise RuntimeError("S72 representation/cache mismatch")

            corr_opt.zero_grad(set_to_none=True)
            base_loss,_ce,_js=s50._private_loss(op,"treatment",canonical,paraphrase)
            cgrads=torch.autograd.grad(base_loss,cparams,allow_unused=True)
            if not any(g is not None and float(g.detach().abs().sum())>0 for g in cgrads):
                raise RuntimeError("S72 correction gradient vanished")
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
                raise RuntimeError("S72 shared pairwise gradient vanished")
            for p,g in zip(hparams,hgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(hparams,s35.GRAD_CLIP)
            head_opt.step()

            with torch.no_grad():
                _a,_b,fused_c=s50._branch_logits(op,"treatment",canonical)
                _a,_b,fused_p=s50._branch_logits(op,"treatment",paraphrase)
                matrix_c=head.pairwise_logits(pair_c_rep)
                matrix_p=head.pairwise_logits(pair_p_rep)

            ref_opt.zero_grad(set_to_none=True)
            ref_loss,ref_diag=multistat_responsibility_loss(
                reference,fused_c,fused_p,matrix_c,matrix_p,
                context_c_rep,context_p_rep,canonical.gold,
            )
            rgrads=torch.autograd.grad(ref_loss,rparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in rgrads):
                raise RuntimeError("S72 reference composer gradient invalid")
            if float(rgrads[2].abs().sum())<=0 or float(rgrads[3].abs().sum())<=0:
                raise RuntimeError("S72 reference composer output gradient vanished")
            if float(rgrads[0].abs().sum())>0 or float(rgrads[1].abs().sum())>0:
                ref_phi_live=True
            for p,g in zip(rparams,rgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(rparams,s35.GRAD_CLIP)
            ref_opt.step()

            trt_opt.zero_grad(set_to_none=True)
            trt_loss,trt_diag=multistat_responsibility_loss(
                treatment,fused_c,fused_p,matrix_c,matrix_p,
                context_c_rep,context_p_rep,canonical.gold,
            )
            tgrads=torch.autograd.grad(trt_loss,tparams,allow_unused=True)
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in tgrads):
                raise RuntimeError("S72 treatment composer gradient invalid")
            if float(tgrads[2].abs().sum())<=0 or float(tgrads[3].abs().sum())<=0:
                raise RuntimeError("S72 treatment composer output gradient vanished")
            if float(tgrads[0].abs().sum())>0 or float(tgrads[1].abs().sum())>0:
                trt_phi_live=True
            for p,g in zip(tparams,tgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(tparams,s35.GRAD_CLIP)
            trt_opt.step()

            q=int(canonical.gold.numel())
            base_sum+=float(base_loss.detach())*n
            pair_sum+=float(pair_loss.detach())*n
            ref_sum+=float(ref_loss.detach())*n
            trt_sum+=float(trt_loss.detach())*n
            ref_disagree+=float(ref_diag["target_disagreement_fraction"])*q
            trt_disagree+=float(trt_diag["target_disagreement_fraction"])*q
            ref_count+=q
            trt_count+=q
            case_count+=n

        if not ref_phi_live or not trt_phi_live:
            raise RuntimeError("S72 composer phi gradient never activated")

        fused_shadow=s50._private_metrics(op,"treatment",dev_cache)
        shared_pair_train=s59._pairwise_train_diagnostics(head,train_pair_rep)
        shared_aggregate=_aggregate_metrics(op,head,dev_pair_rep)
        ref_metrics=_hybrid_metrics(
            op,head,reference,dev_pair_rep,dev_context_rep,
            use_opponent_profile_direction=False,
        )
        trt_metrics=_hybrid_metrics(
            op,head,treatment,dev_pair_rep,dev_context_rep,
            use_opponent_profile_direction=True,
        )

        for key in ("pairwise_gold_pair_accuracy","pairwise_mean_gold_pair_margin"):
            if abs(float(ref_metrics[key])-float(trt_metrics[key]))>1e-12:
                raise RuntimeError(f"S72 pairwise evidence diverged: {key}")

        corr_state=op.correction_state_dict()
        head_state=head.state_dict_exact()
        ref_state=reference.state_dict_exact()
        trt_state=treatment.state_dict_exact()
        corr_digest=s59._state_digest(corr_state)
        head_digest=s59._state_digest(head_state)
        ref_digest=s59._state_digest(ref_state)
        trt_digest=s59._state_digest(trt_state)
        if ref_digest!=trt_digest:
            raise RuntimeError("S72 matched composer states diverged")

        record={
            "epoch":epoch,
            "train_mean_base_correction_loss":base_sum/case_count,
            "train_mean_shared_pairwise_loss":pair_sum/case_count,
            "train_mean_reference_composer_loss":ref_sum/case_count,
            "train_mean_treatment_composer_loss":trt_sum/case_count,
            "train_reference_responsibility_disagreement_fraction":ref_disagree/max(1,ref_count),
            "train_treatment_responsibility_disagreement_fraction":trt_disagree/max(1,trt_count),
            "reference_composer_phi_gradient_ever_live":ref_phi_live,
            "treatment_composer_phi_gradient_ever_live":trt_phi_live,
            "shared_train_pairwise":shared_pair_train,
            "shared_pairwise_aggregate_dev":shared_aggregate,
            "fused_shadow_dev":fused_shadow,
            "reference_dev":ref_metrics,
            "treatment_dev":trt_metrics,
            "correction_state_sha256":corr_digest,
            "shared_pairwise_head_state_sha256":head_digest,
            "reference_composer_state_sha256":ref_digest,
            "treatment_composer_state_sha256":trt_digest,
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
                "shared_aggregate":dict(shared_aggregate),
                "fused_shadow":dict(fused_shadow),
                "correction_state_sha256":corr_digest,
                "head_state_sha256":head_digest,
                "reference_state_sha256":ref_digest,
                "treatment_state_sha256":trt_digest,
            }

        print("HIRA_V1_S72_EPOCH="+json.dumps({
            "epoch":epoch,
            "shared_pairwise_gold_pair_accuracy":shared_pair_train["gold_pair_accuracy"],
            "reference":ref_metrics,
            "treatment":trt_metrics,
        },sort_keys=True),flush=True)

    if best is None:
        raise RuntimeError("S72 shared selector failed")

    replay_op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    replay_op.load_correction_state_dict(best["correction_state"],freeze=True)
    replay_head=ExplicitPairwiseDecisionHead(trainable=True)
    replay_head.load_state_dict_exact(best["head_state"],freeze=True)

    results={}
    for arm,use_profile,state,selected_metrics in (
        ("reference",False,best["reference_state"],best["reference_metrics"]),
        ("treatment",True,best["treatment_state"],best["treatment_metrics"]),
    ):
        replay=MultiStatPairwiseRowComposer(use_multistat=False,trainable=True)
        replay.load_state_dict_exact(state,freeze=True)
        metrics=_hybrid_metrics(
            replay_op,replay_head,replay,dev_pair_rep,dev_context_rep,
            use_opponent_profile_direction=use_profile,
        )
        for key,value in selected_metrics.items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(f"S72 {arm} replay changed: {key}")
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(f"S72 {arm} replay changed: {key}")

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        torch.save({
            "schema_version":"hira-v1-s72-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "shared_selected_dev_epoch":best["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "representation_family":"s69_query_gated_identity_interaction",
            "representation_parameter_count":0,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "pairwise_family":"s59_explicit_pairwise",
            "shared_pairwise_head":True,
            "composer_family":"s71_mean_only_pairwise_row",
            "composer_parameter_count":COMPOSER_PARAMS,
            "use_multistat":False,
            "row_channels":"normalized_mean_plus_three_zero_channels",
            "residual_direction":
                "fused_opponent_profile_confidence" if use_profile
                else "uniform_row_mean",
            "alpha_max":0.35,
            "composer_objective":"per_view_counterfactual_responsibility_bce",
            "composer_context_source":"exact_s59_reference_representation",
            "native_parameter_count_in_optimizer":0,
            "teacher_dependency":False,
            "correction_state_dict":best["correction_state"],
            "pairwise_head_state_dict":best["head_state"],
            "composer_state_dict":state,
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
            "composer_state_sha256":
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
    if a0.get("outcome")!="HIRA_V1_S72_A0_OPPONENT_PROFILE_VECTOR_RESIDUAL_READY":
        raise RuntimeError("S72 A0 not qualified")
    if int(a0.get("reference_composer_parameter_count",-1))!=COMPOSER_PARAMS:
        raise RuntimeError("S72 A0 reference composer capacity changed")
    if int(a0.get("treatment_composer_parameter_count",-1))!=COMPOSER_PARAMS:
        raise RuntimeError("S72 A0 treatment composer capacity changed")
    if a0.get("composer_parameter_initialization_bit_identical") is not True:
        raise RuntimeError("S72 A0 composer initialization changed")
    if float(a0.get("alpha_match_max_abs_error",1.0))!=0.0:
        raise RuntimeError("S72 A0 alpha matching changed")
    if float(a0.get("reference_direction_permutation_max_abs_error",1.0))>2e-6:
        raise RuntimeError("S72 A0 reference direction permutation changed")
    if float(a0.get("treatment_direction_permutation_max_abs_error",1.0))>2e-6:
        raise RuntimeError("S72 A0 treatment direction permutation changed")
    if a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S72 A0 exposed fresh data")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S72 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S72 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S72 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S72 parent authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s72_cases("train")
    dev_rows=generate_s72_cases("dev")
    validate_s72_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S72 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S72 loaded parent runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S72 native remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    (
        train_pair_rep,train_context_rep,
        train_pair_digest,train_context_digest,train_rep_difference,
    )=_materialize_rep_caches(train_cache)
    (
        dev_pair_rep,dev_context_rep,
        dev_pair_digest,dev_context_digest,dev_rep_difference,
    )=_materialize_rep_caches(dev_cache)

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
            raise RuntimeError(f"S72 selected pairwise evidence not matched: {key}")
    if reference["pairwise_head_state_sha256"]!=treatment["pairwise_head_state_sha256"]:
        raise RuntimeError("S72 selected pairwise head states diverged")
    if reference["correction_state_sha256"]!=treatment["correction_state_sha256"]:
        raise RuntimeError("S72 selected correction states diverged")
    if reference["selected_dev_epoch"]!=treatment["selected_dev_epoch"]:
        raise RuntimeError("S72 selected epochs diverged")
    if reference["composer_state_sha256"]!=treatment["composer_state_sha256"]:
        raise RuntimeError("S72 selected composer states diverged")
    for key in (
        "composer_mean_alpha","composer_min_alpha",
        "composer_max_alpha","composer_std_alpha",
    ):
        if abs(float(reference["selected_dev"][key])-float(treatment["selected_dev"][key]))>1e-12:
            raise RuntimeError(f"S72 selected alpha policy diverged: {key}")

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
    )
    deltas=_delta(reference["selected_dev"],treatment["selected_dev"],keys)
    overall_ready=bool(reference["dev_ready"]) and bool(treatment["dev_ready"])
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S72_FRESH_OPPONENT_PROFILE_VECTOR_RESIDUAL",
        "seed":SEED,
        "parent_s71":{
            "merged_main":"ef2e8a4b71ba9a46dd9caea07698df742abb6461",
            "scientific_run":37459447097,
            "artifact_id":11412505173,
            "artifact_digest":"sha256:aa116237d5508131db18f88a23893d9107d05ae67f46ccffe638520fe8098a50",
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
            "reference_row_channels":"normalized_mean_plus_three_zero_channels",
            "treatment_row_channels":"normalized_mean_plus_three_zero_channels",
            "composer_trainable_parameters_per_arm":COMPOSER_PARAMS,
            "composer_parameter_initialization_bit_identical":True,
            "reference_residual_direction":"uniform_row_mean",
            "treatment_residual_direction":"fused_opponent_profile_confidence",
            "alpha_max":0.35,
            "representation_family_both_arms":"s69_query_gated_identity_interaction",
            "representation_parameters_both_arms":0,
            "pairwise_family_both_arms":"s59_explicit_pairwise",
            "pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "shared_pairwise_head":True,
            "shared_pairwise_training_trajectory":True,
            "shared_correction_trajectory":True,
            "shared_selection_epoch":True,
            "shared_selected_dev_epoch":shared_epoch,
            "pairwise_objective":"gold_vs_distractor_pairwise_softplus",
            "composer_objective_both_arms":"per_view_counterfactual_responsibility_bce",
            "composer_context_source_both_arms":"exact_s59_reference_representation",
            "correction_trainable_parameters":CORRECTION_PARAMS,
            "same_train_rows_and_order":True,
            "same_optimizer_lr_weight_decay":True,
            "same_frozen_selector":True,
            "teacher_dependency":False,
            "pseudo_target_dependency":False,
            "reliability_target_reopened":False,
        },
        "training_trajectory":history,
        "reference_branch":reference,
        "treatment_branch":treatment,
        "selected_dev_delta_treatment_minus_reference":deltas,
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "exact_s71_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "opponent_profile_sweep_performed":False,
        "composer_width_changed":False,
        "direction_normalization_changed":False,
        "reliability_target_reopened":False,
        "representation_or_pairwise_head_changed":False,
        "residual_direction_changed":True,
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
    print("HIRA_V1_S72_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
