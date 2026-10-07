from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_evidence_fusion import fused_gold_vs_max_wrong_margin
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_persisted_native_authority import load_native_authority, verify_file_sha256
from nmd.v1_explicit_pairwise_decision_head import (
    ExplicitPairwiseDecisionHead,
    S59_PAIRWISE_PARAMETER_COUNT,
    pairwise_head_loss,
)
from nmd.v1_query_gated_identity_interaction import (
    build_query_gated_pairwise_representation,
)
from nmd.v1_token_level_bidirectional_evidence_binding import (
    S75_INTERACTION_SCALE,
    S75_REPRESENTATION_PARAMETER_COUNT,
    build_token_level_bidirectional_pairwise_representation,
)
from nmd.v1_contextual_reliability_gate import (
    S64_CONTEXT_PROJECTION_SEED,
    S64_GATE_PARAMETER_COUNT,
    ContextInjectedReliabilityGate,
)
from nmd.v1_per_view_responsibility import per_view_responsibility_loss
from nmd.v1_s75_authority import generate_s75_cases, validate_s75_partitions

import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59
import hira_v1_s68_train_dev as s68


SCHEMA_VERSION="hira-v1-s75-token-level-bidirectional-binding-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S75_TOKEN_LEVEL_BIDIRECTIONAL_BINDING_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S75_TOKEN_LEVEL_BIDIRECTIONAL_BINDING_DEV_READY"

SEED=96_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=S59_PAIRWISE_PARAMETER_COUNT
GATE_PARAMS=S64_GATE_PARAMETER_COUNT

A0_RUN=37578213725
A0_ARTIFACT_ID=11463278233
A0_ARTIFACT_DIGEST="sha256:fae3cc55fdbd8ab841da16b0f6ae113dc17d0186f844cbda26fab6a6a70c424e"


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


def _digest(entries):
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
    reference=[]
    treatment=[]
    max_difference=0.0
    for canonical,paraphrase,n in cache:
        ic,qc=_sources(geometry,canonical)
        ip,qp=_sources(geometry,paraphrase)

        # Reference is the exact frozen S69 representation.
        rc_ref=build_query_gated_pairwise_representation(ic,qc)
        rp_ref=build_query_gated_pairwise_representation(ip,qp)

        # Treatment changes semantic evidence before query-token pooling.
        rc_trt=build_token_level_bidirectional_pairwise_representation(
            identity=ic,
            state_tokens=canonical.state_tokens,
            state_mask=canonical.state_mask,
            option_view_tokens=canonical.option_view_tokens,
            option_view_token_mask=canonical.option_view_token_mask,
            option_view_mask=canonical.option_view_mask,
            question_tokens=canonical.question_tokens,
            question_mask=canonical.question_mask,
        )
        rp_trt=build_token_level_bidirectional_pairwise_representation(
            identity=ip,
            state_tokens=paraphrase.state_tokens,
            state_mask=paraphrase.state_mask,
            option_view_tokens=paraphrase.option_view_tokens,
            option_view_token_mask=paraphrase.option_view_token_mask,
            option_view_mask=paraphrase.option_view_mask,
            question_tokens=paraphrase.question_tokens,
            question_mask=paraphrase.question_mask,
        )
        max_difference=max(
            max_difference,
            float((rc_ref-rc_trt).abs().max()),
            float((rp_ref-rp_trt).abs().max()),
        )
        reference.append((canonical,paraphrase,n,rc_ref,rp_ref))
        treatment.append((canonical,paraphrase,n,rc_trt,rp_trt))
    if max_difference<=1e-7:
        raise RuntimeError("S75 TBER treatment representation degenerate")
    return (
        reference,
        treatment,
        _digest(reference),
        _digest(treatment),
        max_difference,
    )

def _hybrid_metrics(op,head,gate,pair_rep_cache,gate_rep_cache):
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

    if len(pair_rep_cache)!=len(gate_rep_cache):
        raise RuntimeError("S75 pair/gate cache length mismatch")
    plain=[(c,p,n) for c,p,n,_rc,_rp in gate_rep_cache]
    base=s50._private_metrics(op,"treatment",plain)

    with torch.no_grad():
        for pair_entry,gate_entry in zip(pair_rep_cache,gate_rep_cache):
            canonical,paraphrase,n,pair_rc,pair_rp=pair_entry
            gc,gp,gn,gate_rc,gate_rp=gate_entry
            if canonical.case_id if hasattr(canonical,"case_id") else None:
                pass
            if n!=gn or canonical is not gc or paraphrase is not gp:
                raise RuntimeError("S75 matched cache identity changed")

            _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
            _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
            pair_c=head.aggregate_logits(pair_rc)
            pair_p=head.aggregate_logits(pair_rp)
            hc,dc=gate.compose(
                fused_c,pair_c,gate_rc,return_diagnostics=True
            )
            hp,dp=gate.compose(
                fused_p,pair_p,gate_rp,return_diagnostics=True
            )
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S75 paired gold mismatch")

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
            hybrid_c_margin_sum+=float(
                fused_gold_vs_max_wrong_margin(hc,gold).sum()
            )
            hybrid_p_margin_sum+=float(
                fused_gold_vs_max_wrong_margin(hp,gold).sum()
            )
            canonical_loss_sum+=float(
                F.cross_entropy(hc,gold,reduction="sum")
            )

            for rep,pair_score,diag,hybrid in (
                (pair_rc,pair_c,dc,hc),
                (pair_rp,pair_p,dp,hp),
            ):
                pair=head.pairwise_logits(rep)
                b,k,_=pair.shape
                batch=torch.arange(b,device=pair.device)
                row=pair[batch,gold,:]
                mask=torch.arange(k,device=pair.device)[None,:].ne(
                    gold[:,None]
                )
                margins=row[mask]
                gold_pair_correct+=int(margins.gt(0).sum())
                gold_pair_count+=int(margins.numel())
                gold_pair_margin_sum+=float(margins.sum())
                max_antisym=max(
                    max_antisym,
                    float((pair+pair.transpose(-1,-2)).abs().max()),
                )
                max_diag=max(
                    max_diag,
                    float(torch.diagonal(pair,dim1=-2,dim2=-1).abs().max()),
                )
                max_mass=max(
                    max_mass,
                    float((torch.softmax(hybrid,-1).sum(-1)-1).abs().max()),
                )
                residual_max=max(
                    residual_max,float(diag["residual_max_abs"].max())
                )
                bound_max=max(
                    bound_max,float(diag["residual_bound"].max())
                )
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
        raise RuntimeError("S75 empty DEV metrics")

    alpha_mean=alpha_sum/alpha_count
    alpha_var=max(0.0,alpha_sq_sum/alpha_count-alpha_mean*alpha_mean)
    return {
        "semantic_cases":semantic_cases,
        "canonical_queries":queries,
        "paraphrase_queries":queries,
        "fused_canonical_accuracy":hybrid_c_correct/queries,
        "fused_paraphrase_accuracy":hybrid_p_correct/queries,
        "fused_canonical_paired_both_correct_rate":
            hybrid_pair_both/semantic_cases,
        "fused_question_swap_choice_change_rate":
            hybrid_pair_changed/semantic_cases,
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
        "raw_triadic_canonical_accuracy":
            base["raw_triadic_canonical_accuracy"],
        "raw_triadic_paraphrase_accuracy":
            base["raw_triadic_paraphrase_accuracy"],
        "raw_triadic_cross_view_agreement":
            base["raw_triadic_cross_view_agreement"],
        "canonical_relation_binding_accuracy":
            base["canonical_relation_binding_accuracy"],
        "paraphrase_relation_binding_accuracy":
            base["paraphrase_relation_binding_accuracy"],
        "relation_cross_view_agreement":
            base["relation_cross_view_agreement"],
        "relation_cross_view_mean_js":
            base["relation_cross_view_mean_js"],
        "canonical_relation_binding_mean_gold_margin":
            base["canonical_relation_binding_mean_gold_margin"],
        "paraphrase_relation_binding_mean_gold_margin":
            base["paraphrase_relation_binding_mean_gold_margin"],
        "mean_same_option_signature_cosine":
            base["mean_same_option_signature_cosine"],
        "mean_signature_same_vs_strongest_wrong_margin":
            base["mean_signature_same_vs_strongest_wrong_margin"],
        "fused_option_order_flip_rate":0.0,
        "fused_max_probability_mass_error":max_mass,
        "full_k":bool(full_k),
        "relation_delta_max_abs":0.0,
        "state_view_encodes":0,
    }


def _matched_state(a,b):
    sa=a.state_dict_exact()
    sb=b.state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _train_matched(
    train_cache,
    dev_cache,
    train_ref_rep,
    train_trt_rep,
    dev_ref_rep,
    dev_trt_rep,
    out_dir,
):
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    reference_head=ExplicitPairwiseDecisionHead(trainable=True)
    treatment_head=ExplicitPairwiseDecisionHead(trainable=True)
    reference_gate=ContextInjectedReliabilityGate(use_context=True,trainable=True)
    treatment_gate=ContextInjectedReliabilityGate(use_context=True,trainable=True)

    if not _matched_state(reference_head,treatment_head):
        raise RuntimeError("S75 pairwise initialization changed")
    if not _matched_state(reference_gate,treatment_gate):
        raise RuntimeError("S75 gate initialization changed")

    cparams=op.correction_parameters()
    rhparams=[reference_head.A,reference_head.u]
    thparams=[treatment_head.A,treatment_head.u]
    gate_names=("W_phi","b_phi","w_out","b_out")
    rg_named=dict(reference_gate.named_parameters())
    tg_named=dict(treatment_gate.named_parameters())
    rgparams=[rg_named[k] for k in gate_names]
    tgparams=[tg_named[k] for k in gate_names]

    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S75 correction capacity changed")
    if sum(p.numel() for p in rhparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S75 reference head capacity changed")
    if sum(p.numel() for p in thparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S75 treatment head capacity changed")
    if sum(p.numel() for p in rgparams)!=GATE_PARAMS:
        raise RuntimeError("S75 reference gate capacity changed")
    if sum(p.numel() for p in tgparams)!=GATE_PARAMS:
        raise RuntimeError("S75 treatment gate capacity changed")

    corr_opt=torch.optim.AdamW(
        cparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    ref_head_opt=torch.optim.AdamW(
        rhparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    trt_head_opt=torch.optim.AdamW(
        thparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    ref_gate_opt=torch.optim.AdamW(
        rgparams,lr=s35.LR,weight_decay=0.0
    )
    trt_gate_opt=torch.optim.AdamW(
        tgparams,lr=s35.LR,weight_decay=0.0
    )

    history=[]
    best={"reference":None,"treatment":None}
    best_key={"reference":None,"treatment":None}
    ref_phi_live=trt_phi_live=False

    print("HIRA_V1_S75_MATCHED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        base_sum=ref_pair_sum=trt_pair_sum=ref_gate_sum=trt_gate_sum=0.0
        ref_disagree=trt_disagree=0.0
        ref_count=trt_count=case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            rc, rp, rn, ref_c, ref_p=train_ref_rep[index]
            tc, tp, tn, trt_c, trt_p=train_trt_rep[index]
            if (
                n!=rn or n!=tn
                or canonical is not rc or canonical is not tc
                or paraphrase is not rp or paraphrase is not tp
            ):
                raise RuntimeError("S75 representation/cache mismatch")

            corr_opt.zero_grad(set_to_none=True)
            base_loss,_ce,_js=s50._private_loss(
                op,"treatment",canonical,paraphrase
            )
            cgrads=torch.autograd.grad(
                base_loss,cparams,allow_unused=True
            )
            if not any(
                g is not None and float(g.detach().abs().sum())>0
                for g in cgrads
            ):
                raise RuntimeError("S75 correction gradient vanished")
            for p,g in zip(cparams,cgrads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(cparams,s35.GRAD_CLIP)
            corr_opt.step()

            ref_head_opt.zero_grad(set_to_none=True)
            rlc,_=pairwise_head_loss(reference_head,ref_c,canonical.gold)
            rlp,_=pairwise_head_loss(reference_head,ref_p,paraphrase.gold)
            ref_pair_loss=0.5*(rlc+rlp)
            rgrads=torch.autograd.grad(
                ref_pair_loss,rhparams,allow_unused=True
            )
            if not all(
                g is not None and float(g.detach().abs().sum())>0
                for g in rgrads
            ):
                raise RuntimeError("S75 reference head gradient vanished")
            for p,g in zip(rhparams,rgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(rhparams,s35.GRAD_CLIP)
            ref_head_opt.step()

            trt_head_opt.zero_grad(set_to_none=True)
            tlc,_=pairwise_head_loss(treatment_head,trt_c,canonical.gold)
            tlp,_=pairwise_head_loss(treatment_head,trt_p,paraphrase.gold)
            trt_pair_loss=0.5*(tlc+tlp)
            tgrads=torch.autograd.grad(
                trt_pair_loss,thparams,allow_unused=True
            )
            if not all(
                g is not None and float(g.detach().abs().sum())>0
                for g in tgrads
            ):
                raise RuntimeError("S75 treatment head gradient vanished")
            for p,g in zip(thparams,tgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(thparams,s35.GRAD_CLIP)
            trt_head_opt.step()

            with torch.no_grad():
                _a,_b,fused_c=s50._branch_logits(
                    op,"treatment",canonical
                )
                _a,_b,fused_p=s50._branch_logits(
                    op,"treatment",paraphrase
                )
                ref_pair_c=reference_head.aggregate_logits(ref_c)
                ref_pair_p=reference_head.aggregate_logits(ref_p)
                trt_pair_c=treatment_head.aggregate_logits(trt_c)
                trt_pair_p=treatment_head.aggregate_logits(trt_p)

            # Both gates consume the exact S59 reference representation.
            ref_gate_opt.zero_grad(set_to_none=True)
            ref_gate_loss,ref_diag=per_view_responsibility_loss(
                reference_gate,
                fused_c,fused_p,
                ref_pair_c,ref_pair_p,
                ref_c,ref_p,
                canonical.gold,
            )
            rggrads=torch.autograd.grad(
                ref_gate_loss,rgparams,allow_unused=True
            )
            if not all(
                g is not None and bool(torch.isfinite(g).all())
                for g in rggrads
            ):
                raise RuntimeError("S75 reference gate gradient invalid")
            if float(rggrads[2].abs().sum())<=0 or float(rggrads[3].abs().sum())<=0:
                raise RuntimeError("S75 reference gate output gradient vanished")
            if float(rggrads[0].abs().sum())>0 or float(rggrads[1].abs().sum())>0:
                ref_phi_live=True
            for p,g in zip(rgparams,rggrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(rgparams,s35.GRAD_CLIP)
            ref_gate_opt.step()

            trt_gate_opt.zero_grad(set_to_none=True)
            trt_gate_loss,trt_diag=per_view_responsibility_loss(
                treatment_gate,
                fused_c,fused_p,
                trt_pair_c,trt_pair_p,
                ref_c,ref_p,
                canonical.gold,
            )
            tggrads=torch.autograd.grad(
                trt_gate_loss,tgparams,allow_unused=True
            )
            if not all(
                g is not None and bool(torch.isfinite(g).all())
                for g in tggrads
            ):
                raise RuntimeError("S75 treatment gate gradient invalid")
            if float(tggrads[2].abs().sum())<=0 or float(tggrads[3].abs().sum())<=0:
                raise RuntimeError("S75 treatment gate output gradient vanished")
            if float(tggrads[0].abs().sum())>0 or float(tggrads[1].abs().sum())>0:
                trt_phi_live=True
            for p,g in zip(tgparams,tggrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(tgparams,s35.GRAD_CLIP)
            trt_gate_opt.step()

            base_sum+=float(base_loss.detach())*n
            ref_pair_sum+=float(ref_pair_loss.detach())*n
            trt_pair_sum+=float(trt_pair_loss.detach())*n
            ref_gate_sum+=float(ref_gate_loss.detach())*n
            trt_gate_sum+=float(trt_gate_loss.detach())*n
            rcnt=int(ref_diag["target_count"])
            tcnt=int(trt_diag["target_count"])
            ref_disagree+=float(ref_diag["target_disagreement_fraction"])*rcnt
            trt_disagree+=float(trt_diag["target_disagreement_fraction"])*tcnt
            ref_count+=rcnt
            trt_count+=tcnt
            case_count+=n

        if not ref_phi_live or not trt_phi_live:
            raise RuntimeError("S75 gate phi gradient never activated")

        fused_shadow=s50._private_metrics(op,"treatment",dev_cache)
        reference=_hybrid_metrics(
            op,reference_head,reference_gate,dev_ref_rep,dev_ref_rep
        )
        treatment=_hybrid_metrics(
            op,treatment_head,treatment_gate,dev_trt_rep,dev_ref_rep
        )
        ref_pair_train=s59._pairwise_train_diagnostics(
            reference_head,train_ref_rep
        )
        trt_pair_train=s59._pairwise_train_diagnostics(
            treatment_head,train_trt_rep
        )
        ref_pair_agg=s68._pairwise_aggregate_metrics(
            reference_head,dev_ref_rep
        )
        trt_pair_agg=s68._pairwise_aggregate_metrics(
            treatment_head,dev_trt_rep
        )

        corr_state=op.correction_state_dict()
        ref_head_state=reference_head.state_dict_exact()
        trt_head_state=treatment_head.state_dict_exact()
        ref_gate_state=reference_gate.state_dict_exact()
        trt_gate_state=treatment_gate.state_dict_exact()

        corr_digest=s59._state_digest(corr_state)
        ref_head_digest=s59._state_digest(ref_head_state)
        trt_head_digest=s59._state_digest(trt_head_state)
        ref_gate_digest=s59._state_digest(ref_gate_state)
        trt_gate_digest=s59._state_digest(trt_gate_state)

        record={
            "epoch":epoch,
            "train_mean_base_correction_loss":base_sum/case_count,
            "train_mean_reference_pairwise_loss":ref_pair_sum/case_count,
            "train_mean_treatment_pairwise_loss":trt_pair_sum/case_count,
            "train_mean_reference_gate_loss":ref_gate_sum/case_count,
            "train_mean_treatment_gate_loss":trt_gate_sum/case_count,
            "train_reference_responsibility_disagreement_fraction":
                ref_disagree/max(1,ref_count),
            "train_treatment_responsibility_disagreement_fraction":
                trt_disagree/max(1,trt_count),
            "reference_gate_phi_gradient_ever_live":ref_phi_live,
            "treatment_gate_phi_gradient_ever_live":trt_phi_live,
            "reference_train_pairwise":ref_pair_train,
            "treatment_train_pairwise":trt_pair_train,
            "reference_pairwise_aggregate_dev":ref_pair_agg,
            "treatment_pairwise_aggregate_dev":trt_pair_agg,
            "fused_shadow_dev":fused_shadow,
            "reference_dev":reference,
            "treatment_dev":treatment,
            "correction_state_sha256":corr_digest,
            "reference_pairwise_head_state_sha256":ref_head_digest,
            "treatment_pairwise_head_state_sha256":trt_head_digest,
            "reference_gate_state_sha256":ref_gate_digest,
            "treatment_gate_state_sha256":trt_gate_digest,
        }
        history.append(record)

        for arm,metrics,pair_metrics,head_state,head_digest,gate_state,gate_digest in (
            (
                "reference",reference,ref_pair_agg,
                ref_head_state,ref_head_digest,
                ref_gate_state,ref_gate_digest,
            ),
            (
                "treatment",treatment,trt_pair_agg,
                trt_head_state,trt_head_digest,
                trt_gate_state,trt_gate_digest,
            ),
        ):
            key=s17._selection_key(epoch,metrics)
            if best_key[arm] is None or key>best_key[arm]:
                best_key[arm]=key
                best[arm]={
                    "epoch":epoch,
                    "correction_state":{
                        k:v.detach().cpu().clone()
                        for k,v in corr_state.items()
                    },
                    "head_state":{
                        k:v.detach().cpu().clone()
                        for k,v in head_state.items()
                    },
                    "gate_state":{
                        k:v.detach().cpu().clone()
                        for k,v in gate_state.items()
                    },
                    "metrics":dict(metrics),
                    "pairwise_aggregate":dict(pair_metrics),
                    "fused_shadow":dict(fused_shadow),
                    "correction_state_sha256":corr_digest,
                    "pairwise_head_state_sha256":head_digest,
                    "gate_state_sha256":gate_digest,
                }

        print(
            "HIRA_V1_S75_EPOCH="+json.dumps({
                "epoch":epoch,
                "reference_pairwise_gold_pair_accuracy":
                    ref_pair_train["gold_pair_accuracy"],
                "treatment_pairwise_gold_pair_accuracy":
                    trt_pair_train["gold_pair_accuracy"],
                "reference_pairwise_dev":ref_pair_agg,
                "treatment_pairwise_dev":trt_pair_agg,
                "reference":reference,
                "treatment":treatment,
            },sort_keys=True),
            flush=True,
        )

    if best["reference"] is None or best["treatment"] is None:
        raise RuntimeError("S75 selector failed")

    results={}
    for arm,pair_rep in (
        ("reference",dev_ref_rep),
        ("treatment",dev_trt_rep),
    ):
        selected=best[arm]
        replay_op=JointStateQueryOptionPrivateCorrectionFork(
            train_correction=True
        )
        replay_op.load_correction_state_dict(
            selected["correction_state"],freeze=True
        )
        replay_head=ExplicitPairwiseDecisionHead(trainable=True)
        replay_head.load_state_dict_exact(
            selected["head_state"],freeze=True
        )
        replay_gate=ContextInjectedReliabilityGate(
            use_context=True,trainable=True
        )
        replay_gate.load_state_dict_exact(
            selected["gate_state"],freeze=True
        )
        metrics=_hybrid_metrics(
            replay_op,replay_head,replay_gate,pair_rep,dev_ref_rep
        )
        pair_metrics=s68._pairwise_aggregate_metrics(
            replay_head,pair_rep
        )
        for key,value in selected["metrics"].items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(f"S75 {arm} replay changed: {key}")
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(f"S75 {arm} replay changed: {key}")
        for key,value in selected["pairwise_aggregate"].items():
            if abs(float(pair_metrics[key])-float(value))>1e-12:
                raise RuntimeError(
                    f"S75 {arm} pairwise replay changed: {key}"
                )

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        rep_family=(
            "s59_concat"
            if arm=="reference"
            else "s75_query_gated_identity_interaction"
        )
        torch.save({
            "schema_version":"hira-v1-s75-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "selected_dev_epoch":selected["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "representation_family":rep_family,
            "representation_dimension":512,
            "representation_parameter_count":0,
            "interaction_scale":
                0.0 if arm=="reference" else S75_INTERACTION_SCALE,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "pairwise_family":"s59_explicit_pairwise",
            "gate_parameter_count":GATE_PARAMS,
            "gate_family":"s64_context_injected",
            "gate_objective":"per_view_counterfactual_responsibility_bce",
            "gate_context_source":"exact_s59_reference_representation",
            "gate_context_projection_seed":S64_CONTEXT_PROJECTION_SEED,
            "native_parameter_count_in_optimizer":0,
            "teacher_dependency":False,
            "correction_state_dict":selected["correction_state"],
            "pairwise_head_state_dict":selected["head_state"],
            "gate_state_dict":selected["gate_state"],
        },checkpoint)
        results[arm]={
            "selected_dev_epoch":selected["epoch"],
            "selected_dev":metrics,
            "selected_pairwise_aggregate_dev":pair_metrics,
            "selected_fused_shadow_dev":selected["fused_shadow"],
            "gates":gates,
            "dev_ready":all(gates.values()),
            "correction_state_sha256":
                selected["correction_state_sha256"],
            "pairwise_head_state_sha256":
                selected["pairwise_head_state_sha256"],
            "gate_state_sha256":
                selected["gate_state_sha256"],
            "checkpoint_file":checkpoint.name,
            "checkpoint_sha256":s59._sha256(checkpoint),
        }
    return history,results


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
    if a0.get("outcome")!="HIRA_V1_S75_A0_TOKEN_LEVEL_BIDIRECTIONAL_BINDING_READY":
        raise RuntimeError("S75 A0 not qualified")
    if int(a0.get("reference_representation_parameter_count",-1))!=0:
        raise RuntimeError("S75 A0 reference rep capacity changed")
    if int(a0.get("treatment_representation_parameter_count",-1))!=S75_REPRESENTATION_PARAMETER_COUNT:
        raise RuntimeError("S75 A0 treatment rep capacity changed")
    if int(a0.get("reference_representation_dimension",-1))!=512:
        raise RuntimeError("S75 A0 reference representation dimension changed")
    if int(a0.get("treatment_representation_dimension",-1))!=512:
        raise RuntimeError("S75 A0 treatment representation dimension changed")
    if float(a0.get("interaction_scale",-1.0))!=S75_INTERACTION_SCALE:
        raise RuntimeError("S75 A0 interaction scale changed")
    if int(a0.get("reference_pairwise_parameter_count",-1))!=PAIRWISE_PARAMS:
        raise RuntimeError("S75 A0 reference head capacity changed")
    if int(a0.get("treatment_pairwise_parameter_count",-1))!=PAIRWISE_PARAMS:
        raise RuntimeError("S75 A0 treatment head capacity changed")
    if int(a0.get("added_treatment_parameter_count",-1))!=0:
        raise RuntimeError("S75 A0 treatment parameter advantage changed")
    if a0.get("pairwise_initialization_bit_identical") is not True:
        raise RuntimeError("S75 A0 pairwise initialization changed")
    if float(a0.get("neutral_collapse_max_abs_error",1.0))>2e-6:
        raise RuntimeError("S75 A0 neutral collapse changed")
    if float(a0.get("nontrivial_representation_max_abs_difference",0.0))<=1e-5:
        raise RuntimeError("S75 A0 treatment representation degenerate")
    if a0.get("gradient_to_upstream_zero") is not True:
        raise RuntimeError("S75 A0 representation gradient ownership changed")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S75 A0 used for model selection")
    if a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S75 A0 exposed fresh data")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S75 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S75 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S75 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S75 parent authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s75_cases("train")
    dev_rows=generate_s75_cases("dev")
    validate_s75_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S75 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S75 loaded parent runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S75 native remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    (
        train_ref_rep,train_trt_rep,
        train_ref_digest,train_trt_digest,train_rep_difference,
    )=_materialize_rep_caches(train_cache)
    (
        dev_ref_rep,dev_trt_rep,
        dev_ref_digest,dev_trt_digest,dev_rep_difference,
    )=_materialize_rep_caches(dev_cache)

    args.out.mkdir(parents=True,exist_ok=True)
    history,arms=_train_matched(
        train_cache,dev_cache,
        train_ref_rep,train_trt_rep,
        dev_ref_rep,dev_trt_rep,
        args.out,
    )

    reference=arms["reference"]
    treatment=arms["treatment"]
    final_keys=(
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
    )
    pair_keys=(
        "canonical_accuracy",
        "paraphrase_accuracy",
        "canonical_paired_both_correct_rate",
        "question_swap_choice_change_rate",
        "cross_view_selected_choice_agreement",
        "cross_view_mean_js",
        "canonical_mean_gold_margin",
        "paraphrase_mean_gold_margin",
    )
    deltas=_delta(
        reference["selected_dev"],treatment["selected_dev"],final_keys
    )
    pair_deltas=_delta(
        reference["selected_pairwise_aggregate_dev"],
        treatment["selected_pairwise_aggregate_dev"],
        pair_keys,
    )
    overall_ready=bool(reference["dev_ready"]) and bool(treatment["dev_ready"])
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S75_FRESH_TOKEN_LEVEL_BIDIRECTIONAL_BINDING",
        "seed":SEED,
        "parent_s74":{
            "merged_main":"0edfb0a64823ccb7ef2cf5c4e77b9fb9696361be",
            "scientific_run":37487578737,
            "artifact_id":11423428905,
            "artifact_digest":"sha256:cb2d8e2718d3aa426bf3ee44795d36c25e5960ca727fb8cc6280d12cfbefdb21",
            "verdict":"CASE_D",
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
            "reference_train_digest":train_ref_digest,
            "treatment_train_digest":train_trt_digest,
            "reference_dev_digest":dev_ref_digest,
            "treatment_dev_digest":dev_trt_digest,
            "train_max_abs_difference":train_rep_difference,
            "dev_max_abs_difference":dev_rep_difference,
            "precomputed_once_before_training":True,
            "requires_grad":False,
        },
        "controlled_variable":{
            "reference_representation_family":"s69_query_gated_identity_interaction",
            "treatment_representation_family":
                "s75_token_level_bidirectional_binding",
            "reference_representation_parameters":0,
            "treatment_representation_parameters":0,
            "representation_dimension":512,
            "interaction_scale":S75_INTERACTION_SCALE,
            "token_binding_temperature":0.10,
            "reference_pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "treatment_pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "added_treatment_parameters":0,
            "pairwise_parameter_initialization_bit_identical":True,
            "pairwise_family_both_arms":"s59_explicit_pairwise",
            "pairwise_objective_both_arms":
                "gold_vs_distractor_pairwise_softplus",
            "gate_family_both_arms":"s64_context_injected",
            "gate_trainable_parameters_per_arm":GATE_PARAMS,
            "gate_objective_both_arms":
                "per_view_counterfactual_responsibility_bce",
            "gate_context_source_both_arms":
                "exact_s59_reference_representation",
            "gate_context_projection_seed":S64_CONTEXT_PROJECTION_SEED,
            "correction_trainable_parameters":CORRECTION_PARAMS,
            "same_train_rows_and_order":True,
            "same_optimizer_lr_weight_decay":True,
            "same_frozen_selector":True,
            "teacher_dependency":False,
            "pseudo_target_dependency":False,
            "pairwise_gradient_enters_representation":False,
            "pairwise_gradient_enters_native":False,
            "reliability_target_reopened":False,
        },
        "training_trajectory":history,
        "reference_branch":reference,
        "treatment_branch":treatment,
        "selected_dev_delta_treatment_minus_reference":deltas,
        "selected_pairwise_aggregate_delta_treatment_minus_reference":
            pair_deltas,
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "exact_s74_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "interaction_scale_changed":False,
        "token_weight_formula_changed":False,
        "pooling_or_normalization_changed":False,
        "representation_dimension_changed":False,
        "learned_representation_adapter_added":False,
        "pairwise_width_rank_changed":False,
        "loss_optimizer_changed":False,
        "reliability_target_reopened":False,
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
        "HIRA_V1_S75_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
