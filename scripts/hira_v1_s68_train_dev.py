from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_persisted_native_authority import load_native_authority, verify_file_sha256
from nmd.v1_context_modulated_pairwise_head import (
    S68_CONTEXT_DIMENSION,
    S68_CONTEXT_PROJECTION_SEED,
    S68_MODULATION_SCALE,
    S68_PAIRWISE_PARAMETER_COUNT,
    ContextModulatedPairwiseHead,
    context_modulated_pairwise_loss,
)
from nmd.v1_contextual_reliability_gate import (
    S64_CONTEXT_PROJECTION_SEED,
    S64_GATE_PARAMETER_COUNT,
    ContextInjectedReliabilityGate,
)
from nmd.v1_per_view_responsibility import per_view_responsibility_loss
from nmd.v1_s68_authority import generate_s68_cases, validate_s68_partitions

import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59
import hira_v1_s66_train_dev as s66


SCHEMA_VERSION="hira-v1-s68-context-modulated-pairwise-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S68_CONTEXT_MODULATED_PAIRWISE_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S68_CONTEXT_MODULATED_PAIRWISE_DEV_READY"

SEED=89_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=S68_PAIRWISE_PARAMETER_COUNT
GATE_PARAMS=S64_GATE_PARAMETER_COUNT

A0_RUN=37388612141
A0_ARTIFACT_ID=11380079442
A0_ARTIFACT_DIGEST="sha256:06cd73759293281780a4ba8e9aa1c375dd95ea1a632061c2351cf21fd5787da5"


def _matched_param_state(a,b)->bool:
    pa=dict(a.named_parameters())
    pb=dict(b.named_parameters())
    return pa.keys()==pb.keys() and all(torch.equal(pa[k],pb[k]) for k in pa)


def _pairwise_aggregate_metrics(head,rep_cache):
    canonical_correct=paraphrase_correct=0
    paired_both=question_swap=0
    agreement_sum=js_sum=0.0
    canonical_margin_sum=paraphrase_margin_sum=0.0
    semantic_cases=queries=0
    max_antisym=max_diag=max_mass=0.0

    with torch.no_grad():
        for canonical,paraphrase,n,rep_c,rep_p in rep_cache:
            sc=head.aggregate_logits(rep_c)
            sp=head.aggregate_logits(rep_p)
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S68 pairwise aggregate paired gold mismatch")

            pc=sc.argmax(dim=-1)
            pp=sp.argmax(dim=-1)
            canonical_correct+=int((pc==gold).sum())
            paraphrase_correct+=int((pp==gold).sum())

            rows=pc.reshape(n,2)
            gold_rows=gold.reshape(n,2)
            paired_both+=int(((rows==gold_rows).all(-1)).sum())
            question_swap+=int((rows[:,0]!=rows[:,1]).sum())

            q=2*n
            agreement_sum+=float(selected_choice_agreement(sc,sp))*q
            js_sum+=float(symmetric_js_divergence(sc,sp))*q

            for score,rep in ((sc,rep_c),(sp,rep_p)):
                pair=head.pairwise_logits(rep)
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
                    float((torch.softmax(score,-1).sum(-1)-1.0).abs().max()),
                )

            batch=torch.arange(sc.shape[0],device=sc.device)
            mask=torch.arange(sc.shape[-1],device=sc.device)[None,:].ne(gold[:,None])
            canonical_margin_sum+=float(
                (
                    sc[batch,gold]
                    -sc.masked_fill(~mask,float("-inf")).max(-1).values
                ).sum()
            )
            paraphrase_margin_sum+=float(
                (
                    sp[batch,gold]
                    -sp.masked_fill(~mask,float("-inf")).max(-1).values
                ).sum()
            )

            semantic_cases+=n
            queries+=q

    if queries<1:
        raise RuntimeError("S68 pairwise aggregate metrics empty")

    return {
        "semantic_cases":semantic_cases,
        "canonical_queries":queries,
        "paraphrase_queries":queries,
        "canonical_accuracy":canonical_correct/queries,
        "paraphrase_accuracy":paraphrase_correct/queries,
        "canonical_paired_both_correct_rate":paired_both/semantic_cases,
        "question_swap_choice_change_rate":question_swap/semantic_cases,
        "cross_view_selected_choice_agreement":agreement_sum/queries,
        "cross_view_mean_js":js_sum/queries,
        "canonical_mean_gold_margin":canonical_margin_sum/queries,
        "paraphrase_mean_gold_margin":paraphrase_margin_sum/queries,
        "antisymmetry_max_abs_error":max_antisym,
        "diagonal_max_abs_error":max_diag,
        "max_probability_mass_error":max_mass,
    }


def _train_matched(train_cache,dev_cache,train_rep,dev_rep,out_dir):
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)

    reference_head=ContextModulatedPairwiseHead(
        use_joint_context=False,trainable=True
    )
    treatment_head=ContextModulatedPairwiseHead(
        use_joint_context=True,trainable=True
    )
    reference_gate=ContextInjectedReliabilityGate(
        use_context=True,trainable=True
    )
    treatment_gate=ContextInjectedReliabilityGate(
        use_context=True,trainable=True
    )

    cparams=op.correction_parameters()
    rhparams=[reference_head.A,reference_head.u,reference_head.G]
    thparams=[treatment_head.A,treatment_head.u,treatment_head.G]

    gate_names=("W_phi","b_phi","w_out","b_out")
    rg_named=dict(reference_gate.named_parameters())
    tg_named=dict(treatment_gate.named_parameters())
    rgparams=[rg_named[k] for k in gate_names]
    tgparams=[tg_named[k] for k in gate_names]

    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S68 correction capacity changed")
    if sum(p.numel() for p in rhparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S68 reference pairwise capacity changed")
    if sum(p.numel() for p in thparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S68 treatment pairwise capacity changed")
    if not _matched_param_state(reference_head,treatment_head):
        raise RuntimeError("S68 pairwise initialization changed")
    if not torch.equal(
        reference_head.context_projection,
        treatment_head.context_projection,
    ):
        raise RuntimeError("S68 pairwise context projection changed")
    if sum(p.numel() for p in rgparams)!=GATE_PARAMS:
        raise RuntimeError("S68 reference gate capacity changed")
    if sum(p.numel() for p in tgparams)!=GATE_PARAMS:
        raise RuntimeError("S68 treatment gate capacity changed")
    if any(not torch.equal(rg_named[k],tg_named[k]) for k in gate_names):
        raise RuntimeError("S68 gate initialization changed")
    if not torch.equal(
        reference_gate.context_projection,
        treatment_gate.context_projection,
    ):
        raise RuntimeError("S68 downstream gate context path changed")

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
    reference_gate_phi_ever_live=False
    treatment_gate_phi_ever_live=False
    reference_G_ever_live=False
    treatment_G_ever_live=False

    print("HIRA_V1_S68_MATCHED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)

        base_loss_sum=0.0
        ref_pair_loss_sum=trt_pair_loss_sum=0.0
        ref_gate_loss_sum=trt_gate_loss_sum=0.0
        ref_resp_disagree_sum=trt_resp_disagree_sum=0.0
        ref_resp_count=trt_resp_count=0
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            _c2,_p2,n2,rep_c,rep_p=train_rep[index]
            if n2!=n:
                raise RuntimeError("S68 representation/cache batch mismatch")

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
                raise RuntimeError("S68 correction gradient vanished")
            for p,g in zip(cparams,cgrads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(cparams,s35.GRAD_CLIP)
            corr_opt.step()

            ref_head_opt.zero_grad(set_to_none=True)
            rlc,_=context_modulated_pairwise_loss(
                reference_head,rep_c,canonical.gold
            )
            rlp,_=context_modulated_pairwise_loss(
                reference_head,rep_p,paraphrase.gold
            )
            ref_pair_loss=0.5*(rlc+rlp)
            rhgrads=torch.autograd.grad(
                ref_pair_loss,rhparams,allow_unused=True
            )
            if not all(
                g is not None and bool(torch.isfinite(g).all())
                for g in rhgrads
            ):
                raise RuntimeError("S68 reference pairwise gradient invalid")
            if not all(float(g.detach().abs().sum())>0 for g in rhgrads):
                raise RuntimeError("S68 reference pairwise gradient vanished")
            reference_G_ever_live=True
            for p,g in zip(rhparams,rhgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(rhparams,s35.GRAD_CLIP)
            ref_head_opt.step()

            trt_head_opt.zero_grad(set_to_none=True)
            tlc,_=context_modulated_pairwise_loss(
                treatment_head,rep_c,canonical.gold
            )
            tlp,_=context_modulated_pairwise_loss(
                treatment_head,rep_p,paraphrase.gold
            )
            trt_pair_loss=0.5*(tlc+tlp)
            thgrads=torch.autograd.grad(
                trt_pair_loss,thparams,allow_unused=True
            )
            if not all(
                g is not None and bool(torch.isfinite(g).all())
                for g in thgrads
            ):
                raise RuntimeError("S68 treatment pairwise gradient invalid")
            if not all(float(g.detach().abs().sum())>0 for g in thgrads):
                raise RuntimeError("S68 treatment pairwise gradient vanished")
            treatment_G_ever_live=True
            for p,g in zip(thparams,thgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(thparams,s35.GRAD_CLIP)
            trt_head_opt.step()

            with torch.no_grad():
                _rc,_sc,fused_c=s50._branch_logits(
                    op,"treatment",canonical
                )
                _rp,_sp,fused_p=s50._branch_logits(
                    op,"treatment",paraphrase
                )
                ref_pair_c=reference_head.aggregate_logits(rep_c)
                ref_pair_p=reference_head.aggregate_logits(rep_p)
                trt_pair_c=treatment_head.aggregate_logits(rep_c)
                trt_pair_p=treatment_head.aggregate_logits(rep_p)

            ref_gate_opt.zero_grad(set_to_none=True)
            ref_gate_loss,ref_diag=per_view_responsibility_loss(
                reference_gate,
                fused_c,fused_p,
                ref_pair_c,ref_pair_p,
                rep_c,rep_p,
                canonical.gold,
            )
            rggrads=torch.autograd.grad(
                ref_gate_loss,rgparams,allow_unused=True
            )
            if not all(
                g is not None and bool(torch.isfinite(g).all())
                for g in rggrads
            ):
                raise RuntimeError("S68 reference gate gradient invalid")
            if float(rggrads[2].detach().abs().sum())<=0.0 or float(
                rggrads[3].detach().abs().sum()
            )<=0.0:
                raise RuntimeError("S68 reference gate output gradient vanished")
            if float(rggrads[0].detach().abs().sum())>0.0 or float(
                rggrads[1].detach().abs().sum()
            )>0.0:
                reference_gate_phi_ever_live=True
            for p,g in zip(rgparams,rggrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(rgparams,s35.GRAD_CLIP)
            ref_gate_opt.step()

            trt_gate_opt.zero_grad(set_to_none=True)
            trt_gate_loss,trt_diag=per_view_responsibility_loss(
                treatment_gate,
                fused_c,fused_p,
                trt_pair_c,trt_pair_p,
                rep_c,rep_p,
                canonical.gold,
            )
            tggrads=torch.autograd.grad(
                trt_gate_loss,tgparams,allow_unused=True
            )
            if not all(
                g is not None and bool(torch.isfinite(g).all())
                for g in tggrads
            ):
                raise RuntimeError("S68 treatment gate gradient invalid")
            if float(tggrads[2].detach().abs().sum())<=0.0 or float(
                tggrads[3].detach().abs().sum()
            )<=0.0:
                raise RuntimeError("S68 treatment gate output gradient vanished")
            if float(tggrads[0].detach().abs().sum())>0.0 or float(
                tggrads[1].detach().abs().sum()
            )>0.0:
                treatment_gate_phi_ever_live=True
            for p,g in zip(tgparams,tggrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(tgparams,s35.GRAD_CLIP)
            trt_gate_opt.step()

            base_loss_sum+=float(base_loss.detach())*n
            ref_pair_loss_sum+=float(ref_pair_loss.detach())*n
            trt_pair_loss_sum+=float(trt_pair_loss.detach())*n
            ref_gate_loss_sum+=float(ref_gate_loss.detach())*n
            trt_gate_loss_sum+=float(trt_gate_loss.detach())*n

            rcnt=int(ref_diag["target_count"])
            tcnt=int(trt_diag["target_count"])
            ref_resp_disagree_sum+=float(
                ref_diag["target_disagreement_fraction"]
            )*rcnt
            trt_resp_disagree_sum+=float(
                trt_diag["target_disagreement_fraction"]
            )*tcnt
            ref_resp_count+=rcnt
            trt_resp_count+=tcnt
            case_count+=n

        if not reference_G_ever_live or not treatment_G_ever_live:
            raise RuntimeError("S68 pairwise modulation never received gradient")
        if not reference_gate_phi_ever_live:
            raise RuntimeError("S68 reference gate encoder never received gradient")
        if not treatment_gate_phi_ever_live:
            raise RuntimeError("S68 treatment gate encoder never received gradient")

        fused_shadow=s50._private_metrics(op,"treatment",dev_cache)
        reference=s66._hybrid_metrics(
            op,reference_head,reference_gate,dev_rep
        )
        treatment=s66._hybrid_metrics(
            op,treatment_head,treatment_gate,dev_rep
        )
        ref_pair_train=s59._pairwise_train_diagnostics(
            reference_head,train_rep
        )
        trt_pair_train=s59._pairwise_train_diagnostics(
            treatment_head,train_rep
        )
        ref_pair_agg=_pairwise_aggregate_metrics(
            reference_head,dev_rep
        )
        trt_pair_agg=_pairwise_aggregate_metrics(
            treatment_head,dev_rep
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
            "train_mean_base_correction_loss":base_loss_sum/case_count,
            "train_mean_reference_pairwise_loss":
                ref_pair_loss_sum/case_count,
            "train_mean_treatment_pairwise_loss":
                trt_pair_loss_sum/case_count,
            "train_mean_reference_gate_loss":
                ref_gate_loss_sum/case_count,
            "train_mean_treatment_gate_loss":
                trt_gate_loss_sum/case_count,
            "train_reference_responsibility_disagreement_fraction":
                ref_resp_disagree_sum/max(1,ref_resp_count),
            "train_treatment_responsibility_disagreement_fraction":
                trt_resp_disagree_sum/max(1,trt_resp_count),
            "reference_G_gradient_ever_live":reference_G_ever_live,
            "treatment_G_gradient_ever_live":treatment_G_ever_live,
            "reference_gate_phi_gradient_ever_live":
                reference_gate_phi_ever_live,
            "treatment_gate_phi_gradient_ever_live":
                treatment_gate_phi_ever_live,
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
            "HIRA_V1_S68_EPOCH="+json.dumps({
                "epoch":epoch,
                "reference_pairwise_gold_pair_accuracy":
                    ref_pair_train["gold_pair_accuracy"],
                "treatment_pairwise_gold_pair_accuracy":
                    trt_pair_train["gold_pair_accuracy"],
                "reference_pairwise_dev":ref_pair_agg,
                "treatment_pairwise_dev":trt_pair_agg,
                "reference":reference,
                "treatment":treatment,
                "correction_state_sha256":corr_digest,
                "reference_pairwise_head_state_sha256":ref_head_digest,
                "treatment_pairwise_head_state_sha256":trt_head_digest,
            },sort_keys=True),
            flush=True,
        )

    if best["reference"] is None or best["treatment"] is None:
        raise RuntimeError("S68 selector failed")

    results={}
    for arm,use_joint in (("reference",False),("treatment",True)):
        selected=best[arm]

        replay_op=JointStateQueryOptionPrivateCorrectionFork(
            train_correction=True
        )
        replay_op.load_correction_state_dict(
            selected["correction_state"],freeze=True
        )
        replay_head=ContextModulatedPairwiseHead(
            use_joint_context=use_joint,trainable=True
        )
        replay_head.load_state_dict_exact(
            selected["head_state"],freeze=True
        )
        replay_gate=ContextInjectedReliabilityGate(
            use_context=True,trainable=True
        )
        replay_gate.load_state_dict_exact(
            selected["gate_state"],freeze=True
        )

        metrics=s66._hybrid_metrics(
            replay_op,replay_head,replay_gate,dev_rep
        )
        pair_metrics=_pairwise_aggregate_metrics(
            replay_head,dev_rep
        )

        for key,value in selected["metrics"].items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(
                        f"S68 {arm} selected replay changed: {key}"
                    )
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(
                    f"S68 {arm} selected replay changed: {key}"
                )
        for key,value in selected["pairwise_aggregate"].items():
            if abs(float(pair_metrics[key])-float(value))>1e-12:
                raise RuntimeError(
                    f"S68 {arm} pairwise replay changed: {key}"
                )

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        torch.save({
            "schema_version":"hira-v1-s68-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "selected_dev_epoch":selected["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "pairwise_family":"s68_context_modulated",
            "pairwise_use_joint_context":bool(use_joint),
            "pairwise_context_projection_seed":
                S68_CONTEXT_PROJECTION_SEED,
            "pairwise_modulation_scale":S68_MODULATION_SCALE,
            "gate_parameter_count":GATE_PARAMS,
            "gate_family":"s64_context_injected",
            "gate_objective":
                "per_view_counterfactual_responsibility_bce",
            "gate_context_projection_seed":
                S64_CONTEXT_PROJECTION_SEED,
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
    return {
        k:float(treatment[k])-float(reference[k])
        for k in keys
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S68_A0_CONTEXT_MODULATED_PAIRWISE_READY":
        raise RuntimeError("S68 A0 not qualified")
    if int(a0.get("reference_pairwise_parameter_count",-1))!=PAIRWISE_PARAMS:
        raise RuntimeError("S68 A0 reference capacity changed")
    if int(a0.get("treatment_pairwise_parameter_count",-1))!=PAIRWISE_PARAMS:
        raise RuntimeError("S68 A0 treatment capacity changed")
    if int(a0.get("added_treatment_parameter_count",-1))!=0:
        raise RuntimeError("S68 A0 parameter advantage changed")
    if a0.get("pairwise_trainable_tensor_names")!=["A","G","u"]:
        raise RuntimeError("S68 A0 pairwise tensor surface changed")
    if int(a0.get("context_dimension",-1))!=S68_CONTEXT_DIMENSION:
        raise RuntimeError("S68 A0 context dimension changed")
    if int(a0.get("context_projection_seed",-1))!=S68_CONTEXT_PROJECTION_SEED:
        raise RuntimeError("S68 A0 context projection changed")
    if float(a0.get("modulation_scale",-1.0))!=S68_MODULATION_SCALE:
        raise RuntimeError("S68 A0 modulation scale changed")
    if a0.get("arm_parameter_initialization_bit_identical") is not True:
        raise RuntimeError("S68 A0 matched initialization changed")
    if a0.get("base_A_u_exact_s59_initialization") is not True:
        raise RuntimeError("S68 A0 base S59 initialization changed")
    if a0.get("G_initial_exact_zero") is not True:
        raise RuntimeError("S68 A0 G initialization changed")
    if float(a0.get("initial_real_cache_s59_identity_max_abs_error",1.0))!=0.0:
        raise RuntimeError("S68 A0 initial S59 identity changed")
    if float(a0.get("reference_treatment_context_max_abs_difference",0.0))<=1e-7:
        raise RuntimeError("S68 A0 context intervention vanished")
    if float(a0.get("joint_context_ablation_max_abs_error",1.0))!=0.0:
        raise RuntimeError("S68 A0 context ablation contract changed")
    if float(a0.get("reference_gradient_G_l1",0.0))<=0.0:
        raise RuntimeError("S68 A0 reference G gradient vanished")
    if float(a0.get("treatment_gradient_G_l1",0.0))<=0.0:
        raise RuntimeError("S68 A0 treatment G gradient vanished")
    if float(a0.get("reference_treatment_G_gradient_l1_difference",0.0))<=1e-7:
        raise RuntimeError("S68 A0 context gradient intervention vanished")
    if a0.get("representation_received_pairwise_gradient") is not False:
        raise RuntimeError("S68 A0 representation ownership changed")
    if a0.get("correction_received_pairwise_gradient") is not False:
        raise RuntimeError("S68 A0 correction ownership changed")
    if a0.get("teacher_dependency") is not False:
        raise RuntimeError("S68 A0 teacher dependency changed")
    if a0.get("pseudo_target_dependency") is not False:
        raise RuntimeError("S68 A0 pseudo-target dependency changed")
    if a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S68 A0 exposed fresh TRAIN DEV")

    authority=json.loads(
        args.authority_receipt.read_text(encoding="utf-8")
    )
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S68 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S68 parent native runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S68 parent native checkpoint changed")
    if authority.get("dev_encoded") is not False:
        raise RuntimeError("S68 parent native authority encoded DEV")
    if authority.get("dev_scored") is not False:
        raise RuntimeError("S68 parent native authority scored DEV")
    verify_file_sha256(
        args.authority_checkpoint,expected_checkpoint
    )

    train_rows=generate_s68_cases("train")
    dev_rows=generate_s68_cases("dev")
    validate_s68_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S68 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S68 loaded native runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S68 native runtime remained trainable")

    train_cache,train_digest=s50._materialize_cache(
        runtime,train_rows
    )
    dev_cache,dev_digest=s50._materialize_cache(
        runtime,dev_rows
    )
    del runtime

    train_rep,train_rep_digest=s59._materialize_rep_cache(
        train_cache
    )
    dev_rep,dev_rep_digest=s59._materialize_rep_cache(
        dev_cache
    )
    if any(
        rep.requires_grad
        for entries in (train_rep,dev_rep)
        for _c,_p,_n,rc,rp in entries
        for rep in (rc,rp)
    ):
        raise RuntimeError("S68 representation gained gradient")

    args.out.mkdir(parents=True,exist_ok=True)
    history,arms=_train_matched(
        train_cache,dev_cache,train_rep,dev_rep,args.out
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

    final_delta=_delta(
        reference["selected_dev"],
        treatment["selected_dev"],
        final_keys,
    )
    pair_delta=_delta(
        reference["selected_pairwise_aggregate_dev"],
        treatment["selected_pairwise_aggregate_dev"],
        pair_keys,
    )

    outcome=(
        OUTCOME_READY
        if reference["dev_ready"] and treatment["dev_ready"]
        else OUTCOME_COMPLETE
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":
            "V1_S68_FRESH_CONTEXT_MODULATED_PAIRWISE",
        "seed":SEED,
        "parent_s67":{
            "merged_main":
                "942871bb44bbef3e28e74698f161b02c6796cdcb",
            "scientific_run":37336779415,
            "verdict":"CASE_B",
            "actions_artifact_available":False,
            "raw_receipt_frozen_in_repo":True,
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
            "train_digest":train_rep_digest,
            "dev_digest":dev_rep_digest,
            "requires_grad":False,
            "source":
                "s59_detached_identity_plus_joint_state_query_option_context",
        },
        "controlled_variable":{
            "reference_pairwise_family":
                "s68_context_modulated_identity_only_set_context",
            "treatment_pairwise_family":
                "s68_context_modulated_full_joint_set_context",
            "reference_pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "treatment_pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "added_treatment_pairwise_parameters":0,
            "pairwise_context_dimension":S68_CONTEXT_DIMENSION,
            "pairwise_context_projection_seed":
                S68_CONTEXT_PROJECTION_SEED,
            "pairwise_modulation_scale":S68_MODULATION_SCALE,
            "pairwise_objective_both_arms":
                "gold_vs_distractor_pairwise_softplus",
            "pairwise_parameter_initialization_bit_identical":True,
            "gate_family_both_arms":"s64_context_injected",
            "gate_objective_both_arms":
                "per_view_counterfactual_responsibility_bce",
            "gate_trainable_parameters_per_arm":GATE_PARAMS,
            "gate_context_projection_seed":
                S64_CONTEXT_PROJECTION_SEED,
            "gate_parameter_initialization_bit_identical":True,
            "correction_trainable_parameters":CORRECTION_PARAMS,
            "shared_correction_trajectory":True,
            "same_train_rows_and_order":True,
            "same_optimizer_lr_weight_decay":True,
            "same_frozen_selector":True,
            "teacher_dependency":False,
            "pseudo_target_dependency":False,
            "oracle_alpha_target_reopened":False,
        },
        "training_trajectory":history,
        "reference_branch":reference,
        "treatment_branch":treatment,
        "selected_dev_delta_treatment_minus_reference":final_delta,
        "selected_pairwise_aggregate_delta_treatment_minus_reference":
            pair_delta,
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "exact_s67_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "context_projection_seed_changed":False,
        "context_dimension_changed":False,
        "modulation_scale_changed":False,
        "additive_context_bias_added":False,
        "pairwise_hidden_width_changed":False,
        "pairwise_rank_changed":False,
        "context_source_retrofit_performed":False,
        "loss_coefficient_changed":False,
        "optimizer_lr_weight_decay_changed":False,
        "reliability_target_reopened":False,
        "native_retraining_performed":False,
        "selector_changed":False,
        "second_dev_run_performed":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
    }

    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S68_TRAIN_DEV_RECEIPT="
        +json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
