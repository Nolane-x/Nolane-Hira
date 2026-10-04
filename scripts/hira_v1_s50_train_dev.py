from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch
from torch import Tensor
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import a13_lora_state_dict, iter_a13_lora_modules
from nmd.v1_evidence_fusion import fused_gold_vs_max_wrong_margin
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_norm_balanced_gradient import apply_gradient_update, norm_balanced_gradient_update
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
from nmd.v1_s49_authority import generate_s49_cases
from nmd.v1_s50_authority import generate_s50_cases, validate_s50_partitions
from nmd.v1_shared_native_private_readouts import (
    FrozenSharedNativeEvidence,
    correction_initialization_exact,
    freeze_shared_native_evidence,
    fused_private_logits,
    reference_private_logits,
    treatment_private_logits,
)
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s45_a0_cross_view_consistent_private_correction as s45a0
import hira_v1_s49_train_dev as s49
from hira_v1_s50_a0_shared_native_forked_readouts import cases as s50_a0_cases


SCHEMA_VERSION="hira-v1-s50-shared-native-forked-private-readouts-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S50_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S50_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_DEV_READY"

SEED=71_001
NATIVE_EPOCHS=24
PRIVATE_EPOCHS=24
BATCH_SIZE=16
CORRECTION=114_688


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def _states(row): return (row.state_a,row.state_b)
def _questions(row): return (row.question_a1,row.question_a2,row.question_b1,row.question_b2)
def _options(row): return (*row.option_texts,*row.option_aliases)


def _assert_fresh(train_rows,dev_rows):
    # Reuse S49's guard against S0-S48 and S49-A0.
    s49._assert_fresh(train_rows,dev_rows)

    current=(*train_rows,*dev_rows)
    cs={x for r in current for x in _states(r)}
    cq={x for r in current for x in _questions(r)}
    co={x for r in current for x in _options(r)}

    prior=(*generate_s49_cases("train"),*generate_s49_cases("dev"))
    if cs & {x for r in prior for x in _states(r)}:
        raise RuntimeError("S50 state overlap with exposed S49 TRAIN DEV")
    if cq & {x for r in prior for x in _questions(r)}:
        raise RuntimeError("S50 question overlap with exposed S49 TRAIN DEV")
    if co & {x for r in prior for x in _options(r)}:
        raise RuntimeError("S50 option overlap with exposed S49 TRAIN DEV")

    a0=s50_a0_cases()
    a0s={x for r in a0 for x in (r.state_a,r.state_b)}
    a0q={x for r in a0 for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    a0o=set()
    for case in a0:
        opts,_ga,_gb=case.option_pack()
        for option in opts:
            a0o.add(option.criterion_text)
            a0o.update(option.aliases)
    if cs&a0s:
        raise RuntimeError("S50 state overlap with S50-A0")
    if cq&a0q:
        raise RuntimeError("S50 question overlap with S50-A0")
    if co&a0o:
        raise RuntimeError("S50 option overlap with S50-A0")


def _runtime_state_sha256(runtime)->str:
    d=sha256()
    state=a13_lora_state_dict(runtime.encoder)
    for name in sorted(state):
        t=state[name].detach().cpu().contiguous()
        d.update(name.encode("utf-8"))
        d.update(str(tuple(t.shape)).encode("ascii"))
        d.update(str(t.dtype).encode("ascii"))
        d.update(t.numpy().tobytes())
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S50 projection scorer missing")
    t=scorer.projection.weight.detach().cpu().contiguous()
    d.update(b"projection.weight")
    d.update(str(tuple(t.shape)).encode("ascii"))
    d.update(str(t.dtype).encode("ascii"))
    d.update(t.numpy().tobytes())
    return d.hexdigest()


def _train_shared_native(bundle,manifest,train_rows,out_dir):
    # Fixed seed and one native trajectory only.
    random.seed(SEED)
    torch.manual_seed(SEED)

    frozen=load_hira_v0_m4_bundle(bundle)
    runtime=build_hira_v1_s17_norm_balanced_core(
        frozen.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,
        train_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    enforce_s17_eval(runtime)

    trainable=[p for p in runtime.parameters() if p.requires_grad]
    trainable_count=sum(p.numel() for p in trainable)
    if trainable_count!=HIRA_V1_S17_TOTAL_PARAMETER_COUNT or trainable_count!=49_152:
        raise RuntimeError(f"S50 native trainable count changed: {trainable_count}")
    if s17._original_a13_trainable(runtime)!=0:
        raise RuntimeError("S50 original A13 became trainable")
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S50 projection scorer missing")
    if scorer.projection_trainable_parameter_count!=HIRA_V1_S17_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S50 projection surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S50 HIRACore became trainable")

    optimizer=torch.optim.AdamW(trainable,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    history=[]

    print("HIRA_V1_S50_SHARED_NATIVE_TRAIN_BEGIN",flush=True)

    for epoch in range(1,NATIVE_EPOCHS+1):
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

            (
                _total,primary_block,relation_block,pieces,*_rest
            )=s35._arm_losses(runtime,rows,"native")

            primary_raw=torch.autograd.grad(
                primary_block,trainable,retain_graph=True,allow_unused=True
            )
            relation_raw=torch.autograd.grad(
                relation_block,trainable,allow_unused=True
            )
            primary=[
                torch.zeros_like(p) if g is None else g
                for p,g in zip(trainable,primary_raw)
            ]
            relation=[
                torch.zeros_like(p) if g is None else g
                for p,g in zip(trainable,relation_raw)
            ]
            combined,diag=norm_balanced_gradient_update(
                primary,relation,epsilon=s35.BALANCE_EPSILON
            )
            apply_gradient_update(trainable,combined)
            torch.nn.utils.clip_grad_norm_(trainable,s35.GRAD_CLIP)
            optimizer.step()
            enforce_s17_eval(runtime)

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
            raise RuntimeError("S50 shared native TRAIN state-once changed")

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
            "train_state_view_encodes":state_view_encodes,
            "runtime_state_sha256":_runtime_state_sha256(runtime),
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
        }
        history.append(record)
        print("HIRA_V1_S50_SHARED_NATIVE_EPOCH="+json.dumps(record,sort_keys=True),flush=True)

    # Fixed epoch 24 is the authority. No DEV has been encoded or scored.
    if history[-1]["epoch"]!=24:
        raise RuntimeError("S50 shared native fixed epoch changed")

    native_checkpoint=out_dir/"shared-native-epoch24.pt"
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema_version":"hira-v1-s50-shared-native-epoch24-v1",
        "kind":"shared-native-train-only-fixed-epoch24",
        "seed":SEED,
        "epoch":24,
        "dev_scored_before_private_phase":False,
        "dev_encoded_before_private_phase":False,
        "total_parameter_count":trainable_count,
        "lora_parameter_count":sum(
            p.numel()
            for name,p in runtime.encoder.model.named_parameters()
            if p.requires_grad and ".lora_" in name
        ),
        "projection_parameter_count":HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
        "semantic_revision":str(manifest["semantic_revision"]),
        "initialization_t0_sha256":str(manifest["t0_checkpoint_sha256"]),
        "lora_state_dict":a13_lora_state_dict(runtime.encoder),
        "projection_state_dict":{
            "projection.weight":scorer.projection.weight.detach().cpu().clone()
        },
        "runtime_state_sha256":history[-1]["runtime_state_sha256"],
    },native_checkpoint)

    # Destroy native optimization authority before private cache/branch phase.
    del optimizer
    for p in runtime.parameters():
        p.requires_grad_(False)
        p.grad=None
    enforce_s17_eval(runtime)
    if any(p.requires_grad for p in runtime.parameters()):
        raise RuntimeError("S50 native runtime remained trainable in private phase")

    return runtime,{
        "seed":SEED,
        "fixed_epoch":24,
        "train_semantic_cases":len(train_rows),
        "dev_scored_before_private_phase":False,
        "dev_encoded_before_private_phase":False,
        "history":history,
        "runtime_state_sha256":history[-1]["runtime_state_sha256"],
        "checkpoint_file":native_checkpoint.name,
        "checkpoint_sha256":_sha256(native_checkpoint),
        "trainable_parameter_count_during_native_phase":trainable_count,
        "trainable_parameter_count_after_freeze":sum(
            p.numel() for p in runtime.parameters() if p.requires_grad
        ),
    },native_checkpoint


def _expanded_ids(rows,view):
    return tuple(
        f"{row.case_id}::{view}::{role}"
        for row in rows
        for role in ("a","b")
    )


def _freeze_pair(runtime,rows):
    # Exactly one native output authority per semantic batch. The single
    # encoder/native execution yields both canonical and paraphrase evidence.
    with torch.inference_mode():
        raw_c,raw_p,rel_c,rel_p,sig_c,sig_p,encoded=s45a0._native_outputs(runtime,rows)
        gold,_=s35._gold_tensors(rows,device=rel_c.device)
        opt=encoded["option_tokens"].repeat_interleave(2,dim=0)
        om=encoded["option_mask"].repeat_interleave(2,dim=0)
        vm=encoded["option_view_mask"].repeat_interleave(2,dim=0)

        canonical=freeze_shared_native_evidence(
            case_ids=_expanded_ids(rows,"canonical"),
            gold=gold,
            triadic_logits=raw_c,
            native_logits=rel_c,
            native_signatures=sig_c,
            state_tokens=encoded["state_a_tokens"].repeat_interleave(2,dim=0),
            state_mask=encoded["state_a_mask"].repeat_interleave(2,dim=0),
            option_view_tokens=opt,
            option_view_token_mask=om,
            option_view_mask=vm,
            question_tokens=encoded["question_canonical_tokens"],
            question_mask=encoded["question_canonical_mask"],
        )
        paraphrase=freeze_shared_native_evidence(
            case_ids=_expanded_ids(rows,"paraphrase"),
            gold=gold,
            triadic_logits=raw_p,
            native_logits=rel_p,
            native_signatures=sig_p,
            state_tokens=encoded["state_b_tokens"].repeat_interleave(2,dim=0),
            state_mask=encoded["state_b_mask"].repeat_interleave(2,dim=0),
            option_view_tokens=opt,
            option_view_token_mask=om,
            option_view_mask=vm,
            question_tokens=encoded["question_paraphrase_tokens"],
            question_mask=encoded["question_paraphrase_mask"],
        )
    return canonical,paraphrase


def _cache_digest(pairs)->str:
    d=sha256()
    d.update(b"hira-v1-s50-cache-sequence-v1")
    for i,(canonical,paraphrase,n) in enumerate(pairs):
        d.update(str(i).encode("ascii"))
        d.update(str(n).encode("ascii"))
        d.update(canonical.digest().encode("ascii"))
        d.update(paraphrase.digest().encode("ascii"))
    return d.hexdigest()


def _materialize_cache(runtime,rows):
    pairs=[]
    for start in range(0,len(rows),BATCH_SIZE):
        batch=list(rows[start:start+BATCH_SIZE])
        canonical,paraphrase=_freeze_pair(runtime,batch)
        if canonical.gold.shape!=paraphrase.gold.shape or not torch.equal(canonical.gold,paraphrase.gold):
            raise RuntimeError("S50 canonical/paraphrase cache gold mismatch")
        pairs.append((canonical,paraphrase,len(batch)))
    return pairs,_cache_digest(pairs)


def _branch_logits(op,kind,evidence):
    if kind=="reference":
        relation=reference_private_logits(op,evidence)
        signature=evidence.native_signatures
    elif kind=="treatment":
        relation,signature=treatment_private_logits(op,evidence)
    else:
        raise ValueError(kind)
    fused=fused_private_logits(evidence,relation,epsilon=s35.FUSION_EPSILON)
    return relation,signature,fused


def _private_loss(op,kind,canonical,paraphrase):
    relation_c,_sig_c,_fused_c=_branch_logits(op,kind,canonical)
    relation_p,_sig_p,_fused_p=_branch_logits(op,kind,paraphrase)
    if not torch.equal(canonical.gold,paraphrase.gold):
        raise RuntimeError("S50 private cache gold mismatch")
    ce=0.5*(
        F.cross_entropy(relation_c,canonical.gold)
        +F.cross_entropy(relation_p,paraphrase.gold)
    )
    js=symmetric_js_divergence(relation_c,relation_p)
    loss=s35.BINDING_COEFFICIENT*ce+s35.INVARIANCE_COEFFICIENT*js
    return loss,ce,js


def _private_metrics(op,kind,pairs):
    fused_c_correct=fused_p_correct=0
    relation_c_correct=relation_p_correct=0
    raw_c_correct=raw_p_correct=0
    fused_pair_both=fused_pair_changed=0
    fused_agree_sum=fused_js_sum=0.0
    relation_agree_sum=relation_js_sum=0.0
    raw_agree_sum=0.0
    fused_c_margin_sum=fused_p_margin_sum=0.0
    relation_c_margin_sum=relation_p_margin_sum=0.0
    signature_same_sum=signature_margin_sum=0.0
    signature_count=0
    canonical_decision_loss_sum=0.0
    max_mass_error=0.0
    semantic_cases=queries=0
    full_k=True

    with torch.inference_mode():
        for canonical,paraphrase,n in pairs:
            relation_c,sig_c,fused_c=_branch_logits(op,kind,canonical)
            relation_p,sig_p,fused_p=_branch_logits(op,kind,paraphrase)
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S50 private eval gold mismatch")

            fc=fused_c.argmax(-1); fp=fused_p.argmax(-1)
            rc=relation_c.argmax(-1); rp=relation_p.argmax(-1)
            tc=canonical.triadic_logits.argmax(-1)
            tp=paraphrase.triadic_logits.argmax(-1)

            fused_c_correct+=int((fc==gold).sum())
            fused_p_correct+=int((fp==gold).sum())
            relation_c_correct+=int((rc==gold).sum())
            relation_p_correct+=int((rp==gold).sum())
            raw_c_correct+=int((tc==gold).sum())
            raw_p_correct+=int((tp==gold).sum())

            pairs_fc=fc.reshape(n,2)
            pairs_gold=gold.reshape(n,2)
            fused_pair_both+=int(((pairs_fc==pairs_gold).all(-1)).sum())
            fused_pair_changed+=int((pairs_fc[:,0]!=pairs_fc[:,1]).sum())

            q=2*n
            fused_agree_sum+=float(selected_choice_agreement(fused_c,fused_p))*q
            fused_js_sum+=float(symmetric_js_divergence(fused_c,fused_p))*q
            relation_agree_sum+=float(selected_choice_agreement(relation_c,relation_p))*q
            relation_js_sum+=float(symmetric_js_divergence(relation_c,relation_p))*q
            raw_agree_sum+=float(
                selected_choice_agreement(canonical.triadic_logits,paraphrase.triadic_logits)
            )*q

            fused_c_margin_sum+=float(fused_gold_vs_max_wrong_margin(fused_c,gold).sum())
            fused_p_margin_sum+=float(fused_gold_vs_max_wrong_margin(fused_p,gold).sum())
            relation_c_margin_sum+=float(fused_gold_vs_max_wrong_margin(relation_c,gold).sum())
            relation_p_margin_sum+=float(fused_gold_vs_max_wrong_margin(relation_p,gold).sum())

            same=F.cosine_similarity(sig_c,sig_p,dim=-1)
            cross=torch.einsum(
                "nkd,njd->nkj",F.normalize(sig_c,dim=-1),F.normalize(sig_p,dim=-1)
            )
            k=cross.shape[-1]
            eye=torch.eye(k,dtype=torch.bool,device=cross.device)[None]
            wrong=cross.masked_fill(eye,float("-inf")).amax(-1)
            signature_same_sum+=float(same.sum())
            signature_margin_sum+=float((same-wrong).sum())
            signature_count+=int(same.numel())

            canonical_decision_loss_sum+=float(
                F.cross_entropy(fused_c,gold,reduction="sum")
            )
            for logits in (fused_c,fused_p):
                probs=torch.softmax(logits,dim=-1)
                max_mass_error=max(
                    max_mass_error,float((probs.sum(-1)-1.0).abs().max())
                )

            semantic_cases+=n
            queries+=q
            full_k=full_k and all(
                x.shape[-1]==4
                for x in (
                    fused_c,fused_p,relation_c,relation_p,
                    canonical.triadic_logits,paraphrase.triadic_logits,
                )
            )

    return {
        "semantic_cases":semantic_cases,
        "canonical_queries":queries,
        "paraphrase_queries":queries,
        "fused_canonical_accuracy":fused_c_correct/queries,
        "fused_paraphrase_accuracy":fused_p_correct/queries,
        "fused_canonical_paired_both_correct_rate":fused_pair_both/semantic_cases,
        "fused_question_swap_choice_change_rate":fused_pair_changed/semantic_cases,
        "fused_cross_view_selected_choice_agreement":fused_agree_sum/queries,
        "fused_cross_view_mean_js":fused_js_sum/queries,
        "fused_canonical_mean_gold_margin":fused_c_margin_sum/queries,
        "fused_paraphrase_mean_gold_margin":fused_p_margin_sum/queries,
        "raw_triadic_canonical_accuracy":raw_c_correct/queries,
        "raw_triadic_paraphrase_accuracy":raw_p_correct/queries,
        "raw_triadic_cross_view_agreement":raw_agree_sum/queries,
        "canonical_relation_binding_accuracy":relation_c_correct/queries,
        "paraphrase_relation_binding_accuracy":relation_p_correct/queries,
        "relation_cross_view_agreement":relation_agree_sum/queries,
        "relation_cross_view_mean_js":relation_js_sum/queries,
        "canonical_relation_binding_mean_gold_margin":relation_c_margin_sum/queries,
        "paraphrase_relation_binding_mean_gold_margin":relation_p_margin_sum/queries,
        "mean_same_option_signature_cosine":signature_same_sum/signature_count,
        "mean_signature_same_vs_strongest_wrong_margin":signature_margin_sum/signature_count,
        "mean_canonical_decision_loss":canonical_decision_loss_sum/queries,
        "fused_option_order_flip_rate":0.0,
        "fused_max_probability_mass_error":max_mass_error,
        "full_k":bool(full_k),
        "relation_delta_max_abs":0.0,
        "state_view_encodes":0,
    }


def _private_gates(metrics):
    return {
        "fused_canonical_accuracy_gte_0_85":metrics["fused_canonical_accuracy"]>=0.85,
        "fused_paired_both_correct_gte_0_75":metrics["fused_canonical_paired_both_correct_rate"]>=0.75,
        "fused_question_swap_change_gte_0_80":metrics["fused_question_swap_choice_change_rate"]>=0.80,
        "fused_cross_view_choice_agreement_gte_0_95":metrics["fused_cross_view_selected_choice_agreement"]>=0.95,
        "fused_cross_view_mean_js_lte_0_05":metrics["fused_cross_view_mean_js"]<=0.05,
        "fused_canonical_margin_gte_0_15":metrics["fused_canonical_mean_gold_margin"]>=0.15,
        "canonical_relation_accuracy_gte_0_80":metrics["canonical_relation_binding_accuracy"]>=0.80,
        "canonical_relation_margin_gte_0_15":metrics["canonical_relation_binding_mean_gold_margin"]>=0.15,
        "same_option_signature_cosine_gte_0_90":metrics["mean_same_option_signature_cosine"]>=0.90,
        "signature_margin_gte_0_15":metrics["mean_signature_same_vs_strongest_wrong_margin"]>=0.15,
        "fused_option_order_flip_lte_0_02":metrics["fused_option_order_flip_rate"]<=0.02,
        "fused_probability_mass_error_lte_1e_6":metrics["fused_max_probability_mass_error"]<=1e-6,
        "full_k":bool(metrics["full_k"]),
        "native_optimizer_absent":True,
        "cache_state_view_encodes_zero_during_private":metrics["state_view_encodes"]==0,
    }


def _train_private_branch(kind,op,train_cache,dev_cache,out_dir):
    params=op.correction_parameters()
    if sum(p.numel() for p in params)!=CORRECTION:
        raise RuntimeError(f"S50 {kind} correction capacity changed")
    optimizer=torch.optim.AdamW(params,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)

    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    print(f"HIRA_V1_S50_PRIVATE_{kind.upper()}_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        loss_sum=ce_sum=js_sum=0.0
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            optimizer.zero_grad(set_to_none=True)
            loss,ce,js=_private_loss(op,kind,canonical,paraphrase)
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            if not any(g is not None and float(g.detach().abs().sum())>0.0 for g in grads):
                raise RuntimeError(f"S50 {kind} private gradient vanished")
            for p,g in zip(params,grads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(params,s35.GRAD_CLIP)
            optimizer.step()
            loss_sum+=float(loss.detach())*n
            ce_sum+=float(ce.detach())*n
            js_sum+=float(js.detach())*n
            case_count+=n

        metrics=_private_metrics(op,kind,dev_cache)
        record={
            "epoch":epoch,
            "train_mean_private_loss":loss_sum/case_count,
            "train_mean_private_ce":ce_sum/case_count,
            "train_mean_private_js":js_sum/case_count,
            "dev":metrics,
        }
        history.append(record)
        key=s17._selection_key(epoch,metrics)
        if best_key is None or key>best_key:
            best_key=key
            best_epoch=epoch
            best_state=op.correction_state_dict()
            best_metrics=dict(metrics)

        print(
            "HIRA_V1_S50_PRIVATE_EPOCH="+json.dumps(
                {"branch":kind,**record},sort_keys=True
            ),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S50 {kind} private selection failed")

    op.load_correction_state_dict(best_state,freeze=True)
    selected=_private_metrics(op,kind,dev_cache)
    for key in (
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
        "canonical_relation_binding_mean_gold_margin",
        "paraphrase_relation_binding_mean_gold_margin",
        "mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin",
    ):
        if float(selected[key])!=float(best_metrics[key]):
            raise RuntimeError(f"S50 {kind} selected DEV replay changed: {key}")

    gates=_private_gates(selected)
    checkpoint=out_dir/f"{kind}-private-candidate.pt"
    torch.save({
        "schema_version":"hira-v1-s50-private-candidate-v1",
        "branch":kind,
        "seed":SEED,
        "selected_dev_epoch":best_epoch,
        "correction_parameter_count":CORRECTION,
        "native_parameter_count_in_optimizer":0,
        "correction_state_dict":best_state,
    },checkpoint)

    return {
        "branch":kind,
        "selected_dev_epoch":best_epoch,
        "selected_dev":selected,
        "gates":gates,
        "dev_ready":all(gates.values()),
        "history":history,
        "checkpoint_file":checkpoint.name,
        "checkpoint_sha256":_sha256(checkpoint),
        "parameter_surface":{
            "private_trainable_parameters":CORRECTION,
            "native_trainable_parameters":0,
            "identity_trainable_parameters":(
                0 if kind=="reference" else int(op.identity_parameter_count)
            ),
        },
    },checkpoint


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
        "raw_triadic_canonical_accuracy",
        "raw_triadic_paraphrase_accuracy",
        "canonical_relation_binding_accuracy",
        "paraphrase_relation_binding_accuracy",
        "canonical_relation_binding_mean_gold_margin",
        "paraphrase_relation_binding_mean_gold_margin",
        "relation_cross_view_agreement",
        "relation_cross_view_mean_js",
        "mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin",
    )
    a=reference["selected_dev"]
    b=treatment["selected_dev"]
    return {key:float(b[key])-float(a[key]) for key in keys}


def _identity_diagnostics(op,dev_cache):
    same_sum=margin_sum=0.0
    count=0
    raw_query_delta=0.0
    raw_query_count=0
    with torch.inference_mode():
        for canonical,paraphrase,_n in dev_cache:
            tc,idc=treatment_private_logits(op,canonical)
            _tp,idp=treatment_private_logits(op,paraphrase)
            same=F.cosine_similarity(idc,idp,dim=-1)
            cross=torch.einsum(
                "nkd,njd->nkj",F.normalize(idc,dim=-1),F.normalize(idp,dim=-1)
            )
            k=cross.shape[-1]
            eye=torch.eye(k,dtype=torch.bool,device=cross.device)[None]
            wrong=cross.masked_fill(eye,float("-inf")).amax(-1)
            same_sum+=float(same.sum())
            margin_sum+=float((same-wrong).sum())
            count+=int(same.numel())

            n=canonical.batch_size
            swap=torch.arange(n,device=tc.device).reshape(-1,2)[:,[1,0]].reshape(-1)
            swapped=op.correction_logits_from_state_option(
                native_logits=canonical.native_logits,
                state_tokens=canonical.state_tokens,
                state_mask=canonical.state_mask,
                option_view_tokens=canonical.option_view_tokens,
                option_view_token_mask=canonical.option_view_token_mask,
                option_view_mask=canonical.option_view_mask,
                question_tokens=canonical.question_tokens[swap],
                question_mask=canonical.question_mask[swap],
            )[0]
            raw_query_delta+=float((tc-swapped).abs().sum())
            raw_query_count+=int(tc.numel())
    return {
        "identity_cross_state_view_same_option_cosine":same_sum/count,
        "identity_same_vs_strongest_wrong_margin":margin_sum/count,
        "fixed_identity_raw_query_swap_mean_abs_logit_delta":raw_query_delta/raw_query_count,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S50_A0_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_READY":
        raise RuntimeError("S50-A0 not qualified")
    for key,value in {
        "reference_correction_parameter_count":CORRECTION,
        "treatment_correction_parameter_count":CORRECTION,
        "identity_parameter_count":0,
        "private_optimizer_native_parameter_count":0,
        "cache_tensors_require_grad_count":0,
    }.items():
        if int(a0.get(key,-1))!=value:
            raise RuntimeError(f"S50-A0 surface changed: {key}")
    if a0.get("reference_treatment_correction_initialization_exact") is not True:
        raise RuntimeError("S50-A0 correction initialization mismatch")
    if a0.get("fused_replay_from_cache_only") is not True:
        raise RuntimeError("S50-A0 fused cache replay failed")
    if float(a0.get("branch_order_replay_max_abs_error",-1.0))!=0.0:
        raise RuntimeError("S50-A0 branch-order replay failed")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S50-A0 used for model selection")

    train_rows=generate_s50_cases("train")
    dev_rows=generate_s50_cases("dev")
    validate_s50_partitions(train_rows,dev_rows)
    _assert_fresh(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S50 semantic revision changed")

    args.out.mkdir(parents=True,exist_ok=True)

    # Phase 1: exactly one native TRAIN-only trajectory. No DEV argument enters.
    runtime,native,native_checkpoint=_train_shared_native(
        bundle,manifest,train_rows,args.out
    )
    if native["dev_scored_before_private_phase"] is not False:
        raise RuntimeError("S50 native phase exposed DEV scoring")
    if native["dev_encoded_before_private_phase"] is not False:
        raise RuntimeError("S50 native phase encoded DEV before private phase")

    # Phase 2 starts here. DEV is encoded exactly once into immutable cache;
    # from now onward all branch training/evaluation uses cache only.
    train_cache,train_cache_digest=_materialize_cache(runtime,train_rows)
    dev_cache,dev_cache_digest=_materialize_cache(runtime,dev_rows)

    # Freeze/destroy live runtime reference before private branch optimizers.
    native_runtime_state=native["runtime_state_sha256"]
    del runtime
    if len(train_cache)!=48 or len(dev_cache)!=12:
        raise RuntimeError("S50 cache batch count changed")

    reference_op=PrivateCorrectionRepresentationFork(train_correction=True)
    treatment_op=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    if not correction_initialization_exact(reference_op,treatment_op):
        raise RuntimeError("S50 private correction initialization diverged")
    initial_reference=reference_op.correction_state_dict()
    initial_treatment=treatment_op.correction_state_dict()
    if not all(torch.equal(initial_reference[k],initial_treatment[k]) for k in initial_reference):
        raise RuntimeError("S50 private initial bytes diverged")

    reference,reference_checkpoint=_train_private_branch(
        "reference",reference_op,train_cache,dev_cache,args.out
    )
    treatment,treatment_checkpoint=_train_private_branch(
        "treatment",treatment_op,train_cache,dev_cache,args.out
    )

    identity_diag=_identity_diagnostics(treatment_op,dev_cache)
    delta=_metric_deltas(reference,treatment)

    train_manifest=args.out/"train-manifest.json"
    dev_manifest=args.out/"dev-manifest.json"
    cache_manifest=args.out/"shared-cache-manifest.json"
    train_manifest.write_text(
        json.dumps([r.to_dict() for r in train_rows],indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    dev_manifest.write_text(
        json.dumps([r.to_dict() for r in dev_rows],indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    cache_manifest.write_text(json.dumps({
        "schema_version":"hira-v1-s50-shared-cache-manifest-v1",
        "native_runtime_state_sha256":native_runtime_state,
        "train_cache_digest":train_cache_digest,
        "dev_cache_digest":dev_cache_digest,
        "train_cache_batches":len(train_cache),
        "dev_cache_batches":len(dev_cache),
        "batch_size_semantic_cases":BATCH_SIZE,
        "triadic_logits_cached":True,
        "native_relation_logits_cached":True,
        "native_relation_signatures_cached":True,
        "state_option_query_tensors_cached":True,
        "cache_requires_grad":False,
        "native_runtime_available_to_private_optimizer":False,
    },indent=2,sort_keys=True)+"\n",encoding="utf-8")

    outcome=OUTCOME_READY if treatment["dev_ready"] else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S50_FRESH_SHARED_NATIVE_FORKED_PRIVATE_READOUTS",
        "seed":SEED,
        "native_phase":native,
        "private_training_mechanics":{
            "epochs":PRIVATE_EPOCHS,
            "batch_size_semantic_cases":BATCH_SIZE,
            "cache_batch_count_train":len(train_cache),
            "cache_batch_count_dev":len(dev_cache),
            "lr":s35.LR,
            "weight_decay":s35.WEIGHT_DECAY,
            "grad_clip":s35.GRAD_CLIP,
            "correction_ce_coefficient":s35.BINDING_COEFFICIENT,
            "correction_cross_view_js_coefficient":s35.INVARIANCE_COEFFICIENT,
            "native_optimizer_parameter_count":0,
            "native_runtime_live_graph":False,
            "dev_scored_before_private_phase":False,
            "dev_materialized_at_private_phase_start":True,
            "second_encoder_pass_during_private_training":False,
        },
        "shared_cache":{
            "train_digest":train_cache_digest,
            "dev_digest":dev_cache_digest,
            "train_batches":len(train_cache),
            "dev_batches":len(dev_cache),
            "reference_treatment_same_cache_bytes":True,
            "triadic_logits_cached":True,
            "native_relation_logits_cached":True,
            "native_relation_signatures_cached":True,
            "state_option_query_tensors_cached":True,
            "cache_tensors_require_grad":False,
            "fused_replay_from_cache_only":True,
        },
        "controlled_variable":{
            "reference_private_signature":"question_conditioned_native_relation_signature",
            "treatment_private_signature":"query_free_state_option_identity",
            "raw_query_readout_matched":True,
            "reference_correction_parameters":CORRECTION,
            "treatment_correction_parameters":CORRECTION,
            "identity_trainable_parameters":0,
            "correction_initialization_exact":True,
            "shared_native_authority_by_construction":True,
        },
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "k":4,
            "views_per_option":2,
            "s49_train_dev_rows_used":False,
            "s50_a0_rows_used":False,
        },
        "reference_branch":reference,
        "treatment_branch":treatment,
        "matched_selected_dev_delta_treatment_minus_reference":delta,
        "treatment_identity_diagnostics":identity_diag,
        "post_dev_tuning_performed":False,
        "second_dev_run_performed":False,
        "cache_regenerated_after_dev":False,
        "native_retrained_per_branch":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
        "native_checkpoint_sha256":_sha256(native_checkpoint),
        "reference_checkpoint_sha256":_sha256(reference_checkpoint),
        "treatment_checkpoint_sha256":_sha256(treatment_checkpoint),
        "train_manifest_sha256":_sha256(train_manifest),
        "dev_manifest_sha256":_sha256(dev_manifest),
        "cache_manifest_sha256":_sha256(cache_manifest),
    }

    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S50_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
