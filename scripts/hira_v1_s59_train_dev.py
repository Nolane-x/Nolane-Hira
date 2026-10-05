from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_evidence_fusion import fused_gold_vs_max_wrong_margin
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority, verify_file_sha256
from nmd.v1_explicit_pairwise_decision_head import (
    S59_PAIRWISE_PARAMETER_COUNT,
    ExplicitPairwiseDecisionHead,
    build_pairwise_representation,
    pairwise_head_loss,
)
from nmd.v1_s59_authority import generate_s59_cases, validate_s59_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50


SCHEMA_VERSION="hira-v1-s59-explicit-learned-pairwise-decision-head-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S59_EXPLICIT_LEARNED_PAIRWISE_DECISION_HEAD_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S59_EXPLICIT_LEARNED_PAIRWISE_DECISION_HEAD_DEV_READY"

SEED=80_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
PAIRWISE_PARAMS=32_832

A0_RUN=37278474963
A0_ARTIFACT_ID=11331042601
A0_ARTIFACT_DIGEST="sha256:5271081c4bcacbced9ca85c146ddd36b5dc9ecdc0c6dc9038c9b3fb6219a11b6"


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def _state_digest(state:dict[str,torch.Tensor])->str:
    d=sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        d.update(key.encode("utf-8"))
        d.update(str(tuple(value.shape)).encode("ascii"))
        d.update(str(value.dtype).encode("ascii"))
        d.update(value.numpy().tobytes())
    return d.hexdigest()


def _representation(op,evidence):
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
    rep=build_pairwise_representation(identity,context)
    if rep.requires_grad:
        raise RuntimeError("S59 pairwise representation retained gradient")
    return rep


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


def _materialize_rep_cache(cache):
    geometry=JointStateQueryOptionPrivateCorrectionFork(train_correction=False)
    entries=[]
    for canonical,paraphrase,n in cache:
        rc=_representation(geometry,canonical)
        rp=_representation(geometry,paraphrase)
        entries.append((canonical,paraphrase,n,rc,rp))
    return entries,_rep_digest(entries)


def _pairwise_metrics(op,head,rep_cache):
    fused_c_correct=fused_p_correct=0
    fused_pair_both=fused_pair_changed=0
    fused_agree_sum=fused_js_sum=0.0
    fused_c_margin_sum=fused_p_margin_sum=0.0
    canonical_loss_sum=0.0
    gold_pair_correct=gold_pair_count=0
    gold_pair_margin_sum=0.0
    top_tie_count=key_fallback_count=0
    max_antisym=max_diag=max_mass=0.0
    semantic_cases=queries=0
    full_k=True

    # Relation/identity diagnostics remain the exact base correction shell and
    # enter the frozen S17 selector unchanged.
    base=s50._private_metrics(
        op,"treatment",[(c,p,n) for c,p,n,_rc,_rp in rep_cache]
    )

    with torch.no_grad():
        for canonical,paraphrase,n,rep_c,rep_p in rep_cache:
            score_c=head.aggregate_logits(rep_c)
            score_p=head.aggregate_logits(rep_p)
            pair_c=head.pairwise_logits(rep_c)
            pair_p=head.pairwise_logits(rep_p)
            gold=canonical.gold
            if not torch.equal(gold,paraphrase.gold):
                raise RuntimeError("S59 paired gold mismatch")

            pc,dc=head.select_with_tiebreak(rep_c)
            pp,dp=head.select_with_tiebreak(rep_p)

            fused_c_correct+=int((pc==gold).sum())
            fused_p_correct+=int((pp==gold).sum())

            pairs_pc=pc.reshape(n,2)
            pairs_gold=gold.reshape(n,2)
            fused_pair_both+=int(((pairs_pc==pairs_gold).all(-1)).sum())
            fused_pair_changed+=int((pairs_pc[:,0]!=pairs_pc[:,1]).sum())

            q=2*n
            fused_agree_sum+=float(selected_choice_agreement(score_c,score_p))*q
            fused_js_sum+=float(symmetric_js_divergence(score_c,score_p))*q
            fused_c_margin_sum+=float(fused_gold_vs_max_wrong_margin(score_c,gold).sum())
            fused_p_margin_sum+=float(fused_gold_vs_max_wrong_margin(score_p,gold).sum())
            canonical_loss_sum+=float(F.cross_entropy(score_c,gold,reduction="sum"))

            for pair,score,rep,diag in (
                (pair_c,score_c,rep_c,dc),
                (pair_p,score_p,rep_p,dp),
            ):
                b,k,_=pair.shape
                batch=torch.arange(b,device=pair.device)
                gold_row=pair[batch,gold,:]
                mask=torch.arange(k,device=pair.device)[None,:].ne(gold[:,None])
                margins=gold_row[mask]
                gold_pair_correct+=int(margins.gt(0).sum())
                gold_pair_count+=int(margins.numel())
                gold_pair_margin_sum+=float(margins.sum())
                top_tie_count+=int(diag["top_score_tie_count"].gt(1).sum())
                key_fallback_count+=int(diag["degenerate_index_fallback"].sum())
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
                    float((torch.softmax(score,dim=-1).sum(-1)-1.0).abs().max()),
                )
                full_k=full_k and score.shape[-1]==4 and rep.shape[-2]==4

            semantic_cases+=n
            queries+=q

    if queries<1 or gold_pair_count<1:
        raise RuntimeError("S59 empty DEV metrics")

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
        "mean_canonical_decision_loss":canonical_loss_sum/queries,
        "pairwise_gold_pair_accuracy":gold_pair_correct/gold_pair_count,
        "pairwise_mean_gold_pair_margin":gold_pair_margin_sum/gold_pair_count,
        "pairwise_top_score_tie_fraction":top_tie_count/(2*queries),
        "pairwise_representation_key_fallback_fraction":key_fallback_count/(2*queries),
        "pairwise_antisymmetry_max_abs_error":max_antisym,
        "pairwise_diagonal_max_abs_error":max_diag,
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


def _pairwise_train_diagnostics(head,rep_cache):
    loss_sum=margin_sum=accuracy_sum=0.0
    pair_count=case_count=0
    max_antisym=0.0
    with torch.no_grad():
        for canonical,paraphrase,n,rep_c,rep_p in rep_cache:
            for evidence,rep in ((canonical,rep_c),(paraphrase,rep_p)):
                pair=head.pairwise_logits(rep)
                b,k,_=pair.shape
                batch=torch.arange(b,device=pair.device)
                gold=evidence.gold
                row=pair[batch,gold,:]
                mask=torch.arange(k,device=pair.device)[None,:].ne(gold[:,None])
                margins=row[mask]
                loss=F.softplus(-margins)
                loss_sum+=float(loss.sum())
                margin_sum+=float(margins.sum())
                accuracy_sum+=float(margins.gt(0).sum())
                pair_count+=int(margins.numel())
                max_antisym=max(max_antisym,float((pair+pair.transpose(-1,-2)).abs().max()))
            case_count+=n
    return {
        "gold_pair_loss":loss_sum/pair_count,
        "gold_pair_accuracy":accuracy_sum/pair_count,
        "mean_gold_pair_margin":margin_sum/pair_count,
        "supervised_pair_count":pair_count,
        "semantic_case_count":case_count,
        "antisymmetry_max_abs_error":max_antisym,
    }


def _train_shared_trajectory(train_cache,dev_cache,train_rep,dev_rep,out_dir):
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)

    cparams=op.correction_parameters()
    hparams=[head.A,head.u]
    if sum(p.numel() for p in cparams)!=CORRECTION_PARAMS:
        raise RuntimeError("S59 correction capacity changed")
    if sum(p.numel() for p in hparams)!=PAIRWISE_PARAMS:
        raise RuntimeError("S59 pairwise capacity changed")

    corr_opt=torch.optim.AdamW(cparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)
    head_opt=torch.optim.AdamW(hparams,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY)

    history=[]
    best={"reference":None,"treatment":None}
    best_key={"reference":None,"treatment":None}

    print("HIRA_V1_S59_SHARED_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        base_loss_sum=pair_loss_sum=0.0
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            _c2,_p2,n2,rep_c,rep_p=train_rep[index]
            if n2!=n:
                raise RuntimeError("S59 representation/cache batch mismatch")

            corr_opt.zero_grad(set_to_none=True)
            base_loss,_ce,_js=s50._private_loss(
                op,"treatment",canonical,paraphrase
            )
            cgrads=torch.autograd.grad(base_loss,cparams,allow_unused=True)
            if not any(g is not None and float(g.detach().abs().sum())>0 for g in cgrads):
                raise RuntimeError("S59 correction gradient vanished")
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
                raise RuntimeError("S59 pairwise head gradient vanished")
            for p,g in zip(hparams,hgrads):
                p.grad=g.detach().clone()
            torch.nn.utils.clip_grad_norm_(hparams,s35.GRAD_CLIP)
            head_opt.step()

            base_loss_sum+=float(base_loss.detach())*n
            pair_loss_sum+=float(pair_loss.detach())*n
            case_count+=n

        reference=s50._private_metrics(op,"treatment",dev_cache)
        treatment=_pairwise_metrics(op,head,dev_rep)
        train_pair_diag=_pairwise_train_diagnostics(head,train_rep)

        corr_state=op.correction_state_dict()
        head_state=head.state_dict_exact()
        corr_digest=_state_digest(corr_state)
        head_digest=_state_digest(head_state)

        record={
            "epoch":epoch,
            "train_mean_base_correction_loss":base_loss_sum/case_count,
            "train_mean_pairwise_loss":pair_loss_sum/case_count,
            "train_pairwise":train_pair_diag,
            "reference_dev":reference,
            "treatment_dev":treatment,
            "correction_state_sha256":corr_digest,
            "pairwise_head_state_sha256":head_digest,
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
                    "metrics":dict(metrics),
                    "correction_state_sha256":corr_digest,
                    "pairwise_head_state_sha256":head_digest,
                }

        print(
            "HIRA_V1_S59_EPOCH="+json.dumps({
                "epoch":epoch,
                "reference":reference,
                "treatment":treatment,
                "correction_state_sha256":corr_digest,
                "pairwise_head_state_sha256":head_digest,
            },sort_keys=True),
            flush=True,
        )

    if best["reference"] is None or best["treatment"] is None:
        raise RuntimeError("S59 selector failed")

    results={}
    for arm in ("reference","treatment"):
        selected=best[arm]
        replay_op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
        replay_op.load_correction_state_dict(selected["correction_state"],freeze=True)
        replay_head=ExplicitPairwiseDecisionHead(trainable=True)
        replay_head.load_state_dict_exact(selected["head_state"],freeze=True)

        if arm=="reference":
            metrics=s50._private_metrics(replay_op,"treatment",dev_cache)
        else:
            metrics=_pairwise_metrics(replay_op,replay_head,dev_rep)

        for key,value in selected["metrics"].items():
            if isinstance(value,bool):
                if metrics[key] is not value:
                    raise RuntimeError(f"S59 {arm} selected replay changed: {key}")
            elif abs(float(metrics[key])-float(value))>1e-12:
                raise RuntimeError(f"S59 {arm} selected replay changed: {key}")

        gates=s50._private_gates(metrics)
        checkpoint=out_dir/f"{arm}-candidate.pt"
        torch.save({
            "schema_version":"hira-v1-s59-private-candidate-v1",
            "branch":arm,
            "seed":SEED,
            "selected_dev_epoch":selected["epoch"],
            "correction_parameter_count":CORRECTION_PARAMS,
            "pairwise_parameter_count":PAIRWISE_PARAMS,
            "native_parameter_count_in_optimizer":0,
            "teacher_dependency":False,
            "decision_shell":"fused" if arm=="reference" else "explicit_pairwise",
            "correction_state_dict":selected["correction_state"],
            "pairwise_head_state_dict":selected["head_state"],
        },checkpoint)
        results[arm]={
            "selected_dev_epoch":selected["epoch"],
            "selected_dev":metrics,
            "gates":gates,
            "dev_ready":all(gates.values()),
            "correction_state_sha256":selected["correction_state_sha256"],
            "pairwise_head_state_sha256":selected["pairwise_head_state_sha256"],
            "checkpoint_file":checkpoint.name,
            "checkpoint_sha256":_sha256(checkpoint),
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
    if a0.get("outcome")!="HIRA_V1_S59_A0_EXPLICIT_LEARNED_PAIRWISE_DECISION_HEAD_READY":
        raise RuntimeError("S59 A0 not qualified")
    if int(a0.get("pairwise_parameter_count",-1))!=PAIRWISE_PARAMS:
        raise RuntimeError("S59 A0 pairwise capacity changed")
    if int(a0.get("correction_trainable_parameter_count",-1))!=CORRECTION_PARAMS:
        raise RuntimeError("S59 A0 correction capacity changed")
    if a0.get("pairwise_has_bias") is not False:
        raise RuntimeError("S59 A0 bias contract changed")
    if float(a0.get("antisymmetry_max_abs_error",-1))!=0.0:
        raise RuntimeError("S59 A0 antisymmetry changed")
    if a0.get("pairwise_representation_requires_grad") is not False:
        raise RuntimeError("S59 A0 representation gained gradient")
    if a0.get("correction_received_pairwise_gradient") is not False:
        raise RuntimeError("S59 A0 correction received pairwise gradient")
    if a0.get("teacher_dependency") is not False:
        raise RuntimeError("S59 A0 teacher dependency changed")
    if any(a0.get(k) is not False for k in (
        "raw_native_logit_input_to_head",
        "raw_fused_logit_input_to_head",
        "raw_corrected_logit_input_to_head",
    )):
        raise RuntimeError("S59 A0 raw logit bypass changed")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S59 A0 used for model selection")
    if a0.get("fresh_train_dev_exposed") is not False:
        raise RuntimeError("S59 A0 exposed fresh TRAIN DEV")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S59 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S59 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S59 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S59 parent authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s59_cases("train")
    dev_rows=generate_s59_cases("dev")
    validate_s59_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S59 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S59 loaded parent runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S59 native runtime remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    train_rep,train_rep_digest=_materialize_rep_cache(train_cache)
    dev_rep,dev_rep_digest=_materialize_rep_cache(dev_cache)

    if any(
        rep.requires_grad
        for entries in (train_rep,dev_rep)
        for _c,_p,_n,rc,rp in entries
        for rep in (rc,rp)
    ):
        raise RuntimeError("S59 materialized representation gained gradient")

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
        "scientific_authority":"V1_S59_FRESH_EXPLICIT_LEARNED_PAIRWISE_DECISION_HEAD",
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
            "native_fused_corrected_logits_used":False,
        },
        "controlled_variable":{
            "reference_decision_shell":"existing_fused",
            "treatment_decision_shell":"explicit_learned_pairwise",
            "correction_trainable_parameters":CORRECTION_PARAMS,
            "pairwise_trainable_parameters":PAIRWISE_PARAMS,
            "pairwise_representation_dimension":512,
            "pairwise_hidden_dimension":64,
            "pairwise_bias":False,
            "teacher_dependency":False,
            "pseudo_target_dependency":False,
            "self_anchor_dependency":False,
            "pairwise_gradient_enters_correction":False,
            "pairwise_gradient_enters_native":False,
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
            "exact_s58_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "width_sweep_performed":False,
        "activation_sweep_performed":False,
        "loss_variant_performed":False,
        "tiebreak_variant_performed":False,
        "base_logit_blend_performed":False,
        "joint_gradient_variant_performed":False,
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
        "HIRA_V1_S59_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
