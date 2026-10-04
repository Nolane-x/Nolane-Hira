from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import load_native_authority, verify_file_sha256
from nmd.v1_learned_joint_relation_interaction import (
    LearnedJointRelationPrivateCorrectionFork,
    learned_joint_initialization_exact,
)
from nmd.v1_s55_authority import generate_s55_cases, validate_s55_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50


SCHEMA_VERSION="hira-v1-s55-learned-joint-relation-interaction-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S55_LEARNED_JOINT_RELATION_INTERACTION_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S55_LEARNED_JOINT_RELATION_INTERACTION_DEV_READY"

SEED=76_001
PRIVATE_EPOCHS=24
CORRECTION_PARAMS=114_688
LEARNED_PARAMS=65_536
PRIVATE_TRAINABLE=180_224


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


@torch.inference_mode()
def _learned_diagnostics(op,pairs):
    cross_cos_sum=q_code_cos_sum=residual_norm_sum=0.0
    state_norm_sum=state_sensitivity_sum=0.0
    reference_zero_error=0.0
    count=0

    for canonical,paraphrase,_n in pairs:
        outputs=[]
        for evidence in (canonical,paraphrase):
            values=op.correction_logits_from_state_option(
                native_logits=evidence.native_logits,
                state_tokens=evidence.state_tokens,
                state_mask=evidence.state_mask,
                option_view_tokens=evidence.option_view_tokens,
                option_view_token_mask=evidence.option_view_token_mask,
                option_view_mask=evidence.option_view_mask,
                question_tokens=evidence.question_tokens,
                question_mask=evidence.question_mask,
                return_context=True,
            )
            _logits,identity,code,qctx,sctx,*_rest=values
            outputs.append(code)

            qn=F.normalize(qctx,dim=-1)
            cn=F.normalize(code,dim=-1)
            q_code_cos_sum+=float(F.cosine_similarity(qn,cn,dim=-1).sum())
            residual_norm_sum+=float((cn-qn).norm(dim=-1).sum())
            count+=code.shape[0]*code.shape[1]

            if op.use_state_joint_context:
                state_norm_sum+=float(sctx.norm(dim=-1).sum())
                zero=torch.zeros_like(sctx)
                zero_code=op.learned_joint_transform(
                    query_context=qctx,
                    state_joint_context=zero,
                    option_identity=identity,
                )
                state_sensitivity_sum+=float((code-zero_code).abs().mean(dim=-1).sum())
            else:
                reference_zero_error=max(
                    reference_zero_error,
                    float(sctx.abs().max()),
                )

        cross_cos_sum+=float(
            F.cosine_similarity(outputs[0],outputs[1],dim=-1).sum()
        )

    if count<1:
        raise RuntimeError("S55 learned diagnostics empty")
    cross_count=count//2
    result={
        "relation_code_cross_view_same_option_cosine":cross_cos_sum/cross_count,
        "query_relation_code_cosine":q_code_cos_sum/count,
        "mean_relation_residual_norm":residual_norm_sum/count,
        "reference_zero_state_channel_max_abs":reference_zero_error,
    }
    if op.use_state_joint_context:
        result.update({
            "treatment_mean_explicit_joint_channel_norm":state_norm_sum/count,
            "treatment_relation_code_joint_channel_sensitivity":state_sensitivity_sum/count,
        })
    else:
        result.update({
            "treatment_mean_explicit_joint_channel_norm":0.0,
            "treatment_relation_code_joint_channel_sensitivity":0.0,
        })
    return result


def _train_branch(name,op,train_cache,dev_cache,out_dir):
    params=op.private_parameters()
    if sum(p.numel() for p in params)!=PRIVATE_TRAINABLE:
        raise RuntimeError(f"S55 {name} private parameter count changed")
    if op.correction_parameter_count!=CORRECTION_PARAMS:
        raise RuntimeError(f"S55 {name} correction count changed")
    if op.learned_joint_parameter_count!=LEARNED_PARAMS:
        raise RuntimeError(f"S55 {name} learned count changed")
    if op.identity_parameter_count!=0:
        raise RuntimeError(f"S55 {name} identity gained parameters")

    optimizer=torch.optim.AdamW(
        params,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    print(f"HIRA_V1_S55_{name.upper()}_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        loss_sum=ce_sum=js_sum=0.0
        case_count=0
        learned_gradient_live=False

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            optimizer.zero_grad(set_to_none=True)
            loss,ce,js=s50._private_loss(
                op,"treatment",canonical,paraphrase
            )
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            correction_grads=grads[:len(op.correction_parameters())]
            learned_grads=grads[len(op.correction_parameters()):]
            if not any(g is not None and float(g.detach().abs().sum())>0.0 for g in correction_grads):
                raise RuntimeError(f"S55 {name} correction gradient vanished")
            if any(g is not None and float(g.detach().abs().sum())>0.0 for g in learned_grads):
                learned_gradient_live=True
            for p,g in zip(params,grads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(params,s35.GRAD_CLIP)
            optimizer.step()

            loss_sum+=float(loss.detach())*n
            ce_sum+=float(ce.detach())*n
            js_sum+=float(js.detach())*n
            case_count+=n

        # B is zero-initialized by contract, so the learned transform may
        # legitimately receive zero gradient on the very first warm-start
        # update while the correction shell is also still at its neutral
        # initialization. Require the path to become live within the TRAIN
        # epoch, before any DEV metric is evaluated.
        if not learned_gradient_live:
            raise RuntimeError(
                f"S55 {name} learned-joint gradient never became live in epoch {epoch}"
            )

        metrics=s50._private_metrics(op,"treatment",dev_cache)
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
            best_state=op.private_state_dict()
            best_metrics=dict(metrics)

        print(
            "HIRA_V1_S55_PRIVATE_EPOCH="
            +json.dumps({"branch":name,**record},sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S55 {name} private selection failed")

    op.load_private_state_dict(best_state,freeze=True)
    selected=s50._private_metrics(op,"treatment",dev_cache)
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
            raise RuntimeError(f"S55 {name} selected DEV replay changed: {key}")

    gates=s50._private_gates(selected)
    checkpoint=out_dir/f"{name}-private-candidate.pt"
    torch.save({
        "schema_version":"hira-v1-s55-private-candidate-v1",
        "branch":name,
        "seed":SEED,
        "selected_dev_epoch":best_epoch,
        "private_trainable_parameter_count":PRIVATE_TRAINABLE,
        "correction_parameter_count":CORRECTION_PARAMS,
        "learned_joint_parameter_count":LEARNED_PARAMS,
        "native_parameter_count_in_optimizer":0,
        "private_state_dict":best_state,
    },checkpoint)

    return {
        "branch":name,
        "selected_dev_epoch":best_epoch,
        "selected_dev":selected,
        "gates":gates,
        "dev_ready":all(gates.values()),
        "history":history,
        "checkpoint_file":checkpoint.name,
        "checkpoint_sha256":_sha256(checkpoint),
        "parameter_surface":{
            "private_trainable_parameters":PRIVATE_TRAINABLE,
            "correction_trainable_parameters":CORRECTION_PARAMS,
            "learned_joint_trainable_parameters":LEARNED_PARAMS,
            "native_trainable_parameters":0,
            "identity_trainable_parameters":0,
        },
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
    if a0.get("outcome")!="HIRA_V1_S55_A0_LEARNED_JOINT_RELATION_INTERACTION_READY":
        raise RuntimeError("S55 A0 not qualified")
    if int(a0.get("seed",-1))!=SEED:
        raise RuntimeError("S55 A0 seed changed")
    for key in ("reference_private_trainable_parameter_count","treatment_private_trainable_parameter_count"):
        if int(a0.get(key,-1))!=PRIVATE_TRAINABLE:
            raise RuntimeError(f"S55 A0 private surface changed: {key}")
    for key in ("reference_learned_joint_parameter_count","treatment_learned_joint_parameter_count"):
        if int(a0.get(key,-1))!=LEARNED_PARAMS:
            raise RuntimeError(f"S55 A0 learned surface changed: {key}")
    if a0.get("initialization_bit_identical") is not True:
        raise RuntimeError("S55 A0 initialization changed")
    if float(a0.get("treatment_state_channel_sensitivity",0.0))<=0.0:
        raise RuntimeError("S55 A0 treatment state channel not live")
    if float(a0.get("reference_zero_state_channel_invariant_error",1.0))!=0.0:
        raise RuntimeError("S55 A0 reference neutral channel changed")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S55 A0 used for model selection")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S55 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S55 parent native runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S55 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S55 parent native authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s55_cases("train")
    dev_rows=generate_s55_cases("dev")
    validate_s55_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S55 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S55 loaded parent runtime hash changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S55 native runtime remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    reference=LearnedJointRelationPrivateCorrectionFork(
        use_state_joint_context=False,
        train_correction=True,
    )
    treatment=LearnedJointRelationPrivateCorrectionFork(
        use_state_joint_context=True,
        train_correction=True,
    )
    if not learned_joint_initialization_exact(reference,treatment):
        raise RuntimeError("S55 reference/treatment initialization changed")
    if reference.private_trainable_parameter_count!=PRIVATE_TRAINABLE:
        raise RuntimeError("S55 reference private surface changed")
    if treatment.private_trainable_parameter_count!=PRIVATE_TRAINABLE:
        raise RuntimeError("S55 treatment private surface changed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)

    reference_result=_train_branch("reference",reference,train_cache,dev_cache,out)
    treatment_result=_train_branch("treatment",treatment,train_cache,dev_cache,out)

    deltas=s50._metric_deltas(reference_result,treatment_result)
    reference_diag=_learned_diagnostics(reference,dev_cache)
    treatment_diag=_learned_diagnostics(treatment,dev_cache)

    overall_ready=bool(reference_result["dev_ready"]) and bool(treatment_result["dev_ready"])
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S55_FRESH_LEARNED_JOINT_RELATION_INTERACTION",
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
            "run":37211474503,
            "artifact_id":11306837784,
            "artifact_digest":"sha256:66ad8bbc2a2046ba00fdc17b60bef991dc246cd2182268edf3965ea91e29b812",
        },
        "shared_cache":{
            "train_digest":train_digest,
            "dev_digest":dev_digest,
            "reference_treatment_same_cache_bytes":True,
            "private_state_view_encodes":0,
            "cache_regenerated_after_dev":False,
        },
        "reference_branch":reference_result,
        "treatment_branch":treatment_result,
        "selected_dev_delta_treatment_minus_reference":deltas,
        "reference_learned_diagnostics":reference_diag,
        "treatment_learned_diagnostics":treatment_diag,
        "controlled_variable":{
            "reference_state_channel":"zeros",
            "treatment_state_channel":"s54_joint_state_query_option_context",
            "reference_private_trainable_parameters":PRIVATE_TRAINABLE,
            "treatment_private_trainable_parameters":PRIVATE_TRAINABLE,
            "correction_parameters_each":CORRECTION_PARAMS,
            "learned_joint_parameters_each":LEARNED_PARAMS,
            "initialization_bit_identical":True,
        },
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "second_dev_run_performed":False,
        "native_retraining_performed":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S55_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
