from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import load_native_authority, verify_file_sha256, file_sha256
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_teacher_consensus_ranking import (
    S58_TEACHER_ACTIVE_THRESHOLD,
    S58_STUDENT_MARGIN_FLOOR,
    S58_CONSENSUS_COEFFICIENT,
    teacher_consensus_pairwise_loss,
    weighted_teacher_consensus_auxiliary,
)
from nmd.v1_s58_authority import generate_s58_cases, validate_s58_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50


SCHEMA_VERSION="hira-v1-s58-consensus-teacher-pairwise-ranking-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S58_CONSENSUS_TEACHER_PAIRWISE_RANKING_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S58_CONSENSUS_TEACHER_PAIRWISE_RANKING_DEV_READY"

SEED=79_001
PRIVATE_EPOCHS=24
PRIVATE_TRAINABLE=114_688
REFERENCE_COEFFICIENT=0.0
TREATMENT_COEFFICIENT=0.05

TEACHER_RUN=37271509208
TEACHER_ARTIFACT_ID=11327849211
TEACHER_CHECKPOINT_SHA="804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa"
TEACHER_SELECTED_EPOCH=19

A0_RUN=37274963437
A0_ARTIFACT_ID=11330280320
A0_ARTIFACT_DIGEST="sha256:f4785071bafcf3674cecefbba98468c9d2dcb42a5c1212f0c446177bde5f5ebc"


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def _coefficient(name:str)->float:
    if name=="reference":
        return REFERENCE_COEFFICIENT
    if name=="treatment":
        return TREATMENT_COEFFICIENT
    raise ValueError(name)


def _load_teacher(path:Path):
    if file_sha256(path)!=TEACHER_CHECKPOINT_SHA:
        raise RuntimeError("S58 teacher checkpoint SHA changed")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("schema_version")!="hira-v1-s57-private-candidate-v1":
        raise RuntimeError("S58 teacher schema changed")
    if payload.get("branch")!="reference":
        raise RuntimeError("S58 teacher branch changed")
    if int(payload.get("seed",-1))!=78001:
        raise RuntimeError("S58 teacher seed changed")
    if int(payload.get("selected_dev_epoch",-1))!=TEACHER_SELECTED_EPOCH:
        raise RuntimeError("S58 teacher selected epoch changed")
    if int(payload.get("correction_parameter_count",-1))!=PRIVATE_TRAINABLE:
        raise RuntimeError("S58 teacher correction surface changed")
    if float(payload.get("ordinal_consistency_coefficient",-1.0))!=0.0:
        raise RuntimeError("S58 teacher is not S57 reference")
    teacher=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    teacher.load_correction_state_dict(payload["correction_state_dict"],freeze=True)
    if any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError("S58 teacher remained trainable")
    return teacher,payload


def _teacher_targets(teacher,cache):
    out=[]
    d=sha256()
    with torch.inference_mode():
        for canonical,paraphrase,n in cache:
            if not torch.equal(canonical.gold,paraphrase.gold):
                raise RuntimeError("S58 teacher cache paired gold changed")
            _rc,_sc,teacher_c=s50._branch_logits(teacher,"treatment",canonical)
            _rp,_sp,teacher_p=s50._branch_logits(teacher,"treatment",paraphrase)
            teacher_c=teacher_c.detach().clone()
            teacher_p=teacher_p.detach().clone()
            if teacher_c.requires_grad or teacher_p.requires_grad:
                raise RuntimeError("S58 teacher target retained gradient")
            for tensor in (teacher_c,teacher_p):
                t=tensor.detach().cpu().contiguous()
                d.update(str(tuple(t.shape)).encode("utf-8"))
                d.update(str(t.dtype).encode("utf-8"))
                d.update(t.numpy().tobytes())
            out.append((canonical,paraphrase,n,teacher_c,teacher_p))
    return out,d.hexdigest()


def _loss(name,op,entry):
    canonical,paraphrase,_n,teacher_c,teacher_p=entry
    if not torch.equal(canonical.gold,paraphrase.gold):
        raise RuntimeError("S58 paired gold changed")
    base,ce,relation_js=s50._private_loss(op,"treatment",canonical,paraphrase)
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    aux,diag=weighted_teacher_consensus_auxiliary(
        fused_c,fused_p,teacher_c,teacher_p,canonical.gold,
        coefficient=_coefficient(name),
    )
    return base+aux,ce,relation_js,aux,diag


@torch.inference_mode()
def _consensus_diagnostics(op,teacher_cache):
    loss_sum=active_sum=wrong_sum=non_gold_sum=violation_sum=0.0
    case_count=0
    for canonical,paraphrase,n,teacher_c,teacher_p in teacher_cache:
        _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
        _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
        loss,diag=teacher_consensus_pairwise_loss(
            fused_c,fused_p,teacher_c,teacher_p,canonical.gold
        )
        loss_sum+=float(loss)*n
        active_sum+=float(diag["teacher_active_consensus_fraction"])*n
        wrong_sum+=float(diag["teacher_wrong_gold_filtered_fraction"])*n
        non_gold_sum+=float(diag["teacher_non_gold_active_fraction"])*n
        violation_sum+=float(diag["student_violation_fraction"])*n
        case_count+=n
    if case_count<1:
        raise RuntimeError("S58 consensus diagnostics empty")
    return {
        "mean_teacher_consensus_pairwise_loss":loss_sum/case_count,
        "mean_teacher_active_consensus_fraction":active_sum/case_count,
        "mean_teacher_wrong_gold_filtered_fraction":wrong_sum/case_count,
        "mean_teacher_non_gold_active_fraction":non_gold_sum/case_count,
        "mean_student_violation_fraction":violation_sum/case_count,
    }


def _train_branch(name,op,train_teacher_cache,dev_cache,dev_teacher_cache,out_dir):
    params=op.correction_parameters()
    if sum(p.numel() for p in params)!=PRIVATE_TRAINABLE:
        raise RuntimeError(f"S58 {name} private parameter count changed")
    if op.correction_parameter_count!=PRIVATE_TRAINABLE:
        raise RuntimeError(f"S58 {name} correction count changed")
    if op.identity_parameter_count!=0:
        raise RuntimeError(f"S58 {name} identity gained parameters")

    optimizer=torch.optim.AdamW(
        params,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    print(f"HIRA_V1_S58_{name.upper()}_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_teacher_cache)))
        random.Random(SEED+epoch).shuffle(order)
        total_sum=ce_sum=relation_js_sum=aux_sum=consensus_sum=0.0
        active_sum=wrong_sum=non_gold_sum=violation_sum=0.0
        case_count=0

        for index in order:
            entry=train_teacher_cache[index]
            _canonical,_paraphrase,n,_teacher_c,_teacher_p=entry
            optimizer.zero_grad(set_to_none=True)
            loss,ce,relation_js,aux,diag=_loss(name,op,entry)
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            if not any(
                g is not None and float(g.detach().abs().sum())>0.0
                for g in grads
            ):
                raise RuntimeError(f"S58 {name} private gradient vanished")
            for p,g in zip(params,grads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(params,s35.GRAD_CLIP)
            optimizer.step()

            total_sum+=float(loss.detach())*n
            ce_sum+=float(ce.detach())*n
            relation_js_sum+=float(relation_js.detach())*n
            aux_sum+=float(aux.detach())*n
            consensus_sum+=float(diag["consensus_pairwise_loss"])*n
            active_sum+=float(diag["teacher_active_consensus_fraction"])*n
            wrong_sum+=float(diag["teacher_wrong_gold_filtered_fraction"])*n
            non_gold_sum+=float(diag["teacher_non_gold_active_fraction"])*n
            violation_sum+=float(diag["student_violation_fraction"])*n
            case_count+=n

        metrics=s50._private_metrics(op,"treatment",dev_cache)
        record={
            "epoch":epoch,
            "train_mean_total_loss":total_sum/case_count,
            "train_mean_private_ce":ce_sum/case_count,
            "train_mean_relation_js":relation_js_sum/case_count,
            "train_mean_weighted_teacher_auxiliary":aux_sum/case_count,
            "train_mean_teacher_consensus_pairwise_loss":consensus_sum/case_count,
            "train_mean_teacher_active_consensus_fraction":active_sum/case_count,
            "train_mean_teacher_wrong_gold_filtered_fraction":wrong_sum/case_count,
            "train_mean_teacher_non_gold_active_fraction":non_gold_sum/case_count,
            "train_mean_student_violation_fraction":violation_sum/case_count,
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
            "HIRA_V1_S58_PRIVATE_EPOCH="
            +json.dumps({"branch":name,**record},sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S58 {name} private selection failed")

    op.load_correction_state_dict(best_state,freeze=True)
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
            raise RuntimeError(f"S58 {name} selected DEV replay changed: {key}")

    gates=s50._private_gates(selected)
    diag=_consensus_diagnostics(op,dev_teacher_cache)
    checkpoint=out_dir/f"{name}-private-candidate.pt"
    torch.save({
        "schema_version":"hira-v1-s58-private-candidate-v1",
        "branch":name,
        "seed":SEED,
        "selected_dev_epoch":best_epoch,
        "private_trainable_parameter_count":PRIVATE_TRAINABLE,
        "correction_parameter_count":PRIVATE_TRAINABLE,
        "teacher_consensus_coefficient":_coefficient(name),
        "teacher_run":TEACHER_RUN,
        "teacher_checkpoint_sha256":TEACHER_CHECKPOINT_SHA,
        "native_parameter_count_in_optimizer":0,
        "correction_state_dict":best_state,
    },checkpoint)

    return {
        "branch":name,
        "selected_dev_epoch":best_epoch,
        "selected_dev":selected,
        "teacher_consensus_diagnostics":diag,
        "gates":gates,
        "dev_ready":all(gates.values()),
        "history":history,
        "checkpoint_file":checkpoint.name,
        "checkpoint_sha256":_sha256(checkpoint),
        "parameter_surface":{
            "private_trainable_parameters":PRIVATE_TRAINABLE,
            "correction_trainable_parameters":PRIVATE_TRAINABLE,
            "added_trainable_parameters":0,
            "native_trainable_parameters":0,
            "teacher_trainable_parameters":0,
            "identity_trainable_parameters":0,
        },
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--teacher-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S58_A0_CONSENSUS_TEACHER_PAIRWISE_RANKING_READY":
        raise RuntimeError("S58 A0 not qualified")
    if int(a0.get("seed",-1))!=SEED:
        raise RuntimeError("S58 A0 seed changed")
    for key in (
        "reference_private_trainable_parameter_count",
        "treatment_private_trainable_parameter_count",
    ):
        if int(a0.get(key,-1))!=PRIVATE_TRAINABLE:
            raise RuntimeError(f"S58 A0 private surface changed: {key}")
    if int(a0.get("added_trainable_parameter_count",-1))!=0:
        raise RuntimeError("S58 A0 added parameters changed")
    if int(a0.get("teacher_run",-1))!=TEACHER_RUN:
        raise RuntimeError("S58 A0 teacher run changed")
    if int(a0.get("teacher_artifact_id",-1))!=TEACHER_ARTIFACT_ID:
        raise RuntimeError("S58 A0 teacher artifact changed")
    if a0.get("teacher_checkpoint_sha256")!=TEACHER_CHECKPOINT_SHA:
        raise RuntimeError("S58 A0 teacher checkpoint changed")
    if int(a0.get("teacher_selected_epoch",-1))!=TEACHER_SELECTED_EPOCH:
        raise RuntimeError("S58 A0 teacher epoch changed")
    if int(a0.get("teacher_trainable_parameter_count",-1))!=0:
        raise RuntimeError("S58 A0 teacher became trainable")
    if float(a0.get("teacher_deterministic_replay_max_abs_error",-1.0))!=0.0:
        raise RuntimeError("S58 A0 teacher replay changed")
    if a0.get("teacher_logits_require_grad") is not False:
        raise RuntimeError("S58 A0 teacher logits gained gradient")
    if a0.get("consensus_sign_detached") is not True:
        raise RuntimeError("S58 A0 consensus sign detach changed")
    if a0.get("reference_auxiliary_exact_zero") is not True:
        raise RuntimeError("S58 A0 reference auxiliary changed")
    if float(a0.get("treatment_auxiliary_gradient_l1",0.0))<=0.0:
        raise RuntimeError("S58 A0 treatment auxiliary not live")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S58 A0 used for model selection")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S58 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S58 parent native runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S58 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S58 parent native authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    teacher,teacher_payload=_load_teacher(args.teacher_checkpoint)

    train_rows=generate_s58_cases("train")
    dev_rows=generate_s58_cases("dev")
    validate_s58_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S58 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S58 loaded parent runtime hash changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S58 native runtime remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    train_teacher_cache,train_teacher_digest=_teacher_targets(teacher,train_cache)
    dev_teacher_cache,dev_teacher_digest=_teacher_targets(teacher,dev_cache)
    if any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError("S58 teacher gained trainable parameters after target materialization")
    del teacher

    reference=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    treatment=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S58 reference/treatment initialization changed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)

    reference_result=_train_branch(
        "reference",reference,train_teacher_cache,dev_cache,dev_teacher_cache,out
    )
    treatment_result=_train_branch(
        "treatment",treatment,train_teacher_cache,dev_cache,dev_teacher_cache,out
    )

    deltas=s50._metric_deltas(reference_result,treatment_result)
    overall_ready=bool(reference_result["dev_ready"]) and bool(treatment_result["dev_ready"])
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S58_FRESH_CONSENSUS_TEACHER_PAIRWISE_RANKING",
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
        "teacher_authority":{
            "run":TEACHER_RUN,
            "artifact_id":TEACHER_ARTIFACT_ID,
            "branch":teacher_payload["branch"],
            "selected_dev_epoch":TEACHER_SELECTED_EPOCH,
            "checkpoint_sha256":TEACHER_CHECKPOINT_SHA,
            "private_trainable_parameters":teacher_payload["private_trainable_parameter_count"],
            "teacher_trainable_parameters_in_s58":0,
            "selected_using_s58_data":False,
        },
        "shared_cache":{
            "train_digest":train_digest,
            "dev_digest":dev_digest,
            "reference_treatment_same_cache_bytes":True,
            "private_state_view_encodes":0,
            "cache_regenerated_after_dev":False,
        },
        "teacher_targets":{
            "train_digest":train_teacher_digest,
            "dev_digest":dev_teacher_digest,
            "precomputed_once_before_student_training":True,
            "reference_treatment_same_teacher_target_bytes":True,
            "teacher_updated_during_student_training":False,
        },
        "reference_branch":reference_result,
        "treatment_branch":treatment_result,
        "selected_dev_delta_treatment_minus_reference":deltas,
        "controlled_variable":{
            "reference_teacher_consensus_coefficient":REFERENCE_COEFFICIENT,
            "treatment_teacher_consensus_coefficient":TREATMENT_COEFFICIENT,
            "teacher_active_threshold":S58_TEACHER_ACTIVE_THRESHOLD,
            "student_target_margin":S58_STUDENT_MARGIN_FLOOR,
            "consensus_coefficient_constant":S58_CONSENSUS_COEFFICIENT,
            "teacher_branch":"reference",
            "teacher_selected_epoch":TEACHER_SELECTED_EPOCH,
            "teacher_checkpoint_sha256":TEACHER_CHECKPOINT_SHA,
            "wrong_gold_teacher_pairs_filtered":True,
            "teacher_consensus_sign_detached":True,
            "reference_private_trainable_parameters":PRIVATE_TRAINABLE,
            "treatment_private_trainable_parameters":PRIVATE_TRAINABLE,
            "added_trainable_parameters":0,
            "teacher_trainable_parameters":0,
            "initialization_bit_identical":True,
        },
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "exact_s57_state_question_option_overlap":0,
            "k":4,
            "views_per_option":2,
        },
        "post_dev_tuning_performed":False,
        "teacher_swap_performed":False,
        "teacher_threshold_sweep_performed":False,
        "target_margin_sweep_performed":False,
        "coefficient_sweep_performed":False,
        "consensus_mask_variant_performed":False,
        "second_dev_run_performed":False,
        "native_retraining_performed":False,
        "selector_changed":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(
        "HIRA_V1_S58_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
