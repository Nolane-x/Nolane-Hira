from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import load_native_authority, verify_file_sha256
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_pairwise_ranking_consistency import (
    decision_discrimination_diagnostics,
    pairwise_ordinal_consistency,
    weighted_pairwise_ordinal_auxiliary,
)
from nmd.v1_s57_authority import generate_s57_cases, validate_s57_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50


SCHEMA_VERSION="hira-v1-s57-discrete-pairwise-ranking-consistency-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S57_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S57_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_DEV_READY"

SEED=78_001
PRIVATE_EPOCHS=24
PRIVATE_TRAINABLE=114_688
REFERENCE_ORDINAL_COEFFICIENT=0.0
TREATMENT_ORDINAL_COEFFICIENT=0.05


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def _coefficient(name:str)->float:
    if name=="reference":
        return REFERENCE_ORDINAL_COEFFICIENT
    if name=="treatment":
        return TREATMENT_ORDINAL_COEFFICIENT
    raise ValueError(name)


def _loss(name,op,canonical,paraphrase):
    if not torch.equal(canonical.gold,paraphrase.gold):
        raise RuntimeError("S57 paired gold changed")
    base,ce,relation_js=s50._private_loss(op,"treatment",canonical,paraphrase)
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    aux,diag=weighted_pairwise_ordinal_auxiliary(
        fused_c,fused_p,canonical.gold,
        coefficient=_coefficient(name),
    )
    return base+aux,ce,relation_js,aux,diag


@torch.inference_mode()
def _ordinal_diagnostics(op,pairs):
    entropy_sum=gap_sum=rms_sum=0.0
    ordinal_sum=active_sum=filtered_sum=disagree_sum=non_gold_sum=0.0
    query_count=pair_count=0

    for canonical,paraphrase,_n in pairs:
        if not torch.equal(canonical.gold,paraphrase.gold):
            raise RuntimeError("S57 DEV paired gold changed")
        _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
        _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)

        for logits in (fused_c,fused_p):
            d=decision_discrimination_diagnostics(logits)
            q=logits.shape[0]
            entropy_sum+=float(d["mean_entropy"])*q
            gap_sum+=float(d["mean_top1_top2_probability_gap"])*q
            rms_sum+=float(d["mean_unregularized_logit_rms"])*q
            query_count+=q

        ordinal,diag=pairwise_ordinal_consistency(
            fused_c,fused_p,canonical.gold
        )
        ordinal_sum+=float(ordinal)
        active_sum+=float(diag["active_directional_anchor_fraction"])
        filtered_sum+=float(diag["gold_filtered_direction_fraction"])
        disagree_sum+=float(diag["sign_disagreement_fraction"])
        non_gold_sum+=float(diag["non_gold_active_direction_fraction"])
        pair_count+=1

    if query_count<1 or pair_count<1:
        raise RuntimeError("S57 ordinal diagnostics empty")

    return {
        "mean_fused_entropy":entropy_sum/query_count,
        "mean_top1_top2_probability_gap":gap_sum/query_count,
        "mean_unregularized_logit_rms":rms_sum/query_count,
        "mean_pairwise_ordinal_loss":ordinal_sum/pair_count,
        "mean_active_directional_anchor_fraction":active_sum/pair_count,
        "mean_gold_filtered_direction_fraction":filtered_sum/pair_count,
        "mean_sign_disagreement_fraction":disagree_sum/pair_count,
        "mean_non_gold_active_direction_fraction":non_gold_sum/pair_count,
    }


def _train_branch(name,op,train_cache,dev_cache,out_dir):
    params=op.correction_parameters()
    if sum(p.numel() for p in params)!=PRIVATE_TRAINABLE:
        raise RuntimeError(f"S57 {name} private parameter count changed")
    if op.correction_parameter_count!=PRIVATE_TRAINABLE:
        raise RuntimeError(f"S57 {name} correction count changed")
    if op.identity_parameter_count!=0:
        raise RuntimeError(f"S57 {name} identity gained parameters")

    optimizer=torch.optim.AdamW(
        params,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    print(f"HIRA_V1_S57_{name.upper()}_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        total_sum=ce_sum=relation_js_sum=aux_sum=ordinal_sum=0.0
        active_sum=filtered_sum=disagree_sum=0.0
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            optimizer.zero_grad(set_to_none=True)
            loss,ce,relation_js,aux,diag=_loss(
                name,op,canonical,paraphrase
            )
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            if not any(
                g is not None and float(g.detach().abs().sum())>0.0
                for g in grads
            ):
                raise RuntimeError(f"S57 {name} private gradient vanished")
            for p,g in zip(params,grads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(params,s35.GRAD_CLIP)
            optimizer.step()

            total_sum+=float(loss.detach())*n
            ce_sum+=float(ce.detach())*n
            relation_js_sum+=float(relation_js.detach())*n
            aux_sum+=float(aux.detach())*n
            ordinal_sum+=float(diag["ordinal_loss"])*n
            active_sum+=float(diag["active_directional_anchor_fraction"])*n
            filtered_sum+=float(diag["gold_filtered_direction_fraction"])*n
            disagree_sum+=float(diag["sign_disagreement_fraction"])*n
            case_count+=n

        metrics=s50._private_metrics(op,"treatment",dev_cache)
        record={
            "epoch":epoch,
            "train_mean_total_loss":total_sum/case_count,
            "train_mean_private_ce":ce_sum/case_count,
            "train_mean_relation_js":relation_js_sum/case_count,
            "train_mean_weighted_ordinal_auxiliary":aux_sum/case_count,
            "train_mean_ordinal_loss":ordinal_sum/case_count,
            "train_mean_active_directional_anchor_fraction":active_sum/case_count,
            "train_mean_gold_filtered_direction_fraction":filtered_sum/case_count,
            "train_mean_sign_disagreement_fraction":disagree_sum/case_count,
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
            "HIRA_V1_S57_PRIVATE_EPOCH="
            +json.dumps({"branch":name,**record},sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S57 {name} private selection failed")

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
            raise RuntimeError(f"S57 {name} selected DEV replay changed: {key}")

    gates=s50._private_gates(selected)
    diag=_ordinal_diagnostics(op,dev_cache)
    checkpoint=out_dir/f"{name}-private-candidate.pt"
    torch.save({
        "schema_version":"hira-v1-s57-private-candidate-v1",
        "branch":name,
        "seed":SEED,
        "selected_dev_epoch":best_epoch,
        "private_trainable_parameter_count":PRIVATE_TRAINABLE,
        "correction_parameter_count":PRIVATE_TRAINABLE,
        "ordinal_consistency_coefficient":_coefficient(name),
        "native_parameter_count_in_optimizer":0,
        "correction_state_dict":best_state,
    },checkpoint)

    return {
        "branch":name,
        "selected_dev_epoch":best_epoch,
        "selected_dev":selected,
        "ordinal_diagnostics":diag,
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
    if a0.get("outcome")!="HIRA_V1_S57_A0_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_READY":
        raise RuntimeError("S57 A0 not qualified")
    if int(a0.get("seed",-1))!=SEED:
        raise RuntimeError("S57 A0 seed changed")
    if int(a0.get("reference_private_trainable_parameter_count",-1))!=PRIVATE_TRAINABLE:
        raise RuntimeError("S57 A0 reference surface changed")
    if int(a0.get("treatment_private_trainable_parameter_count",-1))!=PRIVATE_TRAINABLE:
        raise RuntimeError("S57 A0 treatment surface changed")
    if int(a0.get("added_trainable_parameter_count",-1))!=0:
        raise RuntimeError("S57 A0 added parameters changed")
    if a0.get("anchor_sign_detached") is not True:
        raise RuntimeError("S57 A0 anchor detach changed")
    if int(a0.get("wrong_gold_filtered_direction_count",0))<=0:
        raise RuntimeError("S57 A0 gold filter not live")
    if float(a0.get("treatment_auxiliary_gradient_l1",0.0))<=0.0:
        raise RuntimeError("S57 A0 treatment auxiliary not live")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S57 A0 used for model selection")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S57 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S57 parent native runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S57 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S57 parent native authority exposed DEV")
    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s57_cases("train")
    dev_rows=generate_s57_cases("dev")
    validate_s57_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S57 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S57 loaded parent runtime hash changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S57 native runtime remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    reference=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    treatment=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S57 reference/treatment initialization changed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)

    reference_result=_train_branch("reference",reference,train_cache,dev_cache,out)
    treatment_result=_train_branch("treatment",treatment,train_cache,dev_cache,out)

    deltas=s50._metric_deltas(reference_result,treatment_result)
    overall_ready=bool(reference_result["dev_ready"]) and bool(treatment_result["dev_ready"])
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S57_FRESH_DISCRETE_PAIRWISE_RANKING_CONSISTENCY",
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
            "run":37266891469,
            "artifact_id":11326757222,
            "artifact_digest":"sha256:fcfbececa46f5bac0e1fe75ae51348d1b4c543b5b7dc3166410c148a35dd75ce",
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
        "controlled_variable":{
            "reference_ordinal_consistency_coefficient":REFERENCE_ORDINAL_COEFFICIENT,
            "treatment_ordinal_consistency_coefficient":TREATMENT_ORDINAL_COEFFICIENT,
            "standardization_epsilon":1e-6,
            "active_threshold":0.25,
            "preserved_margin_floor":0.05,
            "gold_order_protection":True,
            "reference_private_trainable_parameters":PRIVATE_TRAINABLE,
            "treatment_private_trainable_parameters":PRIVATE_TRAINABLE,
            "added_trainable_parameters":0,
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
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(
        "HIRA_V1_S57_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
