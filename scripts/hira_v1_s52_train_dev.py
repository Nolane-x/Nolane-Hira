from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import (
    load_native_authority,
    verify_file_sha256,
)
from nmd.v1_query_relation_canonicalization import (
    CanonicalizedQueryFreeIdentityPrivateCorrectionFork,
    weighted_relation_code_auxiliary,
)
from nmd.v1_s52_authority import generate_s52_cases, validate_s52_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50


SCHEMA_VERSION="hira-v1-s52-query-relation-canonicalization-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S52_QUERY_RELATION_CANONICALIZATION_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S52_QUERY_RELATION_CANONICALIZATION_DEV_READY"

SEED=73_001
PRIVATE_EPOCHS=24
PRIVATE_TRAINABLE=147_456
CORRECTION=114_688
CANONICALIZER=32_768
REFERENCE_AUX_COEFFICIENT=0.0
TREATMENT_AUX_COEFFICIENT=0.10


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def _branch_metrics(op,pairs):
    # Both S52 arms use the exact same query-free option identity path.
    return s50._private_metrics(op,"treatment",pairs)


def _relation_auxiliary(op,canonical,paraphrase,n,coefficient):
    cc=op.query_summary(
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
    ).reshape(n,2,256)
    pc=op.query_summary(
        question_tokens=paraphrase.question_tokens,
        question_mask=paraphrase.question_mask,
    ).reshape(n,2,256)
    return weighted_relation_code_auxiliary(
        a1=cc[:,0],
        a2=pc[:,0],
        b1=cc[:,1],
        b2=pc[:,1],
        coefficient=coefficient,
    )


def _private_loss(op,canonical,paraphrase,n,coefficient):
    base,ce,js=s50._private_loss(op,"treatment",canonical,paraphrase)
    aux,aux_diag=_relation_auxiliary(
        op,canonical,paraphrase,n,coefficient
    )
    return base+aux,ce,js,aux,aux_diag


@torch.inference_mode()
def _query_diagnostics(op,pairs):
    same_a_sum=same_b_sum=cross_sum=raw_code_sum=residual_sum=0.0
    count=0
    for canonical,paraphrase,n in pairs:
        raw_c,code_c=op.query_code_pair(
            question_tokens=canonical.question_tokens,
            question_mask=canonical.question_mask,
        )
        raw_p,code_p=op.query_code_pair(
            question_tokens=paraphrase.question_tokens,
            question_mask=paraphrase.question_mask,
        )
        raw_c=raw_c.reshape(n,2,256)
        raw_p=raw_p.reshape(n,2,256)
        code_c=code_c.reshape(n,2,256)
        code_p=code_p.reshape(n,2,256)

        same_a=F.cosine_similarity(code_c[:,0],code_p[:,0],dim=-1)
        same_b=F.cosine_similarity(code_c[:,1],code_p[:,1],dim=-1)
        ac=F.normalize(code_c[:,0]+code_p[:,0],dim=-1)
        bc=F.normalize(code_c[:,1]+code_p[:,1],dim=-1)
        cross=F.cosine_similarity(ac,bc,dim=-1)

        raw=torch.cat([raw_c.reshape(-1,256),raw_p.reshape(-1,256)],dim=0)
        code=torch.cat([code_c.reshape(-1,256),code_p.reshape(-1,256)],dim=0)
        raw_code=F.cosine_similarity(raw,code,dim=-1)
        residual=(code-raw).norm(dim=-1)

        same_a_sum+=float(same_a.sum())
        same_b_sum+=float(same_b.sum())
        cross_sum+=float(cross.sum())
        raw_code_sum+=float(raw_code.sum())
        residual_sum+=float(residual.sum())
        count+=n

    if count<1:
        raise RuntimeError("S52 query diagnostics empty")
    query_count=4*count
    mean_a=same_a_sum/count
    mean_b=same_b_sum/count
    mean_same=0.5*(mean_a+mean_b)
    mean_cross=cross_sum/count
    return {
        "same_relation_a_cosine":mean_a,
        "same_relation_b_cosine":mean_b,
        "mean_same_relation_cosine":mean_same,
        "relation_centroid_cross_cosine":mean_cross,
        "relation_separation_margin":mean_same-mean_cross,
        "raw_vs_canonicalized_query_cosine":raw_code_sum/query_count,
        "canonicalizer_residual_norm":residual_sum/query_count,
    }


def _train_branch(name,op,coefficient,train_cache,dev_cache,out_dir):
    params=op.private_parameters()
    if sum(p.numel() for p in params)!=PRIVATE_TRAINABLE:
        raise RuntimeError(f"S52 {name} private parameter count changed")
    if op.correction_parameter_count!=CORRECTION:
        raise RuntimeError(f"S52 {name} correction count changed")
    if op.canonicalizer_parameter_count!=CANONICALIZER:
        raise RuntimeError(f"S52 {name} canonicalizer count changed")
    if op.identity_parameter_count!=0:
        raise RuntimeError(f"S52 {name} identity gained parameters")

    optimizer=torch.optim.AdamW(
        params,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    print(f"HIRA_V1_S52_{name.upper()}_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        totals={
            "loss":0.0,
            "ce":0.0,
            "js":0.0,
            "aux":0.0,
            "same":0.0,
            "separation":0.0,
        }
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            optimizer.zero_grad(set_to_none=True)
            loss,ce,js,aux,aux_diag=_private_loss(
                op,canonical,paraphrase,n,coefficient
            )
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            if not any(
                g is not None and float(g.detach().abs().sum())>0.0
                for g in grads
            ):
                raise RuntimeError(f"S52 {name} private gradient vanished")
            for p,g in zip(params,grads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(params,s35.GRAD_CLIP)
            optimizer.step()

            totals["loss"]+=float(loss.detach())*n
            totals["ce"]+=float(ce.detach())*n
            totals["js"]+=float(js.detach())*n
            totals["aux"]+=float(aux.detach())*n
            totals["same"]+=float(aux_diag["same_relation_loss"])*n
            totals["separation"]+=float(aux_diag["different_relation_hinge"])*n
            case_count+=n

        metrics=_branch_metrics(op,dev_cache)
        query_diag=_query_diagnostics(op,dev_cache)
        record={
            "epoch":epoch,
            "train_mean_private_loss":totals["loss"]/case_count,
            "train_mean_private_ce":totals["ce"]/case_count,
            "train_mean_private_js":totals["js"]/case_count,
            "train_mean_relation_auxiliary":totals["aux"]/case_count,
            "train_mean_same_relation_loss":totals["same"]/case_count,
            "train_mean_relation_separation_hinge":totals["separation"]/case_count,
            "dev":metrics,
            "query_code_diagnostics":query_diag,
        }
        history.append(record)

        key=s17._selection_key(epoch,metrics)
        if best_key is None or key>best_key:
            best_key=key
            best_epoch=epoch
            best_state=op.private_state_dict()
            best_metrics=dict(metrics)

        print(
            "HIRA_V1_S52_PRIVATE_EPOCH="
            +json.dumps({"branch":name,**record},sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S52 {name} private selection failed")

    op.load_private_state_dict(best_state,freeze=True)
    selected=_branch_metrics(op,dev_cache)
    selected_query_diag=_query_diagnostics(op,dev_cache)

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
            raise RuntimeError(f"S52 {name} selected DEV replay changed: {key}")

    gates=s50._private_gates(selected)
    checkpoint=out_dir/f"{name}-private-candidate.pt"
    torch.save({
        "schema_version":"hira-v1-s52-private-candidate-v1",
        "branch":name,
        "seed":SEED,
        "auxiliary_coefficient":coefficient,
        "selected_dev_epoch":best_epoch,
        "private_trainable_parameter_count":PRIVATE_TRAINABLE,
        "correction_parameter_count":CORRECTION,
        "canonicalizer_parameter_count":CANONICALIZER,
        "native_parameter_count_in_optimizer":0,
        "private_state_dict":best_state,
    },checkpoint)

    return {
        "branch":name,
        "auxiliary_coefficient":coefficient,
        "selected_dev_epoch":best_epoch,
        "selected_dev":selected,
        "selected_query_code_diagnostics":selected_query_diag,
        "gates":gates,
        "dev_ready":all(gates.values()),
        "history":history,
        "checkpoint_file":checkpoint.name,
        "checkpoint_sha256":_sha256(checkpoint),
        "parameter_surface":{
            "private_trainable_parameters":PRIVATE_TRAINABLE,
            "correction_trainable_parameters":CORRECTION,
            "canonicalizer_trainable_parameters":CANONICALIZER,
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
    if a0.get("outcome")!="HIRA_V1_S52_A0_QUERY_RELATION_CANONICALIZATION_READY":
        raise RuntimeError("S52 A0 not qualified")
    if int(a0.get("seed",-1))!=SEED:
        raise RuntimeError("S52 A0 seed changed")
    if int(a0.get("reference_private_trainable_parameter_count",-1))!=PRIVATE_TRAINABLE:
        raise RuntimeError("S52 A0 reference surface changed")
    if int(a0.get("treatment_private_trainable_parameter_count",-1))!=PRIVATE_TRAINABLE:
        raise RuntimeError("S52 A0 treatment surface changed")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S52 A0 used for model selection")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S52 parent native authority changed")
    if int(authority.get("seed",-1))!=72001 or int(authority.get("epoch",-1))!=24:
        raise RuntimeError("S52 parent native seed/epoch changed")
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S52 parent runtime hash changed")
    if authority.get("native_tensor_digest")!=expected_runtime:
        raise RuntimeError("S52 parent logical digest changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S52 parent checkpoint SHA changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S52 parent authority exposed DEV")

    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s52_cases("train")
    dev_rows=generate_s52_cases("dev")
    validate_s52_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S52 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S52 loaded parent runtime hash changed")
    if payload["native_tensor_digest"]!=expected_runtime:
        raise RuntimeError("S52 loaded parent logical digest changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S52 native runtime remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    reference=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    treatment=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    if not all(
        torch.equal(a.detach(),b.detach())
        for a,b in zip(reference.private_parameters(),treatment.private_parameters())
    ):
        raise RuntimeError("S52 reference/treatment initialization changed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)

    reference_result=_train_branch(
        "reference",
        reference,
        REFERENCE_AUX_COEFFICIENT,
        train_cache,
        dev_cache,
        out,
    )
    treatment_result=_train_branch(
        "treatment",
        treatment,
        TREATMENT_AUX_COEFFICIENT,
        train_cache,
        dev_cache,
        out,
    )

    deltas=s50._metric_deltas(reference_result,treatment_result)
    query_delta={
        key:float(treatment_result["selected_query_code_diagnostics"][key])
        -float(reference_result["selected_query_code_diagnostics"][key])
        for key in reference_result["selected_query_code_diagnostics"]
    }

    overall_ready=(
        bool(reference_result["dev_ready"])
        and bool(treatment_result["dev_ready"])
    )
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S52_FRESH_PAIRED_VIEW_QUERY_RELATION_CANONICALIZATION",
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
            "run":37199183308,
            "artifact_id":11302402275,
            "artifact_digest":"sha256:b99bb216580bbd7092126fc028c9581519ca23bfa72a0a8e174c7175daa76b5a",
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
        "selected_query_code_delta_treatment_minus_reference":query_delta,
        "controlled_variable":{
            "reference_auxiliary_coefficient":REFERENCE_AUX_COEFFICIENT,
            "treatment_auxiliary_coefficient":TREATMENT_AUX_COEFFICIENT,
            "separation_ceiling":0.25,
            "reference_private_trainable_parameters":PRIVATE_TRAINABLE,
            "treatment_private_trainable_parameters":PRIVATE_TRAINABLE,
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
        "HIRA_V1_S52_TRAIN_DEV_RECEIPT="
        +json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
