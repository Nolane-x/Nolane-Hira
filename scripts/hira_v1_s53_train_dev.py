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
from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_token_query_option_late_interaction import (
    TokenLateInteractionQueryFreePrivateCorrectionFork,
)
from nmd.v1_s53_authority import generate_s53_cases, validate_s53_partitions
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35
import hira_v1_s50_train_dev as s50


SCHEMA_VERSION="hira-v1-s53-token-query-option-late-interaction-train-dev-v1"
OUTCOME_COMPLETE="HIRA_V1_S53_TOKEN_QUERY_OPTION_LATE_INTERACTION_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S53_TOKEN_QUERY_OPTION_LATE_INTERACTION_DEV_READY"

SEED=74_001
PRIVATE_EPOCHS=24
PRIVATE_TRAINABLE=114_688


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


@torch.inference_mode()
def _token_diagnostics(op,pairs):
    entropy_sum=max_weight_sum=cross_view_cos_sum=0.0
    context_count=0
    max_norm_error=0.0
    for canonical,paraphrase,_n in pairs:
        id_c=op.identity_signatures(
            state_tokens=canonical.state_tokens,
            state_mask=canonical.state_mask,
            option_view_tokens=canonical.option_view_tokens,
            option_view_token_mask=canonical.option_view_token_mask,
            option_view_mask=canonical.option_view_mask,
        )
        id_p=op.identity_signatures(
            state_tokens=paraphrase.state_tokens,
            state_mask=paraphrase.state_mask,
            option_view_tokens=paraphrase.option_view_tokens,
            option_view_token_mask=paraphrase.option_view_token_mask,
            option_view_mask=paraphrase.option_view_mask,
        )
        ctx_c,w_c=op.option_conditioned_query_context(
            option_identity=id_c,
            question_tokens=canonical.question_tokens,
            question_mask=canonical.question_mask,
            return_weights=True,
        )
        ctx_p,w_p=op.option_conditioned_query_context(
            option_identity=id_p,
            question_tokens=paraphrase.question_tokens,
            question_mask=paraphrase.question_mask,
            return_weights=True,
        )
        eps=1e-12
        for w in (w_c,w_p):
            entropy=-(w*w.clamp_min(eps).log()).sum(-1)
            entropy_sum+=float(entropy.sum())
            max_weight_sum+=float(w.max(-1).values.sum())
            context_count+=w.shape[0]*w.shape[1]
        cross_view_cos_sum+=float(
            F.cosine_similarity(ctx_c,ctx_p,dim=-1).sum()
        )
        max_norm_error=max(
            max_norm_error,
            float((ctx_c.norm(dim=-1)-1.0).abs().max()),
            float((ctx_p.norm(dim=-1)-1.0).abs().max()),
        )

    if context_count<1:
        raise RuntimeError("S53 token diagnostics empty")
    # cross-view cosine has one value per canonical option-context; half of
    # context_count because context_count includes canonical + paraphrase.
    cross_count=context_count//2
    return {
        "mean_token_attention_entropy":entropy_sum/context_count,
        "mean_max_token_weight":max_weight_sum/context_count,
        "context_cross_view_same_option_cosine":cross_view_cos_sum/cross_count,
        "context_norm_max_error":max_norm_error,
    }


def _train_branch(name,op,train_cache,dev_cache,out_dir):
    params=op.correction_parameters()
    if sum(p.numel() for p in params)!=PRIVATE_TRAINABLE:
        raise RuntimeError(f"S53 {name} private parameter count changed")
    if op.correction_parameter_count!=PRIVATE_TRAINABLE:
        raise RuntimeError(f"S53 {name} correction count changed")
    if op.identity_parameter_count!=0:
        raise RuntimeError(f"S53 {name} identity gained parameters")

    optimizer=torch.optim.AdamW(
        params,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    print(f"HIRA_V1_S53_{name.upper()}_TRAIN_BEGIN",flush=True)

    for epoch in range(1,PRIVATE_EPOCHS+1):
        order=list(range(len(train_cache)))
        random.Random(SEED+epoch).shuffle(order)
        loss_sum=ce_sum=js_sum=0.0
        case_count=0

        for index in order:
            canonical,paraphrase,n=train_cache[index]
            optimizer.zero_grad(set_to_none=True)
            # Both arms use query-free option identity, identical CE+JS and
            # identical fusion. Only the op's query readout differs.
            loss,ce,js=s50._private_loss(
                op,"treatment",canonical,paraphrase
            )
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            if not any(
                g is not None and float(g.detach().abs().sum())>0.0
                for g in grads
            ):
                raise RuntimeError(f"S53 {name} private gradient vanished")
            for p,g in zip(params,grads):
                p.grad=None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(params,s35.GRAD_CLIP)
            optimizer.step()

            loss_sum+=float(loss.detach())*n
            ce_sum+=float(ce.detach())*n
            js_sum+=float(js.detach())*n
            case_count+=n

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
            best_state=op.correction_state_dict()
            best_metrics=dict(metrics)

        print(
            "HIRA_V1_S53_PRIVATE_EPOCH="
            +json.dumps({"branch":name,**record},sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S53 {name} private selection failed")

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
            raise RuntimeError(f"S53 {name} selected DEV replay changed: {key}")

    gates=s50._private_gates(selected)
    checkpoint=out_dir/f"{name}-private-candidate.pt"
    torch.save({
        "schema_version":"hira-v1-s53-private-candidate-v1",
        "branch":name,
        "seed":SEED,
        "selected_dev_epoch":best_epoch,
        "private_trainable_parameter_count":PRIVATE_TRAINABLE,
        "correction_parameter_count":PRIVATE_TRAINABLE,
        "native_parameter_count_in_optimizer":0,
        "correction_state_dict":best_state,
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
            "correction_trainable_parameters":PRIVATE_TRAINABLE,
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
    if a0.get("outcome")!="HIRA_V1_S53_A0_TOKEN_QUERY_OPTION_LATE_INTERACTION_READY":
        raise RuntimeError("S53 A0 not qualified")
    if int(a0.get("seed",-1))!=SEED:
        raise RuntimeError("S53 A0 seed changed")
    if int(a0.get("reference_private_trainable_parameter_count",-1))!=PRIVATE_TRAINABLE:
        raise RuntimeError("S53 A0 reference surface changed")
    if int(a0.get("treatment_private_trainable_parameter_count",-1))!=PRIVATE_TRAINABLE:
        raise RuntimeError("S53 A0 treatment surface changed")
    if int(a0.get("late_interaction_parameter_count",-1))!=0:
        raise RuntimeError("S53 A0 late interaction gained parameters")
    if a0.get("pooled_query_bypass_absent") is not True:
        raise RuntimeError("S53 A0 pooled-query bypass appeared")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S53 A0 used for model selection")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S53 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S53 parent native runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S53 parent checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S53 parent native authority exposed DEV")

    verify_file_sha256(args.authority_checkpoint,expected_checkpoint)

    train_rows=generate_s53_cases("train")
    dev_rows=generate_s53_cases("dev")
    validate_s53_partitions(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S53 semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S53 loaded parent runtime hash changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S53 native runtime remained trainable")

    train_cache,train_digest=s50._materialize_cache(runtime,train_rows)
    dev_cache,dev_digest=s50._materialize_cache(runtime,dev_rows)
    del runtime

    reference=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    treatment=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S53 reference/treatment initialization changed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)

    reference_result=_train_branch(
        "reference",reference,train_cache,dev_cache,out
    )
    treatment_result=_train_branch(
        "treatment",treatment,train_cache,dev_cache,out
    )

    deltas=s50._metric_deltas(reference_result,treatment_result)
    token_diag=_token_diagnostics(treatment,dev_cache)

    overall_ready=(
        bool(reference_result["dev_ready"])
        and bool(treatment_result["dev_ready"])
    )
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S53_FRESH_TOKEN_QUERY_OPTION_LATE_INTERACTION",
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
            "run":37205099128,
            "artifact_id":11304188134,
            "artifact_digest":"sha256:08dbedcf29c7c71646c0c17a7e079de7df4d763ea0888db01282887867dbc841",
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
        "treatment_token_diagnostics":token_diag,
        "controlled_variable":{
            "reference_query_context":"pooled_global_query_context",
            "treatment_query_context":"option_conditioned_token_level_query_context",
            "token_temperature":0.10,
            "reference_private_trainable_parameters":PRIVATE_TRAINABLE,
            "treatment_private_trainable_parameters":PRIVATE_TRAINABLE,
            "late_interaction_trainable_parameters":0,
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
        "HIRA_V1_S53_TRAIN_DEV_RECEIPT="
        +json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
