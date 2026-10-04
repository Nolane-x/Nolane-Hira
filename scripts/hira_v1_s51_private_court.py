from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import (
    load_native_authority,
    verify_file_sha256,
)
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_s51_authority import (
    generate_s51_train_cases,
    generate_s51_dev_cases,
    validate_s51_partitions,
)
from nmd.v1_shared_native_private_readouts import correction_initialization_exact
from hira_v1_s50_train_dev import (
    _identity_diagnostics,
    _materialize_cache,
    _metric_deltas,
    _private_gates,
    _private_loss,
    _private_metrics,
)


SCHEMA_VERSION="hira-v1-s51-artifact-pinned-private-court-v1"
OUTCOME_COMPLETE="HIRA_V1_S51_ARTIFACT_PINNED_PRIVATE_COURT_DEV_COMPLETE"
OUTCOME_READY="HIRA_V1_S51_ARTIFACT_PINNED_PRIVATE_COURT_DEV_READY"

SEED=72_001
PRIVATE_EPOCHS=24
CORRECTION=114_688


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def _manifest_bytes(rows)->bytes:
    return (
        json.dumps([r.to_dict() for r in rows],indent=2,sort_keys=True)+"\n"
    ).encode("utf-8")


def _train_private_branch_s51(kind,op,train_cache,dev_cache,out_dir):
    params=op.correction_parameters()
    if sum(p.numel() for p in params)!=CORRECTION:
        raise RuntimeError(f"S51 {kind} correction capacity changed")
    import hira_v1_s35_train_dev as s35

    optimizer=torch.optim.AdamW(
        params,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )

    history=[]
    best_key=None
    best_epoch=None
    best_state=None
    best_metrics=None

    import hira_v1_s17_train_dev as s17

    print(f"HIRA_V1_S51_PRIVATE_{kind.upper()}_TRAIN_BEGIN",flush=True)

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
            if not any(
                g is not None and float(g.detach().abs().sum())>0.0
                for g in grads
            ):
                raise RuntimeError(f"S51 {kind} private gradient vanished")
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
            "HIRA_V1_S51_PRIVATE_EPOCH="
            +json.dumps({"branch":kind,**record},sort_keys=True),
            flush=True,
        )

    if best_state is None or best_epoch is None or best_metrics is None:
        raise RuntimeError(f"S51 {kind} private selection failed")

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
            raise RuntimeError(f"S51 {kind} selected DEV replay changed: {key}")

    gates=_private_gates(selected)
    checkpoint=out_dir/f"{kind}-private-candidate.pt"
    torch.save({
        "schema_version":"hira-v1-s51-private-candidate-v1",
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
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--authority-train-manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    receipt=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema_version")!="hira-v1-s51-native-authority-receipt-v1":
        raise RuntimeError("S51 Phase-B authority receipt schema changed")
    if receipt.get("status")!="PASS":
        raise RuntimeError("S51 Phase-B authority not PASS")
    if receipt.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S51 Phase-B authority outcome changed")
    if int(receipt.get("seed",-1))!=SEED or int(receipt.get("epoch",-1))!=24:
        raise RuntimeError("S51 Phase-B authority seed/epoch changed")
    if receipt.get("dev_generated") is not False:
        raise RuntimeError("S51 Phase-A unexpectedly generated DEV")
    if receipt.get("dev_encoded") is not False or receipt.get("dev_scored") is not False:
        raise RuntimeError("S51 Phase-A unexpectedly exposed DEV")
    if receipt.get("private_correction_constructed") is not False:
        raise RuntimeError("S51 Phase-A unexpectedly constructed private correction")

    verify_file_sha256(
        args.authority_checkpoint,
        str(receipt["checkpoint_file_sha256"]),
    )
    if _sha256(args.authority_train_manifest)!=str(receipt["train_manifest_sha256"]):
        raise RuntimeError("S51 authority TRAIN manifest file digest changed")

    train_rows=generate_s51_train_cases()
    dev_rows=generate_s51_dev_cases()
    validate_s51_partitions(train_rows,dev_rows)
    replay_train_sha=sha256(_manifest_bytes(train_rows)).hexdigest()
    if replay_train_sha!=str(receipt["train_manifest_sha256"]):
        raise RuntimeError("S51 Phase-B regenerated TRAIN authority changed")

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S51 Phase-B semantic revision changed")

    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=str(receipt["checkpoint_file_sha256"]),
        expected_seed=SEED,
    )
    if payload["runtime_state_sha256"]!=receipt["runtime_state_sha256"]:
        raise RuntimeError("S51 Phase-B loaded runtime authority hash changed")
    if payload["native_tensor_digest"]!=receipt["native_tensor_digest"]:
        raise RuntimeError("S51 Phase-B loaded logical native digest changed")

    native_trainable=sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if native_trainable!=0:
        raise RuntimeError("S51 Phase-B loaded native runtime remained trainable")

    # Native authority is only used for one immutable evidence materialization.
    train_cache,train_cache_digest=_materialize_cache(runtime,train_rows)
    dev_cache,dev_cache_digest=_materialize_cache(runtime,dev_rows)

    # Destroy the live native runtime reference before private optimization.
    del runtime

    reference=PrivateCorrectionRepresentationFork(train_correction=True)
    treatment=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    if not correction_initialization_exact(reference,treatment):
        raise RuntimeError("S51 correction initialization changed")
    if reference.correction_parameter_count!=CORRECTION:
        raise RuntimeError("S51 reference correction capacity changed")
    if treatment.correction_parameter_count!=CORRECTION:
        raise RuntimeError("S51 treatment correction capacity changed")
    if treatment.identity_parameter_count!=0:
        raise RuntimeError("S51 query-free identity gained parameters")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)

    reference_result=_train_private_branch_s51(
        "reference",reference,train_cache,dev_cache,out
    )
    treatment_result=_train_private_branch_s51(
        "treatment",treatment,train_cache,dev_cache,out
    )
    deltas=_metric_deltas(reference_result,treatment_result)
    identity_diag=_identity_diagnostics(treatment,dev_cache)

    overall_ready=(
        bool(reference_result["dev_ready"])
        and bool(treatment_result["dev_ready"])
    )
    outcome=OUTCOME_READY if overall_ready else OUTCOME_COMPLETE

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S51_ARTIFACT_PINNED_SHARED_NATIVE_PRIVATE_COURT",
        "seed":SEED,
        "phase_a_authority":{
            "runtime_state_sha256":receipt["runtime_state_sha256"],
            "native_tensor_digest":receipt["native_tensor_digest"],
            "checkpoint_file_sha256":receipt["checkpoint_file_sha256"],
            "train_manifest_sha256":receipt["train_manifest_sha256"],
            "epoch":receipt["epoch"],
            "dev_encoded":receipt["dev_encoded"],
            "dev_scored":receipt["dev_scored"],
        },
        "phase_b_loaded_authority":{
            "runtime_state_sha256":payload["runtime_state_sha256"],
            "native_tensor_digest":payload["native_tensor_digest"],
            "native_trainable_parameters":0,
            "native_optimizer_constructed":False,
            "native_training_performed":False,
        },
        "shared_cache":{
            "train_digest":train_cache_digest,
            "dev_digest":dev_cache_digest,
            "reference_treatment_same_cache_bytes":True,
            "cache_regenerated_after_dev":False,
            "private_state_view_encodes":0,
        },
        "reference_branch":reference_result,
        "treatment_branch":treatment_result,
        "selected_dev_delta_treatment_minus_reference":deltas,
        "treatment_identity_diagnostics":identity_diag,
        "private_surface":{
            "reference_correction_parameters":CORRECTION,
            "treatment_correction_parameters":CORRECTION,
            "treatment_identity_parameters":0,
            "correction_initialization_bit_identical":True,
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
        "authority_regenerated_after_private_exposure":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(
        "HIRA_V1_S51_PRIVATE_COURT_RECEIPT="
        +json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
