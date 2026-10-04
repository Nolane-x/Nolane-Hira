from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from nmd.local_runtime import A13_REVISION, load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_norm_balanced_gradient import apply_gradient_update, norm_balanced_gradient_update
from nmd.v1_persisted_native_authority import (
    S51_NATIVE_FIXED_EPOCH,
    S51_NATIVE_TRAINABLE_PARAMETER_COUNT,
    make_native_authority_payload,
    save_native_authority,
)
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
from nmd.v1_s51_authority import generate_s51_train_cases
import hira_v1_s17_train_dev as s17
import hira_v1_s35_train_dev as s35


SCHEMA_VERSION="hira-v1-s51-native-authority-receipt-v1"
OUTCOME="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY"
SEED=72_001
BATCH_SIZE=16


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def _train_manifest(rows,out:Path)->str:
    out.write_text(
        json.dumps([r.to_dict() for r in rows],indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    return _sha256(out)


def _build_trainable_runtime(bundle:Path,manifest:dict):
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
    count=sum(p.numel() for p in trainable)
    if count!=S51_NATIVE_TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError(f"S51 Phase-A native trainable count changed: {count}")
    if count!=HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S51 Phase-A inherited native surface changed")
    if s17._original_a13_trainable(runtime)!=0:
        raise RuntimeError("S51 Phase-A original A13 became trainable")
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S51 Phase-A projection scorer missing")
    if scorer.projection_trainable_parameter_count!=HIRA_V1_S17_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S51 Phase-A projection surface changed")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S51 Phase-A HIRACore became trainable")
    return runtime,trainable


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    # Phase A intentionally has a TRAIN-only authority API. There is no import
    # or call to S51 DEV generation in this file.
    train_rows=generate_s51_train_cases()
    if len(train_rows)!=768 or any(r.split!="train" for r in train_rows):
        raise RuntimeError("S51 Phase-A TRAIN authority changed")
    if len({r.domain for r in train_rows})!=12:
        raise RuntimeError("S51 Phase-A domain count changed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    train_manifest=out/"train-manifest.json"
    train_manifest_sha=_train_manifest(train_rows,train_manifest)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S51 Phase-A semantic revision changed")

    runtime,trainable=_build_trainable_runtime(bundle,manifest)
    optimizer=torch.optim.AdamW(
        trainable,lr=s35.LR,weight_decay=s35.WEIGHT_DECAY
    )
    history=[]

    print("HIRA_V1_S51_NATIVE_AUTHORITY_TRAIN_BEGIN",flush=True)

    for epoch in range(1,S51_NATIVE_FIXED_EPOCH+1):
        order=list(range(len(train_rows)))
        random.Random(SEED+epoch).shuffle(order)
        totals={
            "total":0.0,"decision":0.0,"ce":0.0,"swap":0.0,
            "option_alignment":0.0,"binding":0.0,"canonicalization":0.0,
            "consistency_js":0.0,"primary_block":0.0,"relation_block":0.0,
        }
        state_view_encodes=0

        for start in range(0,len(order),BATCH_SIZE):
            rows=[train_rows[i] for i in order[start:start+BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True)
            _total,primary_block,relation_block,pieces,*_rest=s35._arm_losses(
                runtime,rows,"native"
            )
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
            combined,_diag=norm_balanced_gradient_update(
                primary,relation,epsilon=s35.BALANCE_EPSILON
            )
            apply_gradient_update(trainable,combined)
            torch.nn.utils.clip_grad_norm_(trainable,s35.GRAD_CLIP)
            optimizer.step()
            enforce_s17_eval(runtime)

            n=len(rows)
            for key in totals:
                totals[key]+=pieces[key]*n
            state_view_encodes+=2*n

        if state_view_encodes!=2*len(train_rows):
            raise RuntimeError("S51 Phase-A TRAIN state-once changed")

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
            "train_state_view_encodes":state_view_encodes,
        }
        history.append(record)
        print(
            "HIRA_V1_S51_NATIVE_AUTHORITY_EPOCH="
            +json.dumps(record,sort_keys=True),
            flush=True,
        )

    if history[-1]["epoch"]!=S51_NATIVE_FIXED_EPOCH:
        raise RuntimeError("S51 Phase-A fixed epoch changed")

    payload=make_native_authority_payload(
        runtime,
        seed=SEED,
        epoch=S51_NATIVE_FIXED_EPOCH,
        semantic_revision=str(manifest["semantic_revision"]),
        initialization_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_manifest_sha256=train_manifest_sha,
    )

    checkpoint=out/"native-authority.pt"
    checkpoint_file_sha=save_native_authority(checkpoint,payload)

    receipt={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S51_PHASE_A_TRAIN_ONLY_PERSISTED_NATIVE_AUTHORITY",
        "seed":SEED,
        "epoch":S51_NATIVE_FIXED_EPOCH,
        "train_semantic_cases":len(train_rows),
        "domains":sorted({r.domain for r in train_rows}),
        "dev_generated":False,
        "dev_encoded":False,
        "dev_scored":False,
        "private_correction_constructed":False,
        "native_trainable_parameter_count":S51_NATIVE_TRAINABLE_PARAMETER_COUNT,
        "projection_parameter_count":HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
        "native_tensor_digest":payload["native_tensor_digest"],
        "runtime_state_sha256":payload["runtime_state_sha256"],
        "checkpoint_file_sha256":checkpoint_file_sha,
        "train_manifest_sha256":train_manifest_sha,
        "semantic_revision":str(manifest["semantic_revision"]),
        "initialization_t0_sha256":str(manifest["t0_checkpoint_sha256"]),
        "history":history,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    (out/"authority-receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(
        "HIRA_V1_S51_NATIVE_AUTHORITY_RECEIPT="
        +json.dumps(receipt,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
