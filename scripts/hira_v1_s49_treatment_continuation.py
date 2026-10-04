from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import torch

from nmd.local_runtime import A13_REVISION, read_runtime_bundle_manifest
from nmd.v1_s49_authority import generate_s49_cases, validate_s49_partitions
import hira_v1_s45_train_dev as s45
import hira_v1_s49_train_dev as s49


SCHEMA_VERSION="hira-v1-s49-treatment-only-mechanical-continuation-v1"


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--recovered-reference",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S49_A0_PRIVATE_QUERY_FREE_OPTION_IDENTITY_READY":
        raise RuntimeError("S49-A0 not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S49-A0 used for model selection")
    for key,value in {
        "native_trainable_parameter_count":49152,
        "correction_parameter_count":114688,
        "treatment_total_trainable_parameter_count":163840,
        "identity_parameter_count":0,
    }.items():
        if int(a0.get(key,-1))!=value:
            raise RuntimeError(f"S49-A0 surface changed: {key}")
    if a0.get("identity_api_question_inputs_absent") is not True:
        raise RuntimeError("S49-A0 query leak")
    if bool(a0.get("second_encoder_pass",True)):
        raise RuntimeError("S49-A0 second encoder")

    recovered=json.loads(args.recovered_reference.read_text(encoding="utf-8"))
    if recovered.get("schema_version")!="hira-v1-s49-recovered-reference-arm-v1":
        raise RuntimeError("S49 recovered reference schema changed")
    if recovered.get("status")!="RECOVERED_FROM_PARTIAL_SCIENTIFIC_RUN":
        raise RuntimeError("S49 recovered reference status changed")
    if int(recovered.get("scientific_run",-1))!=37178974852:
        raise RuntimeError("S49 recovered reference run changed")
    if recovered.get("scientific_head")!="7ece9b16fd5ccfd5b93d9ff9eb1cb73ff15105ab":
        raise RuntimeError("S49 recovered reference head changed")
    if recovered.get("reference_arm_completed") is not True:
        raise RuntimeError("S49 recovered reference incomplete")
    if int(recovered.get("reference_epoch_records",-1))!=24:
        raise RuntimeError("S49 recovered reference epoch count changed")
    if int(recovered.get("reference_selected_epoch",-1))!=6:
        raise RuntimeError("S49 recovered reference selected epoch changed")
    if recovered.get("treatment_arm_started") is not False:
        raise RuntimeError("S49 prior treatment unexpectedly started")
    if recovered.get("treatment_dev_exposed") is not False:
        raise RuntimeError("S49 prior treatment DEV unexpectedly exposed")
    governance=recovered.get("governance",{})
    if governance.get("reference_rerun_authorized") is not False:
        raise RuntimeError("S49 reference rerun unexpectedly authorized")
    if governance.get("treatment_only_mechanical_continuation_authorized") is not True:
        raise RuntimeError("S49 treatment continuation not authorized")
    frozen_hashes=list(recovered.get("reference_runtime_trajectory_sha256",[]))
    if len(frozen_hashes)!=24 or len(set(frozen_hashes))!=24:
        raise RuntimeError("S49 recovered reference trajectory malformed")

    train_rows=generate_s49_cases("train")
    dev_rows=generate_s49_cases("dev")
    validate_s49_partitions(train_rows,dev_rows)
    s49._assert_fresh(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S49 semantic revision changed")

    args.out.mkdir(parents=True,exist_ok=True)
    treatment_dir=args.out/"treatment-query-free-identity"
    treatment_dir.mkdir(parents=True,exist_ok=True)

    # This continuation intentionally never calls _run_reference.
    s45.SEED=s49.SEED
    treatment=s49._run_treatment(
        bundle,manifest,train_rows,dev_rows,treatment_dir
    )
    treatment["arm"]="treatment_query_free_identity"

    observed=list(treatment["runtime_trajectory_sha256"])
    if observed!=frozen_hashes:
        mismatches=[
            i+1 for i,(a,b) in enumerate(zip(observed,frozen_hashes)) if a!=b
        ]
        raise RuntimeError(
            f"S49 continuation native trajectory differs from frozen reference at epochs {mismatches}"
        )
    treatment["gates"]["runtime_trajectory_identity"]=True
    treatment["dev_ready"]=all(treatment["gates"].values())

    treatment_cp=treatment_dir/"treatment-private-correction-candidate.pt"
    root_treatment=args.out/"treatment-query-free-identity-private-candidate.pt"
    shutil.copy2(treatment_cp,root_treatment)

    runtime,correction,payload=s49._load_treatment(
        bundle,manifest,treatment_cp
    )
    identity_diag=s49._identity_diagnostics(
        runtime,correction,dev_rows
    )

    reference_selected=dict(recovered["reference_selected_dev"])
    treatment_selected=dict(treatment["selected_dev"])
    delta=s45._metric_delta_values(reference_selected,treatment_selected)

    train_manifest=args.out/"train-manifest.json"
    dev_manifest=args.out/"dev-manifest.json"
    train_manifest.write_text(
        json.dumps([r.to_dict() for r in train_rows],indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    dev_manifest.write_text(
        json.dumps([r.to_dict() for r in dev_rows],indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )

    outcome=(
        "HIRA_V1_S49_PRIVATE_QUERY_FREE_OPTION_IDENTITY_DEV_READY"
        if treatment["dev_ready"]
        else "HIRA_V1_S49_PRIVATE_QUERY_FREE_OPTION_IDENTITY_DEV_COMPLETE"
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S49_TREATMENT_ONLY_CONTINUATION_AGAINST_FROZEN_REFERENCE_37178974852",
        "seed":s49.SEED,
        "continuation_reason":"constructor_compatibility_failure_after_reference_24_epochs_before_treatment_train_begin",
        "reference_rerun_performed":False,
        "reference_source":{
            "scientific_run":37178974852,
            "scientific_head":"7ece9b16fd5ccfd5b93d9ff9eb1cb73ff15105ab",
            "epoch_records":24,
            "selected_epoch":int(recovered["reference_selected_epoch"]),
            "selected_dev":reference_selected,
            "runtime_trajectory_sha256":frozen_hashes,
        },
        "treatment_arm":treatment,
        "matched_selected_dev_delta_treatment_minus_reference":delta,
        "selected_treatment_checkpoint":{
            "epoch":int(payload["selected_dev_epoch"]),
            "total_parameter_count":int(payload["total_parameter_count"]),
            "correction_parameter_count":int(payload["correction_parameter_count"]),
        },
        "treatment_identity_diagnostics":identity_diag,
        "trajectory_invariant":{
            "all_24_epoch_runtime_state_sha256_equal":True,
            "reference_runtime_trajectory_sha256":frozen_hashes,
            "treatment_runtime_trajectory_sha256":observed,
            "epoch_count":24,
        },
        "controlled_variable":{
            "reference_private_signature":"question_conditioned_native_relation_signature",
            "treatment_private_signature":"query_free_state_option_identity",
            "raw_query_readout_matched":True,
            "reference_correction_parameters":114688,
            "treatment_correction_parameters":114688,
            "identity_trainable_parameters":0,
            "semantic_change_after_reference_exposure":False,
            "constructor_compatibility_fix_only":True,
        },
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "k":4,
            "views_per_option":2,
            "same_frozen_s49_rows_as_partial_run":True,
            "s48_train_dev_rows_used":False,
            "s49_a0_rows_used":False,
        },
        "post_dev_tuning_performed":False,
        "second_full_s49_run_performed":False,
        "second_treatment_continuation_performed":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
        "train_manifest_sha256":_sha256(train_manifest),
        "dev_manifest_sha256":_sha256(dev_manifest),
        "treatment_checkpoint_sha256":_sha256(root_treatment),
    }

    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S49_TREATMENT_CONTINUATION_RECEIPT="
        +json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
