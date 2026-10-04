from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import load_a13_lora_state_dict
from nmd.v1_invariance import selected_choice_agreement
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_query_quotient_private_correction import QueryQuotientPrivateCorrectionFork
from nmd.v1_s17_semantic_core import build_hira_v1_s17_norm_balanced_core, enforce_s17_eval
from nmd.v1_s47_authority import generate_s47_cases
from nmd.v1_s48_authority import generate_s48_cases, validate_s48_partitions
import hira_v1_s35_train_dev as s35
import hira_v1_s45_train_dev as s45
from hira_v1_s48_a0_query_quotient_option_evidence import cases as s48_a0_cases

SCHEMA_VERSION="hira-v1-s48-matched-query-quotient-option-evidence-train-dev-v1"
SEED=69_001


def _sha256(path: Path) -> str:
    digest=sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_texts(row):
    return (row.state_a,row.state_b)


def _question_texts(row):
    return (row.question_a1,row.question_a2,row.question_b1,row.question_b2)


def _option_texts(row):
    return (*row.option_texts,*row.option_aliases)


def _assert_s48_fresh(train_rows,dev_rows):
    s45._assert_s45_fresh(train_rows,dev_rows)
    current=(*train_rows,*dev_rows)
    states={x for r in current for x in _state_texts(r)}
    questions={x for r in current for x in _question_texts(r)}
    options={x for r in current for x in _option_texts(r)}

    prior47=(*generate_s47_cases("train"),*generate_s47_cases("dev"))
    prior_states={x for r in prior47 for x in _state_texts(r)}
    prior_questions={x for r in prior47 for x in _question_texts(r)}
    prior_options={x for r in prior47 for x in _option_texts(r)}
    if states&prior_states:
        raise RuntimeError("S48 exact state overlap with exposed S47 rows")
    if questions&prior_questions:
        raise RuntimeError("S48 exact question overlap with exposed S47 rows")
    if options&prior_options:
        raise RuntimeError("S48 exact option overlap with exposed S47 rows")

    a0=s48_a0_cases()
    a0_states={x for r in a0 for x in (r.state_a,r.state_b)}
    a0_questions={x for r in a0 for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    a0_options=set()
    for case in a0:
        opts,_ga,_gb=case.option_pack()
        for option in opts:
            a0_options.add(option.criterion_text)
            a0_options.update(option.aliases)
    if states&a0_states:
        raise RuntimeError("S48 TRAIN DEV state overlap with S48-A0")
    if questions&a0_questions:
        raise RuntimeError("S48 TRAIN DEV question overlap with S48-A0")
    if options&a0_options:
        raise RuntimeError("S48 TRAIN DEV option overlap with S48-A0")


def _raw_factory(*,train=True):
    return PrivateCorrectionRepresentationFork(
        native_dimension=256,
        hidden_dimension=64,
        query_norm_epsilon=1e-12,
        private_norm_epsilon=1e-12,
        residual_scale=1.0,
        adapter_seed=65_044,
        train_correction=train,
    )


def _quotient_factory(*,train=True):
    return QueryQuotientPrivateCorrectionFork(
        native_dimension=256,
        hidden_dimension=64,
        query_norm_epsilon=1e-12,
        private_norm_epsilon=1e-12,
        residual_scale=1.0,
        adapter_seed=65_044,
        train_correction=train,
    )


def _state_equal(a,b):
    sa=a.correction_state_dict()
    sb=b.correction_state_dict()
    return set(sa)==set(sb) and all(torch.equal(sa[k],sb[k]) for k in sa)


def _run_correction_arm(label,factory,bundle,manifest,train_rows,dev_rows,out_dir):
    original_factory=s45._new_correction
    original_seed=s45.SEED
    try:
        s45._new_correction=factory
        s45.SEED=SEED
        result=s45._train_arm(
            "treatment",
            bundle,
            manifest,
            train_rows,
            dev_rows,
            out_dir,
        )
    finally:
        s45._new_correction=original_factory
        s45.SEED=original_seed
    result["scientific_arm"]=label
    result["query_interface"]=(
        "raw_normalized_mean_query"
        if label=="reference_raw_query"
        else "option_difference_covariance_direction_quotient"
    )
    return result


def _load_selected(bundle,manifest,checkpoint_path,correction):
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if payload["arm"]!="treatment":
        raise RuntimeError("S48 selected checkpoint arm changed")
    if int(payload["total_parameter_count"])!=163_840:
        raise RuntimeError("S48 selected checkpoint total changed")
    if int(payload["correction_parameter_count"])!=114_688:
        raise RuntimeError("S48 selected checkpoint correction capacity changed")

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
    load_a13_lora_state_dict(runtime.encoder,payload["lora_state_dict"],freeze=True)
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S48 projection scorer missing")
    scorer.load_projection_state_dict(payload["projection_state_dict"],freeze=True)
    correction.load_correction_state_dict(payload["correction_state_dict"],freeze=True)
    enforce_s17_eval(runtime)
    correction.eval()
    return runtime,payload


@torch.inference_mode()
def _quotient_diagnostics(runtime,correction,rows):
    norm_sum=0.0
    count=0
    zero_count=0
    raw_vs_q_sum=0.0
    cross_q_sum=0.0
    corrected_agree_sum=0.0
    state_view_encodes=0

    for start in range(0,len(rows),s35.BATCH_SIZE):
        batch=list(rows[start:start+s35.BATCH_SIZE])
        (
            _total,_primary,_relation,_pieces,_fc,_fp,
            _raw_c,_raw_p,native_c,native_p,
            sig_c,sig_p,encoded,
        )=s45._native_losses(runtime,batch)

        qc,rawc=correction.query_quotient(
            signatures=sig_c,
            question_tokens=encoded["question_canonical_tokens"],
            question_mask=encoded["question_canonical_mask"],
        )
        qp,rawp=correction.query_quotient(
            signatures=sig_p,
            question_tokens=encoded["question_paraphrase_tokens"],
            question_mask=encoded["question_paraphrase_mask"],
        )
        cc=correction.correction_logits(
            native_logits=native_c,signatures=sig_c,
            question_tokens=encoded["question_canonical_tokens"],
            question_mask=encoded["question_canonical_mask"],
        )
        cp=correction.correction_logits(
            native_logits=native_p,signatures=sig_p,
            question_tokens=encoded["question_paraphrase_tokens"],
            question_mask=encoded["question_paraphrase_mask"],
        )

        qn=torch.cat([qc.norm(dim=-1),qp.norm(dim=-1)])
        norm_sum+=float(qn.sum().cpu())
        zero_count+=int((qn==0).sum().item())
        count+=int(qn.numel())
        raw_vs_q_sum+=float(
            torch.cat([
                F.cosine_similarity(rawc,qc,dim=-1),
                F.cosine_similarity(rawp,qp,dim=-1),
            ]).sum().cpu()
        )
        cross_q_sum+=float(F.cosine_similarity(qc,qp,dim=-1).sum().cpu())
        corrected_agree_sum+=float(selected_choice_agreement(cc,cp))*cc.shape[0]
        state_view_encodes+=2*len(batch)

    if count!=2*len(rows):
        raise RuntimeError("S48 quotient diagnostic count changed")
    if state_view_encodes!=2*len(rows):
        raise RuntimeError("S48 quotient diagnostic state-once changed")

    return {
        "quotient_norm_mean":norm_sum/count,
        "quotient_zero_fraction":zero_count/count,
        "raw_query_vs_quotient_mean_cosine":raw_vs_q_sum/count,
        "quotient_cross_view_mean_cosine":cross_q_sum/len(rows),
        "corrected_option_ranking_cross_view_agreement":corrected_agree_sum/len(rows),
        "state_view_encodes":state_view_encodes,
    }


def _delta(reference,treatment):
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
        "canonical_relation_binding_mean_gold_margin",
        "paraphrase_relation_binding_mean_gold_margin",
        "relation_cross_view_agreement",
        "relation_cross_view_mean_js",
        "mean_same_option_signature_cosine",
        "mean_signature_same_vs_strongest_wrong_margin",
        "fused_option_order_flip_rate",
        "fused_max_probability_mass_error",
    )
    return {k:float(treatment[k])-float(reference[k]) for k in keys}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S48_A0_QUERY_QUOTIENT_OPTION_EVIDENCE_READY":
        raise RuntimeError("S48-A0 authority is not qualified")
    if a0.get("used_for_model_selection") is not False:
        raise RuntimeError("S48-A0 unexpectedly used for model selection")
    expected={
        "native_trainable_parameter_count":49_152,
        "reference_correction_parameter_count":114_688,
        "treatment_correction_parameter_count":114_688,
        "treatment_total_trainable_parameter_count":163_840,
        "quotient_trainable_parameter_count":0,
    }
    for key,value in expected.items():
        if int(a0.get(key,-1))!=value:
            raise RuntimeError(f"S48-A0 capacity changed: {key}")
    if a0.get("raw_query_bypass") is not False:
        raise RuntimeError("S48-A0 raw query bypass opened")
    if bool(a0.get("second_encoder_pass",True)):
        raise RuntimeError("S48-A0 opened a second encoder pass")
    for key in ("arbitrary_k3_pass","arbitrary_k7_pass","arbitrary_k255_pass","actual_shell_one_encoder_batch"):
        if not bool(a0.get(key,False)):
            raise RuntimeError(f"S48-A0 mechanics failed: {key}")
    if float(a0.get("orthogonal_nuisance_quotient_max_abs_error",1.0))>2e-6:
        raise RuntimeError("S48-A0 nuisance quotient invariant failed")
    if float(a0.get("orthogonal_nuisance_corrected_logit_max_abs_error",1.0))>2e-6:
        raise RuntimeError("S48-A0 raw query bypass invariant failed")

    train_rows=generate_s48_cases("train")
    dev_rows=generate_s48_cases("dev")
    validate_s48_partitions(train_rows,dev_rows)
    _assert_s48_fresh(train_rows,dev_rows)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION:
        raise RuntimeError("S48 semantic revision changed")

    if not _state_equal(_raw_factory(train=True),_quotient_factory(train=True)):
        raise RuntimeError("S48 correction initialization differs across arms")

    args.out.mkdir(parents=True,exist_ok=True)
    reference_dir=args.out/"reference-raw-query"
    treatment_dir=args.out/"treatment-query-quotient"

    reference=_run_correction_arm(
        "reference_raw_query",_raw_factory,bundle,manifest,
        train_rows,dev_rows,reference_dir,
    )
    treatment=_run_correction_arm(
        "treatment_query_quotient",_quotient_factory,bundle,manifest,
        train_rows,dev_rows,treatment_dir,
    )

    if reference["runtime_trajectory_sha256"]!=treatment["runtime_trajectory_sha256"]:
        raise RuntimeError("S48 matched native runtime trajectory diverged")
    reference["gates"]["runtime_trajectory_identity"]=True
    treatment["gates"]["runtime_trajectory_identity"]=True
    reference["dev_ready"]=all(reference["gates"].values())
    treatment["dev_ready"]=all(treatment["gates"].values())

    ref_metrics=reference["selected_dev"]
    trt_metrics=treatment["selected_dev"]

    quotient= _quotient_factory(train=False)
    runtime,payload=_load_selected(
        bundle,manifest,
        treatment_dir/"treatment-private-correction-candidate.pt",
        quotient,
    )
    diagnostics=_quotient_diagnostics(runtime,quotient,dev_rows)

    treatment_dev_ready=bool(treatment["dev_ready"])
    outcome=(
        "HIRA_V1_S48_QUERY_QUOTIENT_OPTION_EVIDENCE_DEV_READY"
        if treatment_dev_ready
        else "HIRA_V1_S48_QUERY_QUOTIENT_OPTION_EVIDENCE_DEV_COMPLETE"
    )

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

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":outcome,
        "scientific_authority":"V1_S48_FRESH_MATCHED_RAW_QUERY_VS_QUERY_QUOTIENT_CORRECTION",
        "seed":SEED,
        "training_mechanics":{
            "family":"matched_private_correction_query_interface",
            "reference":"S45_raw_normalized_mean_query",
            "treatment":"S48_option_difference_covariance_direction_quotient",
            "epochs":s35.EPOCHS,
            "batch_size_semantic_cases":s35.BATCH_SIZE,
            "lr":s35.LR,
            "weight_decay":s35.WEIGHT_DECAY,
            "grad_clip":s35.GRAD_CLIP,
            "correction_ce_coefficient":s35.BINDING_COEFFICIENT,
            "correction_cross_view_js_coefficient":s35.INVARIANCE_COEFFICIENT,
            "checkpoint_selection_rule":"exact_s35_selection_key_per_arm",
            "correction_initialization_identical":True,
            "second_encoder_pass":False,
        },
        "parameter_surface":{
            "native_trainable_parameters":49_152,
            "reference_correction_parameters":114_688,
            "treatment_correction_parameters":114_688,
            "reference_total_trainable_parameters":163_840,
            "treatment_total_trainable_parameters":163_840,
            "quotient_trainable_parameters":0,
        },
        "partitions":{
            "train_semantic_cases":len(train_rows),
            "dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),
            "k":4,
            "views_per_option":2,
            "identical_rows_across_arms":True,
            "prior_track_exact_rows_used":False,
            "s47_train_dev_rows_used":False,
            "s48_a0_rows_used":False,
        },
        "reference_arm":reference,
        "treatment_arm":treatment,
        "matched_dev":{
            "reference_raw_query":ref_metrics,
            "treatment_query_quotient":trt_metrics,
            "delta_treatment_minus_reference":_delta(ref_metrics,trt_metrics),
            "treatment_quotient_diagnostics":diagnostics,
            "treatment_dev_ready":treatment_dev_ready,
        },
        "selected_treatment_checkpoint":{
            "epoch":int(payload["selected_dev_epoch"]),
            "total_parameter_count":int(payload["total_parameter_count"]),
            "correction_parameter_count":int(payload["correction_parameter_count"]),
        },
        "trajectory_invariant":{
            "all_epoch_runtime_state_sha256_equal":True,
            "reference_runtime_trajectory_sha256":reference["runtime_trajectory_sha256"],
            "treatment_runtime_trajectory_sha256":treatment["runtime_trajectory_sha256"],
            "epoch_count":len(reference["history"]),
        },
        "post_dev_tuning_performed":False,
        "second_dev_run_performed":False,
        "sealed_confirm_opened":False,
        "external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
        "train_manifest_sha256":_sha256(train_manifest),
        "dev_manifest_sha256":_sha256(dev_manifest),
    }

    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S48_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
