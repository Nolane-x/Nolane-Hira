from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F

from nmd.local_runtime import A13_REVISION, load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import load_a13_lora_state_dict
from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_s17_semantic_core import build_hira_v1_s17_norm_balanced_core, enforce_s17_eval
from nmd.v1_s48_authority import generate_s48_cases
from nmd.v1_s49_authority import generate_s49_cases, validate_s49_partitions
import hira_v1_s17_train_dev as s17mod
import hira_v1_s35_train_dev as s35
import hira_v1_s45_train_dev as s45
from hira_v1_s49_a0_query_free_option_identity import cases as s49_a0_cases

SCHEMA_VERSION="hira-v1-s49-matched-private-query-free-option-identity-train-dev-v1"
SEED=70_001
CORRECTION=114_688
TOTAL=163_840


def _sha256(path:Path)->str:
    d=sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda:h.read(1024*1024),b""): d.update(chunk)
    return d.hexdigest()


def _states(r): return (r.state_a,r.state_b)
def _questions(r): return (r.question_a1,r.question_a2,r.question_b1,r.question_b2)
def _options(r): return (*r.option_texts,*r.option_aliases)


def _assert_fresh(train_rows,dev_rows):
    s45._assert_s45_fresh(train_rows,dev_rows)
    cur=(*train_rows,*dev_rows)
    cs={x for r in cur for x in _states(r)}
    cq={x for r in cur for x in _questions(r)}
    co={x for r in cur for x in _options(r)}
    prior=(*generate_s48_cases("train"),*generate_s48_cases("dev"))
    if cs & {x for r in prior for x in _states(r)}: raise RuntimeError("S49 state overlap with S48")
    if cq & {x for r in prior for x in _questions(r)}: raise RuntimeError("S49 question overlap with S48")
    if co & {x for r in prior for x in _options(r)}: raise RuntimeError("S49 option overlap with S48")

    a0=s49_a0_cases()
    a0s={x for r in a0 for x in (r.state_a,r.state_b)}
    a0q={x for r in a0 for x in (r.qa1,r.qa2,r.qb1,r.qb2)}
    a0o=set()
    for case in a0:
        opts,_ga,_gb=case.option_pack()
        for o in opts:
            a0o.add(o.criterion_text); a0o.update(o.aliases)
    if cs&a0s: raise RuntimeError("S49 state overlap with S49-A0")
    if cq&a0q: raise RuntimeError("S49 question overlap with S49-A0")
    if co&a0o: raise RuntimeError("S49 option overlap with S49-A0")


def _s49_correction_block(correction,relation_c,relation_p,_signature_c,_signature_p,encoded,rows):
    gold,_=s35._gold_tensors(rows,device=relation_c.device)
    opt=encoded["option_tokens"].repeat_interleave(2,dim=0)
    om=encoded["option_mask"].repeat_interleave(2,dim=0)
    vm=encoded["option_view_mask"].repeat_interleave(2,dim=0)
    cc,_=correction.correction_logits_from_state_option(
        native_logits=relation_c,
        state_tokens=encoded["state_a_tokens"].repeat_interleave(2,dim=0),
        state_mask=encoded["state_a_mask"].repeat_interleave(2,dim=0),
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    cp,_=correction.correction_logits_from_state_option(
        native_logits=relation_p,
        state_tokens=encoded["state_b_tokens"].repeat_interleave(2,dim=0),
        state_mask=encoded["state_b_mask"].repeat_interleave(2,dim=0),
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    ce=0.5*(F.cross_entropy(cc,gold)+F.cross_entropy(cp,gold))
    js=s17mod.symmetric_js_divergence(cc,cp)
    return s35.BINDING_COEFFICIENT*ce+s35.INVARIANCE_COEFFICIENT*js,ce,js


def _s49_treatment_relation_outputs(runtime,*,state_tokens,state_mask,question_tokens,question_mask,encoded):
    correction=s45._ACTIVE_CORRECTION
    if correction is None or not isinstance(correction,QueryFreeIdentityPrivateCorrectionFork):
        raise RuntimeError("S49 active correction missing")
    native_logits,native_signature,diag=s35._native_relation_outputs(
        runtime,
        state_tokens=state_tokens,
        state_mask=state_mask,
        question_tokens=question_tokens,
        question_mask=question_mask,
        encoded=encoded,
    )
    corrected,_identity=correction.correction_logits_from_state_option(
        native_logits=native_logits,
        state_tokens=state_tokens.repeat_interleave(2,dim=0),
        state_mask=state_mask.repeat_interleave(2,dim=0),
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2,dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2,dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2,dim=0),
        question_tokens=question_tokens,
        question_mask=question_mask,
    )
    # Keep native signatures in the frozen selection/gating semantics.
    return corrected,native_signature,diag


def _run_reference(bundle,manifest,train_rows,dev_rows,out_dir):
    return s45._train_arm("treatment",bundle,manifest,train_rows,dev_rows,out_dir)


def _run_treatment(bundle,manifest,train_rows,dev_rows,out_dir):
    original_cls=s45.PrivateCorrectionRepresentationFork
    original_block=s45._correction_block
    original_relation=s45._treatment_relation_outputs
    s45.PrivateCorrectionRepresentationFork=QueryFreeIdentityPrivateCorrectionFork
    s45._correction_block=_s49_correction_block
    s45._treatment_relation_outputs=_s49_treatment_relation_outputs
    try:
        return s45._train_arm("treatment",bundle,manifest,train_rows,dev_rows,out_dir)
    finally:
        s45.PrivateCorrectionRepresentationFork=original_cls
        s45._correction_block=original_block
        s45._treatment_relation_outputs=original_relation


def _load_treatment(bundle,manifest,checkpoint):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if payload["arm"]!="treatment": raise RuntimeError("S49 checkpoint arm changed")
    if int(payload["total_parameter_count"])!=TOTAL: raise RuntimeError("S49 total changed")
    if int(payload["correction_parameter_count"])!=CORRECTION: raise RuntimeError("S49 correction changed")
    frozen=load_hira_v0_m4_bundle(bundle)
    runtime=build_hira_v1_s17_norm_balanced_core(
        frozen.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=True,train_projection=True,
    )
    del frozen
    runtime.clear_schema_cache()
    load_a13_lora_state_dict(runtime.encoder,payload["lora_state_dict"],freeze=True)
    scorer=runtime.projection_triadic_scorer
    if scorer is None: raise RuntimeError("S49 projection scorer missing")
    scorer.load_projection_state_dict(payload["projection_state_dict"],freeze=True)
    correction=QueryFreeIdentityPrivateCorrectionFork(train_correction=False)
    correction.load_correction_state_dict(payload["correction_state_dict"],freeze=True)
    enforce_s17_eval(runtime); correction.eval()
    return runtime,correction,payload


@torch.inference_mode()
def _identity_diagnostics(runtime,correction,rows):
    same_sum=margin_sum=0.0
    count=0
    raw_query_delta_sum=0.0
    raw_query_count=0
    s45._set_relation_arm("reference",None)
    for start in range(0,len(rows),s35.BATCH_SIZE):
        batch=list(rows[start:start+s35.BATCH_SIZE])
        (
            _total,_primary,_relation,_pieces,_fc,_fp,_rawc,_rawp,
            native_c,native_p,_sigc,_sigp,encoded,
        )=s45._native_losses(runtime,batch)
        opt=encoded["option_tokens"].repeat_interleave(2,dim=0)
        om=encoded["option_mask"].repeat_interleave(2,dim=0)
        vm=encoded["option_view_mask"].repeat_interleave(2,dim=0)
        idc=correction.identity_signatures(
            state_tokens=encoded["state_a_tokens"].repeat_interleave(2,dim=0),
            state_mask=encoded["state_a_mask"].repeat_interleave(2,dim=0),
            option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        )
        idp=correction.identity_signatures(
            state_tokens=encoded["state_b_tokens"].repeat_interleave(2,dim=0),
            state_mask=encoded["state_b_mask"].repeat_interleave(2,dim=0),
            option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        )
        same=F.cosine_similarity(idc,idp,dim=-1)
        cross=torch.einsum("nkd,njd->nkj",F.normalize(idc,dim=-1),F.normalize(idp,dim=-1))
        k=cross.shape[-1]
        eye=torch.eye(k,dtype=torch.bool,device=cross.device)[None]
        wrong=cross.masked_fill(eye,float("-inf")).amax(-1)
        same_sum+=float(same.sum().cpu())
        margin_sum+=float((same-wrong).sum().cpu())
        count+=int(same.numel())

        cc,_=correction.correction_logits_from_state_option(
            native_logits=native_c,
            state_tokens=encoded["state_a_tokens"].repeat_interleave(2,dim=0),
            state_mask=encoded["state_a_mask"].repeat_interleave(2,dim=0),
            option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
            question_tokens=encoded["question_canonical_tokens"],
            question_mask=encoded["question_canonical_mask"],
        )
        # Swap the two relation-query rows inside each semantic case while
        # holding query-free identity inputs fixed.
        n=len(batch)
        swap=torch.arange(2*n,device=cc.device).reshape(n,2)[:,[1,0]].reshape(-1)
        cc_swap,_=correction.correction_logits_from_state_option(
            native_logits=native_c,
            state_tokens=encoded["state_a_tokens"].repeat_interleave(2,dim=0),
            state_mask=encoded["state_a_mask"].repeat_interleave(2,dim=0),
            option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
            question_tokens=encoded["question_canonical_tokens"][swap],
            question_mask=encoded["question_canonical_mask"][swap],
        )
        raw_query_delta_sum+=float((cc-cc_swap).abs().sum().cpu())
        raw_query_count+=int(cc.numel())

    return {
        "identity_cross_state_view_same_option_cosine":same_sum/count,
        "identity_same_vs_strongest_wrong_margin":margin_sum/count,
        "fixed_identity_raw_query_swap_mean_abs_logit_delta":raw_query_delta_sum/raw_query_count,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--a0-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    a0=json.loads(args.a0_result.read_text(encoding="utf-8"))
    if a0.get("outcome")!="HIRA_V1_S49_A0_PRIVATE_QUERY_FREE_OPTION_IDENTITY_READY": raise RuntimeError("S49-A0 not qualified")
    if a0.get("used_for_model_selection") is not False: raise RuntimeError("S49-A0 used for selection")
    for key,value in {
        "native_trainable_parameter_count":49152,
        "correction_parameter_count":CORRECTION,
        "treatment_total_trainable_parameter_count":TOTAL,
        "identity_parameter_count":0,
    }.items():
        if int(a0.get(key,-1))!=value: raise RuntimeError(f"S49-A0 surface changed: {key}")
    if a0.get("identity_api_question_inputs_absent") is not True: raise RuntimeError("S49-A0 query leak")
    if bool(a0.get("second_encoder_pass",True)): raise RuntimeError("S49-A0 second encoder")

    train_rows=generate_s49_cases("train"); dev_rows=generate_s49_cases("dev")
    validate_s49_partitions(train_rows,dev_rows); _assert_fresh(train_rows,dev_rows)

    bundle=args.bundle.resolve(); manifest=read_runtime_bundle_manifest(bundle)
    if str(manifest["semantic_revision"])!=A13_REVISION: raise RuntimeError("S49 semantic revision changed")
    args.out.mkdir(parents=True,exist_ok=True)
    ref_dir=args.out/"reference-question-conditioned-signature"
    trt_dir=args.out/"treatment-query-free-identity"
    ref_dir.mkdir(parents=True,exist_ok=True); trt_dir.mkdir(parents=True,exist_ok=True)

    # Exact correction initialization equality before any TRAIN exposure.
    ref_init=PrivateCorrectionRepresentationFork(train_correction=True).correction_state_dict()
    trt_init=QueryFreeIdentityPrivateCorrectionFork(train_correction=True).correction_state_dict()
    init_equal=all(torch.equal(ref_init[k],trt_init[k]) for k in ref_init)
    if not init_equal: raise RuntimeError("S49 correction initialization diverged")

    s45.SEED=SEED
    reference=_run_reference(bundle,manifest,train_rows,dev_rows,ref_dir)
    treatment=_run_treatment(bundle,manifest,train_rows,dev_rows,trt_dir)
    reference["arm"]="reference_question_conditioned_signature"
    treatment["arm"]="treatment_query_free_identity"

    if reference["runtime_trajectory_sha256"]!=treatment["runtime_trajectory_sha256"]:
        raise RuntimeError("S49 native trajectories diverged")
    treatment["gates"]["runtime_trajectory_identity"]=True
    treatment["dev_ready"]=all(treatment["gates"].values())

    ref_cp=ref_dir/"treatment-private-correction-candidate.pt"
    trt_cp=trt_dir/"treatment-private-correction-candidate.pt"
    root_ref=args.out/"reference-question-conditioned-private-candidate.pt"
    root_trt=args.out/"treatment-query-free-identity-private-candidate.pt"
    shutil.copy2(ref_cp,root_ref); shutil.copy2(trt_cp,root_trt)

    runtime,correction,payload=_load_treatment(bundle,manifest,trt_cp)
    identity_diag=_identity_diagnostics(runtime,correction,dev_rows)

    train_manifest=args.out/"train-manifest.json"; dev_manifest=args.out/"dev-manifest.json"
    train_manifest.write_text(json.dumps([r.to_dict() for r in train_rows],indent=2,sort_keys=True)+"\n",encoding="utf-8")
    dev_manifest.write_text(json.dumps([r.to_dict() for r in dev_rows],indent=2,sort_keys=True)+"\n",encoding="utf-8")

    delta=s45._metric_delta_values(reference["selected_dev"],treatment["selected_dev"])
    outcome=("HIRA_V1_S49_PRIVATE_QUERY_FREE_OPTION_IDENTITY_DEV_READY" if treatment["dev_ready"] else "HIRA_V1_S49_PRIVATE_QUERY_FREE_OPTION_IDENTITY_DEV_COMPLETE")

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS","outcome":outcome,
        "scientific_authority":"V1_S49_FRESH_MATCHED_QUESTION_CONDITIONED_SIGNATURE_VS_QUERY_FREE_IDENTITY",
        "seed":SEED,
        "training_mechanics":{
            "epochs":s35.EPOCHS,"batch_size_semantic_cases":s35.BATCH_SIZE,
            "lr":s35.LR,"weight_decay":s35.WEIGHT_DECAY,"grad_clip":s35.GRAD_CLIP,
            "correction_ce_coefficient":s35.BINDING_COEFFICIENT,
            "correction_cross_view_js_coefficient":s35.INVARIANCE_COEFFICIENT,
            "same_checkpoint_selection_rule":True,"second_encoder_pass":False,
        },
        "controlled_variable":{
            "reference_private_signature":"question_conditioned_native_relation_signature",
            "treatment_private_signature":"query_free_state_option_identity",
            "raw_query_readout_matched":True,
            "reference_correction_parameters":CORRECTION,
            "treatment_correction_parameters":CORRECTION,
            "identity_trainable_parameters":0,
            "correction_initialization_exact":init_equal,
        },
        "partitions":{
            "train_semantic_cases":len(train_rows),"dev_semantic_cases":len(dev_rows),
            "domains":sorted({r.domain for r in train_rows}),"k":4,"views_per_option":2,
            "identical_rows_across_arms":True,"s48_train_dev_rows_used":False,"s49_a0_rows_used":False,
        },
        "reference_arm":reference,"treatment_arm":treatment,
        "matched_selected_dev_delta_treatment_minus_reference":delta,
        "selected_treatment_checkpoint":{
            "epoch":int(payload["selected_dev_epoch"]),
            "total_parameter_count":int(payload["total_parameter_count"]),
            "correction_parameter_count":int(payload["correction_parameter_count"]),
        },
        "treatment_identity_diagnostics":identity_diag,
        "trajectory_invariant":{
            "all_epoch_runtime_state_sha256_equal":True,
            "reference_runtime_trajectory_sha256":reference["runtime_trajectory_sha256"],
            "treatment_runtime_trajectory_sha256":treatment["runtime_trajectory_sha256"],
            "epoch_count":len(reference["history"]),
        },
        "post_dev_tuning_performed":False,"second_dev_run_performed":False,
        "sealed_confirm_opened":False,"external_laya_jev_evaluation_opened":False,
        "production_ready_claimed":False,
        "train_manifest_sha256":_sha256(train_manifest),"dev_manifest_sha256":_sha256(dev_manifest),
        "reference_checkpoint_sha256":_sha256(root_ref),"treatment_checkpoint_sha256":_sha256(root_trt),
    }
    (args.out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("HIRA_V1_S49_TRAIN_DEV_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
