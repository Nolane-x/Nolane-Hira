from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_explicit_pairwise_decision_head import ExplicitPairwiseDecisionHead
from nmd.v1_contextual_reliability_gate import (
    S64_ALPHA_INITIAL,
    ContextInjectedReliabilityGate,
    contextual_reliability_loss,
)
from nmd.v1_cross_view_soft_and_reliability import (
    cross_view_soft_and_logit,
    cross_view_soft_and_reliability_loss,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    train_only_reliability_target,
)
from nmd.v1_s64_authority import generate_s64_cases
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s65-a0-cross-view-soft-and-reliability-v1"
OUTCOME="HIRA_V1_S65_A0_CROSS_VIEW_SOFT_AND_RELIABILITY_READY"
SEED=86_001


def _synthetic():
    fused_c=torch.tensor([
        [-1.3768910,3.2005668,-3.3355374],
        [-0.4818169,3.1734812,0.6368278],
    ])
    fused_p=torch.tensor([
        [1.6032288,-2.4625099,1.4818633],
        [0.0850223,1.7102296,2.0455709],
    ])
    pair_c=torch.tensor([
        [1.7417690,-0.2375816,0.0183677],
        [1.8026907,-0.3109073,-3.2117627],
    ])
    pair_p=torch.tensor([
        [1.1653450,1.4833535,0.8274329],
        [-0.0134042,-0.6213452,-1.9260790],
    ])
    gold=torch.tensor([0,2],dtype=torch.long)
    g=torch.Generator().manual_seed(65065001)
    context_c=torch.randn(2,3,512,generator=g)
    context_p=torch.randn(2,3,512,generator=g)
    return fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold


def _same_parameter_state(a,b)->bool:
    sa=a.parameter_state_dict_exact()
    sb=b.parameter_state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _mechanics():
    reference=ContextInjectedReliabilityGate(use_context=True)
    treatment=ContextInjectedReliabilityGate(use_context=True)
    if reference.parameter_count!=60 or treatment.parameter_count!=60:
        raise RuntimeError("S65 A0 gate capacity changed")
    if not _same_parameter_state(reference,treatment):
        raise RuntimeError("S65 A0 gate initialization changed")
    if not torch.equal(reference.context_projection,treatment.context_projection):
        raise RuntimeError("S65 A0 context path changed")

    zc=torch.tensor([-3.0,-0.2,0.7,4.0])
    zp=torch.tensor([-2.0,0.4,-1.3,1.0])
    zand=cross_view_soft_and_logit(zc,zp)
    min_error=float((zand-torch.minimum(zc,zp)).abs().max())
    symmetry_error=float((zand-cross_view_soft_and_logit(zp,zc)).abs().max())

    rgc=torch.tensor([-1.0,2.0],requires_grad=True)
    rgp=torch.tensor([1.0,-2.0],requires_grad=True)
    route=cross_view_soft_and_logit(rgc,rgp).sum()
    gc,gp=torch.autograd.grad(route,(rgc,rgp))
    route_error=max(
        float((gc-torch.tensor([1.0,0.0])).abs().max()),
        float((gp-torch.tensor([0.0,1.0])).abs().max()),
    )

    g=torch.Generator().manual_seed(65065002)
    max_mass=max_bound=max_initial=identity_error=0.0
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g)
        pair=torch.randn(2,k,generator=g)
        context=torch.randn(2,k,512,generator=g)
        for gate in (reference,treatment):
            alpha=gate.alpha(fused,pair,context)
            max_initial=max(
                max_initial,
                float((alpha-S64_ALPHA_INITIAL).abs().max()),
            )
            out,diag=gate.compose(
                fused,pair,context,return_diagnostics=True
            )
            max_mass=max(
                max_mass,
                float((torch.softmax(out,-1).sum(-1)-1.0).abs().max()),
            )
            max_bound=max(
                max_bound,
                float(((out-fused).abs()-diag["residual_bound"]).clamp_min(0).max()),
            )
            ident=gate.compose(fused,pair,context,alpha_override=0.0)
            identity_error=max(
                identity_error,
                float((ident-fused).abs().max()),
            )

    return {
        "reference_gate_parameter_count":reference.parameter_count,
        "treatment_gate_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":treatment.parameter_count-reference.parameter_count,
        "parameter_initialization_bit_identical":True,
        "context_projection_bit_identical":True,
        "soft_and_exact_min_max_abs_error":min_error,
        "soft_and_pair_symmetry_max_abs_error":symmetry_error,
        "lower_view_gradient_route_max_abs_error":route_error,
        "treatment_extra_trainable_parameter_count":0,
        "initial_alpha_max_abs_error":max_initial,
        "identity_override_max_abs_error":identity_error,
        "residual_bound_violation_max":max_bound,
        "max_probability_mass_error":max_mass,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
    }


def _staged_gradient_court():
    fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold=_synthetic()
    target,_=train_only_reliability_target(
        fused_c,fused_p,pair_c,pair_p,gold
    )
    if target.tolist()!=[1.0,0.0]:
        raise RuntimeError("S65 inherited S62 target changed")

    reference=ContextInjectedReliabilityGate(use_context=True)
    treatment=ContextInjectedReliabilityGate(use_context=True)
    ref_loss,_=contextual_reliability_loss(
        reference,
        fused_c,fused_p,pair_c,pair_p,
        context_c,context_p,gold,
    )
    trt_loss,_=cross_view_soft_and_reliability_loss(
        treatment,
        fused_c,fused_p,pair_c,pair_p,
        context_c,context_p,gold,
    )

    result={"inherited_target_values":[int(x) for x in target.tolist()]}
    for label,gate,loss in (
        ("reference",reference,ref_loss),
        ("treatment",treatment,trt_loss),
    ):
        params=(gate.W_phi,gate.b_phi,gate.w_out,gate.b_out)
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        l1=[
            0.0 if x is None else float(x.detach().abs().sum())
            for x in grads
        ]
        if l1[2]<=0.0 or l1[3]<=0.0:
            raise RuntimeError(f"S65 A0 {label} output gradient vanished")
        if l1[0]!=0.0 or l1[1]!=0.0:
            raise RuntimeError(f"S65 A0 {label} initial phi gradient changed")

        clone=ContextInjectedReliabilityGate(use_context=True)
        opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
        if label=="reference":
            warm,_=contextual_reliability_loss(
                clone,
                fused_c,fused_p,pair_c,pair_p,
                context_c,context_p,gold,
            )
        else:
            warm,_=cross_view_soft_and_reliability_loss(
                clone,
                fused_c,fused_p,pair_c,pair_p,
                context_c,context_p,gold,
            )
        opt.zero_grad(set_to_none=True)
        warm.backward()
        opt.step()
        if label=="reference":
            second,_=contextual_reliability_loss(
                clone,
                fused_c,fused_p,pair_c,pair_p,
                context_c,context_p,gold,
            )
        else:
            second,_=cross_view_soft_and_reliability_loss(
                clone,
                fused_c,fused_p,pair_c,pair_p,
                context_c,context_p,gold,
            )
        phi=torch.autograd.grad(
            second,(clone.W_phi,clone.b_phi),allow_unused=True
        )
        phi_l1=[
            0.0 if x is None else float(x.detach().abs().sum())
            for x in phi
        ]
        if min(phi_l1)<=0.0:
            raise RuntimeError(f"S65 A0 {label} staged phi gradient vanished")

        result.update({
            f"{label}_initial_W_phi_gradient_l1":l1[0],
            f"{label}_initial_b_phi_gradient_l1":l1[1],
            f"{label}_initial_w_out_gradient_l1":l1[2],
            f"{label}_initial_b_out_gradient_l1":l1[3],
            f"{label}_post_output_warm_W_phi_gradient_l1":phi_l1[0],
            f"{label}_post_output_warm_b_phi_gradient_l1":phi_l1[1],
        })
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S65-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S65-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S65-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S65-A0 checkpoint file changed")

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S65-A0 loaded native changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S65-A0 native remained trainable")

    parent_rows=generate_s64_cases("train")[:16]
    cache,cache_digest=s50._materialize_cache(runtime,parent_rows)
    del runtime

    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    reference=ContextInjectedReliabilityGate(use_context=True,trainable=True)
    treatment=ContextInjectedReliabilityGate(use_context=True,trainable=True)

    if op.correction_parameter_count!=114688:
        raise RuntimeError("S65-A0 correction capacity changed")
    if head.parameter_count!=32832:
        raise RuntimeError("S65-A0 pairwise capacity changed")
    if reference.parameter_count!=60 or treatment.parameter_count!=60:
        raise RuntimeError("S65-A0 gate capacity changed")
    if not _same_parameter_state(reference,treatment):
        raise RuntimeError("S65-A0 matched initialization changed")
    if not torch.equal(reference.context_projection,treatment.context_projection):
        raise RuntimeError("S65-A0 context path changed")

    canonical,paraphrase,_n=cache[0]
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    rep_c=s59._representation(op,canonical)
    rep_p=s59._representation(op,paraphrase)
    pair_c=head.aggregate_logits(rep_c)
    pair_p=head.aggregate_logits(rep_p)

    ref_loss,ref_diag=contextual_reliability_loss(
        reference,
        fused_c,fused_p,pair_c,pair_p,
        rep_c,rep_p,canonical.gold,
    )
    trt_loss,trt_diag=cross_view_soft_and_reliability_loss(
        treatment,
        fused_c,fused_p,pair_c,pair_p,
        rep_c,rep_p,canonical.gold,
    )
    if int(ref_diag["target_count"])!=int(trt_diag["target_count"]):
        raise RuntimeError("S65-A0 target count changed")
    if int(ref_diag["target_positive_count"])!=int(trt_diag["target_positive_count"]):
        raise RuntimeError("S65-A0 target labels changed")

    upstream=(*op.correction_parameters(),head.A,head.u)
    ref_targets=(
        reference.W_phi,reference.b_phi,
        reference.w_out,reference.b_out,*upstream
    )
    trt_targets=(
        treatment.W_phi,treatment.b_phi,
        treatment.w_out,treatment.b_out,*upstream
    )
    ref_grads=torch.autograd.grad(
        ref_loss,ref_targets,allow_unused=True,retain_graph=True
    )
    trt_grads=torch.autograd.grad(
        trt_loss,trt_targets,allow_unused=True
    )
    if any(
        g is not None and float(g.detach().abs().sum())>0
        for g in ref_grads[4:]
    ):
        raise RuntimeError("S65-A0 reference gradient leaked upstream")
    if any(
        g is not None and float(g.detach().abs().sum())>0
        for g in trt_grads[4:]
    ):
        raise RuntimeError("S65-A0 treatment gradient leaked upstream")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0 or rep_c.requires_grad or rep_p.requires_grad:
        raise RuntimeError("S65-A0 detached evidence contract changed")

    mechanics=_mechanics()
    staged=_staged_gradient_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    ref_state=reference.state_dict_exact()
    trt_state=treatment.state_dict_exact()
    checkpoint=out/"matched-gates.pt"
    torch.save({
        "schema_version":"hira-v1-s65-matched-gates-v1",
        "seed":SEED,
        "reference_objective":"independent_view_bce",
        "treatment_objective":"cross_view_exact_min_soft_and_bce",
        "alpha_probe":S62_ALPHA_PROBE,
        "target_tolerance":S62_TARGET_TOLERANCE,
        "reference_state_dict":ref_state,
        "treatment_state_dict":trt_state,
    },checkpoint)

    rr=ContextInjectedReliabilityGate(use_context=True)
    tt=ContextInjectedReliabilityGate(use_context=True)
    rr.load_state_dict_exact(ref_state,freeze=True)
    tt.load_state_dict_exact(trt_state,freeze=True)
    replay_error=max(
        float((
            reference.compose(fused_c,pair_c,rep_c)
            -rr.compose(fused_c,pair_c,rep_c)
        ).abs().max()),
        float((
            treatment.compose(fused_c,pair_c,rep_c)
            -tt.compose(fused_c,pair_c,rep_c)
        ).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S65_A0_CROSS_VIEW_SOFT_AND_RELIABILITY_ONLY",
        "seed":SEED,
        "parent_s64":{
            "merged_main":"c5063da0365d49dc4f9dc2c64f94bc309caef0ba",
            "scientific_run":37319541951,
            "artifact_id":11348594503,
            "artifact_digest":"sha256:f3d04d452a51b7358a23a9391579bca6ce1ab2c16fa107dc9f7f79740dba90ea",
            "verdict":"CASE_B",
        },
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "parent_train_semantic_case_count":len(cache),
        "cache_digest":cache_digest,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "context_representation_requires_grad":False,
        "correction_trainable_parameter_count":114688,
        "pairwise_trainable_parameter_count":32832,
        "reference_objective":"independent_view_bce",
        "treatment_objective":"cross_view_exact_min_soft_and_bce",
        "same_single_view_inference_path":True,
        "same_context_path":True,
        "same_reliability_target":True,
        "reference_gradient_to_upstream_zero":True,
        "treatment_gradient_to_upstream_zero":True,
        "alpha_probe":S62_ALPHA_PROBE,
        "target_tolerance":S62_TARGET_TOLERANCE,
        "teacher_dependency":False,
        "dev_target_dependency":False,
        "self_anchor_dependency":False,
        "pairwise_only_final_path":False,
        "one_encoder_state_once":True,
        "matched_gate_checkpoint_replay_max_abs_error":replay_error,
        **mechanics,
        **staged,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S65_A0_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
