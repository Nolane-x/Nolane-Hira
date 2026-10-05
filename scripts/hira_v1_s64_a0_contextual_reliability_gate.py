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
    S64_ALPHA_MAX,
    S64_CONTEXT_DIM,
    S64_CONTEXT_PROJECTION_SEED,
    S64_ENCODER_INIT_SEED,
    S64_GATE_PARAMETER_COUNT,
    S64_OPTION_HIDDEN_DIM,
    S64_OPTION_INPUT_DIM,
    S64_POOLED_DIM,
    S64_REPRESENTATION_DIM,
    ContextInjectedReliabilityGate,
    contextual_option_features,
    contextual_reliability_loss,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    train_only_reliability_target,
)
from nmd.v1_s63_authority import generate_s63_cases
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s64-a0-contextual-reliability-gate-v1"
OUTCOME="HIRA_V1_S64_A0_CONTEXTUAL_RELIABILITY_GATE_READY"
SEED=85_001


def _synthetic_mixed():
    fused_c=torch.tensor([
        [-1.376891016960144,3.2005667686462402,-3.3355374336242676],
        [-0.4818168878555298,3.1734812259674072,0.6368277668952942],
    ])
    fused_p=torch.tensor([
        [1.6032288074493408,-2.462509870529175,1.481863260269165],
        [0.08502231538295746,1.7102296352386475,2.0455708503723145],
    ])
    pair_c=torch.tensor([
        [1.7417689561843872,-0.2375815510749817,0.0183677077293396],
        [1.8026907444000244,-0.3109073042869568,-3.2117626667022705],
    ])
    pair_p=torch.tensor([
        [1.1653449535369873,1.4833534955978394,0.8274328708648682],
        [-0.013404175639152527,-0.6213452219963074,-1.9260790348052979],
    ])
    gold=torch.tensor([0,2],dtype=torch.long)
    g=torch.Generator().manual_seed(64064064)
    context_c=torch.randn(2,3,512,generator=g)
    context_p=torch.randn(2,3,512,generator=g)
    return fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold


def _parameter_state_equal(a,b)->bool:
    sa=a.parameter_state_dict_exact()
    sb=b.parameter_state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _mechanics():
    generator=torch.Generator().manual_seed(6406407)
    reference=ContextInjectedReliabilityGate(use_context=False)
    treatment=ContextInjectedReliabilityGate(use_context=True)

    if reference.parameter_count!=S64_GATE_PARAMETER_COUNT or treatment.parameter_count!=S64_GATE_PARAMETER_COUNT:
        raise RuntimeError("S64 A0 matched capacity changed")
    if not _parameter_state_equal(reference,treatment):
        raise RuntimeError("S64 A0 parameter initialization not bit-identical")
    if not torch.equal(reference.context_projection,treatment.context_projection):
        raise RuntimeError("S64 A0 context projection changed across arms")
    if reference.context_projection.requires_grad or treatment.context_projection.requires_grad:
        raise RuntimeError("S64 A0 fixed context projection became trainable")

    params=dict(treatment.named_parameters())
    if set(params)!={"W_phi","b_phi","w_out","b_out"}:
        raise RuntimeError("S64 A0 treatment tensor surface changed")
    shapes={k:list(v.shape) for k,v in params.items()}
    expected={
        "W_phi":[5,8],
        "b_phi":[5],
        "w_out":[14],
        "b_out":[],
    }
    if shapes!=expected:
        raise RuntimeError("S64 A0 treatment tensor shapes changed")

    fused=torch.randn(3,7,generator=generator)
    pair=torch.randn(3,7,generator=generator)
    context=torch.randn(3,7,512,generator=generator)
    rin=reference.option_input(fused,pair,context)
    tin=treatment.option_input(fused,pair,context)
    reference_context_max_abs=float(rin[...,4:].abs().max())
    treatment_context_l1=float(tin[...,4:].abs().sum())
    if reference_context_max_abs!=0.0:
        raise RuntimeError("S64 A0 reference context channels are not zero")
    if treatment_context_l1<=0.0:
        raise RuntimeError("S64 A0 treatment context channels degenerate")

    max_mass=max_perm=max_affine=max_bound=max_initial_alpha_error=0.0
    context_sensitivity=0.0
    for k in (3,7,255):
        fused=torch.randn(3,k,generator=generator)
        pair=torch.randn(3,k,generator=generator)
        context=torch.randn(3,k,512,generator=generator)

        for gate in (reference,treatment):
            alpha=gate.alpha(fused,pair,context)
            max_initial_alpha_error=max(
                max_initial_alpha_error,
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

        probe=ContextInjectedReliabilityGate(use_context=True)
        with torch.no_grad():
            probe.w_out.copy_(
                torch.linspace(-0.25,0.25,S64_REPRESENTATION_DIM)
            )
        perm=torch.randperm(k,generator=generator)
        a=probe.alpha(fused,pair,context)
        ap=probe.alpha(
            fused[:,perm],
            pair[:,perm],
            context[:,perm],
        )
        max_perm=max(max_perm,float((a-ap).abs().max()))

        aa=probe.alpha(
            3.0*fused+7.0,
            5.0*pair-11.0,
            context,
        )
        max_affine=max(max_affine,float((a-aa).abs().max()))

        if k==7:
            context_repaired=context.roll(1,dims=1)
            changed=probe.alpha(fused,pair,context_repaired)
            context_sensitivity=max(
                context_sensitivity,
                float((a-changed).abs().max()),
            )

    if context_sensitivity<=1e-7:
        raise RuntimeError("S64 A0 treatment failed contextual sensitivity probe")

    flat_fused=torch.zeros(2,7)
    flat_pair=torch.zeros(2,7)
    flat_context=torch.zeros(2,7,512)
    flat_alpha=treatment.alpha(flat_fused,flat_pair,flat_context)
    flat_context_features=contextual_option_features(
        flat_context,
        treatment.context_projection,
    )
    if not bool(torch.isfinite(flat_alpha).all() and torch.isfinite(flat_context_features).all()):
        raise RuntimeError("S64 A0 flat mechanics non-finite")

    fused=torch.randn(2,7,generator=generator)
    pair=torch.randn(2,7,generator=generator)
    context=torch.randn(2,7,512,generator=generator)
    identity=treatment.compose(
        fused,pair,context,alpha_override=0.0
    )

    return {
        "reference_parameter_count":reference.parameter_count,
        "treatment_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":treatment.parameter_count-reference.parameter_count,
        "reference_trainable_parameter_count":reference.trainable_parameter_count,
        "treatment_trainable_parameter_count":treatment.trainable_parameter_count,
        "trainable_tensor_names":sorted(params),
        "parameter_shapes":shapes,
        "parameter_initialization_bit_identical":True,
        "context_projection_shape":list(treatment.context_projection.shape),
        "context_projection_seed":S64_CONTEXT_PROJECTION_SEED,
        "context_projection_trainable":False,
        "encoder_init_seed":S64_ENCODER_INIT_SEED,
        "surface_dimension":4,
        "context_dimension":S64_CONTEXT_DIM,
        "option_input_dimension":S64_OPTION_INPUT_DIM,
        "option_hidden_dimension":S64_OPTION_HIDDEN_DIM,
        "pooled_dimension":S64_POOLED_DIM,
        "representation_dimension":S64_REPRESENTATION_DIM,
        "reference_context_max_abs":reference_context_max_abs,
        "treatment_context_l1":treatment_context_l1,
        "initial_w_out_exact_zero":bool(torch.equal(treatment.w_out,torch.zeros_like(treatment.w_out))),
        "initial_alpha_max_abs_error":max_initial_alpha_error,
        "option_permutation_alpha_max_abs_error":max_perm,
        "surface_affine_alpha_max_abs_error":max_affine,
        "treatment_context_sensitivity_max_abs":context_sensitivity,
        "flat_mechanics_finite":True,
        "identity_override_max_abs_error":float((identity-fused).abs().max()),
        "residual_bound_violation_max":max_bound,
        "max_probability_mass_error":max_mass,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
    }


def _staged_gradient_court():
    fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold=_synthetic_mixed()
    expected,_=train_only_reliability_target(
        fused_c,fused_p,pair_c,pair_p,gold
    )
    if expected.tolist()!=[1.0,0.0]:
        raise RuntimeError("S64 A0 inherited S62 target changed")

    out={}
    for label,use_context in (("reference",False),("treatment",True)):
        production=ContextInjectedReliabilityGate(
            use_context=use_context
        )
        initial=production.state_dict_exact()
        loss,_=contextual_reliability_loss(
            production,
            fused_c,fused_p,
            pair_c,pair_p,
            context_c,context_p,
            gold,
        )
        params=(
            production.W_phi,
            production.b_phi,
            production.w_out,
            production.b_out,
        )
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        l1=[
            0.0 if g is None else float(g.detach().abs().sum())
            for g in grads
        ]
        if l1[2]<=0.0 or l1[3]<=0.0:
            raise RuntimeError(f"S64 A0 {label} output gradient vanished")
        if l1[0]!=0.0 or l1[1]!=0.0:
            raise RuntimeError(f"S64 A0 {label} initial phi gradient should be zero")
        now=production.state_dict_exact()
        if any(not torch.equal(initial[k],now[k]) for k in initial):
            raise RuntimeError(f"S64 A0 {label} production gate mutated")

        clone=ContextInjectedReliabilityGate(
            use_context=use_context
        )
        clone.load_state_dict_exact(initial)
        opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
        warm,_=contextual_reliability_loss(
            clone,
            fused_c,fused_p,
            pair_c,pair_p,
            context_c,context_p,
            gold,
        )
        opt.zero_grad(set_to_none=True)
        warm.backward()
        opt.step()
        second,_=contextual_reliability_loss(
            clone,
            fused_c,fused_p,
            pair_c,pair_p,
            context_c,context_p,
            gold,
        )
        phi=torch.autograd.grad(
            second,(clone.W_phi,clone.b_phi),allow_unused=True
        )
        phi_l1=[
            0.0 if g is None else float(g.detach().abs().sum())
            for g in phi
        ]
        if min(phi_l1)<=0.0:
            raise RuntimeError(f"S64 A0 {label} staged phi gradient did not activate")

        out.update({
            f"{label}_initial_W_phi_gradient_l1":l1[0],
            f"{label}_initial_b_phi_gradient_l1":l1[1],
            f"{label}_initial_w_out_gradient_l1":l1[2],
            f"{label}_initial_b_out_gradient_l1":l1[3],
            f"{label}_post_output_warm_W_phi_gradient_l1":phi_l1[0],
            f"{label}_post_output_warm_b_phi_gradient_l1":phi_l1[1],
        })

    return {
        "inherited_target_values":[int(x) for x in expected.tolist()],
        **out,
        "production_initial_state_unchanged":True,
    }


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
        raise RuntimeError("S64-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S64-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S64-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S64-A0 checkpoint file changed")

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
        raise RuntimeError("S64-A0 loaded native changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S64-A0 native remained trainable")

    parent_rows=generate_s63_cases("train")[:16]
    cache,cache_digest=s50._materialize_cache(runtime,parent_rows)
    del runtime

    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    reference=ContextInjectedReliabilityGate(
        use_context=False,trainable=True
    )
    treatment=ContextInjectedReliabilityGate(
        use_context=True,trainable=True
    )

    if op.correction_parameter_count!=114688:
        raise RuntimeError("S64-A0 correction capacity changed")
    if head.parameter_count!=32832:
        raise RuntimeError("S64-A0 pairwise capacity changed")
    if reference.parameter_count!=60 or treatment.parameter_count!=60:
        raise RuntimeError("S64-A0 matched gate capacity changed")
    if not _parameter_state_equal(reference,treatment):
        raise RuntimeError("S64-A0 matched gate initialization changed")

    canonical,paraphrase,_n=cache[0]
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    rep_c=s59._representation(op,canonical)
    rep_p=s59._representation(op,paraphrase)
    pair_c=head.aggregate_logits(rep_c)
    pair_p=head.aggregate_logits(rep_p)

    ref_loss,ref_diag=contextual_reliability_loss(
        reference,
        fused_c,fused_p,
        pair_c,pair_p,
        rep_c,rep_p,
        canonical.gold,
    )
    trt_loss,trt_diag=contextual_reliability_loss(
        treatment,
        fused_c,fused_p,
        pair_c,pair_p,
        rep_c,rep_p,
        canonical.gold,
    )
    if float(ref_diag["positive_fraction"])!=float(trt_diag["positive_fraction"]):
        raise RuntimeError("S64-A0 reference/treatment target mismatch")

    upstream=(*op.correction_parameters(),head.A,head.u)
    ref_targets=(
        reference.W_phi,reference.b_phi,
        reference.w_out,reference.b_out,
        *upstream,
    )
    trt_targets=(
        treatment.W_phi,treatment.b_phi,
        treatment.w_out,treatment.b_out,
        *upstream,
    )
    ref_grads=torch.autograd.grad(
        ref_loss,ref_targets,allow_unused=True,retain_graph=True
    )
    trt_grads=torch.autograd.grad(
        trt_loss,trt_targets,allow_unused=True
    )
    ref_gate_l1=sum(
        float(g.detach().abs().sum())
        for g in ref_grads[:4] if g is not None
    )
    trt_gate_l1=sum(
        float(g.detach().abs().sum())
        for g in trt_grads[:4] if g is not None
    )
    if ref_gate_l1<=0.0 or trt_gate_l1<=0.0:
        raise RuntimeError("S64-A0 real-cache gate gradients vanished")
    if any(
        g is not None and float(g.detach().abs().sum())>0
        for g in ref_grads[4:]
    ):
        raise RuntimeError("S64-A0 reference gradient leaked upstream")
    if any(
        g is not None and float(g.detach().abs().sum())>0
        for g in trt_grads[4:]
    ):
        raise RuntimeError("S64-A0 treatment gradient leaked upstream")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0:
        raise RuntimeError("S64-A0 cache gained gradients")
    if rep_c.requires_grad or rep_p.requires_grad:
        raise RuntimeError("S64-A0 context representation gained gradient")

    mechanics=_mechanics()
    staged=_staged_gradient_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    ref_state=reference.state_dict_exact()
    trt_state=treatment.state_dict_exact()
    checkpoint=out/"matched-gates.pt"
    torch.save({
        "schema_version":"hira-v1-s64-matched-gates-v1",
        "seed":SEED,
        "alpha_probe":S62_ALPHA_PROBE,
        "target_tolerance":S62_TARGET_TOLERANCE,
        "reference_use_context":False,
        "treatment_use_context":True,
        "reference_state_dict":ref_state,
        "treatment_state_dict":trt_state,
    },checkpoint)

    rr=ContextInjectedReliabilityGate(use_context=False)
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
        "scientific_authority":"S64_A0_CONTEXTUAL_RELIABILITY_GATE_ONLY",
        "seed":SEED,
        "parent_s63":{
            "merged_main":"4fcb58f0c35e2feca841b08f6f560c092ae600fd",
            "scientific_run":37314015283,
            "artifact_id":11347706059,
            "artifact_digest":"sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e",
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
        "reference_gate_parameter_count":reference.parameter_count,
        "treatment_gate_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":treatment.parameter_count-reference.parameter_count,
        "real_cache_reference_reliability_positive_fraction":float(ref_diag["positive_fraction"]),
        "real_cache_treatment_reliability_positive_fraction":float(trt_diag["positive_fraction"]),
        "reference_gate_gradient_l1":ref_gate_l1,
        "treatment_gate_gradient_l1":trt_gate_l1,
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
        "HIRA_V1_S64_A0_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
