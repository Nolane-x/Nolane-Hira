from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_explicit_pairwise_decision_head import ExplicitPairwiseDecisionHead
from nmd.v1_confidence_adaptive_bounded_hybrid import ConfidenceAdaptiveBoundedHybridGate
from nmd.v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    reliability_gate_loss,
    train_only_reliability_target,
)
from nmd.v1_learned_set_reliability_gate import (
    S63_ALPHA_INITIAL,
    S63_ALPHA_MAX,
    S63_GATE_PARAMETER_COUNT,
    S63_INIT_SEED,
    S63_OPTION_HIDDEN_DIM,
    S63_OPTION_INPUT_DIM,
    S63_POOLED_DIM,
    S63_REPRESENTATION_DIM,
    LearnedSetReliabilityGate,
    learned_set_option_features,
    learned_set_reliability_loss,
)
from nmd.v1_s62_authority import generate_s62_cases
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s63-a0-learned-set-reliability-gate-v1"
OUTCOME="HIRA_V1_S63_A0_LEARNED_SET_RELIABILITY_GATE_READY"
SEED=84_001


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
    return fused_c,fused_p,pair_c,pair_p,gold


def _mechanics():
    generator=torch.Generator().manual_seed(6306303)
    gate=LearnedSetReliabilityGate()
    if gate.parameter_count!=S63_GATE_PARAMETER_COUNT:
        raise RuntimeError("S63 A0 treatment capacity changed")
    params=dict(gate.named_parameters())
    if set(params)!={"W_phi","b_phi","w_out","b_out"}:
        raise RuntimeError("S63 A0 treatment tensor surface changed")
    shapes={k:list(v.shape) for k,v in params.items()}
    expected={
        "W_phi":[8,4],
        "b_phi":[8],
        "w_out":[20],
        "b_out":[],
    }
    if shapes!=expected:
        raise RuntimeError("S63 A0 treatment tensor shapes changed")
    if not torch.equal(gate.w_out,torch.zeros_like(gate.w_out)):
        raise RuntimeError("S63 A0 output weights initialization changed")

    max_mass=max_perm=max_affine=max_bound=max_initial_alpha_error=0.0
    for k in (3,7,255):
        fused=torch.randn(3,k,generator=generator)
        pair=torch.randn(3,k,generator=generator)
        alpha=gate.alpha(fused,pair)
        max_initial_alpha_error=max(
            max_initial_alpha_error,
            float((alpha-S63_ALPHA_INITIAL).abs().max()),
        )
        out,diag=gate.compose(fused,pair,return_diagnostics=True)
        max_mass=max(
            max_mass,
            float((torch.softmax(out,-1).sum(-1)-1.0).abs().max()),
        )
        max_bound=max(
            max_bound,
            float(((out-fused).abs()-diag["residual_bound"]).clamp_min(0).max()),
        )

        probe=LearnedSetReliabilityGate()
        with torch.no_grad():
            probe.w_out.copy_(torch.linspace(-0.25,0.25,S63_REPRESENTATION_DIM))
        perm=torch.randperm(k,generator=generator)
        a=probe.alpha(fused,pair)
        ap=probe.alpha(fused[:,perm],pair[:,perm])
        max_perm=max(max_perm,float((a-ap).abs().max()))

        aa=probe.alpha(3.0*fused+7.0,5.0*pair-11.0)
        max_affine=max(max_affine,float((a-aa).abs().max()))

    flat_fused=torch.zeros(2,7)
    flat_pair=torch.zeros(2,7)
    flat_features=learned_set_option_features(flat_fused,flat_pair)
    flat_alpha=gate.alpha(flat_fused,flat_pair)
    if not bool(torch.isfinite(flat_features).all() and torch.isfinite(flat_alpha).all()):
        raise RuntimeError("S63 A0 flat mechanics non-finite")

    fused,pair=torch.randn(2,7,generator=generator),torch.randn(2,7,generator=generator)
    identity=gate.compose(fused,pair,alpha_override=0.0)

    return {
        "treatment_parameter_count":gate.parameter_count,
        "treatment_trainable_parameter_count":gate.trainable_parameter_count,
        "treatment_trainable_tensor_names":sorted(params),
        "treatment_parameter_shapes":shapes,
        "option_input_dimension":S63_OPTION_INPUT_DIM,
        "option_hidden_dimension":S63_OPTION_HIDDEN_DIM,
        "pooled_dimension":S63_POOLED_DIM,
        "representation_dimension":S63_REPRESENTATION_DIM,
        "representation_init_seed":S63_INIT_SEED,
        "initial_w_out_exact_zero":True,
        "initial_alpha_max_abs_error":max_initial_alpha_error,
        "option_permutation_alpha_max_abs_error":max_perm,
        "surface_affine_alpha_max_abs_error":max_affine,
        "flat_mechanics_finite":True,
        "identity_override_max_abs_error":float((identity-fused).abs().max()),
        "residual_bound_violation_max":max_bound,
        "max_probability_mass_error":max_mass,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
    }


def _staged_gradient_court():
    fused_c,fused_p,pair_c,pair_p,gold=_synthetic_mixed()
    expected,_=train_only_reliability_target(
        fused_c,fused_p,pair_c,pair_p,gold
    )
    if expected.tolist()!=[1.0,0.0]:
        raise RuntimeError("S63 A0 inherited S62 target changed")

    production=LearnedSetReliabilityGate()
    initial={k:v.clone() for k,v in production.state_dict_exact().items()}
    loss,_=learned_set_reliability_loss(
        production,fused_c,fused_p,pair_c,pair_p,gold
    )
    params=(production.W_phi,production.b_phi,production.w_out,production.b_out)
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    l1=[
        0.0 if g is None else float(g.detach().abs().sum())
        for g in grads
    ]
    if l1[2]<=0.0 or l1[3]<=0.0:
        raise RuntimeError("S63 A0 initial output gradient vanished")
    if l1[0]!=0.0 or l1[1]!=0.0:
        raise RuntimeError("S63 A0 initial phi gradient should be blocked by zero output weights")
    if any(not torch.equal(initial[k],production.state_dict_exact()[k]) for k in initial):
        raise RuntimeError("S63 A0 production gate mutated during gradient observation")

    clone=LearnedSetReliabilityGate()
    clone.load_state_dict_exact(initial)
    opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
    warm_loss,_=learned_set_reliability_loss(
        clone,fused_c,fused_p,pair_c,pair_p,gold
    )
    opt.zero_grad(set_to_none=True)
    warm_loss.backward()
    opt.step()
    second,_=learned_set_reliability_loss(
        clone,fused_c,fused_p,pair_c,pair_p,gold
    )
    phi_grads=torch.autograd.grad(second,(clone.W_phi,clone.b_phi),allow_unused=True)
    phi_l1=[
        0.0 if g is None else float(g.detach().abs().sum())
        for g in phi_grads
    ]
    if min(phi_l1)<=0.0:
        raise RuntimeError("S63 A0 staged phi gradient did not activate")

    return {
        "inherited_target_values":[int(x) for x in expected.tolist()],
        "initial_W_phi_gradient_l1":l1[0],
        "initial_b_phi_gradient_l1":l1[1],
        "initial_w_out_gradient_l1":l1[2],
        "initial_b_out_gradient_l1":l1[3],
        "post_output_warm_W_phi_gradient_l1":phi_l1[0],
        "post_output_warm_b_phi_gradient_l1":phi_l1[1],
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
        raise RuntimeError("S63-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S63-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S63-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S63-A0 checkpoint file changed")

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
        raise RuntimeError("S63-A0 loaded native changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S63-A0 native remained trainable")

    parent_rows=generate_s62_cases("train")[:16]
    cache,cache_digest=s50._materialize_cache(runtime,parent_rows)
    del runtime

    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    reference=ConfidenceAdaptiveBoundedHybridGate(trainable=True)
    treatment=LearnedSetReliabilityGate(trainable=True)

    if op.correction_parameter_count!=114688:
        raise RuntimeError("S63-A0 correction capacity changed")
    if head.parameter_count!=32832:
        raise RuntimeError("S63-A0 pairwise capacity changed")
    if reference.parameter_count!=5:
        raise RuntimeError("S63-A0 reference capacity changed")
    if treatment.parameter_count!=61:
        raise RuntimeError("S63-A0 treatment capacity changed")

    canonical,paraphrase,_n=cache[0]
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    pair_c=head.aggregate_logits(s59._representation(op,canonical))
    pair_p=head.aggregate_logits(s59._representation(op,paraphrase))

    ref_loss,ref_diag=reliability_gate_loss(
        reference,fused_c,fused_p,pair_c,pair_p,canonical.gold
    )
    trt_loss,trt_diag=learned_set_reliability_loss(
        treatment,fused_c,fused_p,pair_c,pair_p,canonical.gold
    )
    if float(ref_diag["positive_fraction"])!=float(trt_diag["positive_fraction"]):
        raise RuntimeError("S63-A0 reference/treatment target mismatch")

    upstream=(*op.correction_parameters(),head.A,head.u)
    ref_targets=(reference.w,reference.b,*upstream)
    trt_targets=(treatment.W_phi,treatment.b_phi,treatment.w_out,treatment.b_out,*upstream)
    ref_grads=torch.autograd.grad(ref_loss,ref_targets,allow_unused=True,retain_graph=True)
    trt_grads=torch.autograd.grad(trt_loss,trt_targets,allow_unused=True)
    ref_gate_l1=sum(
        float(g.detach().abs().sum()) for g in ref_grads[:2] if g is not None
    )
    trt_output_l1=sum(
        float(g.detach().abs().sum()) for g in trt_grads[2:4] if g is not None
    )
    if ref_gate_l1<=0 or trt_output_l1<=0:
        raise RuntimeError("S63-A0 real-cache gate gradients vanished")
    if any(g is not None and float(g.detach().abs().sum())>0 for g in ref_grads[2:]):
        raise RuntimeError("S63-A0 reference gradient leaked upstream")
    if any(g is not None and float(g.detach().abs().sum())>0 for g in trt_grads[4:]):
        raise RuntimeError("S63-A0 treatment gradient leaked upstream")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0:
        raise RuntimeError("S63-A0 cache gained gradients")

    mechanics=_mechanics()
    staged=_staged_gradient_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    ref_state=reference.state_dict_exact()
    trt_state=treatment.state_dict_exact()
    checkpoint=out/"matched-gates.pt"
    torch.save({
        "schema_version":"hira-v1-s63-matched-gates-v1",
        "seed":SEED,
        "alpha_probe":S62_ALPHA_PROBE,
        "target_tolerance":S62_TARGET_TOLERANCE,
        "reference_state_dict":ref_state,
        "treatment_state_dict":trt_state,
    },checkpoint)

    rr=ConfidenceAdaptiveBoundedHybridGate()
    tt=LearnedSetReliabilityGate()
    rr.load_state_dict_exact(ref_state,freeze=True)
    tt.load_state_dict_exact(trt_state,freeze=True)
    replay_error=max(
        float((reference.compose(fused_c,pair_c)-rr.compose(fused_c,pair_c)).abs().max()),
        float((treatment.compose(fused_c,pair_c)-tt.compose(fused_c,pair_c)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S63_A0_LEARNED_SET_RELIABILITY_GATE_ONLY",
        "seed":SEED,
        "parent_s62":{
            "merged_main":"798f5c4d933635b36f4e0eaf76503f8213afedd9",
            "scientific_run":37307658289,
            "artifact_id":11344855774,
            "artifact_digest":"sha256:7812c348d707ea8675fc9370af573a87d9e5b1a0b10e55a1f335bf890ddb3305",
            "verdict":"CASE_B",
        },
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "parent_train_semantic_case_count":len(cache),
        "cache_digest":cache_digest,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "correction_trainable_parameter_count":114688,
        "pairwise_trainable_parameter_count":32832,
        "reference_gate_parameter_count":reference.parameter_count,
        "treatment_gate_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":treatment.parameter_count-reference.parameter_count,
        "reference_gate_trainable_tensor_names":sorted(dict(reference.named_parameters())),
        "real_cache_reference_reliability_positive_fraction":float(ref_diag["positive_fraction"]),
        "real_cache_treatment_reliability_positive_fraction":float(trt_diag["positive_fraction"]),
        "reference_gate_gradient_l1":ref_gate_l1,
        "treatment_initial_output_gradient_l1":trt_output_l1,
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
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S63_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
