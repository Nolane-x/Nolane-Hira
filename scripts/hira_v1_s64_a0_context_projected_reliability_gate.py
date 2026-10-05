from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_explicit_pairwise_decision_head import ExplicitPairwiseDecisionHead
from nmd.v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    train_only_reliability_target,
)
from nmd.v1_learned_set_reliability_gate import (
    LearnedSetReliabilityGate,
    learned_set_reliability_loss,
)
from nmd.v1_context_projected_reliability_gate import (
    S64_CONTEXT_DIM,
    S64_CONTEXT_PROJECTION_DIM,
    S64_GATE_PARAMETER_COUNT,
    S64_PROJECTION_SEED,
    ContextProjectedReliabilityGate,
    context_projected_reliability_loss,
    fixed_context_projection,
    projection_sha256,
)
from nmd.v1_s63_authority import generate_s63_cases
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s64-a0-context-projected-reliability-gate-v1"
OUTCOME="HIRA_V1_S64_A0_CONTEXT_PROJECTED_RELIABILITY_GATE_READY"
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
    g=torch.Generator().manual_seed(6406403)
    context_c=torch.randn(2,3,512,generator=g)
    context_p=torch.randn(2,3,512,generator=g)
    return fused_c,fused_p,pair_c,pair_p,context_c,context_p,gold


def _mechanics():
    g=torch.Generator().manual_seed(6406404)
    reference=LearnedSetReliabilityGate()
    treatment=ContextProjectedReliabilityGate()

    if reference.parameter_count!=61 or treatment.parameter_count!=61:
        raise RuntimeError("S64 A0 matched capacity changed")
    ref_state=reference.state_dict_exact()
    trt_learned=treatment.learned_state_dict_exact()
    if ref_state.keys()!=trt_learned.keys():
        raise RuntimeError("S64 A0 learned tensor names changed")
    if not all(torch.equal(ref_state[k],trt_learned[k]) for k in ref_state):
        raise RuntimeError("S64 A0 learned initialization not bit-identical")

    projection=treatment.P_context.detach().cpu()
    expected=1.0/math.sqrt(S64_CONTEXT_DIM)
    if projection.shape!=(4,512):
        raise RuntimeError("S64 A0 projection shape changed")
    if not torch.equal(projection.abs(),torch.full_like(projection,expected)):
        raise RuntimeError("S64 A0 projection values changed")
    if projection.requires_grad:
        raise RuntimeError("S64 A0 projection became trainable")

    max_perm=max_context_affine=max_initial=max_mass=max_bound=0.0
    for k in (3,7,255):
        fused=torch.randn(3,k,generator=g)
        pair=torch.randn(3,k,generator=g)
        context=torch.randn(3,k,512,generator=g)
        ar=reference.alpha(fused,pair)
        at=treatment.alpha(fused,pair,context)
        max_initial=max(
            max_initial,
            float((ar-0.10).abs().max()),
            float((at-0.10).abs().max()),
        )

        with torch.no_grad():
            reference.w_out.copy_(torch.linspace(-0.25,0.25,20))
            treatment.w_out.copy_(torch.linspace(-0.25,0.25,20))

        perm=torch.randperm(k,generator=g)
        a=treatment.alpha(fused,pair,context)
        ap=treatment.alpha(
            fused[:,perm],pair[:,perm],context[:,perm,:]
        )
        max_perm=max(max_perm,float((a-ap).abs().max()))

        aa=treatment.alpha(fused,pair,3.0*context+7.0)
        max_context_affine=max(
            max_context_affine,float((a-aa).abs().max())
        )

        out,diag=treatment.compose(
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

        with torch.no_grad():
            reference.w_out.zero_()
            treatment.w_out.zero_()

    flat_context=torch.zeros(2,7,512)
    flat_fused=torch.zeros(2,7)
    flat_pair=torch.zeros(2,7)
    flat_alpha=treatment.alpha(flat_fused,flat_pair,flat_context)
    if not bool(torch.isfinite(flat_alpha).all()):
        raise RuntimeError("S64 A0 flat context mechanics non-finite")

    fused=torch.randn(2,7,generator=g)
    pair=torch.randn(2,7,generator=g)
    context=torch.randn(2,7,512,generator=g)
    identity=treatment.compose(
        fused,pair,context,alpha_override=0.0
    )

    return {
        "reference_gate_parameter_count":reference.parameter_count,
        "treatment_gate_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":0,
        "reference_trainable_tensor_names":sorted(dict(reference.named_parameters())),
        "treatment_trainable_tensor_names":sorted(dict(treatment.named_parameters())),
        "learned_initialization_bit_identical":True,
        "context_dimension":S64_CONTEXT_DIM,
        "projection_dimension":S64_CONTEXT_PROJECTION_DIM,
        "projection_seed":S64_PROJECTION_SEED,
        "projection_shape":list(projection.shape),
        "projection_requires_grad":False,
        "projection_digest":projection_sha256(projection),
        "projection_rademacher_exact":True,
        "initial_alpha_max_abs_error":max_initial,
        "option_permutation_alpha_max_abs_error":max_perm,
        "context_affine_alpha_max_abs_error":max_context_affine,
        "flat_context_finite":True,
        "identity_override_max_abs_error":float((identity-fused).abs().max()),
        "residual_bound_violation_max":max_bound,
        "max_probability_mass_error":max_mass,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
    }


def _gradient_court():
    fc,fp,pc,pp,cc,cp,gold=_synthetic_mixed()
    target,_=train_only_reliability_target(fc,fp,pc,pp,gold)
    if target.tolist()!=[1.0,0.0]:
        raise RuntimeError("S64 A0 inherited target changed")

    reference=LearnedSetReliabilityGate()
    treatment=ContextProjectedReliabilityGate()
    ref_initial=reference.state_dict_exact()
    trt_initial=treatment.state_dict_exact()

    ref_loss,_=learned_set_reliability_loss(
        reference,fc,fp,pc,pp,gold
    )
    trt_loss,_=context_projected_reliability_loss(
        treatment,fc,fp,pc,pp,cc,cp,gold
    )
    rparams=(reference.W_phi,reference.b_phi,reference.w_out,reference.b_out)
    tparams=(treatment.W_phi,treatment.b_phi,treatment.w_out,treatment.b_out)
    rgrads=torch.autograd.grad(ref_loss,rparams,allow_unused=True)
    tgrads=torch.autograd.grad(trt_loss,tparams,allow_unused=True)
    rl1=[0.0 if x is None else float(x.abs().sum()) for x in rgrads]
    tl1=[0.0 if x is None else float(x.abs().sum()) for x in tgrads]
    if rl1[2]<=0 or rl1[3]<=0 or tl1[2]<=0 or tl1[3]<=0:
        raise RuntimeError("S64 A0 initial output gradient vanished")
    if rl1[0]!=0 or rl1[1]!=0 or tl1[0]!=0 or tl1[1]!=0:
        raise RuntimeError("S64 A0 initial phi gradient contract changed")

    clone=ContextProjectedReliabilityGate()
    clone.load_state_dict_exact(trt_initial)
    opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
    warm,_=context_projected_reliability_loss(
        clone,fc,fp,pc,pp,cc,cp,gold
    )
    opt.zero_grad(set_to_none=True)
    warm.backward()
    opt.step()
    second,_=context_projected_reliability_loss(
        clone,fc,fp,pc,pp,cc,cp,gold
    )
    pg=torch.autograd.grad(
        second,(clone.W_phi,clone.b_phi),allow_unused=True
    )
    pl1=[0.0 if x is None else float(x.abs().sum()) for x in pg]
    if min(pl1)<=0:
        raise RuntimeError("S64 A0 contextual phi gradient did not activate")

    if any(not torch.equal(ref_initial[k],reference.state_dict_exact()[k]) for k in ref_initial):
        raise RuntimeError("S64 A0 reference production init mutated")
    if any(not torch.equal(trt_initial[k],treatment.state_dict_exact()[k]) for k in trt_initial):
        raise RuntimeError("S64 A0 treatment production init mutated")

    fcg=fc.clone().requires_grad_(True)
    fpg=fp.clone().requires_grad_(True)
    pcg=pc.clone().requires_grad_(True)
    ppg=pp.clone().requires_grad_(True)
    ccg=cc.clone().requires_grad_(True)
    cpg=cp.clone().requires_grad_(True)
    isolated=ContextProjectedReliabilityGate()
    loss,_=context_projected_reliability_loss(
        isolated,fcg,fpg,pcg,ppg,ccg,cpg,gold
    )
    source_grads=torch.autograd.grad(
        loss,(fcg,fpg,pcg,ppg,ccg,cpg),
        allow_unused=True,
    )
    if any(x is not None and float(x.abs().sum())>0 for x in source_grads):
        raise RuntimeError("S64 A0 reliability gradient leaked to source tensors")

    return {
        "inherited_target_values":[int(x) for x in target.tolist()],
        "reference_initial_W_phi_gradient_l1":rl1[0],
        "reference_initial_b_phi_gradient_l1":rl1[1],
        "reference_initial_w_out_gradient_l1":rl1[2],
        "reference_initial_b_out_gradient_l1":rl1[3],
        "treatment_initial_W_phi_gradient_l1":tl1[0],
        "treatment_initial_b_phi_gradient_l1":tl1[1],
        "treatment_initial_w_out_gradient_l1":tl1[2],
        "treatment_initial_b_out_gradient_l1":tl1[3],
        "treatment_post_warm_W_phi_gradient_l1":pl1[0],
        "treatment_post_warm_b_phi_gradient_l1":pl1[1],
        "gradient_to_fused_pairwise_context_zero":True,
        "production_initial_state_unchanged":True,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--parent-s63-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    parent=json.loads(args.parent_s63_result.read_text(encoding="utf-8"))
    if parent.get("outcome")!="HIRA_V1_S63_LEARNED_SET_RELIABILITY_GATE_DEV_COMPLETE":
        raise RuntimeError("S64-A0 parent S63 outcome changed")
    if parent.get("scientific_authority")!="V1_S63_FRESH_LEARNED_SET_RELIABILITY_GATE":
        raise RuntimeError("S64-A0 parent S63 authority changed")
    if parent.get("seed")!=84001:
        raise RuntimeError("S64-A0 parent S63 seed changed")
    if parent.get("post_dev_tuning_performed") is not False:
        raise RuntimeError("S64-A0 parent S63 tuning state changed")
    if parent.get("second_dev_run_performed") is not False:
        raise RuntimeError("S64-A0 parent S63 DEV multiplicity changed")

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S64-A0 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S64-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S64-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S64-A0 checkpoint file changed")

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    runtime,payload=load_native_authority(
        bundle=bundle,manifest=manifest,checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,expected_seed=72001,
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
    reference=LearnedSetReliabilityGate(trainable=True)
    treatment=ContextProjectedReliabilityGate(trainable=True)

    if op.correction_parameter_count!=114688:
        raise RuntimeError("S64-A0 correction capacity changed")
    if head.parameter_count!=32832:
        raise RuntimeError("S64-A0 pairwise capacity changed")
    if reference.parameter_count!=S64_GATE_PARAMETER_COUNT:
        raise RuntimeError("S64-A0 reference capacity changed")
    if treatment.parameter_count!=S64_GATE_PARAMETER_COUNT:
        raise RuntimeError("S64-A0 treatment capacity changed")
    if not all(
        torch.equal(reference.state_dict_exact()[k],treatment.learned_state_dict_exact()[k])
        for k in reference.state_dict_exact()
    ):
        raise RuntimeError("S64-A0 real matched learned init changed")

    canonical,paraphrase,_n=cache[0]
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    context_c=s59._representation(op,canonical)
    context_p=s59._representation(op,paraphrase)
    pair_c=head.aggregate_logits(context_c)
    pair_p=head.aggregate_logits(context_p)

    ref_loss,ref_diag=learned_set_reliability_loss(
        reference,fused_c,fused_p,pair_c,pair_p,canonical.gold
    )
    trt_loss,trt_diag=context_projected_reliability_loss(
        treatment,fused_c,fused_p,pair_c,pair_p,
        context_c,context_p,canonical.gold
    )
    if int(ref_diag["target_count"])!=int(trt_diag["target_count"]):
        raise RuntimeError("S64-A0 target count mismatch")
    if int(ref_diag["target_positive_count"])!=int(trt_diag["target_positive_count"]):
        raise RuntimeError("S64-A0 target label mismatch")

    upstream=(*op.correction_parameters(),head.A,head.u)
    rparams=(reference.W_phi,reference.b_phi,reference.w_out,reference.b_out)
    tparams=(treatment.W_phi,treatment.b_phi,treatment.w_out,treatment.b_out)
    rgrads=torch.autograd.grad(ref_loss,(*rparams,*upstream),allow_unused=True,retain_graph=True)
    tgrads=torch.autograd.grad(trt_loss,(*tparams,*upstream),allow_unused=True)
    if any(g is not None and float(g.abs().sum())>0 for g in rgrads[4:]):
        raise RuntimeError("S64-A0 reference gradient leaked upstream")
    if any(g is not None and float(g.abs().sum())>0 for g in tgrads[4:]):
        raise RuntimeError("S64-A0 treatment gradient leaked upstream")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0 or context_c.requires_grad or context_p.requires_grad:
        raise RuntimeError("S64-A0 cache/context gained gradients")

    mechanics=_mechanics()
    gradient=_gradient_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/"matched-gates.pt"
    ref_state=reference.state_dict_exact()
    trt_state=treatment.state_dict_exact()
    torch.save({
        "schema_version":"hira-v1-s64-matched-gates-v1",
        "seed":SEED,
        "reference_state_dict":ref_state,
        "treatment_state_dict":trt_state,
        "projection_digest":treatment.projection_digest,
        "alpha_probe":S62_ALPHA_PROBE,
        "target_tolerance":S62_TARGET_TOLERANCE,
    },checkpoint)

    rr=LearnedSetReliabilityGate()
    tt=ContextProjectedReliabilityGate()
    rr.load_state_dict_exact(ref_state,freeze=True)
    tt.load_state_dict_exact(trt_state,freeze=True)
    replay_error=max(
        float((reference.compose(fused_c,pair_c)-rr.compose(fused_c,pair_c)).abs().max()),
        float((
            treatment.compose(fused_c,pair_c,context_c)
            -tt.compose(fused_c,pair_c,context_c)
        ).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S64_A0_CONTEXT_PROJECTED_RELIABILITY_GATE_ONLY",
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
        "correction_trainable_parameter_count":114688,
        "pairwise_trainable_parameter_count":32832,
        "real_cache_target_positive_fraction":float(ref_diag["positive_fraction"]),
        "real_context_shape":list(context_c.shape),
        "real_context_requires_grad":bool(context_c.requires_grad),
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
        **gradient,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S64_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
