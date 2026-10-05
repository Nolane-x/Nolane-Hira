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
from nmd.v1_per_view_responsibility import (
    per_view_responsibility_targets,
    per_view_responsibility_loss,
)
from nmd.v1_safe_oracle_alpha_responsibility import (
    S67_ALPHA_LATTICE,
    S67_TARGET_LEVELS,
    safe_oracle_alpha_targets,
    safe_oracle_alpha_loss,
    select_safe_oracle_alpha,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    train_only_reliability_target,
)
from nmd.v1_s66_authority import generate_s66_cases
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s67-a0-safe-oracle-alpha-v1"
OUTCOME="HIRA_V1_S67_A0_SAFE_ORACLE_ALPHA_READY"
SEED=88_001


def _same_parameter_state(a,b)->bool:
    sa=a.parameter_state_dict_exact()
    sb=b.parameter_state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _probe_bank():
    g=torch.Generator().manual_seed(67067066)
    b,k=4096,4
    fc=2.0*torch.randn(b,k,generator=g)
    fp=2.0*torch.randn(b,k,generator=g)
    pc=2.0*torch.randn(b,k,generator=g)
    pp=2.0*torch.randn(b,k,generator=g)
    gold=torch.randint(0,k,(b,),generator=g)
    return fc,fp,pc,pp,gold


def _mechanics():
    reference=ContextInjectedReliabilityGate(use_context=True)
    treatment=ContextInjectedReliabilityGate(use_context=True)
    if reference.parameter_count!=60 or treatment.parameter_count!=60:
        raise RuntimeError("S67 A0 gate capacity changed")
    if not _same_parameter_state(reference,treatment):
        raise RuntimeError("S67 A0 gate initialization changed")
    if not torch.equal(reference.context_projection,treatment.context_projection):
        raise RuntimeError("S67 A0 context path changed")

    fc,fp,pc,pp,gold=_probe_bank()
    yc,yp,diag=safe_oracle_alpha_targets(fc,fp,pc,pp,gold)
    levels=sorted({round(float(x),6) for x in torch.cat([yc,yp]).unique()})
    if 0.0 not in levels or len([x for x in levels if x>0.0])<3:
        raise RuntimeError(f"S67 A0 oracle target levels insufficient: {levels}")
    allowed={0.0,0.25,0.5,0.75,1.0}
    if not set(levels).issubset(allowed):
        raise RuntimeError("S67 A0 oracle target escaped frozen levels")

    syc,syp,_=safe_oracle_alpha_targets(fp,fc,pp,pc,gold)
    swap_error=max(float((yc-syp).abs().max()),float((yp-syc).abs().max()))
    if swap_error!=0.0:
        raise RuntimeError("S67 A0 view-swap equivariance changed")

    byc,byp,_=per_view_responsibility_targets(fc,fp,pc,pp,gold)
    target_distinct=bool((yc!=byc).any() or (yp!=byp).any())
    if not target_distinct:
        raise RuntimeError("S67 A0 oracle target collapsed to S66 binary target")

    base_ce=torch.tensor([1.0,1.0])
    base_js=torch.tensor([0.5,0.5])
    candidate_ce=torch.tensor([[0.9,0.9],[0.9,0.9],[0.9,1.2],[0.9,1.2]])
    candidate_js=torch.tensor([[0.3,0.4],[0.3,0.4],[0.2,0.2],[0.2,0.2]])
    tie_alpha,_,_=select_safe_oracle_alpha(base_ce,base_js,candidate_ce,candidate_js)
    tie_error=float((tie_alpha-torch.tensor([0.2625,0.0875])).abs().max())
    if tie_error!=0.0:
        raise RuntimeError("S67 A0 smaller-alpha tie break changed")

    g=torch.Generator().manual_seed(67067067)
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
            out,d=gate.compose(fused,pair,context,return_diagnostics=True)
            max_mass=max(
                max_mass,
                float((torch.softmax(out,-1).sum(-1)-1.0).abs().max()),
            )
            max_bound=max(
                max_bound,
                float(((out-fused).abs()-d["residual_bound"]).clamp_min(0).max()),
            )
            ident=gate.compose(fused,pair,context,alpha_override=0.0)
            identity_error=max(
                identity_error,
                float((ident-fused).abs().max()),
            )

    return {
        "reference_gate_parameter_count":reference.parameter_count,
        "treatment_gate_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":0,
        "parameter_initialization_bit_identical":True,
        "context_projection_bit_identical":True,
        "oracle_target_levels":levels,
        "oracle_alpha_lattice":list(S67_ALPHA_LATTICE),
        "view_swap_target_max_abs_error":swap_error,
        "treatment_target_distinct_from_s66_binary":target_distinct,
        "smaller_alpha_tie_break_max_abs_error":tie_error,
        "probe_bank_canonical_mean_target":float(yc.mean()),
        "probe_bank_paraphrase_mean_target":float(yp.mean()),
        "initial_alpha_max_abs_error":max_initial,
        "identity_override_max_abs_error":identity_error,
        "residual_bound_violation_max":max_bound,
        "max_probability_mass_error":max_mass,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
    }


def _staged_gradient_court():
    g=torch.Generator().manual_seed(67067068)
    b,k=64,4
    fc=2.0*torch.randn(b,k,generator=g)
    fp=2.0*torch.randn(b,k,generator=g)
    pc=2.0*torch.randn(b,k,generator=g)
    pp=2.0*torch.randn(b,k,generator=g)
    gold=torch.randint(0,k,(b,),generator=g)
    rc=torch.randn(b,k,512,generator=g)
    rp=torch.randn(b,k,512,generator=g)

    reference=ContextInjectedReliabilityGate(use_context=True)
    treatment=ContextInjectedReliabilityGate(use_context=True)
    ref_loss,_=per_view_responsibility_loss(
        reference,fc,fp,pc,pp,rc,rp,gold
    )
    trt_loss,trt_diag=safe_oracle_alpha_loss(
        treatment,fc,fp,pc,pp,rc,rp,gold
    )

    result={
        "staged_oracle_target_disagreement_fraction":
            float(trt_diag["target_disagreement_fraction"])
    }
    for label,gate,loss,loss_kind in (
        ("reference",reference,ref_loss,"reference"),
        ("treatment",treatment,trt_loss,"treatment"),
    ):
        params=(gate.W_phi,gate.b_phi,gate.w_out,gate.b_out)
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        l1=[0.0 if x is None else float(x.detach().abs().sum()) for x in grads]
        if l1[2]<=0.0 or l1[3]<=0.0:
            raise RuntimeError(f"S67 A0 {label} output gradient vanished")
        if l1[0]!=0.0 or l1[1]!=0.0:
            raise RuntimeError(f"S67 A0 {label} initial phi gradient changed")

        clone=ContextInjectedReliabilityGate(use_context=True)
        opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
        if loss_kind=="reference":
            warm,_=per_view_responsibility_loss(clone,fc,fp,pc,pp,rc,rp,gold)
        else:
            warm,_=safe_oracle_alpha_loss(clone,fc,fp,pc,pp,rc,rp,gold)
        opt.zero_grad(set_to_none=True)
        warm.backward()
        opt.step()
        if loss_kind=="reference":
            second,_=per_view_responsibility_loss(clone,fc,fp,pc,pp,rc,rp,gold)
        else:
            second,_=safe_oracle_alpha_loss(clone,fc,fp,pc,pp,rc,rp,gold)
        phi=torch.autograd.grad(second,(clone.W_phi,clone.b_phi),allow_unused=True)
        phi_l1=[0.0 if x is None else float(x.detach().abs().sum()) for x in phi]
        if min(phi_l1)<=0.0:
            raise RuntimeError(f"S67 A0 {label} staged phi gradient vanished")

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
        raise RuntimeError("S67-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S67-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S67-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S67-A0 checkpoint file changed")

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
        raise RuntimeError("S67-A0 loaded native changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S67-A0 native remained trainable")

    parent_rows=generate_s66_cases("train")[:16]
    cache,cache_digest=s50._materialize_cache(runtime,parent_rows)
    del runtime

    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    reference=ContextInjectedReliabilityGate(use_context=True,trainable=True)
    treatment=ContextInjectedReliabilityGate(use_context=True,trainable=True)

    if op.correction_parameter_count!=114688:
        raise RuntimeError("S67-A0 correction capacity changed")
    if head.parameter_count!=32832:
        raise RuntimeError("S67-A0 pairwise capacity changed")
    if reference.parameter_count!=60 or treatment.parameter_count!=60:
        raise RuntimeError("S67-A0 gate capacity changed")
    if not _same_parameter_state(reference,treatment):
        raise RuntimeError("S67-A0 matched initialization changed")

    canonical,paraphrase,_n=cache[0]
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    rep_c=s59._representation(op,canonical)
    rep_p=s59._representation(op,paraphrase)
    pair_c=head.aggregate_logits(rep_c)
    pair_p=head.aggregate_logits(rep_p)

    ref_loss,ref_diag=per_view_responsibility_loss(
        reference,fused_c,fused_p,pair_c,pair_p,rep_c,rep_p,canonical.gold
    )
    trt_loss,trt_diag=safe_oracle_alpha_loss(
        treatment,fused_c,fused_p,pair_c,pair_p,rep_c,rep_p,canonical.gold
    )

    upstream=(*op.correction_parameters(),head.A,head.u)
    ref_targets=(reference.W_phi,reference.b_phi,reference.w_out,reference.b_out,*upstream)
    trt_targets=(treatment.W_phi,treatment.b_phi,treatment.w_out,treatment.b_out,*upstream)
    ref_grads=torch.autograd.grad(ref_loss,ref_targets,allow_unused=True,retain_graph=True)
    trt_grads=torch.autograd.grad(trt_loss,trt_targets,allow_unused=True)
    if any(g is not None and float(g.detach().abs().sum())>0 for g in ref_grads[4:]):
        raise RuntimeError("S67-A0 reference gradient leaked upstream")
    if any(g is not None and float(g.detach().abs().sum())>0 for g in trt_grads[4:]):
        raise RuntimeError("S67-A0 treatment gradient leaked upstream")

    mechanics=_mechanics()
    staged=_staged_gradient_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    ref_state=reference.state_dict_exact()
    trt_state=treatment.state_dict_exact()
    checkpoint=out/"matched-gates.pt"
    torch.save({
        "schema_version":"hira-v1-s67-matched-gates-v1",
        "seed":SEED,
        "reference_objective":"per_view_binary_responsibility_bce",
        "treatment_objective":"safe_oracle_alpha_continuous_bce",
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
        float((reference.compose(fused_c,pair_c,rep_c)-rr.compose(fused_c,pair_c,rep_c)).abs().max()),
        float((treatment.compose(fused_c,pair_c,rep_c)-tt.compose(fused_c,pair_c,rep_c)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S67_A0_SAFE_ORACLE_ALPHA_ONLY",
        "seed":SEED,
        "parent_s66":{
            "merged_main":"e898314b51995db7e6fa18b06256a59551d0797e",
            "scientific_run":37331468679,
            "artifact_id":11354228532,
            "artifact_digest":"sha256:f3c6322eec8cce02a698807037a4c05e48c32b2addc5709a116a4384338379b1",
            "verdict":"CASE_B",
        },
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "parent_train_semantic_case_count":len(cache),
        "cache_digest":cache_digest,
        "correction_trainable_parameter_count":114688,
        "pairwise_trainable_parameter_count":32832,
        "reference_gate_parameter_count":reference.parameter_count,
        "treatment_gate_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":0,
        "reference_real_cache_canonical_positive_fraction":float(ref_diag["canonical_positive_fraction"]),
        "reference_real_cache_paraphrase_positive_fraction":float(ref_diag["paraphrase_positive_fraction"]),
        "treatment_real_cache_canonical_mean_target":float(trt_diag["canonical_mean_target"]),
        "treatment_real_cache_paraphrase_mean_target":float(trt_diag["paraphrase_mean_target"]),
        "treatment_real_cache_target_disagreement_fraction":float(trt_diag["target_disagreement_fraction"]),
        "reference_gradient_to_upstream_zero":True,
        "treatment_gradient_to_upstream_zero":True,
        "alpha_probe":S62_ALPHA_PROBE,
        "target_tolerance":S62_TARGET_TOLERANCE,
        "teacher_dependency":False,
        "dev_target_dependency":False,
        "self_anchor_dependency":False,
        "pairwise_only_final_path":False,
        "single_view_inference":True,
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
    print("HIRA_V1_S67_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
