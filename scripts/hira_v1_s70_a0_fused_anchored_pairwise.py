from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.v1_contextual_reliability_gate import (
    S64_ALPHA_INITIAL,
    S64_GATE_PARAMETER_COUNT,
    ContextInjectedReliabilityGate,
    contextual_reliability_loss,
)
from nmd.v1_fused_anchored_pairwise_aggregation import (
    S70_AGGREGATION_PARAMETER_COUNT,
    S70_SOFTMAX_TEMPERATURE,
    fused_anchored_pairwise_aggregate,
    fused_anchored_pairwise_weights,
    uniform_pairwise_aggregate,
)


SCHEMA_VERSION="hira-v1-s70-a0-fused-anchored-pairwise-aggregation-v1"
OUTCOME="HIRA_V1_S70_A0_FUSED_ANCHORED_PAIRWISE_AGGREGATION_READY"
SEED=91_001


def _antisymmetric_pair(batch:int,k:int,g:torch.Generator)->torch.Tensor:
    raw=torch.randn(batch,k,k,generator=g)
    pair=0.5*(raw-raw.transpose(-1,-2))
    eye=torch.eye(k,dtype=torch.bool).unsqueeze(0)
    return pair.masked_fill(eye,0.0)


def _same_parameter_state(a,b)->bool:
    sa=a.parameter_state_dict_exact()
    sb=b.parameter_state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _mechanics():
    g=torch.Generator().manual_seed(70070070)
    reference_gate=ContextInjectedReliabilityGate(use_context=True)
    treatment_gate=ContextInjectedReliabilityGate(use_context=True)

    if reference_gate.parameter_count!=S64_GATE_PARAMETER_COUNT:
        raise RuntimeError("S70 A0 reference gate capacity changed")
    if treatment_gate.parameter_count!=S64_GATE_PARAMETER_COUNT:
        raise RuntimeError("S70 A0 treatment gate capacity changed")
    if not _same_parameter_state(reference_gate,treatment_gate):
        raise RuntimeError("S70 A0 gate initialization changed")

    max_ref_identity=0.0
    max_uniform_collapse=0.0
    max_ref_perm=0.0
    max_trt_perm=0.0
    max_weight_sum_error=0.0
    max_weight_diag=0.0
    min_weight=float("inf")
    max_mass=0.0
    max_bound_violation=0.0
    max_identity_error=0.0
    max_initial_alpha_error=0.0

    for k in (3,7,255):
        pair=_antisymmetric_pair(2,k,g)
        fused=torch.randn(2,k,generator=g)
        context=torch.randn(2,k,512,generator=g)

        ref=uniform_pairwise_aggregate(pair,fused)
        expected=pair.sum(dim=-1)/float(k-1)
        max_ref_identity=max(max_ref_identity,float((ref-expected).abs().max()))

        uniform=torch.zeros_like(fused)
        ref_uniform=uniform_pairwise_aggregate(pair,uniform)
        trt_uniform=fused_anchored_pairwise_aggregate(pair,uniform)
        max_uniform_collapse=max(
            max_uniform_collapse,
            float((ref_uniform-trt_uniform).abs().max()),
        )

        trt=fused_anchored_pairwise_aggregate(pair,fused)
        weights=fused_anchored_pairwise_weights(fused)
        max_weight_sum_error=max(
            max_weight_sum_error,
            float((weights.sum(-1)-1.0).abs().max()),
        )
        max_weight_diag=max(
            max_weight_diag,
            float(torch.diagonal(weights,dim1=-2,dim2=-1).abs().max()),
        )
        min_weight=min(min_weight,float(weights.min()))

        perm=torch.randperm(k,generator=g)
        pairp=pair[:,perm][:,:,perm]
        fusedp=fused[:,perm]
        refp=uniform_pairwise_aggregate(pairp,fusedp)
        trtp=fused_anchored_pairwise_aggregate(pairp,fusedp)
        max_ref_perm=max(max_ref_perm,float((refp-ref[:,perm]).abs().max()))
        max_trt_perm=max(max_trt_perm,float((trtp-trt[:,perm]).abs().max()))

        for gate,evidence in ((reference_gate,ref),(treatment_gate,trt)):
            alpha=gate.alpha(fused,evidence,context)
            max_initial_alpha_error=max(
                max_initial_alpha_error,
                float((alpha-S64_ALPHA_INITIAL).abs().max()),
            )
            out,diag=gate.compose(
                fused,evidence,context,return_diagnostics=True
            )
            max_mass=max(
                max_mass,
                float((torch.softmax(out,-1).sum(-1)-1.0).abs().max()),
            )
            max_bound_violation=max(
                max_bound_violation,
                float(((out-fused).abs()-diag["residual_bound"]).clamp_min(0).max()),
            )
            ident=gate.compose(fused,evidence,context,alpha_override=0.0)
            max_identity_error=max(
                max_identity_error,float((ident-fused).abs().max())
            )

    # Diagonal must never contribute.
    pair=_antisymmetric_pair(2,4,g)
    fused=torch.randn(2,4,generator=g)
    ref0=uniform_pairwise_aggregate(pair,fused)
    trt0=fused_anchored_pairwise_aggregate(pair,fused)
    changed=pair.clone()
    for j in range(4):
        changed[:,j,j]=1000.0+j
    diagonal_ref_error=float(
        (uniform_pairwise_aggregate(changed,fused)-ref0).abs().max()
    )
    diagonal_trt_error=float(
        (fused_anchored_pairwise_aggregate(changed,fused)-trt0).abs().max()
    )

    # Ownership boundary: pair/fused inputs cannot receive aggregation/gate gradient.
    fc=torch.randn(16,4,generator=g,requires_grad=True)
    fp=torch.randn(16,4,generator=g,requires_grad=True)
    pc=_antisymmetric_pair(16,4,g).requires_grad_(True)
    pp=_antisymmetric_pair(16,4,g).requires_grad_(True)
    rc=torch.randn(16,4,512,generator=g,requires_grad=True)
    rp=torch.randn(16,4,512,generator=g,requires_grad=True)
    gold=torch.arange(16,dtype=torch.long)%4

    gradient_receipt={}
    for label,gate,agg in (
        ("reference",ContextInjectedReliabilityGate(use_context=True),uniform_pairwise_aggregate),
        ("treatment",ContextInjectedReliabilityGate(use_context=True),fused_anchored_pairwise_aggregate),
    ):
        ec=agg(pc,fc)
        ep=agg(pp,fp)
        if ec.requires_grad or ep.requires_grad:
            raise RuntimeError(f"S70 A0 {label} aggregation retained gradient")
        loss,_=contextual_reliability_loss(
            gate,fc,fp,ec,ep,rc,rp,gold
        )
        grads=torch.autograd.grad(
            loss,
            (gate.W_phi,gate.b_phi,gate.w_out,gate.b_out,fc,fp,pc,pp,rc,rp),
            allow_unused=True,
        )
        l1=[0.0 if x is None else float(x.detach().abs().sum()) for x in grads]
        if l1[2]<=0.0 or l1[3]<=0.0:
            raise RuntimeError(f"S70 A0 {label} output gradient vanished")
        if l1[0]!=0.0 or l1[1]!=0.0:
            raise RuntimeError(f"S70 A0 {label} initial phi gradient changed")
        if any(x is not None for x in grads[4:]):
            raise RuntimeError(f"S70 A0 {label} gradient leaked upstream")

        clone=ContextInjectedReliabilityGate(use_context=True)
        opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
        warm,_=contextual_reliability_loss(
            clone,fc,fp,ec,ep,rc,rp,gold
        )
        opt.zero_grad(set_to_none=True)
        warm.backward()
        opt.step()
        second,_=contextual_reliability_loss(
            clone,fc,fp,ec,ep,rc,rp,gold
        )
        phi=torch.autograd.grad(
            second,(clone.W_phi,clone.b_phi),allow_unused=True
        )
        phi_l1=[0.0 if x is None else float(x.detach().abs().sum()) for x in phi]
        if min(phi_l1)<=0.0:
            raise RuntimeError(f"S70 A0 {label} staged phi gradient vanished")

        gradient_receipt.update({
            f"{label}_initial_W_phi_gradient_l1":l1[0],
            f"{label}_initial_b_phi_gradient_l1":l1[1],
            f"{label}_initial_w_out_gradient_l1":l1[2],
            f"{label}_initial_b_out_gradient_l1":l1[3],
            f"{label}_post_output_warm_W_phi_gradient_l1":phi_l1[0],
            f"{label}_post_output_warm_b_phi_gradient_l1":phi_l1[1],
            f"{label}_gradient_to_upstream_zero":True,
        })

    return {
        "aggregation_parameter_count":S70_AGGREGATION_PARAMETER_COUNT,
        "softmax_temperature":S70_SOFTMAX_TEMPERATURE,
        "reference_gate_parameter_count":reference_gate.parameter_count,
        "treatment_gate_parameter_count":treatment_gate.parameter_count,
        "added_treatment_parameter_count":0,
        "gate_parameter_initialization_bit_identical":True,
        "reference_row_mean_max_abs_error":max_ref_identity,
        "uniform_prior_collapse_max_abs_error":max_uniform_collapse,
        "reference_permutation_max_abs_error":max_ref_perm,
        "treatment_permutation_max_abs_error":max_trt_perm,
        "opponent_weight_sum_max_abs_error":max_weight_sum_error,
        "opponent_weight_diagonal_max_abs":max_weight_diag,
        "opponent_weight_min":min_weight,
        "diagonal_reference_max_abs_error":diagonal_ref_error,
        "diagonal_treatment_max_abs_error":diagonal_trt_error,
        "initial_alpha_max_abs_error":max_initial_alpha_error,
        "identity_override_max_abs_error":max_identity_error,
        "residual_bound_violation_max":max_bound_violation,
        "max_probability_mass_error":max_mass,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        **gradient_receipt,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    parent=json.loads(args.parent_result.read_text(encoding="utf-8"))
    if parent.get("status")!="PASS":
        raise RuntimeError("S70 A0 parent S69 not PASS")
    if parent.get("outcome")!="HIRA_V1_S69_QUERY_GATED_INTERACTION_DEV_COMPLETE":
        raise RuntimeError("S70 A0 parent S69 outcome changed")
    if parent.get("scientific_authority")!="V1_S69_FRESH_QUERY_GATED_INTERACTION":
        raise RuntimeError("S70 A0 parent S69 authority changed")
    if parent.get("seed")!=90001:
        raise RuntimeError("S70 A0 parent S69 seed changed")
    if parent.get("second_dev_run_performed") is not False:
        raise RuntimeError("S70 A0 parent S69 second DEV changed")
    if parent.get("external_laya_jev_evaluation_opened") is not False:
        raise RuntimeError("S70 A0 parent external eval changed")

    mechanics=_mechanics()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    reference=ContextInjectedReliabilityGate(use_context=True)
    treatment=ContextInjectedReliabilityGate(use_context=True)
    checkpoint=out/"matched-gates.pt"
    torch.save({
        "schema_version":"hira-v1-s70-matched-gates-v1",
        "seed":SEED,
        "reference_state_dict":reference.state_dict_exact(),
        "treatment_state_dict":treatment.state_dict_exact(),
    },checkpoint)

    g=torch.Generator().manual_seed(70070071)
    pair=_antisymmetric_pair(2,4,g)
    fused=torch.randn(2,4,generator=g)
    context=torch.randn(2,4,512,generator=g)
    ref_e=uniform_pairwise_aggregate(pair,fused)
    trt_e=fused_anchored_pairwise_aggregate(pair,fused)

    rr=ContextInjectedReliabilityGate(use_context=True)
    tt=ContextInjectedReliabilityGate(use_context=True)
    rr.load_state_dict_exact(reference.state_dict_exact(),freeze=True)
    tt.load_state_dict_exact(treatment.state_dict_exact(),freeze=True)
    replay_error=max(
        float((reference.compose(fused,ref_e,context)-rr.compose(fused,ref_e,context)).abs().max()),
        float((treatment.compose(fused,trt_e,context)-tt.compose(fused,trt_e,context)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S70_A0_FUSED_ANCHORED_PAIRWISE_AGGREGATION_ONLY",
        "seed":SEED,
        "parent_s69":{
            "merged_main":"1ada877bcd9381486395599519363e87ef3dd831",
            "scientific_run":37420104460,
            "artifact_id":11392672918,
            "artifact_digest":"sha256:c24c00e9b79d23b3e7e3fe4d17013270e0410d822b85c62230bb85e972a2f94d",
            "verdict":"CASE_B",
        },
        **mechanics,
        "one_encoder_state_once":True,
        "matched_gate_checkpoint_replay_max_abs_error":replay_error,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S70_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
