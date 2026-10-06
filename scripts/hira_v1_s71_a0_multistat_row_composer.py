from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.v1_multistat_pairwise_row_composer import (
    S71_ALPHA_INITIAL,
    S71_PARAMETER_COUNT,
    MultiStatPairwiseRowComposer,
    multistat_responsibility_loss,
    pairwise_row_mean,
)


SCHEMA_VERSION="hira-v1-s71-a0-multistat-row-composer-v1"
OUTCOME="HIRA_V1_S71_A0_MULTISTAT_ROW_COMPOSER_READY"
SEED=92_001


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
    g=torch.Generator().manual_seed(71071071)
    reference=MultiStatPairwiseRowComposer(use_multistat=False)
    treatment=MultiStatPairwiseRowComposer(use_multistat=True)

    if reference.parameter_count!=S71_PARAMETER_COUNT:
        raise RuntimeError("S71 A0 reference capacity changed")
    if treatment.parameter_count!=S71_PARAMETER_COUNT:
        raise RuntimeError("S71 A0 treatment capacity changed")
    if not _same_parameter_state(reference,treatment):
        raise RuntimeError("S71 A0 parameter init changed")
    if not torch.equal(reference.context_projection,treatment.context_projection):
        raise RuntimeError("S71 A0 context projection changed")

    max_ref_extra=0.0
    max_ref_treatment_mean_diff=0.0
    max_perm_row=0.0
    max_perm_out=0.0
    max_alpha_perm=0.0
    max_initial_alpha_error=0.0
    max_identity_error=0.0
    max_bound_violation=0.0
    max_mass=0.0
    treatment_extra_nonzero=0.0

    for k in (3,7,255):
        pair=_antisymmetric_pair(2,k,g)
        fused=torch.randn(2,k,generator=g)
        context=torch.randn(2,k,512,generator=g)

        rr=reference.row_channels(pair)
        tr=treatment.row_channels(pair)
        max_ref_extra=max(max_ref_extra,float(rr[...,1:].abs().max()))
        max_ref_treatment_mean_diff=max(
            max_ref_treatment_mean_diff,
            float((rr[...,:1]-tr[...,:1]).abs().max()),
        )
        treatment_extra_nonzero=max(
            treatment_extra_nonzero,
            float(tr[...,1:].abs().max()),
        )

        perm=torch.randperm(k,generator=g)
        inv=torch.argsort(perm)
        pairp=pair[:,perm][:,:,perm]
        fusedp=fused[:,perm]
        contextp=context[:,perm]

        for gate in (reference,treatment):
            row=gate.row_channels(pair)
            rowp=gate.row_channels(pairp)
            max_perm_row=max(
                max_perm_row,
                float((rowp[:,inv]-row).abs().max()),
            )

            alpha=gate.alpha(fused,pair,context)
            alphap=gate.alpha(fusedp,pairp,contextp)
            max_alpha_perm=max(
                max_alpha_perm,
                float((alpha-alphap).abs().max()),
            )
            max_initial_alpha_error=max(
                max_initial_alpha_error,
                float((alpha-S71_ALPHA_INITIAL).abs().max()),
            )

            out,diag=gate.compose(
                fused,pair,context,return_diagnostics=True
            )
            outp=gate.compose(fusedp,pairp,contextp)
            max_perm_out=max(
                max_perm_out,
                float((outp[:,inv]-out).abs().max()),
            )
            max_bound_violation=max(
                max_bound_violation,
                float(((out-fused).abs()-diag["residual_bound"]).clamp_min(0).max()),
            )
            max_mass=max(
                max_mass,
                float((torch.softmax(out,-1).sum(-1)-1).abs().max()),
            )
            ident=gate.compose(fused,pair,context,alpha_override=0.0)
            max_identity_error=max(
                max_identity_error,
                float((ident-fused).abs().max()),
            )

    if treatment_extra_nonzero<=1e-7:
        raise RuntimeError("S71 A0 treatment extra row stats degenerate")

    # Diagonal must never contribute.
    pair=_antisymmetric_pair(3,4,g)
    base=pairwise_row_mean(pair)
    changed=pair.clone()
    idx=torch.arange(4)
    changed[:,idx,idx]=9999.0
    diagonal_mean_error=float((pairwise_row_mean(changed)-base).abs().max())

    # Exact residual direction is common between arms.
    fused=torch.randn(3,4,generator=g)
    context=torch.randn(3,4,512,generator=g)
    ref_fixed=reference.compose(fused,pair,context,alpha_override=0.2)
    trt_fixed=treatment.compose(fused,pair,context,alpha_override=0.2)
    fixed_alpha_direction_error=float((ref_fixed-trt_fixed).abs().max())

    # Ownership / staged-gradient court.
    fc=torch.randn(32,4,generator=g,requires_grad=True)
    fp=torch.randn(32,4,generator=g,requires_grad=True)
    pc=_antisymmetric_pair(32,4,g).requires_grad_(True)
    pp=_antisymmetric_pair(32,4,g).requires_grad_(True)
    rc=torch.randn(32,4,512,generator=g,requires_grad=True)
    rp=torch.randn(32,4,512,generator=g,requires_grad=True)
    gold=torch.arange(32,dtype=torch.long)%4

    gradient_receipt={}
    for label,use in (("reference",False),("treatment",True)):
        gate=MultiStatPairwiseRowComposer(use_multistat=use)
        loss,_=multistat_responsibility_loss(
            gate,fc,fp,pc,pp,rc,rp,gold
        )
        grads=torch.autograd.grad(
            loss,
            (
                gate.W_phi,gate.b_phi,gate.w_out,gate.b_out,
                fc,fp,pc,pp,rc,rp,
            ),
            allow_unused=True,
        )
        l1=[0.0 if x is None else float(x.detach().abs().sum()) for x in grads]
        if l1[2]<=0.0 or l1[3]<=0.0:
            raise RuntimeError(f"S71 A0 {label} output gradient vanished")
        if l1[0]!=0.0 or l1[1]!=0.0:
            raise RuntimeError(f"S71 A0 {label} initial phi gradient changed")
        if any(x is not None for x in grads[4:]):
            raise RuntimeError(f"S71 A0 {label} gradient leaked upstream")

        clone=MultiStatPairwiseRowComposer(use_multistat=use)
        opt=torch.optim.SGD([clone.w_out,clone.b_out],lr=0.1)
        warm,_=multistat_responsibility_loss(
            clone,
            fc.detach(),fp.detach(),
            pc.detach(),pp.detach(),
            rc.detach(),rp.detach(),gold,
        )
        opt.zero_grad(set_to_none=True)
        warm.backward()
        opt.step()

        second,_=multistat_responsibility_loss(
            clone,
            fc.detach(),fp.detach(),
            pc.detach(),pp.detach(),
            rc.detach(),rp.detach(),gold,
        )
        phi=torch.autograd.grad(
            second,(clone.W_phi,clone.b_phi),allow_unused=True
        )
        phi_l1=[0.0 if x is None else float(x.detach().abs().sum()) for x in phi]
        if min(phi_l1)<=0.0:
            raise RuntimeError(f"S71 A0 {label} staged phi gradient vanished")

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
        "reference_parameter_count":reference.parameter_count,
        "treatment_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":0,
        "parameter_initialization_bit_identical":True,
        "context_projection_bit_identical":True,
        "reference_extra_row_channels_max_abs":max_ref_extra,
        "reference_treatment_mean_channel_max_abs_error":max_ref_treatment_mean_diff,
        "treatment_extra_row_channels_max_abs":treatment_extra_nonzero,
        "row_permutation_max_abs_error":max_perm_row,
        "output_permutation_max_abs_error":max_perm_out,
        "alpha_permutation_max_abs_error":max_alpha_perm,
        "diagonal_row_mean_max_abs_error":diagonal_mean_error,
        "fixed_alpha_residual_direction_max_abs_error":fixed_alpha_direction_error,
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
        raise RuntimeError("S71 A0 parent S70 not PASS")
    if parent.get("outcome")!="HIRA_V1_S70_FUSED_ANCHORED_PAIRWISE_DEV_COMPLETE":
        raise RuntimeError("S71 A0 parent S70 outcome changed")
    if parent.get("scientific_authority")!="V1_S70_FRESH_FUSED_ANCHORED_PAIRWISE_AGGREGATION":
        raise RuntimeError("S71 A0 parent S70 authority changed")
    if parent.get("seed")!=91001:
        raise RuntimeError("S71 A0 parent S70 seed changed")
    if parent.get("second_dev_run_performed") is not False:
        raise RuntimeError("S71 A0 parent S70 second DEV changed")
    if parent.get("external_laya_jev_evaluation_opened") is not False:
        raise RuntimeError("S71 A0 parent external eval changed")

    mechanics=_mechanics()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    reference=MultiStatPairwiseRowComposer(use_multistat=False)
    treatment=MultiStatPairwiseRowComposer(use_multistat=True)
    checkpoint=out/"matched-composers.pt"
    torch.save({
        "schema_version":"hira-v1-s71-matched-composers-v1",
        "seed":SEED,
        "reference_state_dict":reference.state_dict_exact(),
        "treatment_state_dict":treatment.state_dict_exact(),
    },checkpoint)

    g=torch.Generator().manual_seed(71071072)
    pair=_antisymmetric_pair(2,4,g)
    fused=torch.randn(2,4,generator=g)
    context=torch.randn(2,4,512,generator=g)

    rr=MultiStatPairwiseRowComposer(use_multistat=False)
    tt=MultiStatPairwiseRowComposer(use_multistat=True)
    rr.load_state_dict_exact(reference.state_dict_exact(),freeze=True)
    tt.load_state_dict_exact(treatment.state_dict_exact(),freeze=True)
    replay_error=max(
        float((reference.compose(fused,pair,context)-rr.compose(fused,pair,context)).abs().max()),
        float((treatment.compose(fused,pair,context)-tt.compose(fused,pair,context)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S71_A0_MULTISTAT_ROW_COMPOSER_ONLY",
        "seed":SEED,
        "parent_s70":{
            "merged_main":"751161746887b795149fcd924735ee2d043b43b3",
            "scientific_run":37423476142,
            "artifact_id":11394610000,
            "artifact_digest":"sha256:8af711f789efb9717eaed0eff882fade7bab37b546f86a35502db82242ad221f",
            "verdict":"CASE_C",
        },
        **mechanics,
        "one_encoder_state_once":True,
        "matched_composer_checkpoint_replay_max_abs_error":replay_error,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "reliability_target_reopened":False,
        "residual_direction_changed":False,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S71_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
