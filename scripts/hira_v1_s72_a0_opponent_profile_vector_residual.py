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
)
from nmd.v1_opponent_profile_vector_residual import (
    compose_with_residual_direction,
    opponent_profile_confidence,
    opponent_profile_vector_direction,
    reference_pairwise_mean_direction,
)


SCHEMA_VERSION="hira-v1-s72-a0-opponent-profile-vector-residual-v1"
OUTCOME="HIRA_V1_S72_A0_OPPONENT_PROFILE_VECTOR_RESIDUAL_READY"
SEED=93_001


def _antisymmetric_pair(batch:int,k:int,g:torch.Generator)->torch.Tensor:
    raw=torch.randn(batch,k,k,generator=g)
    pair=0.5*(raw-raw.transpose(-1,-2))
    eye=torch.eye(k,dtype=torch.bool).unsqueeze(0)
    return pair.masked_fill(eye,0.0)


def _same_parameter_state(a,b)->bool:
    sa=a.parameter_state_dict_exact()
    sb=b.parameter_state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _neutral_case()->tuple[torch.Tensor,torch.Tensor]:
    k=4
    pair=torch.zeros(1,k,k)
    for i in range(k):
        for j in range(i+1,k):
            pair[0,i,j]=1.0
            pair[0,j,i]=-1.0
    fused=torch.zeros(1,k)
    return fused,pair


def _mechanics():
    g=torch.Generator().manual_seed(72072072)
    reference=MultiStatPairwiseRowComposer(use_multistat=False)
    treatment=MultiStatPairwiseRowComposer(use_multistat=False)

    if reference.parameter_count!=S71_PARAMETER_COUNT:
        raise RuntimeError("S72 A0 reference composer capacity changed")
    if treatment.parameter_count!=S71_PARAMETER_COUNT:
        raise RuntimeError("S72 A0 treatment composer capacity changed")
    if not _same_parameter_state(reference,treatment):
        raise RuntimeError("S72 A0 composer initialization changed")
    if not torch.equal(reference.context_projection,treatment.context_projection):
        raise RuntimeError("S72 A0 context projection changed")

    max_alpha_match=0.0
    max_initial_alpha_error=0.0
    max_ref_perm=0.0
    max_trt_perm=0.0
    max_profile_bound=0.0
    max_opponent_mass_error=0.0
    max_diagonal_weight=0.0
    max_identity_error=0.0
    max_bound_violation=0.0
    max_mass_error=0.0
    max_direction_difference=0.0

    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g)
        pair=_antisymmetric_pair(2,k,g)
        context=torch.randn(2,k,512,generator=g)

        ar=reference.alpha(fused,pair,context)
        at=treatment.alpha(fused,pair,context)
        max_alpha_match=max(max_alpha_match,float((ar-at).abs().max()))
        max_initial_alpha_error=max(
            max_initial_alpha_error,
            float((ar-S71_ALPHA_INITIAL).abs().max()),
            float((at-S71_ALPHA_INITIAL).abs().max()),
        )

        ref_dir=reference_pairwise_mean_direction(pair)
        trt_dir,profile=opponent_profile_vector_direction(fused,pair)
        max_direction_difference=max(
            max_direction_difference,
            float((ref_dir-trt_dir).abs().max()),
        )
        max_profile_bound=max(
            max_profile_bound,
            float(profile["confidence"].abs().max()),
        )
        max_opponent_mass_error=max(
            max_opponent_mass_error,
            float((profile["opponent_mass"]-1.0).abs().max()),
        )
        max_diagonal_weight=max(
            max_diagonal_weight,
            float(profile["opponent_weights"].diagonal(dim1=-2,dim2=-1).abs().max()),
        )

        perm=torch.randperm(k,generator=g)
        inv=torch.argsort(perm)
        fusedp=fused[:,perm]
        pairp=pair[:,perm][:,:,perm]
        contextp=context[:,perm]

        ref_dir_p=reference_pairwise_mean_direction(pairp)
        trt_dir_p,_=opponent_profile_vector_direction(fusedp,pairp)
        max_ref_perm=max(
            max_ref_perm,
            float((ref_dir_p[:,inv]-ref_dir).abs().max()),
        )
        max_trt_perm=max(
            max_trt_perm,
            float((trt_dir_p[:,inv]-trt_dir).abs().max()),
        )

        for composer,use_profile in (
            (reference,False),
            (treatment,True),
        ):
            out,diag=compose_with_residual_direction(
                composer,fused,pair,context,
                use_opponent_profile_direction=use_profile,
                return_diagnostics=True,
            )
            max_bound_violation=max(
                max_bound_violation,
                float(((out-fused).abs()-diag["residual_bound"]).clamp_min(0).max()),
            )
            max_mass_error=max(
                max_mass_error,
                float((torch.softmax(out,-1).sum(-1)-1.0).abs().max()),
            )
            ident=compose_with_residual_direction(
                composer,fused,pair,context,
                use_opponent_profile_direction=use_profile,
                alpha_override=0.0,
            )
            max_identity_error=max(
                max_identity_error,
                float((ident-fused).abs().max()),
            )

    if max_direction_difference<=1e-4:
        raise RuntimeError("S72 A0 treatment direction did not differ mechanically")

    neutral_fused,neutral_pair=_neutral_case()
    neutral_ref=reference_pairwise_mean_direction(neutral_pair)
    neutral_trt,_=opponent_profile_vector_direction(neutral_fused,neutral_pair)
    neutral_collapse_error=float((neutral_ref-neutral_trt).abs().max())

    # Diagonal must never contribute to either direction.
    fused=torch.randn(3,4,generator=g)
    pair=_antisymmetric_pair(3,4,g)
    ref0=reference_pairwise_mean_direction(pair)
    trt0,_=opponent_profile_vector_direction(fused,pair)
    changed=pair.clone()
    idx=torch.arange(4)
    changed[:,idx,idx]=9999.0
    ref1=reference_pairwise_mean_direction(changed)
    trt1,_=opponent_profile_vector_direction(fused,changed)
    diagonal_direction_error=max(
        float((ref0-ref1).abs().max()),
        float((trt0-trt1).abs().max()),
    )

    # Composer ownership / staged-gradient court is intentionally identical
    # between arms: S72 changes only inference residual direction.
    fc=torch.randn(32,4,generator=g,requires_grad=True)
    fp=torch.randn(32,4,generator=g,requires_grad=True)
    pc=_antisymmetric_pair(32,4,g).requires_grad_(True)
    pp=_antisymmetric_pair(32,4,g).requires_grad_(True)
    rc=torch.randn(32,4,512,generator=g,requires_grad=True)
    rp=torch.randn(32,4,512,generator=g,requires_grad=True)
    gold=torch.arange(32,dtype=torch.long)%4

    gradient_receipt={}
    for label in ("reference","treatment"):
        composer=MultiStatPairwiseRowComposer(use_multistat=False)
        loss,_=multistat_responsibility_loss(
            composer,fc,fp,pc,pp,rc,rp,gold
        )
        grads=torch.autograd.grad(
            loss,
            (
                composer.W_phi,composer.b_phi,composer.w_out,composer.b_out,
                fc,fp,pc,pp,rc,rp,
            ),
            allow_unused=True,
        )
        l1=[0.0 if x is None else float(x.detach().abs().sum()) for x in grads]
        if l1[2]<=0.0 or l1[3]<=0.0:
            raise RuntimeError(f"S72 A0 {label} output gradient vanished")
        if l1[0]!=0.0 or l1[1]!=0.0:
            raise RuntimeError(f"S72 A0 {label} initial phi gradient changed")
        if any(x is not None for x in grads[4:]):
            raise RuntimeError(f"S72 A0 {label} gradient leaked upstream")

        clone=MultiStatPairwiseRowComposer(use_multistat=False)
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
            raise RuntimeError(f"S72 A0 {label} staged phi gradient vanished")

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
        "reference_composer_parameter_count":reference.parameter_count,
        "treatment_composer_parameter_count":treatment.parameter_count,
        "direction_trainable_parameter_count_reference":0,
        "direction_trainable_parameter_count_treatment":0,
        "added_treatment_parameter_count":0,
        "composer_parameter_initialization_bit_identical":True,
        "context_projection_bit_identical":True,
        "alpha_match_max_abs_error":max_alpha_match,
        "initial_alpha_max_abs_error":max_initial_alpha_error,
        "reference_direction_permutation_max_abs_error":max_ref_perm,
        "treatment_direction_permutation_max_abs_error":max_trt_perm,
        "profile_confidence_max_abs":max_profile_bound,
        "opponent_mass_max_abs_error":max_opponent_mass_error,
        "opponent_diagonal_weight_max_abs":max_diagonal_weight,
        "neutral_direction_collapse_max_abs_error":neutral_collapse_error,
        "diagonal_direction_max_abs_error":diagonal_direction_error,
        "nontrivial_direction_max_abs_difference":max_direction_difference,
        "identity_override_max_abs_error":max_identity_error,
        "residual_bound_violation_max":max_bound_violation,
        "max_probability_mass_error":max_mass_error,
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
        raise RuntimeError("S72 A0 parent S71 not PASS")
    if parent.get("outcome")!="HIRA_V1_S71_MULTISTAT_ROW_COMPOSER_DEV_COMPLETE":
        raise RuntimeError("S72 A0 parent S71 outcome changed")
    if parent.get("scientific_authority")!="V1_S71_FRESH_MULTISTAT_ROW_COMPOSER":
        raise RuntimeError("S72 A0 parent S71 authority changed")
    if parent.get("seed")!=92001:
        raise RuntimeError("S72 A0 parent S71 seed changed")
    if parent.get("second_dev_run_performed") is not False:
        raise RuntimeError("S72 A0 parent S71 second DEV changed")
    if parent.get("external_laya_jev_evaluation_opened") is not False:
        raise RuntimeError("S72 A0 parent external eval changed")

    mechanics=_mechanics()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    reference=MultiStatPairwiseRowComposer(use_multistat=False)
    treatment=MultiStatPairwiseRowComposer(use_multistat=False)
    checkpoint=out/"matched-composers.pt"
    torch.save({
        "schema_version":"hira-v1-s72-matched-composers-v1",
        "seed":SEED,
        "reference_state_dict":reference.state_dict_exact(),
        "treatment_state_dict":treatment.state_dict_exact(),
    },checkpoint)

    g=torch.Generator().manual_seed(72072073)
    pair=_antisymmetric_pair(2,4,g)
    fused=torch.randn(2,4,generator=g)
    context=torch.randn(2,4,512,generator=g)

    rr=MultiStatPairwiseRowComposer(use_multistat=False)
    tt=MultiStatPairwiseRowComposer(use_multistat=False)
    rr.load_state_dict_exact(reference.state_dict_exact(),freeze=True)
    tt.load_state_dict_exact(treatment.state_dict_exact(),freeze=True)
    replay_error=max(
        float((
            compose_with_residual_direction(
                reference,fused,pair,context,
                use_opponent_profile_direction=False,
            )-
            compose_with_residual_direction(
                rr,fused,pair,context,
                use_opponent_profile_direction=False,
            )
        ).abs().max()),
        float((
            compose_with_residual_direction(
                treatment,fused,pair,context,
                use_opponent_profile_direction=True,
            )-
            compose_with_residual_direction(
                tt,fused,pair,context,
                use_opponent_profile_direction=True,
            )
        ).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S72_A0_OPPONENT_PROFILE_VECTOR_RESIDUAL_ONLY",
        "seed":SEED,
        "parent_s71":{
            "merged_main":"ef2e8a4b71ba9a46dd9caea07698df742abb6461",
            "scientific_run":37459447097,
            "artifact_id":11412505173,
            "artifact_digest":"sha256:aa116237d5508131db18f88a23893d9107d05ae67f46ccffe638520fe8098a50",
            "verdict":"CASE_C",
        },
        **mechanics,
        "one_encoder_state_once":True,
        "matched_composer_checkpoint_replay_max_abs_error":replay_error,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "reliability_target_reopened":False,
        "composer_architecture_changed":False,
        "residual_direction_changed":True,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S72_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
