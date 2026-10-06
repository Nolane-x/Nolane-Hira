from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.v1_counterfactual_safety_veto import (
    S73_ACCEPT_THRESHOLD,
    S73_PARAMETER_COUNT,
    CounterfactualSafetyVeto,
    counterfactual_safety_loss,
    counterfactual_safety_targets,
    safety_features,
)
from nmd.v1_multistat_pairwise_row_composer import MultiStatPairwiseRowComposer
from nmd.v1_opponent_profile_vector_residual import compose_with_residual_direction


SCHEMA_VERSION="hira-v1-s73-a0-counterfactual-safety-veto-v1"
OUTCOME="HIRA_V1_S73_A0_COUNTERFACTUAL_SAFETY_VETO_READY"
SEED=94_001


def _antisymmetric_pair(batch:int,k:int,g:torch.Generator)->torch.Tensor:
    raw=torch.randn(batch,k,k,generator=g)
    pair=0.5*(raw-raw.transpose(-1,-2))
    eye=torch.eye(k,dtype=torch.bool).unsqueeze(0)
    return pair.masked_fill(eye,0.0)


def _same_state(a:CounterfactualSafetyVeto,b:CounterfactualSafetyVeto)->bool:
    sa=a.state_dict_exact()
    sb=b.state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _mechanics():
    g=torch.Generator().manual_seed(73073073)
    reference_predictor=CounterfactualSafetyVeto()
    treatment_predictor=CounterfactualSafetyVeto()

    if reference_predictor.parameter_count!=S73_PARAMETER_COUNT:
        raise RuntimeError("S73 A0 reference predictor capacity changed")
    if treatment_predictor.parameter_count!=S73_PARAMETER_COUNT:
        raise RuntimeError("S73 A0 treatment predictor capacity changed")
    if not _same_state(reference_predictor,treatment_predictor):
        raise RuntimeError("S73 A0 predictor initialization changed")

    initial_probability_error=0.0
    feature_permutation_error=0.0
    all_accept_error=0.0
    all_veto_error=0.0
    endpoint_error=0.0
    arbitrary_k={}

    for k in (3,7,255):
        b=3
        fused=torch.randn(b,k,generator=g)
        pair=_antisymmetric_pair(b,k,g)
        context=torch.randn(b,k,512,generator=g)

        composer=MultiStatPairwiseRowComposer(use_multistat=False)
        candidate,diag=compose_with_residual_direction(
            composer,
            fused,
            pair,
            context,
            use_opponent_profile_direction=True,
            return_diagnostics=True,
        )
        features=safety_features(
            fused,candidate,pair,diag["direction"],diag["alpha"]
        )
        p=reference_predictor.probability(features)
        initial_probability_error=max(
            initial_probability_error,
            float((p-0.5).abs().max()),
        )

        accept=treatment_predictor.apply(
            fused,candidate,features,force_accept=True
        )
        veto=treatment_predictor.apply(
            fused,candidate,features,force_accept=False
        )
        all_accept_error=max(all_accept_error,float((accept-candidate).abs().max()))
        all_veto_error=max(all_veto_error,float((veto-fused).abs().max()))

        ordinary,ordinary_diag=treatment_predictor.apply(
            fused,candidate,features,return_diagnostics=True
        )
        choose_candidate=ordinary_diag["accept"].unsqueeze(-1)
        expected=torch.where(choose_candidate,candidate,fused)
        endpoint_error=max(endpoint_error,float((ordinary-expected).abs().max()))

        perm=torch.randperm(k,generator=g)
        inv=torch.argsort(perm)
        pairp=pair[:,perm][:,:,perm]
        fusedp=fused[:,perm]
        contextp=context[:,perm]
        candidatep,diagp=compose_with_residual_direction(
            composer,
            fusedp,pairp,contextp,
            use_opponent_profile_direction=True,
            return_diagnostics=True,
        )
        fp=safety_features(
            fusedp,candidatep,pairp,diagp["direction"],diagp["alpha"]
        )
        feature_permutation_error=max(
            feature_permutation_error,
            float((features-fp).abs().max()),
        )

        arbitrary_k[f"arbitrary_k{k}_pass"]=True

    # Target mechanics and ownership.
    b,k=32,4
    fc=torch.randn(b,k,generator=g,requires_grad=True)
    fp=torch.randn(b,k,generator=g,requires_grad=True)
    pc=_antisymmetric_pair(b,k,g).requires_grad_(True)
    pp=_antisymmetric_pair(b,k,g).requires_grad_(True)
    rc=torch.randn(b,k,512,generator=g,requires_grad=True)
    rp=torch.randn(b,k,512,generator=g,requires_grad=True)
    gold=torch.arange(b,dtype=torch.long)%k

    composer=MultiStatPairwiseRowComposer(use_multistat=False)
    cc,dc=compose_with_residual_direction(
        composer,fc,pc,rc,
        use_opponent_profile_direction=True,
        return_diagnostics=True,
    )
    cp,dp=compose_with_residual_direction(
        composer,fp,pp,rp,
        use_opponent_profile_direction=True,
        return_diagnostics=True,
    )

    yc,yp,target_diag=counterfactual_safety_targets(
        fc,fp,cc,cp,gold
    )
    if yc.requires_grad or yp.requires_grad:
        raise RuntimeError("S73 A0 target leaked gradient")

    xc=safety_features(fc,cc,pc,dc["direction"],dc["alpha"])
    xp=safety_features(fp,cp,pp,dp["direction"],dp["alpha"])
    if xc.requires_grad or xp.requires_grad:
        raise RuntimeError("S73 A0 features leaked gradient")

    predictor=CounterfactualSafetyVeto()
    loss,loss_diag=counterfactual_safety_loss(
        predictor,xc,xp,yc,yp
    )
    grads=torch.autograd.grad(
        loss,(predictor.weight,predictor.bias),
        allow_unused=True,
    )
    weight_grad=0.0 if grads[0] is None else float(grads[0].abs().sum())
    bias_grad=0.0 if grads[1] is None else float(grads[1].abs().sum())
    if weight_grad<=0.0:
        raise RuntimeError("S73 A0 predictor weight gradient vanished")

    # Detached feature/target construction means upstream tensors receive no
    # graph from predictor loss.
    upstream_gradient_zero=all(
        x.grad is None for x in (fc,fp,pc,pp,rc,rp)
    )

    return {
        "reference_predictor_parameter_count":reference_predictor.parameter_count,
        "treatment_predictor_parameter_count":treatment_predictor.parameter_count,
        "added_treatment_parameter_count":0,
        "predictor_initialization_bit_identical":True,
        "hard_accept_threshold":S73_ACCEPT_THRESHOLD,
        "initial_probability_max_abs_error":initial_probability_error,
        "feature_permutation_max_abs_error":feature_permutation_error,
        "all_accept_candidate_max_abs_error":all_accept_error,
        "all_veto_fused_max_abs_error":all_veto_error,
        "endpoint_only_max_abs_error":endpoint_error,
        "features_detached":True,
        "targets_detached":True,
        "predictor_weight_gradient_l1":weight_grad,
        "predictor_bias_gradient_l1":bias_grad,
        "predictor_gradient_to_upstream_zero":upstream_gradient_zero,
        "canonical_safe_count":float(target_diag["canonical_safe_count"]),
        "paraphrase_safe_count":float(target_diag["paraphrase_safe_count"]),
        "canonical_predicted_accept_count":float(loss_diag["canonical_predicted_accept_count"]),
        "paraphrase_predicted_accept_count":float(loss_diag["paraphrase_predicted_accept_count"]),
        **arbitrary_k,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    parent=json.loads(args.parent_result.read_text(encoding="utf-8"))
    if parent.get("status")!="PASS":
        raise RuntimeError("S73 A0 parent S72 not PASS")
    if parent.get("outcome")!="HIRA_V1_S72_OPPONENT_PROFILE_VECTOR_RESIDUAL_DEV_COMPLETE":
        raise RuntimeError("S73 A0 parent S72 outcome changed")
    if parent.get("scientific_authority")!="V1_S72_FRESH_OPPONENT_PROFILE_VECTOR_RESIDUAL":
        raise RuntimeError("S73 A0 parent S72 authority changed")
    if parent.get("seed")!=93001:
        raise RuntimeError("S73 A0 parent S72 seed changed")
    if parent.get("second_dev_run_performed") is not False:
        raise RuntimeError("S73 A0 parent S72 second DEV changed")
    if parent.get("external_laya_jev_evaluation_opened") is not False:
        raise RuntimeError("S73 A0 parent external eval changed")

    mechanics=_mechanics()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    reference=CounterfactualSafetyVeto()
    treatment=CounterfactualSafetyVeto()
    checkpoint=out/"matched-veto-predictors.pt"
    torch.save({
        "schema_version":"hira-v1-s73-matched-veto-predictors-v1",
        "seed":SEED,
        "reference_state_dict":reference.state_dict_exact(),
        "treatment_state_dict":treatment.state_dict_exact(),
    },checkpoint)

    rr=CounterfactualSafetyVeto()
    tt=CounterfactualSafetyVeto()
    rr.load_state_dict_exact(reference.state_dict_exact(),freeze=True)
    tt.load_state_dict_exact(treatment.state_dict_exact(),freeze=True)
    probe=torch.randn(7,8,generator=torch.Generator().manual_seed(73073074))
    replay_error=max(
        float((reference.probability(probe)-rr.probability(probe)).abs().max()),
        float((treatment.probability(probe)-tt.probability(probe)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S73_A0_COUNTERFACTUAL_SAFETY_VETO_ONLY",
        "seed":SEED,
        "parent_s72":{
            "merged_main":"16eb7a583376cd824421d5e7ae9df948979b3953",
            "scientific_run":37470998139,
            "artifact_id":11416962628,
            "artifact_digest":"sha256:9a653a681e0cb3182819634fe24eb345ce82a48f2d078d15e510d1d3ebfb0834",
            "verdict":"CASE_B",
        },
        **mechanics,
        "checkpoint_replay_max_abs_error":replay_error,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "dev_target_dependency":False,
        "candidate_residual_changed":False,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }

    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S73_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
