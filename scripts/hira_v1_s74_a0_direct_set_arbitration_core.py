from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.v1_direct_set_arbitration_core import (
    S74_FUSED_FEATURE_DIM,
    S74_INPUT_DIM,
    S74_PARAMETER_COUNT,
    DirectSetArbitrationCore,
    direct_set_arbitration_features,
    direct_set_arbitration_loss,
)


SCHEMA_VERSION="hira-v1-s74-a0-direct-set-arbitration-core-v1"
OUTCOME="HIRA_V1_S74_A0_DIRECT_SET_ARBITRATION_CORE_READY"
SEED=95_001


def _pair(batch:int,k:int,seed:int)->torch.Tensor:
    g=torch.Generator().manual_seed(seed)
    raw=torch.randn(batch,k,k,generator=g)
    pair=0.5*(raw-raw.transpose(-1,-2))
    eye=torch.eye(k,dtype=torch.bool).unsqueeze(0)
    return pair.masked_fill(eye,0.0)


def _same_state(a:DirectSetArbitrationCore,b:DirectSetArbitrationCore)->bool:
    sa=a.state_dict_exact()
    sb=b.state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _mechanics():
    reference=DirectSetArbitrationCore()
    treatment=DirectSetArbitrationCore()
    if reference.parameter_count!=S74_PARAMETER_COUNT:
        raise RuntimeError("S74 A0 reference capacity changed")
    if treatment.parameter_count!=S74_PARAMETER_COUNT:
        raise RuntimeError("S74 A0 treatment capacity changed")
    if not _same_state(reference,treatment):
        raise RuntimeError("S74 A0 initialization mismatch")

    reference_relational_max_abs=0.0
    treatment_relational_max_abs=0.0
    permutation_error=0.0
    max_mass_error=0.0
    arbitrary_k={}

    for k in (3,7,255):
        b=3
        g=torch.Generator().manual_seed(SEED+k)
        fused=torch.randn(b,k,generator=g)
        pair=_pair(b,k,SEED+100+k)

        ref_features=direct_set_arbitration_features(
            fused,pair,use_relational=False
        )
        trt_features=direct_set_arbitration_features(
            fused,pair,use_relational=True
        )

        reference_relational_max_abs=max(
            reference_relational_max_abs,
            float(ref_features[...,S74_FUSED_FEATURE_DIM:].abs().max()),
        )
        treatment_relational_max_abs=max(
            treatment_relational_max_abs,
            float(trt_features[...,S74_FUSED_FEATURE_DIM:].abs().max()),
        )

        for core,use_relational in (
            (reference,False),
            (treatment,True),
        ):
            logits=core(fused,pair,use_relational=use_relational)
            probs=torch.softmax(logits,dim=-1)
            max_mass_error=max(
                max_mass_error,
                float((probs.sum(dim=-1)-1.0).abs().max()),
            )

            perm=torch.randperm(k,generator=g)
            inv=torch.argsort(perm)
            pp=pair[:,perm][:,:,perm]
            got=core(
                fused[:,perm],pp,use_relational=use_relational
            )[:,inv]
            permutation_error=max(
                permutation_error,
                float((logits-got).abs().max()),
            )

        arbitrary_k[f"arbitrary_k{k}_pass"]=True

    # Direct-logit semantics: with zero parameters output must be zero rather
    # than preserving fused logits through a residual/fallback path.
    direct_probe=DirectSetArbitrationCore()
    with torch.no_grad():
        for p in direct_probe.parameters():
            p.zero_()
    fused=torch.tensor([[3.0,1.0,-2.0]],dtype=torch.float32)
    pair=_pair(1,3,SEED+700)
    zero_logits=direct_probe(fused,pair,use_relational=True)
    direct_zero_max_abs=float(zero_logits.abs().max())
    fused_passthrough_max_abs_error=float((zero_logits-fused).abs().max())

    # Gradient ownership.
    g=torch.Generator().manual_seed(SEED+800)
    fc=torch.randn(8,4,generator=g,requires_grad=True)
    fp=torch.randn(8,4,generator=g,requires_grad=True)
    pc=_pair(8,4,SEED+801).requires_grad_(True)
    pp=_pair(8,4,SEED+802).requires_grad_(True)
    gold=torch.arange(8,dtype=torch.long)%4
    core=DirectSetArbitrationCore()
    loss,_=direct_set_arbitration_loss(
        core,fc,fp,pc,pp,gold,use_relational=True
    )
    grads=torch.autograd.grad(
        loss,tuple(core.parameters()),allow_unused=True
    )
    gradient_l1=[
        0.0 if grad is None else float(grad.detach().abs().sum())
        for grad in grads
    ]
    all_core_gradients_live=all(x>0.0 for x in gradient_l1)
    upstream_gradient_zero=all(
        x.grad is None for x in (fc,fp,pc,pp)
    )

    # Exact checkpoint replay.
    replay=DirectSetArbitrationCore()
    replay.load_state_dict_exact(core.state_dict_exact(),freeze=True)
    fused_probe=torch.randn(5,6,generator=g)
    pair_probe=_pair(5,6,SEED+900)
    expected=core(fused_probe,pair_probe,use_relational=True)
    got=replay(fused_probe,pair_probe,use_relational=True)
    replay_error=float((expected-got).abs().max())

    return {
        "reference_parameter_count":reference.parameter_count,
        "treatment_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":0,
        "parameter_initialization_bit_identical":True,
        "reference_relational_channels_max_abs":reference_relational_max_abs,
        "treatment_relational_channels_max_abs":treatment_relational_max_abs,
        "permutation_max_abs_error":permutation_error,
        "max_probability_mass_error":max_mass_error,
        "direct_zero_parameter_logits_max_abs":direct_zero_max_abs,
        "direct_zero_vs_fused_max_abs_difference":fused_passthrough_max_abs_error,
        "all_core_gradients_live":all_core_gradients_live,
        "core_gradient_l1":gradient_l1,
        "gradient_to_upstream_zero":upstream_gradient_zero,
        "checkpoint_replay_max_abs_error":replay_error,
        **arbitrary_k,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    parent=json.loads(args.parent_result.read_text(encoding="utf-8"))
    if parent.get("status")!="PASS":
        raise RuntimeError("S74 parent S73 not PASS")
    if parent.get("outcome")!="HIRA_V1_S73_COUNTERFACTUAL_SAFETY_VETO_DEV_COMPLETE":
        raise RuntimeError("S74 parent S73 outcome changed")
    if parent.get("scientific_authority")!="V1_S73_FRESH_COUNTERFACTUAL_SAFETY_VETO":
        raise RuntimeError("S74 parent S73 authority changed")
    if parent.get("seed")!=94001:
        raise RuntimeError("S74 parent S73 seed changed")
    if parent.get("second_dev_run_performed") is not False:
        raise RuntimeError("S74 parent S73 second DEV changed")
    if parent.get("external_laya_jev_evaluation_opened") is not False:
        raise RuntimeError("S74 parent external eval changed")

    mechanics=_mechanics()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    reference=DirectSetArbitrationCore()
    treatment=DirectSetArbitrationCore()
    checkpoint=out/"matched-dsac-cores.pt"
    torch.save({
        "schema_version":"hira-v1-s74-matched-dsac-cores-v1",
        "seed":SEED,
        "reference_state_dict":reference.state_dict_exact(),
        "treatment_state_dict":treatment.state_dict_exact(),
    },checkpoint)

    rr=DirectSetArbitrationCore()
    tt=DirectSetArbitrationCore()
    rr.load_state_dict_exact(reference.state_dict_exact(),freeze=True)
    tt.load_state_dict_exact(treatment.state_dict_exact(),freeze=True)
    if not _same_state(rr,tt):
        raise RuntimeError("S74 replay states diverged")

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S74_A0_DIRECT_SET_ARBITRATION_CORE_ONLY",
        "seed":SEED,
        "parent_s73":{
            "merged_main":"45ff1bc301972fc8aa04c1456200283f65616937",
            "scientific_run":37480212536,
            "artifact_id":11420533826,
            "artifact_digest":"sha256:0f08d05bea219333495529c6ae2c54d65fd9a276dd5f783adff0b0ac3925982f",
            "verdict":"CASE_C",
        },
        **mechanics,
        "one_encoder_state_once":True,
        "full_k":True,
        "k_specific_learned_tensors":False,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "dev_target_dependency":False,
        "fresh_train_dev_exposed":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }

    if result["reference_parameter_count"]!=S74_PARAMETER_COUNT:
        raise RuntimeError("S74 A0 reference parameter count changed")
    if result["treatment_parameter_count"]!=S74_PARAMETER_COUNT:
        raise RuntimeError("S74 A0 treatment parameter count changed")
    if result["reference_relational_channels_max_abs"]!=0.0:
        raise RuntimeError("S74 A0 reference relational channel leaked")
    if result["treatment_relational_channels_max_abs"]<=1e-5:
        raise RuntimeError("S74 A0 treatment relational channels degenerate")
    if result["permutation_max_abs_error"]>2e-6:
        raise RuntimeError("S74 A0 permutation equivariance failed")
    if result["max_probability_mass_error"]>1e-6:
        raise RuntimeError("S74 A0 probability mass failed")
    if result["direct_zero_parameter_logits_max_abs"]!=0.0:
        raise RuntimeError("S74 A0 direct zero-logit contract failed")
    if result["direct_zero_vs_fused_max_abs_difference"]<=1e-3:
        raise RuntimeError("S74 A0 fused passthrough unexpectedly present")
    if result["all_core_gradients_live"] is not True:
        raise RuntimeError("S74 A0 core gradient vanished")
    if result["gradient_to_upstream_zero"] is not True:
        raise RuntimeError("S74 A0 upstream gradient leaked")
    if result["checkpoint_replay_max_abs_error"]!=0.0:
        raise RuntimeError("S74 A0 checkpoint replay changed")

    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S74_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
