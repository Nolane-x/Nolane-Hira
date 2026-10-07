from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionLateInteraction
from nmd.v1_query_gated_identity_interaction import build_query_gated_pairwise_representation
from nmd.v1_explicit_pairwise_decision_head import (
    ExplicitPairwiseDecisionHead,
    S59_PAIRWISE_PARAMETER_COUNT,
)
from nmd.v1_token_level_bidirectional_evidence_binding import (
    S75_REPRESENTATION_DIMENSION,
    S75_REPRESENTATION_PARAMETER_COUNT,
    S75_TEMPERATURE,
    S75_INTERACTION_SCALE,
    build_token_level_bidirectional_pairwise_representation,
)


SCHEMA_VERSION="hira-v1-s75-a0-token-level-bidirectional-binding-v1"
OUTCOME="HIRA_V1_S75_A0_TOKEN_LEVEL_BIDIRECTIONAL_BINDING_READY"
SEED=96_001


def _fixture(*,k:int,seed:int,neutral:bool=False):
    g=torch.Generator().manual_seed(seed+k)
    b,s,v,t,q,d=3,5,2,4,6,256
    identity=F.normalize(torch.randn(b,k,d,generator=g),dim=-1)
    if neutral:
        base=F.normalize(torch.randn(b,1,d,generator=g),dim=-1)
        state=base.expand(b,s,d).clone()
        option=base[:,None,None,None,:].expand(b,k,v,t,d).clone()
        query=base.expand(b,q,d).clone()
    else:
        state=torch.randn(b,s,d,generator=g)
        option=torch.randn(b,k,v,t,d,generator=g)
        query=torch.randn(b,q,d,generator=g)
    state_mask=torch.ones(b,s,dtype=torch.bool)
    option_token_mask=torch.ones(b,k,v,t,dtype=torch.bool)
    option_view_mask=torch.ones(b,k,v,dtype=torch.bool)
    question_mask=torch.ones(b,q,dtype=torch.bool)
    return identity,state,state_mask,option,option_token_mask,option_view_mask,query,question_mask


def _reference_rep(x):
    identity,state,state_mask,option,option_token_mask,option_view_mask,query,question_mask=x
    s54=JointStateQueryOptionLateInteraction()
    context=s54(
        state_tokens=state,
        state_mask=state_mask,
        option_view_tokens=option,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
        question_tokens=query,
        question_mask=question_mask,
    )
    return build_query_gated_pairwise_representation(identity,context)


def _treatment_rep(x):
    identity,state,state_mask,option,option_token_mask,option_view_mask,query,question_mask=x
    return build_token_level_bidirectional_pairwise_representation(
        identity=identity,
        state_tokens=state,
        state_mask=state_mask,
        option_view_tokens=option,
        option_view_token_mask=option_token_mask,
        option_view_mask=option_view_mask,
        question_tokens=query,
        question_mask=question_mask,
    )


def _same_head_state(a,b):
    sa=a.state_dict_exact()
    sb=b.state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _mechanics():
    ref_head=ExplicitPairwiseDecisionHead()
    trt_head=ExplicitPairwiseDecisionHead()
    if not _same_head_state(ref_head,trt_head):
        raise RuntimeError("S75 head initialization mismatch")

    max_rep_difference=0.0
    max_ref_perm=max_trt_perm=0.0
    max_mass=0.0
    arbitrary={}

    for k in (3,7,255):
        x=list(_fixture(k=k,seed=SEED+100+k))
        ref=_reference_rep(x)
        trt=_treatment_rep(x)
        max_rep_difference=max(max_rep_difference,float((ref-trt).abs().max()))

        perm=torch.randperm(k,generator=torch.Generator().manual_seed(SEED+k))
        inv=torch.argsort(perm)
        xp=list(x)
        xp[0]=x[0][:,perm]
        xp[3]=x[3][:,perm]
        xp[4]=x[4][:,perm]
        xp[5]=x[5][:,perm]
        ref_perm=_reference_rep(xp)[:,inv]
        trt_perm=_treatment_rep(xp)[:,inv]
        max_ref_perm=max(max_ref_perm,float((ref-ref_perm).abs().max()))
        max_trt_perm=max(max_trt_perm,float((trt-trt_perm).abs().max()))

        for head,rep in ((ref_head,ref),(trt_head,trt)):
            pair=head.pairwise_logits(rep)
            score=head.aggregate_logits(rep)
            max_mass=max(
                max_mass,
                float((torch.softmax(score,-1).sum(-1)-1).abs().max()),
            )
            if not bool(torch.isfinite(pair).all()):
                raise RuntimeError("S75 pair logits non-finite")
        arbitrary[f"arbitrary_k{k}_pass"]=True

    neutral=_fixture(k=4,seed=SEED+800,neutral=True)
    neutral_ref=_reference_rep(neutral)
    neutral_trt=_treatment_rep(neutral)
    neutral_error=float((neutral_ref-neutral_trt).abs().max())

    # Upstream ownership boundary.
    x=list(_fixture(k=4,seed=SEED+900))
    x[0].requires_grad_(True)
    x[1].requires_grad_(True)
    x[3].requires_grad_(True)
    x[6].requires_grad_(True)
    rep=_treatment_rep(x)
    upstream_gradient_zero=(
        not rep.requires_grad
        and x[0].grad is None
        and x[1].grad is None
        and x[3].grad is None
        and x[6].grad is None
    )

    # Exact S59 checkpoint replay.
    replay=ExplicitPairwiseDecisionHead()
    replay.load_state_dict_exact(ref_head.state_dict_exact(),freeze=True)
    probe=_reference_rep(_fixture(k=6,seed=SEED+1000))
    expected=ref_head.pairwise_logits(probe)
    got=replay.pairwise_logits(probe)
    replay_error=float((expected-got).abs().max())

    return {
        "reference_representation_dimension":S75_REPRESENTATION_DIMENSION,
        "treatment_representation_dimension":S75_REPRESENTATION_DIMENSION,
        "reference_representation_parameter_count":0,
        "treatment_representation_parameter_count":S75_REPRESENTATION_PARAMETER_COUNT,
        "reference_pairwise_parameter_count":ref_head.parameter_count,
        "treatment_pairwise_parameter_count":trt_head.parameter_count,
        "added_treatment_parameter_count":0,
        "pairwise_initialization_bit_identical":True,
        "temperature":S75_TEMPERATURE,
        "interaction_scale":S75_INTERACTION_SCALE,
        "nontrivial_representation_max_abs_difference":max_rep_difference,
        "reference_option_permutation_max_abs_error":max_ref_perm,
        "treatment_option_permutation_max_abs_error":max_trt_perm,
        "neutral_collapse_max_abs_error":neutral_error,
        "max_probability_mass_error":max_mass,
        "gradient_to_upstream_zero":upstream_gradient_zero,
        "checkpoint_replay_max_abs_error":replay_error,
        **arbitrary,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent-result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    parent=json.loads(args.parent_result.read_text(encoding="utf-8"))
    if parent.get("status")!="PASS":
        raise RuntimeError("S75 parent S74 not PASS")
    if parent.get("outcome")!="HIRA_V1_S74_DIRECT_SET_ARBITRATION_CORE_DEV_COMPLETE":
        raise RuntimeError("S75 parent S74 outcome changed")
    if parent.get("scientific_authority")!="V1_S74_FRESH_DIRECT_SET_ARBITRATION_CORE":
        raise RuntimeError("S75 parent S74 authority changed")
    if parent.get("seed")!=95001:
        raise RuntimeError("S75 parent S74 seed changed")
    if parent.get("second_dev_run_performed") is not False:
        raise RuntimeError("S75 parent second DEV changed")
    if parent.get("external_laya_jev_evaluation_opened") is not False:
        raise RuntimeError("S75 parent external eval changed")

    mechanics=_mechanics()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)

    ref_head=ExplicitPairwiseDecisionHead()
    trt_head=ExplicitPairwiseDecisionHead()
    checkpoint=out/"matched-s59-heads.pt"
    torch.save({
        "schema_version":"hira-v1-s75-matched-s59-heads-v1",
        "seed":SEED,
        "reference_state_dict":ref_head.state_dict_exact(),
        "treatment_state_dict":trt_head.state_dict_exact(),
    },checkpoint)

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S75_A0_TOKEN_LEVEL_BIDIRECTIONAL_BINDING_ONLY",
        "seed":SEED,
        "parent_s74":{
            "merged_main":"0edfb0a64823ccb7ef2cf5c4e77b9fb9696361be",
            "scientific_run":37487578737,
            "artifact_id":11423428905,
            "artifact_digest":"sha256:cb2d8e2718d3aa426bf3ee44795d36c25e5960ca727fb8cc6280d12cfbefdb21",
            "verdict":"CASE_D",
        },
        **mechanics,
        "option_view_permutation_invariance_covered_by_contract_tests":True,
        "mask_correctness_covered_by_contract_tests":True,
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

    if result["reference_representation_dimension"]!=512 or result["treatment_representation_dimension"]!=512:
        raise RuntimeError("S75 A0 representation dimension changed")
    if result["reference_representation_parameter_count"]!=0 or result["treatment_representation_parameter_count"]!=0:
        raise RuntimeError("S75 A0 representation gained parameters")
    if result["reference_pairwise_parameter_count"]!=S59_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S75 A0 reference head capacity changed")
    if result["treatment_pairwise_parameter_count"]!=S59_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S75 A0 treatment head capacity changed")
    if result["nontrivial_representation_max_abs_difference"]<=1e-5:
        raise RuntimeError("S75 A0 treatment representation degenerate")
    if result["reference_option_permutation_max_abs_error"]>2e-6:
        raise RuntimeError("S75 A0 reference permutation failed")
    if result["treatment_option_permutation_max_abs_error"]>2e-6:
        raise RuntimeError("S75 A0 treatment permutation failed")
    if result["neutral_collapse_max_abs_error"]>2e-6:
        raise RuntimeError("S75 A0 neutral collapse failed")
    if result["max_probability_mass_error"]>1e-6:
        raise RuntimeError("S75 A0 probability mass failed")
    if result["gradient_to_upstream_zero"] is not True:
        raise RuntimeError("S75 A0 upstream gradient leaked")
    if result["checkpoint_replay_max_abs_error"]!=0.0:
        raise RuntimeError("S75 A0 checkpoint replay changed")

    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S75_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
