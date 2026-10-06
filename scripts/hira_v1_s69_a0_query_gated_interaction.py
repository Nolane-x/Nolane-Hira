from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_explicit_pairwise_decision_head import (
    ExplicitPairwiseDecisionHead,
    S59_PAIRWISE_PARAMETER_COUNT,
    build_pairwise_representation,
    pairwise_head_loss,
)
from nmd.v1_query_gated_identity_interaction import (
    S69_INTERACTION_SCALE,
    S69_REPRESENTATION_DIMENSION,
    S69_REPRESENTATION_PARAMETER_COUNT,
    build_reference_pairwise_representation,
    build_query_gated_context,
    build_query_gated_pairwise_representation,
)
from nmd.v1_s68_authority import generate_s68_cases
from hira_v1_s50_train_dev import _materialize_cache


SCHEMA_VERSION="hira-v1-s69-a0-query-gated-interaction-rep-v1"
OUTCOME="HIRA_V1_S69_A0_QUERY_GATED_INTERACTION_REP_READY"
SEED=90_001


def _sources(op,evidence):
    identity=op.identity_signatures(
        state_tokens=evidence.state_tokens,
        state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
    )
    context=op.joint_query_context(
        state_tokens=evidence.state_tokens,
        state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
        question_tokens=evidence.question_tokens,
        question_mask=evidence.question_mask,
    )
    return identity,context


def _head_state_equal(a,b):
    sa=a.state_dict_exact()
    sb=b.state_dict_exact()
    return sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa)


def _synthetic_court():
    g=torch.Generator().manual_seed(69069069)
    max_ref_identity=max_ref_perm=max_trt_perm=max_pair_perm=0.0
    max_antisym=max_diag=max_mass=0.0
    max_treatment_difference=0.0

    for k in (3,7,255):
        identity=torch.nn.functional.normalize(
            torch.randn(2,k,256,generator=g),dim=-1
        )
        context=torch.nn.functional.normalize(
            torch.randn(2,k,256,generator=g),dim=-1
        )
        ref=build_reference_pairwise_representation(identity,context)
        s59=build_pairwise_representation(identity,context)
        trt=build_query_gated_pairwise_representation(identity,context)

        max_ref_identity=max(max_ref_identity,float((ref-s59).abs().max()))
        max_treatment_difference=max(
            max_treatment_difference,float((trt-ref).abs().max())
        )

        perm=torch.randperm(k,generator=g)
        refp=build_reference_pairwise_representation(identity[:,perm],context[:,perm])
        trtp=build_query_gated_pairwise_representation(identity[:,perm],context[:,perm])
        max_ref_perm=max(max_ref_perm,float((refp-ref[:,perm]).abs().max()))
        max_trt_perm=max(max_trt_perm,float((trtp-trt[:,perm]).abs().max()))

        head=ExplicitPairwiseDecisionHead(trainable=True)
        for rep in (ref,trt):
            pair=head.pairwise_logits(rep)
            score=head.aggregate_logits(rep)
            pairp=head.pairwise_logits(rep[:,perm])
            restored=pairp[:,torch.argsort(perm)][:,:,torch.argsort(perm)]
            max_pair_perm=max(max_pair_perm,float((pair-restored).abs().max()))
            max_antisym=max(
                max_antisym,float((pair+pair.transpose(-1,-2)).abs().max())
            )
            max_diag=max(
                max_diag,float(torch.diagonal(pair,dim1=-2,dim2=-1).abs().max())
            )
            max_mass=max(
                max_mass,float((torch.softmax(score,-1).sum(-1)-1.0).abs().max())
            )

    identity=torch.zeros(3,5,256)
    context=torch.nn.functional.normalize(
        torch.randn(3,5,256,generator=g),dim=-1
    )
    ref=build_reference_pairwise_representation(identity,context)
    trt=build_query_gated_pairwise_representation(identity,context)
    neutral_error=float((ref-trt).abs().max())

    if max_ref_identity!=0.0:
        raise RuntimeError("S69 A0 reference is not exact S59")
    if max_treatment_difference<=1e-7:
        raise RuntimeError("S69 A0 treatment representation degenerate")
    if neutral_error>1e-7:
        raise RuntimeError("S69 A0 zero-interaction neutralization failed")

    identity=torch.nn.functional.normalize(
        torch.randn(8,4,256,generator=g),dim=-1
    ).requires_grad_(True)
    context=torch.nn.functional.normalize(
        torch.randn(8,4,256,generator=g),dim=-1
    ).requires_grad_(True)
    gold=torch.arange(8,dtype=torch.long)%4

    gradients={}
    for label,builder in (
        ("reference",build_reference_pairwise_representation),
        ("treatment",build_query_gated_pairwise_representation),
    ):
        head=ExplicitPairwiseDecisionHead(trainable=True)
        rep=builder(identity,context)
        loss,_=pairwise_head_loss(head,rep,gold)
        grads=torch.autograd.grad(
            loss,(head.A,head.u,identity,context),allow_unused=True
        )
        a_l1=0.0 if grads[0] is None else float(grads[0].abs().sum())
        u_l1=0.0 if grads[1] is None else float(grads[1].abs().sum())
        if min(a_l1,u_l1)<=0.0:
            raise RuntimeError(f"S69 A0 {label} pairwise gradient vanished")
        if grads[2] is not None or grads[3] is not None:
            raise RuntimeError(f"S69 A0 {label} gradient leaked upstream")
        gradients[label]={"A":a_l1,"u":u_l1}

    return {
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "reference_vs_s59_max_abs_error":max_ref_identity,
        "treatment_reference_max_abs_difference":max_treatment_difference,
        "zero_identity_neutralization_max_abs_error":neutral_error,
        "reference_option_permutation_max_abs_error":max_ref_perm,
        "treatment_option_permutation_max_abs_error":max_trt_perm,
        "pairwise_permutation_max_abs_error":max_pair_perm,
        "antisymmetry_max_abs_error":max_antisym,
        "diagonal_max_abs_error":max_diag,
        "max_probability_mass_error":max_mass,
        "reference_gradient_A_l1":gradients["reference"]["A"],
        "reference_gradient_u_l1":gradients["reference"]["u"],
        "treatment_gradient_A_l1":gradients["treatment"]["A"],
        "treatment_gradient_u_l1":gradients["treatment"]["u"],
        "representation_received_pairwise_gradient":False,
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
        raise RuntimeError("S69 A0 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S69 A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S69 A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S69 A0 checkpoint file changed")

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
        raise RuntimeError("S69 A0 loaded native changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S69 A0 native remained trainable")

    rows=list(generate_s68_cases("train"))[:16]
    cache,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    geometry=JointStateQueryOptionPrivateCorrectionFork(train_correction=False)
    canonical,paraphrase,_n=cache[0]
    identity_c,context_c=_sources(geometry,canonical)
    identity_p,context_p=_sources(geometry,paraphrase)

    ref_c=build_reference_pairwise_representation(identity_c,context_c)
    ref_p=build_reference_pairwise_representation(identity_p,context_p)
    trt_c=build_query_gated_pairwise_representation(identity_c,context_c)
    trt_p=build_query_gated_pairwise_representation(identity_p,context_p)

    if max(
        float((ref_c-build_pairwise_representation(identity_c,context_c)).abs().max()),
        float((ref_p-build_pairwise_representation(identity_p,context_p)).abs().max()),
    )!=0.0:
        raise RuntimeError("S69 A0 real-cache reference changed from S59")

    real_difference=max(
        float((trt_c-ref_c).abs().max()),
        float((trt_p-ref_p).abs().max()),
    )
    if real_difference<=1e-7:
        raise RuntimeError("S69 A0 real-cache interaction degenerate")

    reference=ExplicitPairwiseDecisionHead(trainable=True)
    treatment=ExplicitPairwiseDecisionHead(trainable=True)
    if reference.parameter_count!=S59_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S69 A0 reference head capacity changed")
    if treatment.parameter_count!=S59_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S69 A0 treatment head capacity changed")
    if not _head_state_equal(reference,treatment):
        raise RuntimeError("S69 A0 head initialization changed")

    real_grad={}
    for label,head,rc,rp in (
        ("reference",reference,ref_c,ref_p),
        ("treatment",treatment,trt_c,trt_p),
    ):
        lc,dc=pairwise_head_loss(head,rc,canonical.gold)
        lp,dp=pairwise_head_loss(head,rp,paraphrase.gold)
        loss=0.5*(lc+lp)
        grads=torch.autograd.grad(loss,(head.A,head.u),allow_unused=True)
        vals=[0.0 if g is None else float(g.abs().sum()) for g in grads]
        if min(vals)<=0.0:
            raise RuntimeError(f"S69 A0 {label} real-cache gradient vanished")
        real_grad[label]={
            "A":vals[0],
            "u":vals[1],
            "gold_pair_accuracy_canonical":float(dc["gold_pair_accuracy"]),
            "gold_pair_accuracy_paraphrase":float(dp["gold_pair_accuracy"]),
        }

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0:
        raise RuntimeError("S69 A0 cache gained gradients")

    mechanics=_synthetic_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    ref_state=reference.state_dict_exact()
    trt_state=treatment.state_dict_exact()
    checkpoint=out/"matched-pairwise-heads.pt"
    torch.save({
        "schema_version":"hira-v1-s69-matched-pairwise-heads-v1",
        "seed":SEED,
        "interaction_scale":S69_INTERACTION_SCALE,
        "representation_parameter_count":S69_REPRESENTATION_PARAMETER_COUNT,
        "pairwise_parameter_count_per_arm":S59_PAIRWISE_PARAMETER_COUNT,
        "reference_state_dict":ref_state,
        "treatment_state_dict":trt_state,
    },checkpoint)

    rr=ExplicitPairwiseDecisionHead(trainable=True)
    tt=ExplicitPairwiseDecisionHead(trainable=True)
    rr.load_state_dict_exact(ref_state,freeze=True)
    tt.load_state_dict_exact(trt_state,freeze=True)
    replay_error=max(
        float((reference.pairwise_logits(ref_c)-rr.pairwise_logits(ref_c)).abs().max()),
        float((treatment.pairwise_logits(trt_c)-tt.pairwise_logits(trt_c)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S69_A0_QUERY_GATED_INTERACTION_REP_ONLY",
        "seed":SEED,
        "parent_s68":{
            "merged_main":"4cfdbc6fe100715bd950c9a0e2f504f71c486e9c",
            "scientific_run":37389861004,
            "artifact_id":11381470793,
            "artifact_digest":"sha256:0156bb57125f6b8c4f380649505803ed76cf49b8edf1c5bd4774fd8790386eaa",
            "verdict":"CASE_C",
        },
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "parent_exposed_semantic_case_count":len(cache),
        "cache_digest":cache_digest,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "reference_representation_parameter_count":0,
        "treatment_representation_parameter_count":0,
        "representation_dimension":S69_REPRESENTATION_DIMENSION,
        "interaction_scale":S69_INTERACTION_SCALE,
        "reference_pairwise_parameter_count":reference.parameter_count,
        "treatment_pairwise_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":0,
        "pairwise_trainable_tensor_names":sorted(dict(reference.named_parameters())),
        "head_parameter_initialization_bit_identical":True,
        "real_cache_treatment_reference_max_abs_difference":real_difference,
        "reference_real_cache_gradient_A_l1":real_grad["reference"]["A"],
        "reference_real_cache_gradient_u_l1":real_grad["reference"]["u"],
        "treatment_real_cache_gradient_A_l1":real_grad["treatment"]["A"],
        "treatment_real_cache_gradient_u_l1":real_grad["treatment"]["u"],
        "native_trainable_parameter_count":0,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "one_encoder_state_once":True,
        "matched_head_checkpoint_replay_max_abs_error":replay_error,
        **mechanics,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S69_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
