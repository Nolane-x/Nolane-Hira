from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_explicit_pairwise_decision_head import (
    ExplicitPairwiseDecisionHead,
    build_pairwise_representation,
)
from nmd.v1_context_modulated_pairwise_head import (
    S68_CONTEXT_DIMENSION,
    S68_CONTEXT_PROJECTION_SEED,
    S68_MODULATION_PARAMETER_COUNT,
    S68_MODULATION_SCALE,
    S68_PAIRWISE_PARAMETER_COUNT,
    ContextModulatedPairwiseHead,
    context_modulated_pairwise_loss,
)
from nmd.v1_s67_authority import generate_s67_cases
from hira_v1_s50_train_dev import _materialize_cache


SCHEMA_VERSION="hira-v1-s68-a0-context-modulated-pairwise-v1"
OUTCOME="HIRA_V1_S68_A0_CONTEXT_MODULATED_PAIRWISE_READY"
SEED=89_001


def _representation(op,evidence):
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
    return build_pairwise_representation(identity,context)


def _param_state_equal(a,b)->bool:
    pa=dict(a.named_parameters())
    pb=dict(b.named_parameters())
    return pa.keys()==pb.keys() and all(torch.equal(pa[k],pb[k]) for k in pa)


def _synthetic_court():
    g=torch.Generator().manual_seed(68068068)
    ref=ContextModulatedPairwiseHead(use_joint_context=False)
    trt=ContextModulatedPairwiseHead(use_joint_context=True)
    s59=ExplicitPairwiseDecisionHead()

    if ref.parameter_count!=S68_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S68-A0 reference capacity changed")
    if trt.parameter_count!=S68_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S68-A0 treatment capacity changed")
    if not _param_state_equal(ref,trt):
        raise RuntimeError("S68-A0 arm parameter initialization changed")
    if not torch.equal(ref.A,s59.A) or not torch.equal(ref.u,s59.u):
        raise RuntimeError("S68-A0 base S59 initialization changed")
    if not torch.equal(ref.G,torch.zeros_like(ref.G)):
        raise RuntimeError("S68-A0 reference G init changed")
    if not torch.equal(trt.G,torch.zeros_like(trt.G)):
        raise RuntimeError("S68-A0 treatment G init changed")
    if not torch.equal(ref.context_projection,trt.context_projection):
        raise RuntimeError("S68-A0 context projection differs across arms")
    if ref.context_projection.requires_grad or trt.context_projection.requires_grad:
        raise RuntimeError("S68-A0 context projection became trainable")

    max_pair_identity=max_agg_identity=max_perm=max_antisym=max_diag=max_mass=0.0
    max_gamma_identity=0.0
    max_context_permutation_error=0.0
    context_difference=0.0
    ablated_context_error=0.0

    for k in (3,7,255):
        rep=torch.randn(1,k,512,generator=g)
        baseline_pair=s59.pairwise_logits(rep)
        baseline_agg=s59.aggregate_logits(rep)
        baseline_choice,_=s59.select_with_tiebreak(rep)

        for head in (ref,trt):
            gamma=head.modulation(rep)
            max_gamma_identity=max(
                max_gamma_identity,
                float((gamma-1.0).abs().max()),
            )
            pair=head.pairwise_logits(rep)
            agg=head.aggregate_logits(rep)
            choice,_=head.select_with_tiebreak(rep)
            max_pair_identity=max(
                max_pair_identity,
                float((pair-baseline_pair).abs().max()),
            )
            max_agg_identity=max(
                max_agg_identity,
                float((agg-baseline_agg).abs().max()),
            )
            if not torch.equal(choice,baseline_choice):
                raise RuntimeError(f"S68-A0 K={k} initial choice changed")
            max_antisym=max(
                max_antisym,
                float((pair+pair.transpose(-1,-2)).abs().max()),
            )
            max_diag=max(
                max_diag,
                float(torch.diagonal(pair,dim1=-2,dim2=-1).abs().max()),
            )
            max_mass=max(
                max_mass,
                float((torch.softmax(agg,-1).sum(-1)-1.0).abs().max()),
            )

        zr=ref.context_feature(rep)
        zt=trt.context_feature(rep)
        context_difference=max(
            context_difference,
            float((zr-zt).abs().max()),
        )

        ablated=rep.clone()
        ablated[...,256:]=0.0
        ablated_context_error=max(
            ablated_context_error,
            float((ref.context_feature(ablated)-trt.context_feature(ablated)).abs().max()),
        )

        perm=torch.randperm(k,generator=g)
        z=trt.context_feature(rep)
        zp=trt.context_feature(rep[:,perm])
        max_context_permutation_error=max(
            max_context_permutation_error,
            float((z-zp).abs().max()),
        )

        probe_ref=ContextModulatedPairwiseHead(use_joint_context=False)
        probe_trt=ContextModulatedPairwiseHead(use_joint_context=True)
        with torch.no_grad():
            pattern=torch.linspace(
                -0.08,0.08,
                64*S68_CONTEXT_DIMENSION,
                dtype=probe_ref.G.dtype,
            ).reshape(64,S68_CONTEXT_DIMENSION)
            probe_ref.G.copy_(pattern)
            probe_trt.G.copy_(pattern)

        for head in (probe_ref,probe_trt):
            pair=head.pairwise_logits(rep)
            pairp=head.pairwise_logits(rep[:,perm])
            restored=pairp[:,torch.argsort(perm)][:,:,torch.argsort(perm)]
            max_perm=max(
                max_perm,
                float((pair-restored).abs().max()),
            )

    if context_difference<=1e-7:
        raise RuntimeError("S68-A0 full context did not differ from identity-only control")
    if ablated_context_error!=0.0:
        raise RuntimeError("S68-A0 context ablation identity failed")

    rep=torch.randn(6,4,512,generator=g,requires_grad=True)
    gold=torch.tensor([0,1,2,3,0,1],dtype=torch.long)
    grad_G={}
    grad_A={}
    grad_u={}
    for label,head in (
        ("reference",ContextModulatedPairwiseHead(use_joint_context=False)),
        ("treatment",ContextModulatedPairwiseHead(use_joint_context=True)),
    ):
        loss,_=context_modulated_pairwise_loss(head,rep,gold)
        grads=torch.autograd.grad(
            loss,(head.A,head.u,head.G,rep),allow_unused=True
        )
        grad_A[label]=0.0 if grads[0] is None else float(grads[0].abs().sum())
        grad_u[label]=0.0 if grads[1] is None else float(grads[1].abs().sum())
        grad_G[label]=0.0 if grads[2] is None else float(grads[2].abs().sum())
        if min(grad_A[label],grad_u[label],grad_G[label])<=0.0:
            raise RuntimeError(f"S68-A0 {label} gradient vanished")
        if grads[3] is not None:
            raise RuntimeError(f"S68-A0 {label} gradient leaked into representation")

    G_gradient_difference=float(
        (
            torch.autograd.grad(
                context_modulated_pairwise_loss(
                    ContextModulatedPairwiseHead(use_joint_context=False),
                    rep.detach(),
                    gold,
                )[0],
                ContextModulatedPairwiseHead(use_joint_context=False).G,
                allow_unused=True,
            )[0]
            if False else torch.tensor(0.0)
        )
    )
    # Compare gradients using fresh matched heads so the only difference is
    # the context-source mask.
    hr=ContextModulatedPairwiseHead(use_joint_context=False)
    ht=ContextModulatedPairwiseHead(use_joint_context=True)
    lr,_=context_modulated_pairwise_loss(hr,rep.detach(),gold)
    lt,_=context_modulated_pairwise_loss(ht,rep.detach(),gold)
    gr=torch.autograd.grad(lr,hr.G)[0]
    gt=torch.autograd.grad(lt,ht.G)[0]
    G_gradient_difference=float((gr-gt).abs().sum())
    if G_gradient_difference<=1e-7:
        raise RuntimeError("S68-A0 reference/treatment G gradients did not separate")

    return {
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "initial_pair_vs_s59_max_abs_error":max_pair_identity,
        "initial_aggregate_vs_s59_max_abs_error":max_agg_identity,
        "initial_choice_vs_s59_exact":True,
        "initial_gamma_max_abs_error_from_one":max_gamma_identity,
        "option_permutation_pair_max_abs_error":max_perm,
        "context_permutation_max_abs_error":max_context_permutation_error,
        "antisymmetry_max_abs_error":max_antisym,
        "diagonal_max_abs_error":max_diag,
        "max_probability_mass_error":max_mass,
        "reference_treatment_context_max_abs_difference":context_difference,
        "joint_context_ablation_max_abs_error":ablated_context_error,
        "reference_gradient_A_l1":grad_A["reference"],
        "reference_gradient_u_l1":grad_u["reference"],
        "reference_gradient_G_l1":grad_G["reference"],
        "treatment_gradient_A_l1":grad_A["treatment"],
        "treatment_gradient_u_l1":grad_u["treatment"],
        "treatment_gradient_G_l1":grad_G["treatment"],
        "reference_treatment_G_gradient_l1_difference":G_gradient_difference,
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
        raise RuntimeError("S68-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S68-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S68-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S68-A0 checkpoint file changed")

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
        raise RuntimeError("S68-A0 loaded native runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S68-A0 native remained trainable")

    # A0 uses only already-exposed S67 TRAIN rows. No fresh S68 evidence.
    rows=list(generate_s67_cases("train"))[:16]
    cache,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    correction=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    correction_before={
        k:v.detach().clone()
        for k,v in correction.correction_state_dict().items()
    }
    if correction.correction_parameter_count!=114688:
        raise RuntimeError("S68-A0 correction capacity changed")

    reference=ContextModulatedPairwiseHead(use_joint_context=False)
    treatment=ContextModulatedPairwiseHead(use_joint_context=True)
    baseline=ExplicitPairwiseDecisionHead()

    if reference.parameter_count!=S68_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S68-A0 reference capacity changed")
    if treatment.parameter_count!=S68_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S68-A0 treatment capacity changed")
    if reference.parameter_count-treatment.parameter_count!=0:
        raise RuntimeError("S68-A0 treatment parameter advantage changed")
    if not _param_state_equal(reference,treatment):
        raise RuntimeError("S68-A0 arm parameter initialization changed")
    if not torch.equal(reference.A,baseline.A) or not torch.equal(reference.u,baseline.u):
        raise RuntimeError("S68-A0 base S59 initialization changed")
    if not torch.equal(reference.context_projection,treatment.context_projection):
        raise RuntimeError("S68-A0 context projection changed across arms")

    canonical,paraphrase,_n=cache[0]
    rep_c=_representation(correction,canonical)
    rep_p=_representation(correction,paraphrase)
    if rep_c.requires_grad or rep_p.requires_grad:
        raise RuntimeError("S68-A0 representation retained gradients")

    initial_identity_error=max(
        float((reference.pairwise_logits(rep_c)-baseline.pairwise_logits(rep_c)).abs().max()),
        float((treatment.pairwise_logits(rep_c)-baseline.pairwise_logits(rep_c)).abs().max()),
        float((reference.pairwise_logits(rep_p)-baseline.pairwise_logits(rep_p)).abs().max()),
        float((treatment.pairwise_logits(rep_p)-baseline.pairwise_logits(rep_p)).abs().max()),
    )
    if initial_identity_error!=0.0:
        raise RuntimeError("S68-A0 initial S59 functional identity changed")

    real_grad={}
    for label,head in (("reference",reference),("treatment",treatment)):
        lc,dc=context_modulated_pairwise_loss(head,rep_c,canonical.gold)
        lp,dp=context_modulated_pairwise_loss(head,rep_p,paraphrase.gold)
        loss=0.5*(lc+lp)
        grads=torch.autograd.grad(
            loss,(head.A,head.u,head.G),allow_unused=True
        )
        vals=[
            0.0 if g is None else float(g.detach().abs().sum())
            for g in grads
        ]
        if min(vals)<=0.0:
            raise RuntimeError(f"S68-A0 real-cache {label} gradient vanished")
        real_grad[label]={
            "A":vals[0],
            "u":vals[1],
            "G":vals[2],
            "gold_pair_accuracy_canonical":float(dc["gold_pair_accuracy"]),
            "gold_pair_accuracy_paraphrase":float(dp["gold_pair_accuracy"]),
        }

    correction_after=correction.correction_state_dict()
    if correction_before.keys()!=correction_after.keys() or not all(
        torch.equal(correction_before[k],correction_after[k])
        for k in correction_before
    ):
        raise RuntimeError("S68-A0 pairwise court altered correction state")
    if any(p.grad is not None for p in correction.correction_parameters()):
        raise RuntimeError("S68-A0 correction received pairwise gradient")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0:
        raise RuntimeError("S68-A0 cache gained gradients")

    mechanics=_synthetic_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    ref_state=reference.state_dict_exact()
    trt_state=treatment.state_dict_exact()
    checkpoint=out/"matched-pairwise-heads.pt"
    torch.save({
        "schema_version":"hira-v1-s68-matched-pairwise-heads-v1",
        "seed":SEED,
        "reference_use_joint_context":False,
        "treatment_use_joint_context":True,
        "pairwise_parameter_count_per_arm":S68_PAIRWISE_PARAMETER_COUNT,
        "context_projection_seed":S68_CONTEXT_PROJECTION_SEED,
        "modulation_scale":S68_MODULATION_SCALE,
        "reference_state_dict":ref_state,
        "treatment_state_dict":trt_state,
    },checkpoint)

    rr=ContextModulatedPairwiseHead(use_joint_context=False)
    tt=ContextModulatedPairwiseHead(use_joint_context=True)
    rr.load_state_dict_exact(ref_state,freeze=True)
    tt.load_state_dict_exact(trt_state,freeze=True)
    replay_error=max(
        float((reference.pairwise_logits(rep_c)-rr.pairwise_logits(rep_c)).abs().max()),
        float((treatment.pairwise_logits(rep_c)-tt.pairwise_logits(rep_c)).abs().max()),
        float((reference.pairwise_logits(rep_p)-rr.pairwise_logits(rep_p)).abs().max()),
        float((treatment.pairwise_logits(rep_p)-tt.pairwise_logits(rep_p)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S68_A0_CONTEXT_MODULATED_PAIRWISE_ONLY",
        "seed":SEED,
        "parent_s67":{
            "merged_main":"942871bb44bbef3e28e74698f161b02c6796cdcb",
            "scientific_run":37336779415,
            "verdict":"CASE_B",
            "actions_artifact_available":False,
            "raw_receipt_frozen_in_repo":True,
        },
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "parent_exposed_semantic_case_count":len(cache),
        "cache_digest":cache_digest,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "correction_trainable_parameter_count":114688,
        "reference_pairwise_parameter_count":reference.parameter_count,
        "treatment_pairwise_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":0,
        "pairwise_trainable_tensor_names":sorted(dict(reference.named_parameters())),
        "base_s59_parameter_count":32832,
        "modulation_parameter_count":S68_MODULATION_PARAMETER_COUNT,
        "context_dimension":S68_CONTEXT_DIMENSION,
        "context_projection_seed":S68_CONTEXT_PROJECTION_SEED,
        "context_projection_trainable":False,
        "modulation_scale":S68_MODULATION_SCALE,
        "arm_parameter_initialization_bit_identical":True,
        "base_A_u_exact_s59_initialization":True,
        "G_initial_exact_zero":True,
        "initial_real_cache_s59_identity_max_abs_error":initial_identity_error,
        "reference_real_cache_gradient_A_l1":real_grad["reference"]["A"],
        "reference_real_cache_gradient_u_l1":real_grad["reference"]["u"],
        "reference_real_cache_gradient_G_l1":real_grad["reference"]["G"],
        "treatment_real_cache_gradient_A_l1":real_grad["treatment"]["A"],
        "treatment_real_cache_gradient_u_l1":real_grad["treatment"]["u"],
        "treatment_real_cache_gradient_G_l1":real_grad["treatment"]["G"],
        "reference_real_cache_gold_pair_accuracy_canonical":
            real_grad["reference"]["gold_pair_accuracy_canonical"],
        "reference_real_cache_gold_pair_accuracy_paraphrase":
            real_grad["reference"]["gold_pair_accuracy_paraphrase"],
        "treatment_real_cache_gold_pair_accuracy_canonical":
            real_grad["treatment"]["gold_pair_accuracy_canonical"],
        "treatment_real_cache_gold_pair_accuracy_paraphrase":
            real_grad["treatment"]["gold_pair_accuracy_paraphrase"],
        "correction_received_pairwise_gradient":False,
        "correction_state_unchanged_after_pairwise_backward":True,
        "native_trainable_parameter_count":0,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "raw_native_logit_input_to_head":False,
        "raw_fused_logit_input_to_head":False,
        "raw_corrected_logit_input_to_head":False,
        "one_encoder_state_once":True,
        "matched_head_checkpoint_replay_max_abs_error":replay_error,
        **mechanics,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }

    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S68_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
