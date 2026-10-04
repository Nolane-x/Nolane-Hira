from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import types

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_token_query_option_late_interaction import (
    TokenLateInteractionQueryFreePrivateCorrectionFork,
)
from nmd.v1_joint_state_query_option_interaction import (
    JointStateQueryOptionPrivateCorrectionFork,
)
from hira_v1_s50_train_dev import _materialize_cache


SCHEMA_VERSION="hira-v1-s54-a0-joint-state-query-option-interaction-v1"
OUTCOME="HIRA_V1_S54_A0_JOINT_STATE_QUERY_OPTION_INTERACTION_READY"
SEED=75_001


@dataclass(frozen=True)
class Case:
    case_id:str
    noun:str
    field_a:str
    first:str
    field_b:str
    second:str
    wrong_a:str
    wrong_b:str
    seed:int

    @property
    def state_a(self):
        return f"S54-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"S54 triadic audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."

    @property
    def qa1(self): return f"For S54-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S54-A0 entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For S54-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S54-A0 entry is tagged {self.field_b} for {self.case_id}?"

    def option_pack(self):
        rows=[
            ("a",self.field_a,self.first),
            ("b",self.field_b,self.second),
            ("x",self.field_a,self.wrong_a),
            ("y",self.field_b,self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        options=tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"for this {self.noun}, {field} is {value}",
                aliases=(f"{value} is the S54-A0 {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("TJ11","axion torque camera","sensor","niobium ring","torque floor","2 zNm","steel disk","8 zNm",17101),
        Case("TJ22","spin recoil radar","film","YIG bilayer","recoil blur","0.07 um","bulk Ni","0.28 um",17102),
        Case("TJ33","phonon phase compass","guide","soft SiN ridge","phase floor","0.06 rad","Cu strip","0.24 rad",17103),
        Case("TJ44","Rydberg gradient scope","ensemble","dressed Rb 58D","gradient floor","5 nV/cm","hot vapor","20 nV/cm",17104),
        Case("TJ55","moire heat lens","stack","aligned WSe2-MoSe2","thermal floor","6 nK","bulk Si","24 nK",17105),
        Case("TJ66","neutron phase ruler","target","perfect Ge","phase drift","0.12 mrad","polymer tile","0.48 mrad",17106),
        Case("TJ77","topological pressure map","channel","Chern edge","pressure leak","-49 dB","bulk bar","-21 dB",17107),
        Case("TJ88","molecular torque clock","beam","state-selected YbF","torque spread","3 zNm","thermal SO2","12 zNm",17108),
        Case("TK11","vacuum recoil compass","surface","template Ag","recoil floor","0.11 eV","rough steel","0.44 eV",17109),
        Case("TK22","atomic curvature camera","species","lattice Sr","curvature noise","3 nrad/mm","thermal K","12 nrad/mm",17110),
        Case("TK33","magnon heat lens","guide","low-loss YIG","thermal floor","3 nK","Ni strip","12 nK",17111),
        Case("TK44","optical torque bridge","interface","SiV cavity","torque loss","0.03 rad","free-space link","0.12 rad",17112),
        Case("TK55","superfluid recoil ruler","fluid","He3-B film","recoil noise","5 mm/s","oil film","20 mm/s",17113),
        Case("TK66","Casimir gradient camera","surface","Au-graphene pair","gradient drift","0.14 pN/mm","steel pair","0.56 pN/mm",17114),
        Case("TK77","quantum pressure compass","sensor","nanoSQUID","pressure floor","0.05 nPa","Hall bar","0.20 nPa",17115),
        Case("TK88","ferroelectric torque radar","crystal","strained KTaO3","torque jitter","0.7 zNm","ceramic slab","2.8 zNm",17116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s54-{case.case_id}",
            split="train",
            domain="s54_a0_only",
            language="en",
            state_a=case.state_a,
            state_b=case.state_b,
            question_a1=case.qa1,
            question_a2=case.qa2,
            question_b1=case.qb1,
            question_b2=case.qb2,
            option_texts=tuple(x.criterion_text for x in options),
            option_aliases=tuple(x.aliases[0] for x in options),
            option_ids=tuple(x.option_id for x in options),
            gold_a=ga,
            gold_b=gb,
        ))
    return out


def _activate(op):
    with torch.no_grad():
        op.adapter_b.normal_(
            generator=torch.Generator().manual_seed(75501),
            std=0.01,
        )
        op.bilinear_weight.normal_(
            generator=torch.Generator().manual_seed(75502),
            std=0.01,
        )


def _synthetic_k_court():
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    _activate(op)
    g=torch.Generator().manual_seed(75520)
    max_mass=0.0
    max_context_norm_error=0.0
    for k in (3,7,255):
        b=2
        state=torch.randn(b,5,256,generator=g)
        state_mask=torch.ones(b,5,dtype=torch.bool)
        option=torch.randn(b,k,2,4,256,generator=g)
        token_mask=torch.ones(b,k,2,4,dtype=torch.bool)
        view_mask=torch.ones(b,k,2,dtype=torch.bool)
        native=torch.randn(b,k,generator=g)
        question=torch.randn(b,5,256,generator=g)
        question_mask=torch.ones(b,5,dtype=torch.bool)
        logits,_identity,context,weights,state_support,option_support=(
            op.correction_logits_from_state_option(
                native_logits=native,
                state_tokens=state,
                state_mask=state_mask,
                option_view_tokens=option,
                option_view_token_mask=token_mask,
                option_view_mask=view_mask,
                question_tokens=question,
                question_mask=question_mask,
                return_context=True,
            )
        )
        if tuple(logits.shape)!=(b,k):
            raise RuntimeError("S54-A0 arbitrary-K shape changed")
        if not all(bool(torch.isfinite(x).all()) for x in (
            logits,context,weights,state_support,option_support
        )):
            raise RuntimeError("S54-A0 arbitrary-K non-finite")
        mass=float((torch.softmax(logits,dim=-1).sum(-1)-1.0).abs().max())
        max_mass=max(max_mass,mass)
        max_context_norm_error=max(
            max_context_norm_error,
            float((context.norm(dim=-1)-1.0).abs().max()),
        )
        if float((weights.sum(-1)-1.0).abs().max())>1e-6:
            raise RuntimeError("S54-A0 joint attention mass changed")
    if max_mass>1e-6:
        raise RuntimeError("S54-A0 probability mass failed")
    if max_context_norm_error>1e-6:
        raise RuntimeError("S54-A0 context normalization failed")
    return max_mass,max_context_norm_error


def _activated_path_sensitivity(canonical):
    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    _activate(op)

    base,_,ctx,*_=op.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
        return_context=True,
    )

    q_index=int(torch.nonzero(canonical.question_mask[0],as_tuple=False)[0].item())
    s_index=int(torch.nonzero(canonical.state_mask[0],as_tuple=False)[0].item())
    option_mask=(
        canonical.option_view_token_mask
        & canonical.option_view_mask.unsqueeze(-1)
    )
    active=torch.nonzero(option_mask[0,0],as_tuple=False)[0]
    v_index=int(active[0].item())
    t_index=int(active[1].item())

    state=canonical.state_tokens.clone()
    state[0,s_index]+=8.0*canonical.question_tokens[0,q_index]
    sl,_,sc,*_=op.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=state,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
        return_context=True,
    )

    query=canonical.question_tokens.clone()
    query[0,q_index]+=4.0
    ql,_,qc,*_=op.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=query,
        question_mask=canonical.question_mask,
        return_context=True,
    )

    option=canonical.option_view_tokens.clone()
    option[0,0,v_index,t_index]+=8.0*canonical.question_tokens[0,q_index]
    ol,_,oc,*_=op.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=option,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
        return_context=True,
    )

    out={
        "state_context_sensitivity":float((sc-ctx).abs().max()),
        "state_logit_sensitivity":float((sl-base).abs().max()),
        "query_context_sensitivity":float((qc-ctx).abs().max()),
        "query_logit_sensitivity":float((ql-base).abs().max()),
        "option_context_sensitivity":float((oc-ctx).abs().max()),
        "option_logit_sensitivity":float((ol-base).abs().max()),
    }
    if min(out.values())<=1e-8:
        raise RuntimeError(f"S54-A0 activated triadic path sensitivity vanished: {out}")
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    frozen=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if frozen.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S54-A0 parent native runtime changed")
    if frozen.get("native_tensor_digest")!=expected_runtime:
        raise RuntimeError("S54-A0 parent native digest changed")
    if frozen.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S54-A0 parent checkpoint SHA changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S54-A0 native checkpoint file SHA changed")

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
        raise RuntimeError("S54-A0 loaded native runtime hash changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S54-A0 native runtime remained trainable")

    rows=_rows()
    if len(rows)!=16:
        raise RuntimeError("S54-A0 case count changed")
    cache_pairs,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    reference=TokenLateInteractionQueryFreePrivateCorrectionFork(
        train_correction=True
    )
    treatment=JointStateQueryOptionPrivateCorrectionFork(
        train_correction=True
    )

    if reference.correction_parameter_count!=114688:
        raise RuntimeError("S54-A0 reference correction capacity changed")
    if treatment.correction_parameter_count!=114688:
        raise RuntimeError("S54-A0 treatment correction capacity changed")
    if treatment.joint_interaction_parameter_count!=0:
        raise RuntimeError("S54-A0 joint interaction gained parameters")
    if reference.identity_parameter_count!=0 or treatment.identity_parameter_count!=0:
        raise RuntimeError("S54-A0 identity gained parameters")

    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S54-A0 correction initialization differs")

    canonical=cache_pairs[0][0]
    ref_logits,_ri,ref_context,ref_weights=reference.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
        return_context=True,
    )
    trt_logits,_ti,trt_context,trt_weights,state_support,option_support=(
        treatment.correction_logits_from_state_option(
            native_logits=canonical.native_logits,
            state_tokens=canonical.state_tokens,
            state_mask=canonical.state_mask,
            option_view_tokens=canonical.option_view_tokens,
            option_view_token_mask=canonical.option_view_token_mask,
            option_view_mask=canonical.option_view_mask,
            question_tokens=canonical.question_tokens,
            question_mask=canonical.question_mask,
            return_context=True,
        )
    )

    if not all(bool(torch.isfinite(x).all()) for x in (
        ref_logits,trt_logits,trt_context,trt_weights,state_support,option_support
    )):
        raise RuntimeError("S54-A0 actual cache produced non-finite values")

    context_norm_error=float((trt_context.norm(dim=-1)-1.0).abs().max())
    attention_mass_error=float((trt_weights.sum(-1)-1.0).abs().max())
    context_delta_vs_s53=float((trt_context-ref_context).abs().max())
    if context_norm_error>1e-6 or attention_mass_error>1e-6:
        raise RuntimeError("S54-A0 actual cache context invariant failed")
    if context_delta_vs_s53<=1e-8:
        raise RuntimeError("S54-A0 joint context collapsed to S53 context")

    # Runtime proof that treatment never uses the inherited S53 query↔option-only path.
    def _fail(*_args,**_kwargs):
        raise RuntimeError("S54 S53-query-option bypass")
    treatment.option_conditioned_query_context=types.MethodType(_fail,treatment)
    bypass_logits,_=treatment.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
    )
    if not torch.equal(bypass_logits,trt_logits):
        raise RuntimeError("S54-A0 treatment path changed under S53 bypass disable")

    cache_inference_count=cache_requires_grad_count=0
    for c,p,_n in cache_pairs:
        for evidence in (c,p):
            for tensor in evidence.tensors():
                cache_inference_count+=int(torch.is_inference(tensor))
                cache_requires_grad_count+=int(tensor.requires_grad)
    if cache_inference_count!=0 or cache_requires_grad_count!=0:
        raise RuntimeError("S54-A0 cache ownership changed")

    sensitivity=_activated_path_sensitivity(canonical)
    max_mass,max_norm=_synthetic_k_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S54_A0_JOINT_STATE_QUERY_OPTION_INTERACTION_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "native_trainable_parameter_count":0,
        "semantic_case_count":len(rows),
        "cache_digest":cache_digest,
        "cache_inference_tensor_count":cache_inference_count,
        "cache_requires_grad_tensor_count":cache_requires_grad_count,
        "reference_correction_parameter_count":reference.correction_parameter_count,
        "treatment_correction_parameter_count":treatment.correction_parameter_count,
        "reference_private_trainable_parameter_count":reference.correction_parameter_count,
        "treatment_private_trainable_parameter_count":treatment.correction_parameter_count,
        "identity_parameter_count":treatment.identity_parameter_count,
        "joint_interaction_parameter_count":treatment.joint_interaction_parameter_count,
        "initialization_bit_identical":True,
        "actual_cache_context_norm_max_error":context_norm_error,
        "actual_cache_attention_mass_max_error":attention_mass_error,
        "actual_cache_context_delta_vs_s53_max_abs":context_delta_vs_s53,
        "s53_query_option_bypass_absent":True,
        "mean_state_support":float(state_support.mean()),
        "mean_option_support":float(option_support.mean()),
        **sensitivity,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "max_probability_mass_error":max_mass,
        "max_context_norm_error":max_norm,
        "second_encoder_pass":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S54_A0_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
