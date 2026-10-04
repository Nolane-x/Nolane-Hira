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
from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_token_query_option_late_interaction import (
    TokenLateInteractionQueryFreePrivateCorrectionFork,
)
from hira_v1_s50_train_dev import _materialize_cache


SCHEMA_VERSION="hira-v1-s53-a0-token-query-option-late-interaction-v1"
OUTCOME="HIRA_V1_S53_A0_TOKEN_QUERY_OPTION_LATE_INTERACTION_READY"
SEED=74_001


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
        return f"S53-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"S53 token audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."

    @property
    def qa1(self): return f"For S53-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S53-A0 entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For S53-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S53-A0 entry is tagged {self.field_b} for {self.case_id}?"

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
                aliases=(f"{value} is the S53-A0 {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("LI11","quantum torque lens","rotor","diamond paddle","torque floor","3 zNm","steel vane","12 zNm",16101),
        Case("LI22","spin phase radar","film","PtCo interface","phase jitter","0.08 mrad","bulk Fe","0.32 mrad",16102),
        Case("LI33","phonon recoil compass","guide","soft SiN ridge","recoil blur","0.09 um-1","Cu strip","0.36 um-1",16103),
        Case("LI44","Rydberg heat scope","ensemble","dressed Cs 61D","thermal floor","7 nK","hot vapor","28 nK",16104),
        Case("LI55","moire phase camera","stack","aligned MoTe2-WSe2","phase blur","0.05 rad","bulk Si","0.20 rad",16105),
        Case("LI66","neutron curvature ruler","target","perfect Si","curvature floor","4 nrad/mm","polymer tile","16 nrad/mm",16106),
        Case("LI77","topological charge map","channel","valley edge","charge leak","-51 dB","bulk bar","-23 dB",16107),
        Case("LI88","molecular recoil clock","beam","state-selected OCS","recoil spread","4 mm/s","thermal SO2","16 mm/s",16108),
        Case("LJ11","vacuum phase compass","surface","template Au","phase floor","0.13 mrad","rough steel","0.52 mrad",16109),
        Case("LJ22","atomic spin camera","species","lattice Yb","spin noise","2 nrad","thermal K","8 nrad",16110),
        Case("LJ33","magnon pressure lens","guide","YIG strip","pressure floor","4 nPa","Ni strip","16 nPa",16111),
        Case("LJ44","optical recoil bridge","interface","SiV cavity","recoil loss","0.04 rad","free-space link","0.16 rad",16112),
        Case("LJ55","superfluid heat ruler","fluid","He4 film","thermal noise","7 nK","oil film","28 nK",16113),
        Case("LJ66","Casimir phase camera","surface","Au-Si pair","phase drift","0.17 mrad","steel pair","0.68 mrad",16114),
        Case("LJ77","quantum gradient compass","sensor","nanoSQUID","gradient floor","0.06 pT/mm","Hall bar","0.24 pT/mm",16115),
        Case("LJ88","ferroelectric recoil radar","crystal","strained BaTiO3","recoil jitter","0.8 nm","ceramic slab","3.2 nm",16116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s53-{case.case_id}",
            split="train",
            domain="s53_a0_only",
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


def _synthetic_k_court():
    op=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)
    g=torch.Generator().manual_seed(74520)
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
        logits,_identity,context,weights=op.correction_logits_from_state_option(
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
        if tuple(logits.shape)!=(b,k):
            raise RuntimeError("S53-A0 arbitrary-K shape changed")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("S53-A0 arbitrary-K non-finite")
        mass=float((torch.softmax(logits,dim=-1).sum(-1)-1.0).abs().max())
        max_mass=max(max_mass,mass)
        max_context_norm_error=max(
            max_context_norm_error,
            float((context.norm(dim=-1)-1.0).abs().max()),
        )
        if float((weights.sum(-1)-1.0).abs().max())>1e-6:
            raise RuntimeError("S53-A0 token attention mass changed")
    if max_mass>1e-6:
        raise RuntimeError("S53-A0 probability mass failed")
    if max_context_norm_error>1e-5:
        raise RuntimeError("S53-A0 context normalization failed")
    return max_mass,max_context_norm_error


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
        raise RuntimeError("S53-A0 parent native runtime changed")
    if frozen.get("native_tensor_digest")!=expected_runtime:
        raise RuntimeError("S53-A0 parent native digest changed")
    if frozen.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S53-A0 parent checkpoint SHA changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S53-A0 native checkpoint file SHA changed")

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
        raise RuntimeError("S53-A0 loaded native runtime hash changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S53-A0 native runtime remained trainable")

    rows=_rows()
    if len(rows)!=16:
        raise RuntimeError("S53-A0 case count changed")
    cache_pairs,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    reference=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    treatment=TokenLateInteractionQueryFreePrivateCorrectionFork(train_correction=True)

    if reference.correction_parameter_count!=114688:
        raise RuntimeError("S53-A0 reference correction capacity changed")
    if treatment.correction_parameter_count!=114688:
        raise RuntimeError("S53-A0 treatment correction capacity changed")
    if treatment.late_interaction_parameter_count!=0:
        raise RuntimeError("S53-A0 late interaction gained parameters")
    if reference.identity_parameter_count!=0 or treatment.identity_parameter_count!=0:
        raise RuntimeError("S53-A0 identity gained parameters")
    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S53-A0 correction initialization differs")

    canonical=cache_pairs[0][0]
    ref_logits,_=reference.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
    )
    trt_logits,identity,context,weights=treatment.correction_logits_from_state_option(
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

    if not bool(torch.isfinite(ref_logits).all()) or not bool(torch.isfinite(trt_logits).all()):
        raise RuntimeError("S53-A0 actual cache logits non-finite")
    context_norm_error=float((context.norm(dim=-1)-1.0).abs().max())
    attention_mass_error=float((weights.sum(-1)-1.0).abs().max())
    option_context_diversity=float((context[:,0]-context[:,1]).abs().max())
    if context_norm_error>1e-5 or attention_mass_error>1e-6:
        raise RuntimeError("S53-A0 actual cache context invariant failed")
    if option_context_diversity<=1e-7:
        raise RuntimeError("S53-A0 option-conditioned contexts collapsed")

    # Runtime proof that the treatment path never calls inherited pooled query_summary.
    def _fail(*_args,**_kwargs):
        raise RuntimeError("S53 pooled query bypass")
    treatment.query_summary=types.MethodType(_fail,treatment)
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
        raise RuntimeError("S53-A0 treatment path changed under pooled-query disable")

    # Informative-token perturbation must flow through the documented token context.
    q2=canonical.question_tokens.clone()
    first_active=int(torch.nonzero(canonical.question_mask[0],as_tuple=False)[0].item())
    q2[0,first_active]+=3.0
    pert_logits,_pi,pert_context,_pw=treatment.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=q2,
        question_mask=canonical.question_mask,
        return_context=True,
    )
    context_sensitivity=float((pert_context-context).abs().max())
    logit_sensitivity=float((pert_logits-trt_logits).abs().max())
    if context_sensitivity<=1e-7:
        raise RuntimeError("S53-A0 informative-token context sensitivity vanished")

    cache_inference_count=cache_requires_grad_count=0
    for c,p,_n in cache_pairs:
        for evidence in (c,p):
            for tensor in evidence.tensors():
                cache_inference_count+=int(torch.is_inference(tensor))
                cache_requires_grad_count+=int(tensor.requires_grad)
    if cache_inference_count!=0 or cache_requires_grad_count!=0:
        raise RuntimeError("S53-A0 cache ownership changed")

    max_mass,max_norm=_synthetic_k_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S53_A0_TOKEN_QUERY_OPTION_LATE_INTERACTION_ONLY",
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
        "late_interaction_parameter_count":treatment.late_interaction_parameter_count,
        "initialization_bit_identical":True,
        "actual_cache_context_norm_max_error":context_norm_error,
        "actual_cache_attention_mass_max_error":attention_mass_error,
        "actual_cache_option_context_diversity_max_abs":option_context_diversity,
        "pooled_query_bypass_absent":True,
        "informative_token_context_sensitivity_max_abs":context_sensitivity,
        "informative_token_logit_sensitivity_max_abs":logit_sensitivity,
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
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S53_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
