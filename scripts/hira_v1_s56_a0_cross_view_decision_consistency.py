from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_cross_view_decision_consistency import (
    decision_discrimination_diagnostics,
    pairwise_ordering_consistency,
    symmetric_js_from_standardized_logits,
    weighted_cross_view_decision_auxiliary,
)
from hira_v1_s50_train_dev import _materialize_cache, _branch_logits


SCHEMA_VERSION="hira-v1-s56-a0-cross-view-decision-consistency-v1"
OUTCOME="HIRA_V1_S56_A0_CROSS_VIEW_DECISION_CONSISTENCY_READY"
SEED=77_001


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
        return f"S56-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."
    @property
    def state_b(self):
        return f"S56 consistency audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."
    @property
    def qa1(self): return f"For S56-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S56-A0 entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For S56-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S56-A0 entry is tagged {self.field_b} for {self.case_id}?"

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
                aliases=(f"{value} is the S56-A0 {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("DC11","axion phase meter","sensor","Nb cavity","phase floor","0.07 rad","steel cell","0.28 rad",21101),
        Case("DC22","spin pressure lens","film","PtCo layer","pressure floor","2 nPa","bulk Ni","8 nPa",21102),
        Case("DC33","phonon torque map","guide","soft SiN beam","torque floor","0.04 zNm","Cu strip","0.16 zNm",21103),
        Case("DC44","Rydberg recoil scope","ensemble","dressed Cs 64D","recoil spread","3 mm/s","hot vapor","12 mm/s",21104),
        Case("DC55","moire curvature clock","stack","aligned WSe2-MoSe2","curvature floor","0.03 mm-1","bulk Si","0.12 mm-1",21105),
        Case("DC66","neutron pressure camera","target","perfect Si","pressure blur","0.11 uPa","polymer tile","0.44 uPa",21106),
        Case("DC77","topological gradient compass","channel","valley edge","gradient leak","-46 dB","bulk bar","-18 dB",21107),
        Case("DC88","molecular phase radar","beam","selected ThO","phase drift","0.05 rad","thermal SO2","0.20 rad",21108),
        Case("DD11","vacuum recoil bridge","surface","template Au","recoil floor","0.08 pm","rough steel","0.32 pm",21109),
        Case("DD22","atomic torque lens","species","clocked Yb","torque floor","0.03 zNm","thermal K","0.12 zNm",21110),
        Case("DD33","magnon curvature scope","guide","low-loss YIG","curvature floor","0.02 mm-1","Ni strip","0.08 mm-1",21111),
        Case("DD44","optical pressure ruler","interface","SiV cavity","pressure loss","1.4 nPa","free-space link","5.6 nPa",21112),
        Case("DD55","superfluid recoil map","fluid","He3-B film","recoil noise","0.06 um","oil film","0.24 um",21113),
        Case("DD66","Casimir gradient camera","surface","Au-Si pair","gradient drift","0.10 pN/mm","steel pair","0.40 pN/mm",21114),
        Case("DD77","quantum pressure compass","sensor","NV resonator","pressure floor","0.05 nPa","Hall bar","0.20 nPa",21115),
        Case("DD88","ferroelectric torque radar","crystal","strained BaTiO3","torque jitter","0.5 zNm","ceramic slab","2.0 zNm",21116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s56-{case.case_id}",
            split="train",
            domain="s56_a0_only",
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


def _arms():
    return (
        JointStateQueryOptionPrivateCorrectionFork(train_correction=True),
        JointStateQueryOptionPrivateCorrectionFork(train_correction=True),
    )


def _synthetic_k_court():
    g=torch.Generator().manual_seed(77500)
    max_mass=0.0
    for k in (3,7,255):
        b=2
        op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
        with torch.no_grad():
            op.adapter_b.normal_(generator=torch.Generator().manual_seed(77510+k),std=0.01)
            op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(77520+k),std=0.01)
        data=dict(
            native_logits=torch.randn(b,k,generator=g),
            state_tokens=torch.randn(b,5,256,generator=g),
            state_mask=torch.ones(b,5,dtype=torch.bool),
            option_view_tokens=torch.randn(b,k,2,4,256,generator=g),
            option_view_token_mask=torch.ones(b,k,2,4,dtype=torch.bool),
            option_view_mask=torch.ones(b,k,2,dtype=torch.bool),
            question_tokens=torch.randn(b,5,256,generator=g),
            question_mask=torch.ones(b,5,dtype=torch.bool),
        )
        logits,_=op.correction_logits_from_state_option(**data)
        if tuple(logits.shape)!=(b,k) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("S56-A0 arbitrary-K failed")
        mass=float((torch.softmax(logits,dim=-1).sum(-1)-1.0).abs().max())
        max_mass=max(max_mass,mass)
    if max_mass>1e-6:
        raise RuntimeError("S56-A0 probability mass changed")
    return max_mass


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
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S56-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S56-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S56-A0 checkpoint file changed")

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
        raise RuntimeError("S56-A0 loaded runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S56-A0 native runtime remained trainable")

    rows=_rows()
    cache_pairs,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    reference,treatment=_arms()
    if reference.correction_parameter_count!=114688 or treatment.correction_parameter_count!=114688:
        raise RuntimeError("S56-A0 private capacity changed")
    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S56-A0 initialization differs")
    if reference.identity_parameter_count!=0 or treatment.identity_parameter_count!=0:
        raise RuntimeError("S56-A0 identity gained parameters")

    # Pure loss mechanics.
    identical=torch.tensor([[3.0,1.0,-1.0,0.5]],requires_grad=True)
    identical_js=float(symmetric_js_from_standardized_logits(identical,identical))
    disagree_a=torch.tensor([[3.0,1.0,0.0,-2.0]],requires_grad=True)
    disagree_b=torch.tensor([[-2.0,0.0,1.0,3.0]],requires_grad=True)
    disagreement_js=float(symmetric_js_from_standardized_logits(disagree_a,disagree_b))
    offset_error=abs(
        float(symmetric_js_from_standardized_logits(disagree_a+9.0,disagree_b-13.0))
        -disagreement_js
    )
    scale_error=abs(
        float(symmetric_js_from_standardized_logits(disagree_a*4.0,disagree_b*0.5))
        -disagreement_js
    )

    order_match,_=pairwise_ordering_consistency(
        torch.tensor([[3.0,1.0,-1.0]]),
        torch.tensor([[6.0,2.0,-2.0]]),
    )
    order_flip,flip_diag=pairwise_ordering_consistency(
        torch.tensor([[3.0,1.0,-1.0]]),
        torch.tensor([[-1.0,1.0,3.0]]),
    )
    order_flat,flat_order_diag=pairwise_ordering_consistency(
        torch.zeros(1,4),torch.zeros(1,4)
    )

    uniform_diag=decision_discrimination_diagnostics(torch.zeros(2,7))

    # Real-cache gradient ownership. Reference auxiliary must be exactly zero.
    canonical,paraphrase,_n=cache_pairs[0]
    _rc,_si_rc,fused_rc=_branch_logits(reference,"treatment",canonical)
    _rp,_si_rp,fused_rp=_branch_logits(reference,"treatment",paraphrase)
    ref_aux,_=weighted_cross_view_decision_auxiliary(
        fused_rc,fused_rp,
        decision_coefficient=0.0,
        ordering_coefficient=0.0,
    )
    if float(ref_aux)!=0.0:
        raise RuntimeError("S56-A0 reference auxiliary nonzero")

    _tc,_si_tc,fused_tc=_branch_logits(treatment,"treatment",canonical)
    _tp,_si_tp,fused_tp=_branch_logits(treatment,"treatment",paraphrase)
    trt_aux,trt_diag=weighted_cross_view_decision_auxiliary(
        fused_tc,fused_tp,
        decision_coefficient=0.10,
        ordering_coefficient=0.05,
    )
    grads=torch.autograd.grad(
        trt_aux,treatment.correction_parameters(),allow_unused=True
    )
    treatment_aux_gradient_l1=sum(
        float(g.abs().sum()) for g in grads if g is not None
    )
    if treatment_aux_gradient_l1<=0.0:
        raise RuntimeError("S56-A0 treatment auxiliary gradient vanished")

    cache_inference=cache_requires_grad=0
    for c,p,_n in cache_pairs:
        for evidence in (c,p):
            for tensor in evidence.tensors():
                cache_inference+=int(torch.is_inference(tensor))
                cache_requires_grad+=int(tensor.requires_grad)
    if cache_inference!=0 or cache_requires_grad!=0:
        raise RuntimeError("S56-A0 cache ownership changed")

    max_mass=_synthetic_k_court()

    if identical_js>1e-8:
        raise RuntimeError("S56-A0 identical decision loss nonzero")
    if disagreement_js<=0.01:
        raise RuntimeError("S56-A0 disagreement decision loss not live")
    if offset_error>1e-6:
        raise RuntimeError("S56-A0 offset invariance failed")
    if scale_error>3e-5:
        raise RuntimeError("S56-A0 scale invariance failed")
    if float(order_match)!=0.0 or float(order_flip)<=0.0:
        raise RuntimeError("S56-A0 ordering mechanics failed")
    if float(order_flat)!=0.0 or float(flat_order_diag["active_pair_fraction"])!=0.0:
        raise RuntimeError("S56-A0 inactive ordering failed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S56_A0_CROSS_VIEW_DECISION_CONSISTENCY_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "native_trainable_parameter_count":0,
        "semantic_case_count":len(rows),
        "cache_digest":cache_digest,
        "cache_inference_tensor_count":cache_inference,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "reference_private_trainable_parameter_count":114688,
        "treatment_private_trainable_parameter_count":114688,
        "added_trainable_parameter_count":0,
        "identity_parameter_count":0,
        "initialization_bit_identical":True,
        "identical_decision_js":identical_js,
        "controlled_disagreement_decision_js":disagreement_js,
        "shared_offset_invariance_error":offset_error,
        "positive_scale_invariance_error":scale_error,
        "matching_ordering_loss":float(order_match),
        "sign_flip_ordering_loss":float(order_flip),
        "sign_flip_disagreement_fraction":float(flip_diag["sign_disagreement_fraction"]),
        "inactive_flat_ordering_loss":float(order_flat),
        "inactive_flat_active_pair_fraction":float(flat_order_diag["active_pair_fraction"]),
        "reference_auxiliary_exact_zero":float(ref_aux)==0.0,
        "treatment_auxiliary_value":float(trt_aux.detach()),
        "treatment_auxiliary_gradient_l1":treatment_aux_gradient_l1,
        "treatment_active_pair_fraction":float(trt_diag["active_pair_fraction"]),
        "uniform_top1_top2_probability_gap":float(uniform_diag["mean_top1_top2_probability_gap"]),
        "uniform_logit_rms":float(uniform_diag["mean_unregularized_logit_rms"]),
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "max_probability_mass_error":max_mass,
        "second_encoder_pass":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S56_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
