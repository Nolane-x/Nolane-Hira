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
from nmd.v1_learned_joint_relation_interaction import (
    LearnedJointRelationPrivateCorrectionFork,
    learned_joint_initialization_exact,
)
from hira_v1_s50_train_dev import _materialize_cache


SCHEMA_VERSION="hira-v1-s55-a0-learned-joint-relation-interaction-v1"
OUTCOME="HIRA_V1_S55_A0_LEARNED_JOINT_RELATION_INTERACTION_READY"
SEED=76_001


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
        return f"S55-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."
    @property
    def state_b(self):
        return f"S55 learned-joint audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."
    @property
    def qa1(self): return f"For S55-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S55-A0 entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For S55-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S55-A0 entry is tagged {self.field_b} for {self.case_id}?"

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
                aliases=(f"{value} is the S55-A0 {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("LJ11","axion pressure clock","sensor","Nb cavity","pressure floor","2 nPa","steel cell","8 nPa",19101),
        Case("LJ22","spin torque lens","film","PtCo bilayer","torque floor","0.06 zNm","bulk Ni","0.24 zNm",19102),
        Case("LJ33","phonon recoil camera","guide","soft SiN ridge","recoil blur","0.05 um","Cu strip","0.20 um",19103),
        Case("LJ44","Rydberg curvature radar","ensemble","dressed Cs 62D","curvature floor","4 nrad/mm","hot vapor","16 nrad/mm",19104),
        Case("LJ55","moire pressure compass","stack","aligned MoSe2-WSe2","pressure floor","5 nPa","bulk Si","20 nPa",19105),
        Case("LJ66","neutron gradient scope","target","perfect Si","gradient drift","0.10 mrad","polymer tile","0.40 mrad",19106),
        Case("LJ77","topological torque clock","channel","valley edge","torque leak","-48 dB","bulk bar","-20 dB",19107),
        Case("LJ88","molecular recoil map","beam","selected ThO","recoil spread","2 mm/s","thermal SO2","8 mm/s",19108),
        Case("LK11","vacuum pressure bridge","surface","template Au","pressure floor","0.09 nPa","rough steel","0.36 nPa",19109),
        Case("LK22","atomic phase lens","species","clocked Yb","phase noise","0.04 rad","thermal K","0.16 rad",19110),
        Case("LK33","magnon torque compass","guide","low-loss YIG","torque floor","2 zNm","Ni strip","8 zNm",19111),
        Case("LK44","optical recoil ruler","interface","NV cavity","recoil loss","0.02 rad","free-space link","0.08 rad",19112),
        Case("LK55","superfluid gradient scope","fluid","He3-B film","gradient noise","0.04 mm-1","oil film","0.16 mm-1",19113),
        Case("LK66","Casimir pressure camera","surface","Au-Si pair","pressure drift","0.12 pN","steel pair","0.48 pN",19114),
        Case("LK77","quantum torque compass","sensor","SiV resonator","torque floor","0.04 zNm","Hall bar","0.16 zNm",19115),
        Case("LK88","ferroelectric recoil radar","crystal","strained BaTiO3","recoil jitter","0.6 nm","ceramic slab","2.4 nm",19116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s55-{case.case_id}",
            split="train",
            domain="s55_a0_only",
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
    ref=LearnedJointRelationPrivateCorrectionFork(
        use_state_joint_context=False,
        train_correction=True,
    )
    trt=LearnedJointRelationPrivateCorrectionFork(
        use_state_joint_context=True,
        train_correction=True,
    )
    return ref,trt


def _activate(op,seed):
    with torch.no_grad():
        op.adapter_b.normal_(generator=torch.Generator().manual_seed(seed),std=0.01)
        op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(seed+1),std=0.01)
        op.learned_joint_transform.adapter_b.normal_(
            generator=torch.Generator().manual_seed(seed+2),std=0.01
        )


def _synthetic_k_court():
    ref,trt=_arms()
    _activate(ref,76510)
    _activate(trt,76510)
    g=torch.Generator().manual_seed(76520)
    max_mass=0.0
    max_norm=0.0
    for k in (3,7,255):
        b=2
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
        for op in (ref,trt):
            logits,_identity,code,*_=op.correction_logits_from_state_option(
                **data,return_context=True
            )
            if tuple(logits.shape)!=(b,k):
                raise RuntimeError("S55-A0 arbitrary-K shape changed")
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(code).all()):
                raise RuntimeError("S55-A0 arbitrary-K non-finite")
            mass=float((torch.softmax(logits,dim=-1).sum(-1)-1.0).abs().max())
            max_mass=max(max_mass,mass)
            max_norm=max(max_norm,float((code.norm(dim=-1)-1.0).abs().max()))
    if max_mass>1e-6 or max_norm>1e-6:
        raise RuntimeError("S55-A0 arbitrary-K normalization failed")
    return max_mass,max_norm


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
        raise RuntimeError("S55-A0 parent native runtime changed")
    if frozen.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S55-A0 parent checkpoint SHA changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S55-A0 checkpoint file SHA changed")

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
        raise RuntimeError("S55-A0 loaded native runtime hash changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S55-A0 native runtime remained trainable")

    rows=_rows()
    if len(rows)!=16:
        raise RuntimeError("S55-A0 case count changed")
    cache_pairs,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    reference,treatment=_arms()
    if reference.correction_parameter_count!=114688 or treatment.correction_parameter_count!=114688:
        raise RuntimeError("S55-A0 correction capacity changed")
    if reference.learned_joint_parameter_count!=65536 or treatment.learned_joint_parameter_count!=65536:
        raise RuntimeError("S55-A0 learned joint capacity changed")
    if reference.private_trainable_parameter_count!=180224 or treatment.private_trainable_parameter_count!=180224:
        raise RuntimeError("S55-A0 total private capacity changed")
    if reference.identity_parameter_count!=0 or treatment.identity_parameter_count!=0:
        raise RuntimeError("S55-A0 identity gained params")
    if not learned_joint_initialization_exact(reference,treatment):
        raise RuntimeError("S55-A0 initialization differs")

    canonical=cache_pairs[0][0]

    # Zero-init warm-start: learned relation code must reproduce the normalized
    # S53 query-option context before the transform is activated.
    ref0=reference.correction_logits_from_state_option(
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
    trt0=treatment.correction_logits_from_state_option(
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
    warm_ref=float((ref0[2]-F.normalize(ref0[3],dim=-1)).abs().max())
    warm_trt=float((trt0[2]-F.normalize(trt0[3],dim=-1)).abs().max())
    if max(warm_ref,warm_trt)>1e-7:
        raise RuntimeError("S55-A0 zero-init warm start changed")

    # Controlled explicit state-channel isolation with q and identity held fixed.
    q=F.normalize(ref0[3].detach(),dim=-1)
    identity=F.normalize(ref0[1].detach(),dim=-1)
    s1=F.normalize(trt0[4].detach(),dim=-1)
    s2=torch.roll(s1,shifts=1,dims=1)

    _activate(reference,76600)
    _activate(treatment,76600)

    zero=torch.zeros_like(s1)
    r1=reference.learned_joint_transform(
        query_context=q,state_joint_context=zero,option_identity=identity
    )
    r2=reference.learned_joint_transform(
        query_context=q,state_joint_context=zero,option_identity=identity
    )
    t1=treatment.learned_joint_transform(
        query_context=q,state_joint_context=s1,option_identity=identity
    )
    t2=treatment.learned_joint_transform(
        query_context=q,state_joint_context=s2,option_identity=identity
    )
    reference_zero_channel_error=float((r1-r2).abs().max())
    treatment_state_channel_sensitivity=float((t1-t2).abs().max())
    if reference_zero_channel_error!=0.0:
        raise RuntimeError("S55-A0 reference zero state channel changed")
    if treatment_state_channel_sensitivity<=1e-8:
        raise RuntimeError("S55-A0 treatment state channel is not live")

    # Learned + correction gradients must both be live while cache/native
    # tensors remain detached.
    logits,_=treatment.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
    )
    gold=torch.tensor([0],dtype=torch.long)
    loss=F.cross_entropy(logits[:1],gold)
    params=treatment.private_parameters()
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    nc=len(treatment.correction_parameters())
    correction_grad_l1=sum(float(g.abs().sum()) for g in grads[:nc] if g is not None)
    learned_grad_l1=sum(float(g.abs().sum()) for g in grads[nc:] if g is not None)
    if correction_grad_l1<=0.0 or learned_grad_l1<=0.0:
        raise RuntimeError("S55-A0 private gradient path vanished")

    cache_inference_count=cache_requires_grad_count=0
    for c,p,_n in cache_pairs:
        for evidence in (c,p):
            for tensor in evidence.tensors():
                cache_inference_count+=int(torch.is_inference(tensor))
                cache_requires_grad_count+=int(tensor.requires_grad)
    if cache_inference_count!=0 or cache_requires_grad_count!=0:
        raise RuntimeError("S55-A0 cache ownership changed")

    max_mass,max_norm=_synthetic_k_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S55_A0_LEARNED_JOINT_RELATION_INTERACTION_ONLY",
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
        "reference_learned_joint_parameter_count":reference.learned_joint_parameter_count,
        "treatment_learned_joint_parameter_count":treatment.learned_joint_parameter_count,
        "reference_private_trainable_parameter_count":reference.private_trainable_parameter_count,
        "treatment_private_trainable_parameter_count":treatment.private_trainable_parameter_count,
        "identity_parameter_count":treatment.identity_parameter_count,
        "initialization_bit_identical":True,
        "reference_zero_init_warm_start_max_abs":warm_ref,
        "treatment_zero_init_warm_start_max_abs":warm_trt,
        "reference_zero_state_channel_invariant_error":reference_zero_channel_error,
        "treatment_state_channel_sensitivity":treatment_state_channel_sensitivity,
        "correction_gradient_l1":correction_grad_l1,
        "learned_joint_gradient_l1":learned_grad_l1,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "max_probability_mass_error":max_mass,
        "max_relation_code_norm_error":max_norm,
        "direct_s54_context_bypass_absent":True,
        "second_encoder_pass":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S55_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
