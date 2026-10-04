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
from nmd.v1_persisted_native_authority import (
    file_sha256,
    load_native_authority,
)
from nmd.v1_query_relation_canonicalization import (
    CanonicalizedQueryFreeIdentityPrivateCorrectionFork,
    weighted_relation_code_auxiliary,
)
from nmd.v1_s17_authority import S17FusionCase
from hira_v1_s50_train_dev import _materialize_cache


SCHEMA_VERSION="hira-v1-s52-a0-query-relation-canonicalization-v1"
OUTCOME="HIRA_V1_S52_A0_QUERY_RELATION_CANONICALIZATION_READY"
SEED=73_001


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
        return f"S52-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"S52 relation audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."

    @property
    def qa1(self): return f"For S52-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S52-A0 entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For S52-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S52-A0 entry is tagged {self.field_b} for {self.case_id}?"

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
                aliases=(f"{value} is the S52-A0 {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("QC11","spin echo lens","medium","isotopic Si","echo floor","0.8 urad","steel slab","3.2 urad",15101),
        Case("QC22","phonon phase compass","guide","soft SiN ridge","phase jitter","0.12 mrad","Cu strip","0.48 mrad",15102),
        Case("QC33","Rydberg vector clock","ensemble","dressed Rb 58D","field noise","6 nV/cm","thermal vapor","24 nV/cm",15103),
        Case("QC44","moire charge scope","stack","aligned WSe2-MoSe2","charge blur","0.07 e","bulk Si","0.28 e",15104),
        Case("QC55","vacuum torque camera","rotor","diamond paddle","torque floor","4 zNm","steel vane","16 zNm",15105),
        Case("QC66","nuclear recoil lens","target","enriched Ge","recoil floor","5 eV","plastic tile","20 eV",15106),
        Case("QC77","topological heat map","channel","chiral edge","thermal leak","-49 dB","bulk bar","-21 dB",15107),
        Case("QC88","molecular Stark ruler","beam","state-selected HCN","Stark drift","0.9 Hz","thermal beam","3.6 Hz",15108),
        Case("QD11","quantum pressure clock","membrane","soft-clamped SiN","pressure floor","6 nPa","metal foil","24 nPa",15109),
        Case("QD22","magnon recoil scope","guide","YIG ridge","recoil blur","0.10 um-1","Ni strip","0.40 um-1",15110),
        Case("QD33","atomic curvature camera","species","lattice Sr","curvature noise","3 nrad/mm","thermal K","12 nrad/mm",15111),
        Case("QD44","optical spin bridge","interface","SiV cavity","conversion loss","0.06 rad","free-space link","0.24 rad",15112),
        Case("QD55","superfluid phase ruler","fluid","He4 film","phase noise","0.11 mrad","oil film","0.44 mrad",15113),
        Case("QD66","Casimir gradient camera","surface","template Au","gradient floor","0.20 pN/mm","rough steel","0.80 pN/mm",15114),
        Case("QD77","quantum heat compass","sensor","nanoSQUID","thermal floor","5 nK","thermistor","20 nK",15115),
        Case("QD88","spin texture radar","film","DMI PtCo","texture blur","0.07 um","bulk Fe","0.28 um",15116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s52-{case.case_id}",
            split="train",
            domain="s52_a0_only",
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
    op=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    g=torch.Generator().manual_seed(73520)
    max_mass=0.0
    for k in (3,7,255):
        b=2
        state=torch.randn(b,5,256,generator=g)
        state_mask=torch.ones(b,5,dtype=torch.bool)
        option=torch.randn(b,k,2,4,256,generator=g)
        token_mask=torch.ones(b,k,2,4,dtype=torch.bool)
        view_mask=torch.ones(b,k,2,dtype=torch.bool)
        native=torch.randn(b,k,generator=g)
        question=torch.randn(b,3,256,generator=g)
        question_mask=torch.ones(b,3,dtype=torch.bool)
        logits,_identity=op.correction_logits_from_state_option(
            native_logits=native,
            state_tokens=state,
            state_mask=state_mask,
            option_view_tokens=option,
            option_view_token_mask=token_mask,
            option_view_mask=view_mask,
            question_tokens=question,
            question_mask=question_mask,
        )
        if tuple(logits.shape)!=(b,k):
            raise RuntimeError("S52-A0 arbitrary-K shape changed")
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("S52-A0 arbitrary-K non-finite")
        mass=float((torch.softmax(logits,dim=-1).sum(-1)-1.0).abs().max())
        max_mass=max(max_mass,mass)
    if max_mass>1e-6:
        raise RuntimeError("S52-A0 probability mass failed")
    return max_mass


def _auxiliary_gradient_court():
    op=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    g=torch.Generator().manual_seed(73600)

    # Controlled query-token probes: same-relation views are close but not
    # identical; A/B relation centroids intentionally overlap enough to make
    # the frozen separation hinge nonzero.
    base=torch.randn(3,1,256,generator=g)
    a1=base+0.04*torch.randn(3,1,256,generator=g)
    a2=base+0.09*torch.randn(3,1,256,generator=g)
    bbase=base+0.10*torch.randn(3,1,256,generator=g)
    b1=bbase+0.05*torch.randn(3,1,256,generator=g)
    b2=bbase+0.11*torch.randn(3,1,256,generator=g)
    mask=torch.ones(3,1,dtype=torch.bool)

    c=[]
    for q in (a1,a2,b1,b2):
        c.append(op.query_summary(question_tokens=q,question_mask=mask))

    aux,diag=weighted_relation_code_auxiliary(
        a1=c[0],a2=c[1],b1=c[2],b2=c[3],coefficient=0.10
    )
    if diag["same_relation_loss"]<=0.0:
        raise RuntimeError("S52-A0 same-relation auxiliary vanished")
    if diag["different_relation_hinge"]<=0.0:
        raise RuntimeError("S52-A0 separation auxiliary vanished")

    params=op.canonicalizer_parameters()+op.correction_parameters()
    grads=torch.autograd.grad(aux,params,allow_unused=True)
    cgrads=grads[:2]
    rgrads=grads[2:]
    canonicalizer_grad_l1=sum(
        0.0 if g is None else float(g.detach().abs().sum())
        for g in cgrads
    )
    correction_grad_l1=sum(
        0.0 if g is None else float(g.detach().abs().sum())
        for g in rgrads
    )
    if canonicalizer_grad_l1<=0.0:
        raise RuntimeError("S52-A0 auxiliary did not reach canonicalizer")
    if correction_grad_l1!=0.0:
        raise RuntimeError("S52-A0 auxiliary leaked into correction")

    reference_aux,reference_diag=weighted_relation_code_auxiliary(
        a1=c[0],a2=c[1],b1=c[2],b2=c[3],coefficient=0.0
    )
    if float(reference_aux.detach())!=0.0:
        raise RuntimeError("S52-A0 reference auxiliary is nonzero")

    return {
        "treatment_auxiliary":float(aux.detach()),
        "same_relation_loss":diag["same_relation_loss"],
        "different_relation_hinge":diag["different_relation_hinge"],
        "canonicalizer_gradient_l1":canonicalizer_grad_l1,
        "correction_gradient_l1":correction_grad_l1,
        "reference_auxiliary":float(reference_aux.detach()),
        "reference_auxiliary_coefficient":reference_diag["coefficient"],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    frozen=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected={
        "runtime_state_sha256":"ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628",
        "native_tensor_digest":"ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628",
        "checkpoint_file_sha256":"19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916",
        "train_manifest_sha256":"590aa9464a5d7925eb1028130f0f66957955437f2b62c6338fcd252b7de56c51",
    }
    for key,value in expected.items():
        if frozen.get(key)!=value:
            raise RuntimeError(f"S52-A0 parent native authority changed: {key}")
    if file_sha256(args.authority_checkpoint)!=expected["checkpoint_file_sha256"]:
        raise RuntimeError("S52-A0 native checkpoint file SHA changed")

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected["checkpoint_file_sha256"],
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected["runtime_state_sha256"]:
        raise RuntimeError("S52-A0 loaded native runtime hash changed")
    if payload["native_tensor_digest"]!=expected["native_tensor_digest"]:
        raise RuntimeError("S52-A0 loaded native logical digest changed")
    native_trainable=sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if native_trainable!=0:
        raise RuntimeError("S52-A0 native runtime remained trainable")

    rows=_rows()
    if len(rows)!=16:
        raise RuntimeError("S52-A0 case count changed")
    cache_pairs,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    reference=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    treatment=CanonicalizedQueryFreeIdentityPrivateCorrectionFork(train_correction=True)

    if reference.correction_parameter_count!=114688:
        raise RuntimeError("S52-A0 correction capacity changed")
    if reference.canonicalizer_parameter_count!=32768:
        raise RuntimeError("S52-A0 canonicalizer capacity changed")
    if reference.private_trainable_parameter_count!=147456:
        raise RuntimeError("S52-A0 private total changed")
    if reference.identity_parameter_count!=0:
        raise RuntimeError("S52-A0 identity gained parameters")

    if len(reference.private_parameters())!=len(treatment.private_parameters()):
        raise RuntimeError("S52-A0 private parameter list changed")
    if not all(
        torch.equal(a.detach(),b.detach())
        for a,b in zip(reference.private_parameters(),treatment.private_parameters())
    ):
        raise RuntimeError("S52-A0 reference/treatment initialization differs")

    # Actual cached query zero-init identity behavior.
    canonical=cache_pairs[0][0]
    raw=reference.raw_query_summary(
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
    )
    code=reference.query_summary(
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
    )
    zero_init_error=float((raw-code).abs().max())
    if zero_init_error>1e-7:
        raise RuntimeError("S52-A0 zero-init query identity changed")

    # Same init + same cache means exact branch-order identity before training.
    r0,_=reference.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
    )
    t0,_=treatment.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens,
        question_mask=canonical.question_mask,
    )
    initial_branch_error=float((r0-t0).abs().max())
    if initial_branch_error!=0.0:
        raise RuntimeError("S52-A0 initial branch logits differ")

    cache_inference_count=0
    cache_requires_grad_count=0
    for canonical_e,paraphrase_e,_n in cache_pairs:
        for evidence in (canonical_e,paraphrase_e):
            for tensor in evidence.tensors():
                cache_inference_count+=int(torch.is_inference(tensor))
                cache_requires_grad_count+=int(tensor.requires_grad)
    if cache_inference_count!=0 or cache_requires_grad_count!=0:
        raise RuntimeError("S52-A0 shared cache ownership changed")

    max_mass=_synthetic_k_court()
    aux=_auxiliary_gradient_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S52_A0_QUERY_RELATION_CANONICALIZATION_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected["runtime_state_sha256"],
        "parent_native_checkpoint_sha256":expected["checkpoint_file_sha256"],
        "native_trainable_parameter_count":0,
        "semantic_case_count":len(rows),
        "cache_digest":cache_digest,
        "cache_inference_tensor_count":cache_inference_count,
        "cache_requires_grad_tensor_count":cache_requires_grad_count,
        "reference_correction_parameter_count":reference.correction_parameter_count,
        "treatment_correction_parameter_count":treatment.correction_parameter_count,
        "reference_canonicalizer_parameter_count":reference.canonicalizer_parameter_count,
        "treatment_canonicalizer_parameter_count":treatment.canonicalizer_parameter_count,
        "reference_private_trainable_parameter_count":reference.private_trainable_parameter_count,
        "treatment_private_trainable_parameter_count":treatment.private_trainable_parameter_count,
        "identity_parameter_count":reference.identity_parameter_count,
        "initialization_bit_identical":True,
        "zero_init_raw_query_max_abs_error":zero_init_error,
        "initial_branch_logit_max_abs_error":initial_branch_error,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "max_probability_mass_error":max_mass,
        "auxiliary":aux,
        "second_encoder_pass":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S52_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
