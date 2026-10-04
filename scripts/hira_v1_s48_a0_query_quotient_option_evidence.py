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
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_query_quotient_private_correction import QueryQuotientPrivateCorrectionFork
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import HIRA_V1_S17_TOTAL_PARAMETER_COUNT
import hira_v1_s45_a0_cross_view_consistent_private_correction as s45a0

SCHEMA_VERSION="hira-v1-s48-a0-query-quotient-option-evidence-v1"
OUTCOME="HIRA_V1_S48_A0_QUERY_QUOTIENT_OPTION_EVIDENCE_READY"
SEED=69_001

NATIVE_TRAINABLE=49_152
CORRECTION=114_688
TREATMENT_TOTAL=163_840


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
        return (
            f"S48-A0 {self.noun} record {self.case_id}: "
            f"{self.field_a} {self.first}; {self.field_b} {self.second}."
        )

    @property
    def state_b(self):
        return (
            f"Audit {self.case_id} stores {self.second} for {self.field_b}; "
            f"the same S48-A0 {self.noun} stores {self.first} for {self.field_a}."
        )

    @property
    def qa1(self):
        return f"For S48-A0 {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry is tagged {self.field_a} in S48-A0 {self.case_id}?"

    @property
    def qb1(self):
        return f"For S48-A0 {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry is tagged {self.field_b} in S48-A0 {self.case_id}?"

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
                aliases=(
                    f"{value} is the S48 quotient-audit {field} entry for this {self.noun}",
                ),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("QQ11","vacuum spin compass","probe","diamond NV ring","noise floor","2.6 nT","Hall tile","10.4 nT",88101),
        Case("QQ22","anyon phase lens","device","fractional Hall loop","braid drift","0.7 pct","metal loop","2.8 pct",88102),
        Case("QQ33","phonon torque mapper","rotor","SiN torsion paddle","floor","3 zNm","steel vane","12 zNm",88103),
        Case("QQ44","dark photon camera","mode","nested reentrant mode","noise","9 nV","plain cavity","36 nV",88104),
        Case("QQ55","neutrino timing ruler","target","dual-phase xenon","jitter","4 ns","plastic tile","16 ns",88105),
        Case("QQ66","moire spin radar","stack","twisted WSe2 stack","blur","0.11 deg","bulk Si","0.44 deg",88106),
        Case("QQ77","quantum pressure lens","membrane","soft-clamped SiN","floor","7 nPa","metal foil","28 nPa",88107),
        Case("QQ88","phonon entanglement camera","guide","piezo AlN guide","loss","0.6 dB","ceramic guide","2.4 dB",88108),
        Case("QR11","vacuum birefringence ruler","magnet","nested Halbach ring","rotation","10 nrad","iron yoke","40 nrad",88109),
        Case("QR22","nuclear clock camera","isotope","isomer Th229 crystal","drift","0.9e-18","thermal foil","3.6e-18",88110),
        Case("QR33","thermal quantum compass","sensor","nanoSQUID array","floor","0.7 uK/mm","thermistor","2.8 uK/mm",88111),
        Case("QR44","spin photon bridge","interface","impedance NV cavity","loss","0.08 rad","free-space link","0.32 rad",88112),
        Case("QR55","magnon recoil camera","guide","domain YIG ridge","blur","0.13 um-1","Ni strip","0.52 um-1",88113),
        Case("QR66","Rydberg torque lens","ensemble","dressed Rb cloud","noise","5 zNm","thermal vapor","20 zNm",88114),
        Case("QR77","topological heat radar","channel","chiral edge channel","leakage","-48 dB","bulk bar","-20 dB",88115),
        Case("QR88","molecular phase compass","beam","state-selected HCN","jitter","14 urad","thermal beam","56 urad",88116),
    )


def _rows(suite):
    out=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        out.append(
            S17FusionCase(
                case_id=f"a0-s48-{case.case_id}",
                split="train",
                domain="s48_a0_only",
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
            )
        )
    return out


def _synthetic_mechanics():
    op=QueryQuotientPrivateCorrectionFork(train_correction=True)
    if op.correction_parameter_count!=CORRECTION:
        raise RuntimeError("S48-A0 correction capacity changed")
    if HIRA_V1_S17_TOTAL_PARAMETER_COUNT+op.correction_parameter_count!=TREATMENT_TOTAL:
        raise RuntimeError("S48-A0 treatment total changed")

    g=torch.Generator().manual_seed(69480)
    arbitrary={}
    max_mass_error=0.0
    for k in (3,7,255):
        sig=F.normalize(torch.randn(3,k,256,generator=g),dim=-1)
        native=torch.randn(3,k,generator=g)
        q=torch.randn(3,5,256,generator=g)
        qm=torch.ones(3,5,dtype=torch.bool)
        out=op.correction_logits(
            native_logits=native,signatures=sig,
            question_tokens=q,question_mask=qm,
        )
        quotient,_=op.query_quotient(
            signatures=sig,question_tokens=q,question_mask=qm
        )
        if tuple(out.shape)!=(3,k) or tuple(quotient.shape)!=(3,256):
            raise RuntimeError("S48-A0 arbitrary-K shape changed")
        if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(quotient).all()):
            raise RuntimeError("S48-A0 arbitrary-K non-finite")
        max_mass_error=max(
            max_mass_error,
            float((torch.softmax(out,dim=-1).sum(-1)-1.0).abs().max()),
        )
        arbitrary[f"k{k}_pass"]=True

    with torch.no_grad():
        op.adapter_b.copy_(torch.randn(256,64,generator=g)*0.01)
        op.bilinear_weight.copy_(torch.randn(256,256,generator=g)*0.01)

    sig=torch.zeros(1,4,256)
    sig[0,:,0]=torch.tensor([-3.,-1.,1.,3.])
    sig[0,:,1]=torch.tensor([-1.,2.,-2.,1.])
    q=torch.zeros(1,1,256)
    q[0,0,0]=1.0
    q[0,0,1]=0.5
    qm=torch.ones(1,1,dtype=torch.bool)

    nuisance=q.clone()
    nuisance[0,0,200]=9.0
    quotient,_=op.query_quotient(signatures=sig,question_tokens=q,question_mask=qm)
    nuisance_quotient,_=op.query_quotient(
        signatures=sig,question_tokens=nuisance,question_mask=qm
    )
    nuisance_error=float((quotient-nuisance_quotient).abs().max())

    q2=torch.zeros_like(q); q2[0,0,1]=1.0
    quotient2,_=op.query_quotient(signatures=sig,question_tokens=q2,question_mask=qm)
    distinct_cosine=float(F.cosine_similarity(quotient,quotient2).abs().item())
    relevant_norm=float(quotient.norm().item())

    native=torch.randn(1,4,generator=g)
    base_logits=op.correction_logits(
        native_logits=native,signatures=sig,
        question_tokens=q,question_mask=qm,
    )
    nuisance_logits=op.correction_logits(
        native_logits=native,signatures=sig,
        question_tokens=nuisance,question_mask=qm,
    )
    bypass_error=float((base_logits-nuisance_logits).abs().max())

    perm=torch.tensor([2,0,3,1])
    perm_q,_=op.query_quotient(
        signatures=sig[:,perm],question_tokens=q,question_mask=qm
    )
    perm_logits=op.correction_logits(
        native_logits=native[:,perm],signatures=sig[:,perm],
        question_tokens=q,question_mask=qm,
    )
    permutation_q_error=float((perm_q-quotient).abs().max())
    permutation_logit_error=float((perm_logits-base_logits[:,perm]).abs().max())

    flat=torch.ones(1,7,256)
    flat_q,_=op.query_quotient(signatures=flat,question_tokens=q,question_mask=qm)
    flat_native=torch.randn(1,7,generator=g)
    flat_out=op.correction_logits(
        native_logits=flat_native,signatures=flat,
        question_tokens=q,question_mask=qm,
    )
    zero_quotient_error=float(flat_q.abs().max())
    zero_subspace_logit_error=float((flat_out-flat_native).abs().max())

    if nuisance_error>2e-6 or bypass_error>2e-6:
        raise RuntimeError("S48-A0 orthogonal nuisance bypass detected")
    if relevant_norm<0.99:
        raise RuntimeError("S48-A0 relation-relevant quotient collapsed")
    if distinct_cosine>=0.99:
        raise RuntimeError("S48-A0 distinct relation directions collapsed")
    if permutation_q_error>2e-6 or permutation_logit_error>2e-6:
        raise RuntimeError("S48-A0 option permutation equivariance failed")
    if zero_quotient_error!=0.0 or zero_subspace_logit_error!=0.0:
        raise RuntimeError("S48-A0 zero-subspace behavior changed")
    if max_mass_error>1e-6:
        raise RuntimeError("S48-A0 probability mass failed")

    return {
        "arbitrary_k3_pass":arbitrary["k3_pass"],
        "arbitrary_k7_pass":arbitrary["k7_pass"],
        "arbitrary_k255_pass":arbitrary["k255_pass"],
        "orthogonal_nuisance_quotient_max_abs_error":nuisance_error,
        "orthogonal_nuisance_corrected_logit_max_abs_error":bypass_error,
        "relation_relevant_quotient_norm":relevant_norm,
        "distinct_relation_direction_abs_cosine":distinct_cosine,
        "logical_option_permutation_quotient_max_abs_error":permutation_q_error,
        "logical_option_permutation_corrected_logit_max_abs_error":permutation_logit_error,
        "zero_subspace_quotient_max_abs":zero_quotient_error,
        "zero_subspace_corrected_logit_max_abs_error":zero_subspace_logit_error,
        "max_probability_mass_error":max_mass_error,
    }


def _ownership_court(bundle,manifest,rows):
    original=s45a0.PrivateCorrectionRepresentationFork
    s45a0.PrivateCorrectionRepresentationFork=QueryQuotientPrivateCorrectionFork
    try:
        result=s45a0._ownership_warmstart_court(bundle,manifest,rows)
    finally:
        s45a0.PrivateCorrectionRepresentationFork=original
    return result


@torch.inference_mode()
def _actual_runtime_court(bundle,manifest,rows):
    runtime=s45a0._runtime(bundle=bundle,manifest=manifest,seed=SEED,train=False)
    _raw_c,_raw_p,native_c,native_p,sig_c,sig_p,encoded=s45a0._native_outputs(runtime,rows)
    op=QueryQuotientPrivateCorrectionFork(train_correction=False)

    qc,rawc=op.query_quotient(
        signatures=sig_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    qp,rawp=op.query_quotient(
        signatures=sig_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )

    cc=op.correction_logits(
        native_logits=native_c,signatures=sig_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    cp=op.correction_logits(
        native_logits=native_p,signatures=sig_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )

    raw_q_cos=float(F.cosine_similarity(rawc,rawp,dim=-1).mean().cpu())
    quotient_cos=float(F.cosine_similarity(qc,qp,dim=-1).mean().cpu())
    raw_vs_quotient_c=float(F.cosine_similarity(rawc,qc,dim=-1).mean().cpu())
    raw_vs_quotient_p=float(F.cosine_similarity(rawp,qp,dim=-1).mean().cpu())
    zero_fraction=float(((qc.norm(dim=-1)==0)|(qp.norm(dim=-1)==0)).to(torch.float32).mean().cpu())
    quotient_norm_mean=float(torch.cat([qc.norm(dim=-1),qp.norm(dim=-1)]).mean().cpu())
    mass=max(
        float((torch.softmax(cc,dim=-1).sum(-1)-1.0).abs().max().cpu()),
        float((torch.softmax(cp,dim=-1).sum(-1)-1.0).abs().max().cpu()),
    )

    if not all(torch.isfinite(x).all() for x in (qc,qp,cc,cp)):
        raise RuntimeError("S48-A0 actual runtime non-finite")
    if mass>1e-6:
        raise RuntimeError("S48-A0 actual runtime probability mass failed")

    return {
        "actual_shell_state_view_encodes":2*len(rows),
        "actual_shell_one_encoder_batch":True,
        "actual_quotient_norm_mean":quotient_norm_mean,
        "actual_quotient_zero_fraction":zero_fraction,
        "actual_raw_query_cross_view_cosine":raw_q_cos,
        "actual_quotient_cross_view_cosine":quotient_cos,
        "actual_raw_vs_quotient_canonical_cosine":raw_vs_quotient_c,
        "actual_raw_vs_quotient_paraphrase_cosine":raw_vs_quotient_p,
        "actual_probability_mass_error":mass,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    suite=cases()
    if len(suite)!=16:
        raise RuntimeError("S48-A0 suite size changed")
    rows=_rows(suite)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    mechanics=_synthetic_mechanics()
    ownership=_ownership_court(bundle,manifest,rows)
    actual=_actual_runtime_court(bundle,manifest,rows)

    for key in (
        "matched_native_one_step_parameter_max_abs",
        "matched_native_one_step_output_max_abs",
        "zero_init_correction_native_runtime_gradient_l1",
        "native_objective_correction_gradient_l1",
        "js_only_native_runtime_gradient_l1",
    ):
        if float(ownership.get(key,-1.0))!=0.0:
            raise RuntimeError(f"S48-A0 ownership failed: {key}")

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S48_A0_QUERY_QUOTIENT_OPTION_EVIDENCE_ONLY",
        "semantic_case_count":len(rows),
        "state_view_count":2*len(rows),
        "k":4,
        "views_per_option":2,
        "seed":SEED,
        "native_trainable_parameter_count":NATIVE_TRAINABLE,
        "reference_correction_parameter_count":CORRECTION,
        "treatment_correction_parameter_count":CORRECTION,
        "treatment_total_trainable_parameter_count":TREATMENT_TOTAL,
        "quotient_trainable_parameter_count":0,
        "query_interface":"option_difference_covariance_direction_quotient",
        "raw_query_bypass":False,
        "second_encoder_pass":False,
        **mechanics,
        **actual,
        "ownership":ownership,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }

    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S48_A0_QUERY_QUOTIENT_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
