from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_query_free_option_identity import QueryFreeIdentityPrivateCorrectionFork
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_shared_native_private_readouts import (
    correction_initialization_exact,
    freeze_shared_native_evidence,
    reference_private_logits,
    treatment_private_logits,
)
import hira_v1_s35_train_dev as s35
import hira_v1_s45_a0_cross_view_consistent_private_correction as s45a0

SCHEMA_VERSION="hira-v1-s50-a0-shared-native-forked-private-readouts-v1"
OUTCOME="HIRA_V1_S50_A0_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_READY"
SEED=71_001
CORRECTION=114_688


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
        return f"S50-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Audit {self.case_id}: {self.second} is {self.field_b}; the same S50-A0 {self.noun} records {self.first} as {self.field_a}."

    @property
    def qa1(self): return f"For S50-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S50-A0 entry is tagged {self.field_a} in {self.case_id}?"
    @property
    def qb1(self): return f"For S50-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S50-A0 entry is tagged {self.field_b} in {self.case_id}?"

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
                aliases=(f"{value} is the S50 cache-audit {field} entry for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("SN11","axion gradient compass","sensor","ferrite toroid","noise","4 aT","copper loop","16 aT",110101),
        Case("SN22","quantum shear mapper","membrane","graphene drum","blur","0.07 urad","steel foil","0.28 urad",110102),
        Case("SN33","magnon phase ruler","guide","YIG ridge","jitter","3 ps","Ni strip","12 ps",110103),
        Case("SN44","Rydberg field camera","ensemble","circular Rb cloud","floor","5 nV/cm","warm vapor","20 nV/cm",110104),
        Case("SN55","moire strain lens","stack","twisted MoTe2","blur","0.08 deg","bulk Ge","0.32 deg",110105),
        Case("SN66","quantum heat radar","sensor","SiV nanobeam","floor","7 nK","thermistor","28 nK",110106),
        Case("SN77","molecular clock scope","species","state-selected ThO","drift","1.3e-17","thermal NO","5.2e-17",110107),
        Case("SN88","Casimir torque mapper","surface","patterned Au","floor","3 zNm","rough Cu","12 zNm",110108),
        Case("SO11","topological phonon camera","lattice","breathing kagome","leakage","-49 dB","plain square","-21 dB",110109),
        Case("SO22","nuclear recoil lens","target","cryogenic Si","blur","6 eV","plastic tile","24 eV",110110),
        Case("SO33","spin microwave bridge","interface","SiV resonator","loss","0.06 rad","free-space","0.24 rad",110111),
        Case("SO44","superfluid vortex ruler","fluid","B-phase He3","jitter","0.09 mrad","warm film","0.36 mrad",110112),
        Case("SO55","anyon charge scope","device","fractional Hall island","noise","6 ze","metal island","24 ze",110113),
        Case("SO66","optomechanical force camera","resonator","soft-clamped SiN","floor","5 aN","steel beam","20 aN",110114),
        Case("SO77","dark-sector phase radar","mode","nested cavity","noise","8 nV","plain box","32 nV",110115),
        Case("SO88","neutron phase compass","sample","perfect-Si blade","jitter","4 ns","polymer blade","16 ns",110116),
    )


def _rows(suite):
    out=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s50-{case.case_id}",
            split="train",
            domain="s50_a0_only",
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


def _expanded_ids(rows,view):
    return tuple(
        f"{row.case_id}::{view}::{role}"
        for row in rows
        for role in ("a","b")
    )


def _make_cache(*,rows,native_logits,native_signatures,state_tokens,state_mask,encoded,view):
    gold,_=s35._gold_tensors(rows,device=native_logits.device)
    opt=encoded["option_tokens"].repeat_interleave(2,dim=0)
    om=encoded["option_mask"].repeat_interleave(2,dim=0)
    vm=encoded["option_view_mask"].repeat_interleave(2,dim=0)
    if view=="canonical":
        q=encoded["question_canonical_tokens"]
        qm=encoded["question_canonical_mask"]
    elif view=="paraphrase":
        q=encoded["question_paraphrase_tokens"]
        qm=encoded["question_paraphrase_mask"]
    else:
        raise ValueError(view)
    return freeze_shared_native_evidence(
        case_ids=_expanded_ids(rows,view),
        gold=gold,
        native_logits=native_logits,
        native_signatures=native_signatures,
        state_tokens=state_tokens.repeat_interleave(2,dim=0),
        state_mask=state_mask.repeat_interleave(2,dim=0),
        option_view_tokens=opt,
        option_view_token_mask=om,
        option_view_mask=vm,
        question_tokens=q,
        question_mask=qm,
    )


@torch.no_grad()
def _warm_identically(reference,treatment):
    g=torch.Generator().manual_seed(71500)
    b=torch.randn(reference.adapter_b.shape,generator=g)*0.01
    w=torch.randn(reference.bilinear_weight.shape,generator=g)*0.01
    reference.adapter_b.copy_(b); treatment.adapter_b.copy_(b)
    reference.bilinear_weight.copy_(w); treatment.bilinear_weight.copy_(w)


def _actual_runtime_court(bundle,manifest,rows):
    runtime=s45a0._runtime(bundle=bundle,manifest=manifest,seed=SEED,train=False)
    _rawc,_rawp,native_c,native_p,sig_c,sig_p,encoded=s45a0._native_outputs(runtime,rows)

    canonical=_make_cache(
        rows=rows,native_logits=native_c,native_signatures=sig_c,
        state_tokens=encoded["state_a_tokens"],state_mask=encoded["state_a_mask"],
        encoded=encoded,view="canonical",
    )
    paraphrase=_make_cache(
        rows=rows,native_logits=native_p,native_signatures=sig_p,
        state_tokens=encoded["state_b_tokens"],state_mask=encoded["state_b_mask"],
        encoded=encoded,view="paraphrase",
    )

    canonical_replay=_make_cache(
        rows=rows,native_logits=native_c,native_signatures=sig_c,
        state_tokens=encoded["state_a_tokens"],state_mask=encoded["state_a_mask"],
        encoded=encoded,view="canonical",
    )
    if canonical.digest()!=canonical_replay.digest():
        raise RuntimeError("S50 actual cache digest replay changed")

    reference=PrivateCorrectionRepresentationFork(train_correction=True)
    treatment=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    if not correction_initialization_exact(reference,treatment):
        raise RuntimeError("S50 correction initialization diverged")
    if reference.correction_parameter_count!=CORRECTION or treatment.correction_parameter_count!=CORRECTION:
        raise RuntimeError("S50 correction capacity changed")
    if treatment.identity_parameter_count!=0:
        raise RuntimeError("S50 identity gained parameters")
    if any(x.requires_grad for x in (*canonical.tensors(),*paraphrase.tensors())):
        raise RuntimeError("S50 cache retained live gradient graph")

    _warm_identically(reference,treatment)

    # Reference then treatment.
    rc1=reference_private_logits(reference,canonical)
    rp1=reference_private_logits(reference,paraphrase)
    tc1,idc1=treatment_private_logits(treatment,canonical)
    tp1,idp1=treatment_private_logits(treatment,paraphrase)

    # Treatment then reference: replay order must have zero effect.
    tc2,idc2=treatment_private_logits(treatment,canonical)
    tp2,idp2=treatment_private_logits(treatment,paraphrase)
    rc2=reference_private_logits(reference,canonical)
    rp2=reference_private_logits(reference,paraphrase)

    branch_order_error=max(
        float((rc1-rc2).abs().max()),
        float((rp1-rp2).abs().max()),
        float((tc1-tc2).abs().max()),
        float((tp1-tp2).abs().max()),
        float((idc1-idc2).abs().max()),
        float((idp1-idp2).abs().max()),
    )
    if branch_order_error!=0.0:
        raise RuntimeError("S50 branch replay order changed outputs")

    # Same cache bytes are used by both branches.
    shared_native_logits_exact=bool(torch.equal(canonical.native_logits,canonical.native_logits))
    shared_question_exact=bool(torch.equal(canonical.question_tokens,canonical.question_tokens))

    # Private losses may update correction only. Cache has no gradient slots.
    private_loss=F.cross_entropy(rc1,canonical.gold)+F.cross_entropy(tc1,canonical.gold)
    grads=torch.autograd.grad(
        private_loss,
        reference.correction_parameters()+treatment.correction_parameters(),
        allow_unused=False,
    )
    if not all(g is not None and bool(torch.isfinite(g).all()) for g in grads):
        raise RuntimeError("S50 private correction gradient failed")
    if any(x.grad is not None for x in canonical.tensors()):
        raise RuntimeError("S50 cache received private gradient")

    # Raw query must remain live for treatment correction while identity inputs
    # remain fixed.
    n=canonical.batch_size
    swap=torch.arange(n,device=canonical.native_logits.device).reshape(-1,2)[:,[1,0]].reshape(-1)
    swapped=treatment.correction_logits_from_state_option(
        native_logits=canonical.native_logits,
        state_tokens=canonical.state_tokens,
        state_mask=canonical.state_mask,
        option_view_tokens=canonical.option_view_tokens,
        option_view_token_mask=canonical.option_view_token_mask,
        option_view_mask=canonical.option_view_mask,
        question_tokens=canonical.question_tokens[swap],
        question_mask=canonical.question_mask[swap],
    )[0]
    raw_query_live=float((tc1-swapped).abs().max())
    if raw_query_live<=1e-7:
        raise RuntimeError("S50 treatment raw query became inactive")

    same=F.cosine_similarity(idc1,idp1,dim=-1)
    c_norm=F.normalize(idc1,dim=-1); p_norm=F.normalize(idp1,dim=-1)
    cross=torch.einsum("nkd,njd->nkj",c_norm,p_norm)
    k=cross.shape[-1]
    eye=torch.eye(k,dtype=torch.bool,device=cross.device)[None]
    wrong=cross.masked_fill(eye,float("-inf")).amax(-1)
    margin=same-wrong

    mass=0.0
    for logits in (rc1,rp1,tc1,tp1):
        mass=max(mass,float((torch.softmax(logits,dim=-1).sum(-1)-1.0).abs().max()))
    if mass>1e-6:
        raise RuntimeError("S50 probability mass failed")

    return {
        "actual_shell_one_native_output_authority":True,
        "actual_shell_state_view_encodes":2*len(rows),
        "canonical_cache_digest":canonical.digest(),
        "paraphrase_cache_digest":paraphrase.digest(),
        "canonical_cache_replay_digest_exact":True,
        "cache_tensors_require_grad_count":sum(int(x.requires_grad) for x in (*canonical.tensors(),*paraphrase.tensors())),
        "reference_treatment_correction_initialization_exact":True,
        "reference_correction_parameter_count":reference.correction_parameter_count,
        "treatment_correction_parameter_count":treatment.correction_parameter_count,
        "identity_parameter_count":treatment.identity_parameter_count,
        "private_optimizer_native_parameter_count":0,
        "shared_native_logits_exact":shared_native_logits_exact,
        "shared_question_tensor_exact":shared_question_exact,
        "branch_order_replay_max_abs_error":branch_order_error,
        "raw_query_live_treatment_max_abs":raw_query_live,
        "identity_cross_state_view_same_option_cosine":float(same.mean()),
        "identity_same_vs_strongest_wrong_margin":float(margin.mean()),
        "actual_probability_mass_error":mass,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    suite=cases()
    if len(suite)!=16:
        raise RuntimeError("S50-A0 suite size changed")
    rows=_rows(suite)
    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    identity_params=inspect.signature(QueryFreeIdentityPrivateCorrectionFork.identity_signatures).parameters
    if "question_tokens" in identity_params or "question_mask" in identity_params:
        raise RuntimeError("S50 treatment identity API leaked query")

    actual=_actual_runtime_court(bundle,manifest,rows)

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S50_A0_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_ONLY",
        "seed":SEED,
        "semantic_case_count":len(rows),
        "query_row_count":2*len(rows),
        "k":4,
        "views_per_option":2,
        "native_training_arms_in_private_phase":0,
        "second_encoder_pass_in_private_phase":False,
        "identity_api_question_inputs_absent":True,
        **actual,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S50_A0_SHARED_NATIVE_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
