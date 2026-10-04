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
from nmd.v1_query_free_option_identity import (
    QueryFreeIdentityPrivateCorrectionFork,
    QueryFreeStateOptionIdentity,
)
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import HIRA_V1_S17_TOTAL_PARAMETER_COUNT
import hira_v1_s35_train_dev as s35
import hira_v1_s45_a0_cross_view_consistent_private_correction as s45a0

SCHEMA_VERSION="hira-v1-s49-a0-private-query-free-option-identity-v1"
OUTCOME="HIRA_V1_S49_A0_PRIVATE_QUERY_FREE_OPTION_IDENTITY_READY"
SEED=70_001
NATIVE_TRAINABLE=49_152
CORRECTION=114_688
TOTAL=163_840


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
        return f"S49-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Audit {self.case_id}: {self.second} is {self.field_b}; for the same S49-A0 {self.noun}, {self.first} is {self.field_a}."

    @property
    def qa1(self): return f"For S49-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which entry is tagged {self.field_a} in S49-A0 {self.case_id}?"
    @property
    def qb1(self): return f"For S49-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which entry is tagged {self.field_b} in S49-A0 {self.case_id}?"

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
                aliases=(f"{value} is the S49 identity-audit {field} entry for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("ID11","vacuum momentum compass","probe","diamond loop","noise","3 nT","Hall bar","12 nT",99101),
        Case("ID22","anyon recoil mapper","device","fractional island","blur","0.08 pct","metal island","0.32 pct",99102),
        Case("ID33","phonon phase camera","mode","chiral edge","jitter","4 ps","bulk mode","16 ps",99103),
        Case("ID44","Rydberg vector ruler","ensemble","dressed Cs cloud","floor","6 nV/cm","thermal cell","24 nV/cm",99104),
        Case("ID55","spin texture lens","film","DMI PtCo","blur","0.09 um","Ni sheet","0.36 um",99105),
        Case("ID66","quantum heat camera","sensor","nanoSQUID","floor","8 nK","thermistor","32 nK",99106),
        Case("ID77","molecular parity radar","species","oriented HfF+","drift","1.6e-16","thermal NO","6.4e-16",99107),
        Case("ID88","Casimir gradient lens","surface","annealed Au","floor","0.24 pN/mm","rough Cu","0.96 pN/mm",99108),
        Case("IE11","topological strain camera","lattice","valley-Hall","leakage","-47 dB","plain square","-19 dB",99109),
        Case("IE22","nuclear recoil scope","target","enriched Ge","blur","8 eV","plastic tile","32 eV",99110),
        Case("IE33","spin photon mapper","interface","NV cavity","loss","0.07 rad","free-space","0.28 rad",99111),
        Case("IE44","superfluid phase compass","fluid","vortex-free He4","jitter","0.11 mrad","warm film","0.44 mrad",99112),
        Case("IE55","moire charge radar","stack","twisted WSe2","blur","0.13 deg","bulk Si","0.52 deg",99113),
        Case("IE66","optomechanical torque camera","rotor","SiN paddle","floor","4 zNm","steel vane","16 zNm",99114),
        Case("IE77","dark photon phase lens","mode","reentrant cavity","noise","10 nV","plain box","40 nV",99115),
        Case("IE88","neutron timing compass","sample","perfect-Si grating","jitter","5 ns","polymer grating","20 ns",99116),
    )


def _rows(suite):
    out=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s49-{case.case_id}",split="train",domain="s49_a0_only",language="en",
            state_a=case.state_a,state_b=case.state_b,
            question_a1=case.qa1,question_a2=case.qa2,
            question_b1=case.qb1,question_b2=case.qb2,
            option_texts=tuple(x.criterion_text for x in options),
            option_aliases=tuple(x.aliases[0] for x in options),
            option_ids=tuple(x.option_id for x in options),
            gold_a=ga,gold_b=gb,
        ))
    return out


def _synthetic_mechanics():
    identity=QueryFreeStateOptionIdentity()
    correction=QueryFreeIdentityPrivateCorrectionFork(train_correction=True)
    params=inspect.signature(identity.forward).parameters
    if "question_tokens" in params or "question_mask" in params:
        raise RuntimeError("S49 identity API leaked question inputs")
    if identity.parameter_count!=0 or correction.identity_parameter_count!=0:
        raise RuntimeError("S49 identity gained trainable parameters")
    if correction.correction_parameter_count!=CORRECTION:
        raise RuntimeError("S49 correction capacity changed")
    if HIRA_V1_S17_TOTAL_PARAMETER_COUNT+CORRECTION!=TOTAL:
        raise RuntimeError("S49 total treatment changed")

    g=torch.Generator().manual_seed(70490)
    arbitrary={}
    max_mass=0.0
    for k in (3,7,255):
        st=torch.randn(2,5,256,generator=g)
        sm=torch.ones(2,5,dtype=torch.bool)
        opt=torch.randn(2,k,2,4,256,generator=g)
        om=torch.ones(2,k,2,4,dtype=torch.bool)
        vm=torch.ones(2,k,2,dtype=torch.bool)
        ids=identity(
            state_tokens=st,state_mask=sm,
            option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        )
        q=torch.randn(2,3,256,generator=g)
        qm=torch.ones(2,3,dtype=torch.bool)
        native=torch.randn(2,k,generator=g)
        logits=correction.correction_logits(
            native_logits=native,signatures=ids,
            question_tokens=q,question_mask=qm,
        )
        if tuple(ids.shape)!=(2,k,256) or tuple(logits.shape)!=(2,k):
            raise RuntimeError("S49 arbitrary-K shape changed")
        max_mass=max(max_mass,float((torch.softmax(logits,dim=-1).sum(-1)-1.0).abs().max()))
        arbitrary[f"k{k}_pass"]=True

    with torch.no_grad():
        correction.adapter_b.copy_(torch.randn(256,64,generator=g)*0.01)
        correction.bilinear_weight.copy_(torch.randn(256,256,generator=g)*0.01)

    st=torch.randn(1,5,256,generator=g)
    sm=torch.ones(1,5,dtype=torch.bool)
    opt=torch.randn(1,4,2,4,256,generator=g)
    om=torch.ones(1,4,2,4,dtype=torch.bool)
    vm=torch.ones(1,4,2,dtype=torch.bool)
    ids=identity(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )

    # Arbitrary question substitution cannot affect identity because the
    # identity API has no question input.
    question_substitution_identity_error=0.0

    q1=torch.zeros(2,1,256); q1[0,0,0]=1.; q1[1,0,1]=1.
    qm=torch.ones(2,1,dtype=torch.bool)
    ids2=ids.repeat_interleave(2,dim=0)
    native=torch.zeros(2,4)
    logits=correction.correction_logits(
        native_logits=native,signatures=ids2,
        question_tokens=q1,question_mask=qm,
    )
    raw_query_live_max_abs=float((logits[0]-logits[1]).abs().max())
    raw_query_changes_ranking=bool(logits[0].argmax()!=logits[1].argmax())

    perm=torch.tensor([2,0,3,1])
    moved=identity(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt[:,perm],
        option_view_token_mask=om[:,perm],
        option_view_mask=vm[:,perm],
    )
    option_perm_error=float((moved-ids[:,perm]).abs().max())

    state_perm=torch.tensor([3,0,4,1,2])
    state_moved=identity(
        state_tokens=st[:,state_perm],state_mask=sm[:,state_perm],
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )
    state_perm_error=float((state_moved-ids).abs().max())

    token_perm=torch.tensor([2,0,3,1])
    token_moved=identity(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt[:,:,:,token_perm],
        option_view_token_mask=om[:,:,:,token_perm],
        option_view_mask=vm,
    )
    token_perm_error=float((token_moved-ids).abs().max())

    view_moved=identity(
        state_tokens=st,state_mask=sm,
        option_view_tokens=opt[:,:,torch.tensor([1,0])],
        option_view_token_mask=om[:,:,torch.tensor([1,0])],
        option_view_mask=vm[:,:,torch.tensor([1,0])],
    )
    view_perm_error=float((view_moved-ids).abs().max())

    sm_pad=torch.cat([sm,torch.zeros(1,2,dtype=torch.bool)],dim=1)
    st_pad=torch.cat([st,torch.randn(1,2,256,generator=g)*1e6],dim=1)
    om_pad=torch.cat([om,torch.zeros(1,4,2,2,dtype=torch.bool)],dim=3)
    opt_pad=torch.cat([opt,torch.randn(1,4,2,2,256,generator=g)*1e6],dim=3)
    padded=identity(
        state_tokens=st_pad,state_mask=sm_pad,
        option_view_tokens=opt_pad,option_view_token_mask=om_pad,option_view_mask=vm,
    )
    padding_error=float((padded-ids).abs().max())

    state=correction.correction_state_dict()
    replay=QueryFreeIdentityPrivateCorrectionFork(train_correction=False)
    replay.load_correction_state_dict(state,freeze=True)
    checkpoint_exact=all(
        torch.equal(state[k],replay.correction_state_dict()[k]) for k in state
    )

    if max(option_perm_error,state_perm_error,token_perm_error,view_perm_error,padding_error)>2e-6:
        raise RuntimeError("S49 identity invariance failed")
    if raw_query_live_max_abs<=1e-7:
        raise RuntimeError("S49 raw query became inactive")
    if max_mass>1e-6:
        raise RuntimeError("S49 probability mass failed")
    if not checkpoint_exact:
        raise RuntimeError("S49 checkpoint roundtrip failed")

    return {
        "identity_parameter_count":identity.parameter_count,
        "identity_api_question_inputs_absent":True,
        "arbitrary_k3_pass":arbitrary["k3_pass"],
        "arbitrary_k7_pass":arbitrary["k7_pass"],
        "arbitrary_k255_pass":arbitrary["k255_pass"],
        "question_substitution_identity_max_abs_error":question_substitution_identity_error,
        "logical_option_permutation_max_abs_error":option_perm_error,
        "state_token_permutation_max_abs_error":state_perm_error,
        "option_token_permutation_max_abs_error":token_perm_error,
        "option_view_permutation_max_abs_error":view_perm_error,
        "masked_padding_max_abs_error":padding_error,
        "raw_query_live_corrected_logit_max_abs":raw_query_live_max_abs,
        "raw_query_changes_ranking_on_probe":raw_query_changes_ranking,
        "max_probability_mass_error":max_mass,
        "checkpoint_roundtrip_exact":checkpoint_exact,
    }


def _s49_correction_loss(op,relation_c,relation_p,_signature_c,_signature_p,encoded,rows):
    gold,_=s35._gold_tensors(rows,device=relation_c.device)
    n=len(rows)
    opt=encoded["option_tokens"].repeat_interleave(2,dim=0)
    om=encoded["option_mask"].repeat_interleave(2,dim=0)
    vm=encoded["option_view_mask"].repeat_interleave(2,dim=0)
    cc,_idc=op.correction_logits_from_state_option(
        native_logits=relation_c,
        state_tokens=encoded["state_a_tokens"].repeat_interleave(2,dim=0),
        state_mask=encoded["state_a_mask"].repeat_interleave(2,dim=0),
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    cp,_idp=op.correction_logits_from_state_option(
        native_logits=relation_p,
        state_tokens=encoded["state_b_tokens"].repeat_interleave(2,dim=0),
        state_mask=encoded["state_b_mask"].repeat_interleave(2,dim=0),
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    ce=0.5*(F.cross_entropy(cc,gold)+F.cross_entropy(cp,gold))
    js=s45a0._js_divergence(cc,cp)
    block=s35.BINDING_COEFFICIENT*ce+s35.INVARIANCE_COEFFICIENT*js
    return block,ce,js,cc,cp


def _ownership_court(bundle,manifest,rows):
    original_cls=s45a0.PrivateCorrectionRepresentationFork
    original_loss=s45a0._correction_loss
    s45a0.PrivateCorrectionRepresentationFork=QueryFreeIdentityPrivateCorrectionFork
    s45a0._correction_loss=_s49_correction_loss
    try:
        return s45a0._ownership_warmstart_court(bundle,manifest,rows)
    finally:
        s45a0.PrivateCorrectionRepresentationFork=original_cls
        s45a0._correction_loss=original_loss


@torch.inference_mode()
def _actual_runtime_court(bundle,manifest,rows):
    runtime=s45a0._runtime(bundle=bundle,manifest=manifest,seed=SEED,train=False)
    _rawc,_rawp,native_c,native_p,_sc,_sp,encoded=s45a0._native_outputs(runtime,rows)
    op=QueryFreeIdentityPrivateCorrectionFork(train_correction=False)

    opt=encoded["option_tokens"].repeat_interleave(2,dim=0)
    om=encoded["option_mask"].repeat_interleave(2,dim=0)
    vm=encoded["option_view_mask"].repeat_interleave(2,dim=0)
    idc=op.identity_signatures(
        state_tokens=encoded["state_a_tokens"].repeat_interleave(2,dim=0),
        state_mask=encoded["state_a_mask"].repeat_interleave(2,dim=0),
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )
    idp=op.identity_signatures(
        state_tokens=encoded["state_b_tokens"].repeat_interleave(2,dim=0),
        state_mask=encoded["state_b_mask"].repeat_interleave(2,dim=0),
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
    )

    same=F.cosine_similarity(idc,idp,dim=-1)
    c_norm=F.normalize(idc,dim=-1); p_norm=F.normalize(idp,dim=-1)
    cross=torch.einsum("nkd,njd->nkj",c_norm,p_norm)
    k=cross.shape[-1]
    eye=torch.eye(k,dtype=torch.bool,device=cross.device)[None]
    wrong=cross.masked_fill(eye,float("-inf")).amax(-1)
    margin=same-wrong

    cc,_=op.correction_logits_from_state_option(
        native_logits=native_c,
        state_tokens=encoded["state_a_tokens"].repeat_interleave(2,dim=0),
        state_mask=encoded["state_a_mask"].repeat_interleave(2,dim=0),
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    cp,_=op.correction_logits_from_state_option(
        native_logits=native_p,
        state_tokens=encoded["state_b_tokens"].repeat_interleave(2,dim=0),
        state_mask=encoded["state_b_mask"].repeat_interleave(2,dim=0),
        option_view_tokens=opt,option_view_token_mask=om,option_view_mask=vm,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    mass=max(
        float((torch.softmax(cc,dim=-1).sum(-1)-1.0).abs().max().cpu()),
        float((torch.softmax(cp,dim=-1).sum(-1)-1.0).abs().max().cpu()),
    )
    return {
        "actual_shell_one_encoder_batch":True,
        "actual_shell_state_view_encodes":2*len(rows),
        "identity_cross_state_view_same_option_cosine":float(same.mean().cpu()),
        "identity_same_vs_strongest_wrong_margin":float(margin.mean().cpu()),
        "actual_probability_mass_error":mass,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    suite=cases()
    if len(suite)!=16: raise RuntimeError("S49-A0 suite size changed")
    rows=_rows(suite)
    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    mechanics=_synthetic_mechanics()
    ownership=_ownership_court(bundle,manifest,rows)
    actual=_actual_runtime_court(bundle,manifest,rows)

    if mechanics["identity_parameter_count"]!=0: raise RuntimeError("S49 identity params changed")
    if ownership["js_only_native_runtime_gradient_l1"]!=0.0: raise RuntimeError("S49 correction leaked native gradient")
    if ownership["native_objective_correction_gradient_l1"]!=0.0: raise RuntimeError("S49 native objective leaked private gradient")
    if ownership["matched_native_one_step_parameter_max_abs"]!=0.0: raise RuntimeError("S49 native one-step changed")
    if ownership["matched_native_one_step_output_max_abs"]!=0.0: raise RuntimeError("S49 native one-step outputs changed")

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S49_A0_PRIVATE_QUERY_FREE_OPTION_IDENTITY_ONLY",
        "seed":SEED,
        "semantic_case_count":len(rows),
        "state_view_count":2*len(rows),
        "k":4,
        "views_per_option":2,
        "native_trainable_parameter_count":NATIVE_TRAINABLE,
        "correction_parameter_count":CORRECTION,
        "treatment_total_trainable_parameter_count":TOTAL,
        "second_encoder_pass":False,
        **mechanics,
        **actual,
        "ownership":ownership,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("HIRA_V1_S49_A0_QUERY_FREE_IDENTITY_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
