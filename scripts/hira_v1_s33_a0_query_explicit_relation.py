from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import (
    a13_lora_state_dict,
    iter_a13_lora_modules,
    load_a13_lora_state_dict,
)
from nmd.v1_query_explicit_relation import QueryExplicitRelationCanonicalizer
from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    cross_view_relation_signature_loss,
    relation_signature_same_option_cosine,
)
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
)
from hira_v1_s17_train_dev import (
    BINDING_COEFFICIENT,
    CANONICALIZATION_COEFFICIENT,
    BINDING_CONTRASTIVE_TEMPERATURE,
    PAIR_TEMPERATURE,
    ROLE_TEMPERATURE,
    SIGNATURE_SEPARATION_MARGIN,
    _encode_batch,
    _gold_tensors,
)

SCHEMA_VERSION="hira-v1-s33-a0-query-explicit-relation-v1"
OUTCOME="HIRA_V1_S33_A0_QUERY_EXPLICIT_RELATION_READY"


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
        return f"S33-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Record {self.case_id} lists {self.second} for {self.field_b}. The same S33-A0 {self.noun} lists {self.first} for {self.field_a}."

    @property
    def qa1(self):
        return f"For S33-A0 {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry belongs under {self.field_a} in S33-A0 record {self.case_id}?"

    @property
    def qb1(self):
        return f"For S33-A0 {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry belongs under {self.field_b} in S33-A0 record {self.case_id}?"

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
                aliases=(f"{value} is the recorded {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("QE11","ion microscope","objective","electrostatic","field width","18 um","magnetic","6 um",63101),
        Case("QE22","cryogenic resonator","substrate","sapphire","linewidth","24 kHz","silicon","8 kHz",63102),
        Case("QE33","molecular beam source","nozzle","pulsed","stagnation pressure","12 bar","continuous","4 bar",63103),
        Case("QE44","photonic processor","modulator","Mach-Zehnder","clock","40 GHz","ring","10 GHz",63104),
        Case("QE55","diamond sensor","defect","NV","bias field","36 mT","SiV","9 mT",63105),
        Case("QE66","microcalorimeter","absorber","bismuth","thermal time","280 us","gold","70 us",63106),
        Case("QE77","plasma probe","probe type","triple Langmuir","sweep rate","16 kHz","single","4 kHz",63107),
        Case("QE88","frequency standard","reference","hydrogen maser","gate time","8 s","OCXO","2 s",63108),
        Case("QF11","optical delay","medium","fiber","delay span","240 ns","free space","60 ns",63109),
        Case("QF22","xray detector","sensor","CdTe","frame rate","800 fps","silicon","200 fps",63110),
        Case("QF33","neural probe","electrode","iridium oxide","sample rate","40 kS/s","gold","10 kS/s",63111),
        Case("QF44","microthruster","propellant","xenon","beam current","1.6 A","argon","0.4 A",63112),
        Case("QF55","laser scanner","mirror","galvo","scan angle","32 deg","MEMS","8 deg",63113),
        Case("QF66","gas sensor","film","SnO2","heater","320 C","ZnO","120 C",63114),
        Case("QF77","RF sampler","front end","track-hold","aperture","12 ps","mixer","45 ps",63115),
        Case("QF88","quantum memory","medium","Pr:Y2SiO5","storage","80 us","Rb vapor","20 us",63116),
    )


def _rows(suite):
    out=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s33-{case.case_id}",
            split="train",
            domain="s33_a0_only",
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


def _modules():
    control=CrossViewRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=BINDING_CONTRASTIVE_TEMPERATURE,
    )
    treatment=QueryExplicitRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=BINDING_CONTRASTIVE_TEMPERATURE,
        base_weight=0.50,
        query_option_weight=0.50,
    )
    return control,treatment


def _outputs(runtime,rows):
    encoded=_encode_batch(runtime,rows)
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S33-A0 projection scorer missing")
    control,treatment=_modules()

    def one(state_prefix,question_prefix):
        st=encoded[f"state_{state_prefix}_tokens"].repeat_interleave(2,dim=0)
        sm=encoded[f"state_{state_prefix}_mask"].repeat_interleave(2,dim=0)
        qt=encoded[f"question_{question_prefix}_tokens"]
        qm=encoded[f"question_{question_prefix}_mask"]
        ov=encoded["option_tokens"].repeat_interleave(2,dim=0)
        ovtm=encoded["option_mask"].repeat_interleave(2,dim=0)
        ovm=encoded["option_view_mask"].repeat_interleave(2,dim=0)
        c_logits,c_sig,_=control(
            projection=scorer.projection,
            state_tokens=st,state_mask=sm,
            question_tokens=qt,question_mask=qm,
            option_view_tokens=ov,
            option_view_token_mask=ovtm,
            option_view_mask=ovm,
        )
        t_logits,t_sig,_diag,parts=treatment.forward_with_components(
            projection=scorer.projection,
            state_tokens=st,state_mask=sm,
            question_tokens=qt,question_mask=qm,
            option_view_tokens=ov,
            option_view_token_mask=ovtm,
            option_view_mask=ovm,
        )
        return c_logits,c_sig,t_logits,t_sig,parts,qt,qm

    return encoded,one("a","canonical"),one("b","paraphrase")


def _controlled_question_intervention():
    projection=nn.Linear(4,4,bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(4))
    module=QueryExplicitRelationCanonicalizer()
    state=torch.tensor([[
        [1.0,0.0,0.0,0.0],
        [0.0,0.0,1.0,0.0],
    ]])
    q_a=torch.tensor([[[1.0,0.0,0.0,0.0]]])
    q_b=torch.tensor([[[0.0,1.0,0.0,0.0]]])
    options=torch.tensor([[
        [[[1.0,0.0,0.0,0.0]]],
        [[[0.0,1.0,0.0,0.0]]],
    ]])
    sm=torch.ones(1,2,dtype=torch.bool)
    qm=torch.ones(1,1,dtype=torch.bool)
    otm=torch.ones(1,2,1,1,dtype=torch.bool)
    ovm=torch.ones(1,2,1,dtype=torch.bool)

    def call(q,opt=options):
        return module.forward_with_components(
            projection=projection,
            state_tokens=state,state_mask=sm,
            question_tokens=q,question_mask=qm,
            option_view_tokens=opt,
            option_view_token_mask=otm,
            option_view_mask=ovm,
        )

    la,sa,_da,a=call(q_a)
    lb,sb,_db,b=call(q_b)
    if not (a["query_option_logits"][0,0] > a["query_option_logits"][0,1]):
        raise RuntimeError("S33-A0 qA did not prefer option A")
    if not (b["query_option_logits"][0,1] > b["query_option_logits"][0,0]):
        raise RuntimeError("S33-A0 qB did not prefer option B")

    perm=torch.tensor([1,0])
    pl,ps,_pd,p=call(q_a,options[:,perm])
    perm_logit_error=float((pl-la[:,perm]).abs().max())
    perm_signature_error=float((ps-sa[:,perm]).abs().max())
    if perm_logit_error!=0.0 or perm_signature_error!=0.0:
        raise RuntimeError("S33-A0 option permutation changed")

    return {
        "question_anchor_intervention_max_abs":float(
            (a["query_anchor"]-b["query_anchor"]).abs().max()
        ),
        "query_option_signature_intervention_max_abs":float(
            (a["query_option_signature"]-b["query_option_signature"]).abs().max()
        ),
        "query_option_logits_intervention_max_abs":float(
            (a["query_option_logits"]-b["query_option_logits"]).abs().max()
        ),
        "q_a_option0_minus_option1":float(
            a["query_option_logits"][0,0]-a["query_option_logits"][0,1]
        ),
        "q_b_option1_minus_option0":float(
            b["query_option_logits"][0,1]-b["query_option_logits"][0,0]
        ),
        "option_permutation_logit_error":perm_logit_error,
        "option_permutation_signature_error":perm_signature_error,
    }


def _gradient_probe(bundle,manifest,rows):
    with torch.inference_mode(False),torch.enable_grad():
        fresh=load_hira_v0_m4_bundle(bundle)
        runtime=build_hira_v1_s17_norm_balanced_core(
            fresh.runtime.encoder,
            bundle/str(manifest["t0_checkpoint"]),
            expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
            train_lora=True,
            train_projection=True,
        )
        del fresh
        runtime.clear_schema_cache()
        runtime.eval()
        trainable=[p for p in runtime.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable)!=HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S33-A0 trainable surface changed")

        _encoded,c,p=_outputs(runtime,rows)
        _cc,_cs,tc,tsc,_cp,_qt,_qm=c
        _pc,_ps,tp,tsp,_pp,_pqt,_pqm=p
        gold,_other=_gold_tensors(rows,device=tc.device)
        relation_loss=0.5*(
            F.cross_entropy(tc,gold)+F.cross_entropy(tp,gold)
        )
        canonicalization,_align,_sep=cross_view_relation_signature_loss(
            tsc,tsp,separation_margin=SIGNATURE_SEPARATION_MARGIN
        )
        relation_block=(
            BINDING_COEFFICIENT*relation_loss
            + CANONICALIZATION_COEFFICIENT*canonicalization
        )
        grads=torch.autograd.grad(
            relation_block,trainable,allow_unused=True
        )
        grads=[
            torch.zeros_like(p) if g is None else g
            for p,g in zip(trainable,grads)
        ]
        index={id(p):i for i,p in enumerate(trainable)}
        modules=iter_a13_lora_modules(runtime.encoder)
        b_l1=[
            float(grads[index[id(module.lora_b)]].abs().sum().cpu())
            for module in modules
        ]
        scorer=runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S33-A0 projection scorer missing")
        projection_l1=float(
            grads[index[id(scorer.projection.weight)]].abs().sum().cpu()
        )
        if any(v<=0.0 for v in b_l1):
            raise RuntimeError("S33-A0 LoRA-B gradient vanished")
        if projection_l1<=0.0:
            raise RuntimeError("S33-A0 projection gradient vanished")
        return {
            "relation_block":float(relation_block.detach().cpu()),
            "relation_ce":float(relation_loss.detach().cpu()),
            "canonicalization":float(canonicalization.detach().cpu()),
            "lora_b_gradient_l1":b_l1,
            "projection_gradient_l1":projection_l1,
        }


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    suite=cases()
    if len(suite)!=16:
        raise RuntimeError("S33-A0 suite size changed")
    rows=_rows(suite)
    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    fresh=load_hira_v0_m4_bundle(bundle)
    runtime=build_hira_v1_s17_norm_balanced_core(
        fresh.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    del fresh
    runtime.clear_schema_cache()
    runtime.eval()

    encoded,c,p=_outputs(runtime,rows)
    cc,cs,tc,tsc,cp,cqt,cqm=c
    pc,ps,tp,tsp,pp,pqt,pqm=p

    base_logit_error=max(
        float((cp["base_logits"]-cc).abs().max().cpu()),
        float((pp["base_logits"]-pc).abs().max().cpu()),
    )
    base_signature_error=max(
        float((cp["base_signatures"]-cs).abs().max().cpu()),
        float((pp["base_signatures"]-ps).abs().max().cpu()),
    )
    if base_logit_error!=0.0 or base_signature_error!=0.0:
        raise RuntimeError("S33-A0 inherited S13 base block changed")

    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S33-A0 projection scorer missing")
    projected_q=F.normalize(scorer.projection(cqt),dim=-1)
    w=cqm.to(projected_q.dtype)[...,None]
    expected_q=F.normalize(
        (projected_q*w).sum(1)/w.sum(1).clamp_min(1.0),
        dim=-1,
    )
    query_anchor_error=float(
        (expected_q-cp["query_anchor"]).abs().max().cpu()
    )
    if query_anchor_error!=0.0:
        raise RuntimeError("S33-A0 query anchor formula changed")

    if tsc.shape[-1]!=256 or tsp.shape[-1]!=256:
        raise RuntimeError("S33-A0 treatment signature width changed")
    if cs.shape[-1]!=128 or ps.shape[-1]!=128:
        raise RuntimeError("S33-A0 base signature width changed")

    mass=max(
        float((torch.softmax(tc,dim=-1).sum(-1)-1).abs().max().cpu()),
        float((torch.softmax(tp,dim=-1).sum(-1)-1).abs().max().cpu()),
    )
    if mass>1e-6:
        raise RuntimeError("S33-A0 treatment probability mass changed")

    gold,_=_gold_tensors(rows,device=tc.device)
    treatment_acc=0.5*(
        float((tc.argmax(-1)==gold).float().mean().cpu())
        + float((tp.argmax(-1)==gold).float().mean().cpu())
    )
    same_cos=float(
        relation_signature_same_option_cosine(tsc,tsp).mean().cpu()
    )

    physical=HIRA_V1_S17_TOTAL_PARAMETER_COUNT
    if physical!=49_152:
        raise RuntimeError("S33-A0 physical surface changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S33-A0 frozen runtime became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S33-A0 HIRACore became trainable")
    original=sum(
        p.numel()
        for name,p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original!=0:
        raise RuntimeError("S33-A0 original A13 became trainable")

    # Zero-init checkpoint roundtrip.
    state=a13_lora_state_dict(runtime.encoder)
    replay_fresh=load_hira_v0_m4_bundle(bundle)
    replay=build_hira_v1_s17_norm_balanced_core(
        replay_fresh.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    del replay_fresh
    load_a13_lora_state_dict(replay.encoder,state,freeze=True)
    replay_state=a13_lora_state_dict(replay.encoder)
    checkpoint_roundtrip=all(
        torch.equal(state[k],replay_state[k]) for k in state
    )
    if not checkpoint_roundtrip:
        raise RuntimeError("S33-A0 LoRA checkpoint roundtrip changed")

    intervention=_controlled_question_intervention()
    gradient=_gradient_probe(bundle,manifest,rows)

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S33_A0_QUERY_EXPLICIT_RELATION_ONLY",
        "semantic_case_count":len(suite),
        "state_view_count":2*len(suite),
        "k":4,
        "views_per_option":2,
        "base_relation_logit_identity_max_abs":base_logit_error,
        "base_relation_signature_identity_max_abs":base_signature_error,
        "query_anchor_reference_max_abs":query_anchor_error,
        "base_signature_dimension":128,
        "query_option_block_dimension":128,
        "treatment_signature_dimension":256,
        "base_score_weight":0.50,
        "query_option_score_weight":0.50,
        "operator_added_parameter_count":0,
        "physical_trainable_parameter_count":physical,
        "lora_parameter_count":16_384,
        "projection_parameter_count":HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
        "runtime_trainable_parameter_count":0,
        "original_a13_trainable_parameter_count":original,
        "hira_core_trainable_parameter_count":0,
        "checkpoint_key_count":len(state),
        "checkpoint_roundtrip_exact":checkpoint_roundtrip,
        "treatment_probability_mass_error":mass,
        "full_k":tc.shape[-1]==4 and tp.shape[-1]==4,
        "state_once_expected_view_count":2*len(suite),
        "treatment_a0_relation_accuracy":treatment_acc,
        "treatment_a0_same_option_signature_cosine":same_cos,
        **intervention,
        "gradient":gradient,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S33_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
