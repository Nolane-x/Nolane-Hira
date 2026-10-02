from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import (
    a13_lora_state_dict,
    iter_a13_lora_modules,
    load_a13_lora_state_dict,
)
from nmd.v1_norm_balanced_gradient import norm_balanced_gradient_update
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
)
from nmd.v1_s31_global_relation_contrastive import (
    global_cross_case_relation_contrastive_loss,
)
from nmd.v1_s31_training import s31_global_relation_block
from nmd.v1_s32_global_relation_matrix import (
    global_query_option_relation_contrastive_loss,
)
from nmd.v1_s32_training import (
    S32_GLOBAL_CANONICALIZATION_COEFFICIENT,
    S32_GLOBAL_TEMPERATURE,
    s32_all_option_relation_block,
)
from hira_v1_s17_a0_identity import _collect_decisions, _collect_fusion, _text_bank
from hira_v1_s17_train_dev import _gold_tensors, _losses

SCHEMA_VERSION="hira-v1-s32-a0-global-query-option-matrix-v1"
OUTCOME="HIRA_V1_S32_A0_GLOBAL_QUERY_OPTION_MATRIX_READY"
TEMPERATURE=S32_GLOBAL_TEMPERATURE


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
            f"S32-A0 {self.noun} record {self.case_id}: "
            f"{self.field_a} = {self.first}; {self.field_b} = {self.second}."
        )

    @property
    def state_b(self):
        return (
            f"Record {self.case_id} stores {self.second} under {self.field_b}. "
            f"The same S32-A0 {self.noun} lists {self.first} for {self.field_a}."
        )

    @property
    def qa1(self): return f"For S32-A0 record {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which value is entered for {self.field_a} in S32-A0 record {self.case_id}?"
    @property
    def qb1(self): return f"For S32-A0 record {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which value is entered for {self.field_b} in S32-A0 record {self.case_id}?"

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


def cases()->tuple[Case,...]:
    return (
        Case("QA11","optical gyroscope","coil fiber","PM1550","bias phase","42 deg","SMF28","12 deg",61101),
        Case("QA22","microcantilever readout","coating","gold","resonance","18 kHz","aluminum","6 kHz",61102),
        Case("QA33","frequency counter","reference","OCXO","gate time","250 ms","TCXO","20 ms",61103),
        Case("QA44","UV photometer","detector","SiC","integration","40 ms","silicon","5 ms",61104),
        Case("QA55","ion trap driver","waveform","RF","amplitude","180 Vpp","DC","30 Vpp",61105),
        Case("QA66","microfluidic mixer","geometry","herringbone","flow","24 uL/min","straight","6 uL/min",61106),
        Case("QA77","THz detector","sensor","bolometer","bandwidth","2.4 THz","Schottky","0.4 THz",61107),
        Case("QA88","precision thermometer","element","Cernox","excitation","12 uA","Pt100","1 mA",61108),
        Case("QB11","optical encoder","code","absolute","resolution","20 bit","incremental","12 bit",61109),
        Case("QB22","laser scanner","mirror","galvo","scan rate","28 kHz","polygon","8 kHz",61110),
        Case("QB33","electron multiplier","dynode","channel plate","gain","1e7","discrete","1e4",61111),
        Case("QB44","acoustic source","transducer","CMUT","frequency","18 MHz","speaker","40 kHz",61112),
        Case("QB55","cryogenic mux","switch","SQUID","channels","64","relay","8",61113),
        Case("QB66","spectral lamp","species","neon","current","14 mA","argon","4 mA",61114),
        Case("QB77","beam expander","design","Galilean","magnification","8x","Keplerian","2x",61115),
        Case("QB88","magnetic probe","sensor","fluxgate","range","6 mT","Hall","0.5 T",61116),
    )


def _rows(suite):
    rows=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        rows.append(S17FusionCase(
            case_id=f"a0-s32-{case.case_id}",
            split="train",
            domain="s32_a0_only",
            language="en",
            state_a=case.state_a,
            state_b=case.state_b,
            question_a1=case.qa1,
            question_a2=case.qa2,
            question_b1=case.qb1,
            question_b2=case.qb2,
            option_texts=tuple(o.criterion_text for o in options),
            option_aliases=tuple(o.aliases[0] for o in options),
            option_ids=tuple(o.option_id for o in options),
            gold_a=ga,
            gold_b=gb,
        ))
    return rows


def _runtime(bundle,manifest,train):
    fresh=load_hira_v0_m4_bundle(bundle)
    runtime=build_hira_v1_s17_norm_balanced_core(
        fresh.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=train,
        train_projection=train,
    )
    del fresh
    runtime.clear_schema_cache()
    runtime.eval()
    return runtime


def _treatment_parts(runtime,rows):
    (
        _total,primary_block,_local_relation,_pieces,
        fused_c,fused_p,raw_c,raw_p,relation_c,relation_p,
        signature_c,signature_p,encoded,
    )=_losses(runtime,rows)
    gold,_other=_gold_tensors(rows,device=relation_c.device)
    relation_block,pieces=s32_all_option_relation_block(
        relation_c,relation_p,signature_c,signature_p,gold
    )
    return {
        "primary_block":primary_block,
        "relation_block":relation_block,
        "global_loss":pieces["global_contrastive"],
        "gold":gold,
        "signature_c":signature_c,
        "signature_p":signature_p,
        "fused_c":fused_c,
        "fused_p":fused_p,
        "raw_c":raw_c,
        "raw_p":raw_p,
        "relation_c":relation_c,
        "relation_p":relation_p,
        "encoded":encoded,
    }


def _gradient_probe(bundle,manifest,rows):
    with torch.inference_mode(False),torch.enable_grad():
        runtime=_runtime(bundle,manifest,True)
        trainable=[p for p in runtime.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable)!=HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S32-A0 treatment surface changed")

        parts=_treatment_parts(runtime,rows)
        pg=torch.autograd.grad(
            parts["primary_block"],trainable,retain_graph=True,allow_unused=True
        )
        rg=torch.autograd.grad(
            parts["relation_block"],trainable,allow_unused=True
        )
        primary=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,pg)]
        relation=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,rg)]
        combined,diag=norm_balanced_gradient_update(primary,relation,epsilon=1e-12)

        idx={id(p):i for i,p in enumerate(trainable)}
        modules=iter_a13_lora_modules(runtime.encoder)
        b_l1=[
            float(combined[idx[id(m.lora_b)]].abs().sum().cpu())
            for m in modules
        ]
        scorer=runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S32-A0 scorer missing")
        projection_l1=float(
            combined[idx[id(scorer.projection.weight)]].abs().sum().cpu()
        )

        if any(v<=0 for v in b_l1) or projection_l1<=0:
            raise RuntimeError("S32-A0 intended gradients vanished")
        if diag.primary_norm<=0 or diag.relation_norm<=0:
            raise RuntimeError("S32-A0 gradient partition vanished")

        return {
            "lora_b_gradient_l1":b_l1,
            "projection_gradient_l1":projection_l1,
            "primary_norm":diag.primary_norm,
            "relation_norm":diag.relation_norm,
            "normalized_pre_dot":diag.normalized_pre_dot,
            "normalized_post_dot":diag.normalized_post_dot,
            "projection_coefficient":diag.projection_coefficient,
            "combined_norm":diag.combined_norm,
        }


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    suite=cases()
    if len(suite)!=16:
        raise RuntimeError("S32-A0 suite size changed")
    rows=_rows(suite)
    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    control=_runtime(bundle,manifest,False)
    treatment=_runtime(bundle,manifest,False)

    bank=_text_bank(suite)
    cb=control.encoder.encode_texts(bank)
    tb=treatment.encoder.encode_texts(bank)
    token_error=float((cb.token_embeddings-tb.token_embeddings).abs().max().cpu())
    pooled_error=float((cb.pooled_embeddings-tb.pooled_embeddings).abs().max().cpu())
    if token_error!=0.0 or pooled_error!=0.0:
        raise RuntimeError("S32-A0 encoder identity changed")

    with torch.inference_mode():
        c=_losses(control,rows)
        t=_losses(treatment,rows)
    names=(
        "fused_c","fused_p","raw_c","raw_p",
        "relation_c","relation_p","signature_c","signature_p",
    )
    identity={}
    for name,index in zip(names,range(4,12)):
        diff=float((c[index]-t[index]).abs().max().cpu())
        identity[name+"_max_abs"]=diff
        if diff!=0.0:
            raise RuntimeError(f"S32-A0 inference identity changed: {name}")

    cdec=_collect_decisions(control,suite,"projection_triadic")
    tdec=_collect_decisions(treatment,suite,"projection_triadic")
    exact_logits=exact_choices=0
    for key,(clogits,cchoice) in cdec["records"].items():
        tlogits,tchoice=tdec["records"][key]
        same_logits=torch.equal(clogits,tlogits)
        same_choice=cchoice==tchoice
        exact_logits+=int(same_logits)
        exact_choices+=int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S32-A0 decision identity changed: {key}")

    cfusion=_collect_fusion(control,suite)
    tfusion=_collect_fusion(treatment,suite)
    if cfusion!=tfusion:
        raise RuntimeError("S32-A0 fusion identity changed")

    real=_treatment_parts(treatment,rows)
    real_global=float(real["global_loss"].cpu())

    # Controlled operator discriminator.
    q,k,d=4,4,16
    canonical=torch.zeros(q,k,d)
    paraphrase=torch.zeros(q,k,d)
    for qi in range(q):
        for ki in range(k):
            canonical[qi,ki,qi*k+ki]=1.0
            paraphrase[qi,ki,qi*k+ki]=1.0
    gold=torch.zeros(q,dtype=torch.long)

    gold_good,_,_=global_cross_case_relation_contrastive_loss(
        canonical,paraphrase,gold,temperature=TEMPERATURE
    )
    all_good,_,_=global_query_option_relation_contrastive_loss(
        canonical,paraphrase,temperature=TEMPERATURE
    )

    perturbed=paraphrase.clone()
    perturbed[0,1]=paraphrase[0,2]

    gold_bad,_,_=global_cross_case_relation_contrastive_loss(
        canonical,perturbed,gold,temperature=TEMPERATURE
    )
    all_bad,_,_=global_query_option_relation_contrastive_loss(
        canonical,perturbed,temperature=TEMPERATURE
    )

    gold_non_gold_change=float((gold_bad-gold_good).abs())
    all_non_gold_increase=float(all_bad-all_good)
    if gold_non_gold_change!=0.0:
        raise RuntimeError("S32-A0 gold-only control reacted to non-gold perturbation")
    if all_non_gold_increase<0.10:
        raise RuntimeError("S32-A0 all-option non-gold sensitivity too weak")

    qperm=torch.tensor([2,0,3,1])
    kperm=torch.tensor([3,1,0,2])
    permuted,_,_=global_query_option_relation_contrastive_loss(
        canonical[qperm][:,kperm],
        paraphrase[qperm][:,kperm],
        temperature=TEMPERATURE,
    )
    swapped,_,_=global_query_option_relation_contrastive_loss(
        paraphrase,canonical,temperature=TEMPERATURE
    )
    permutation_error=float((all_good-permuted).abs())
    view_swap_error=float((all_good-swapped).abs())
    if permutation_error>1e-7:
        raise RuntimeError("S32-A0 matched query-option permutation changed loss")
    if view_swap_error>1e-7:
        raise RuntimeError("S32-A0 view swap changed loss")

    # Direct non-gold gradient discriminator.
    with torch.inference_mode(False),torch.enable_grad():
        torch.manual_seed(3203)
        gc=torch.randn(6,4,24,requires_grad=True)
        gp=torch.randn(6,4,24,requires_grad=True)
        gg=torch.zeros(6,dtype=torch.long)

        control_loss,_,_=global_cross_case_relation_contrastive_loss(
            gc,gp,gg,temperature=TEMPERATURE
        )
        cgc,cgp=torch.autograd.grad(
            control_loss,(gc,gp),retain_graph=True
        )
        mask=torch.ones(6,4,dtype=torch.bool)
        mask[:,0]=False
        control_non_gold_c=float(cgc[mask].abs().sum())
        control_non_gold_p=float(cgp[mask].abs().sum())
        if control_non_gold_c!=0.0 or control_non_gold_p!=0.0:
            raise RuntimeError("S32-A0 gold-only control leaked gradient to non-gold")

        treatment_loss,_,_=global_query_option_relation_contrastive_loss(
            gc,gp,temperature=TEMPERATURE
        )
        tgc,tgp=torch.autograd.grad(treatment_loss,(gc,gp))
        treatment_non_gold_c=float(tgc[mask].abs().sum())
        treatment_non_gold_p=float(tgp[mask].abs().sum())
        if treatment_non_gold_c<=0.0 or treatment_non_gold_p<=0.0:
            raise RuntimeError("S32-A0 all-option non-gold gradients vanished")
        if not bool(torch.isfinite(tgc).all() and torch.isfinite(tgp).all()):
            raise RuntimeError("S32-A0 all-option gradients non-finite")

    state=a13_lora_state_dict(control.encoder)
    replay=_runtime(bundle,manifest,False)
    load_a13_lora_state_dict(replay.encoder,state,freeze=True)
    replay_state=a13_lora_state_dict(replay.encoder)
    checkpoint_roundtrip=all(
        torch.equal(state[k],replay_state[k]) for k in state
    )
    if not checkpoint_roundtrip:
        raise RuntimeError("S32-A0 checkpoint roundtrip changed")

    grad=_gradient_probe(bundle,manifest,rows)

    total_decisions=len(suite)*4
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S32_A0_GOLD_ONLY_VS_ALL_QUERY_OPTION_GLOBAL",
        "semantic_case_count":len(suite),
        "decision_count":total_decisions,
        "state_view_count":len(suite)*2,
        "k":4,
        "views_per_option":2,
        "temperature":TEMPERATURE,
        "outer_coefficient":S32_GLOBAL_CANONICALIZATION_COEFFICIENT,
        "operator_parameter_count":0,
        "arm_token_output_identity_max_abs":token_error,
        "arm_pooled_output_identity_max_abs":pooled_error,
        "exact_logit_identity_rate":exact_logits/total_decisions,
        "exact_choice_identity_rate":exact_choices/total_decisions,
        **identity,
        "control_physical_parameter_count":HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        "treatment_physical_parameter_count":HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        "runtime_trainable_parameter_count":sum(
            p.numel() for p in control.parameters() if p.requires_grad
        ),
        "original_a13_trainable_parameter_count":sum(
            p.numel()
            for name,p in control.encoder.model.named_parameters()
            if p.requires_grad and ".lora_" not in name
        ),
        "hira_core_trainable_parameter_count":sum(
            p.numel() for p in control.hira.parameters() if p.requires_grad
        ),
        "checkpoint_key_count":len(state),
        "checkpoint_roundtrip_exact":checkpoint_roundtrip,
        "controlled_gold_good_loss":float(gold_good),
        "controlled_gold_non_gold_perturbed_loss":float(gold_bad),
        "controlled_gold_non_gold_change":gold_non_gold_change,
        "controlled_all_option_good_loss":float(all_good),
        "controlled_all_option_non_gold_perturbed_loss":float(all_bad),
        "controlled_all_option_non_gold_increase":all_non_gold_increase,
        "controlled_matched_query_option_permutation_error":permutation_error,
        "controlled_view_swap_error":view_swap_error,
        "controlled_gold_only_non_gold_canonical_gradient_l1":control_non_gold_c,
        "controlled_gold_only_non_gold_paraphrase_gradient_l1":control_non_gold_p,
        "controlled_all_option_non_gold_canonical_gradient_l1":treatment_non_gold_c,
        "controlled_all_option_non_gold_paraphrase_gradient_l1":treatment_non_gold_p,
        "real_semantic_all_option_loss":real_global,
        "gradient":grad,
        "state_encode_calls":cdec["state_encode_calls"],
        "option_order_flip_rate":cdec["option_order_flip_rate"],
        "max_probability_mass_error":cdec["max_probability_mass_error"],
        "fusion":cfusion,
        "full_k":True,
        "relation_refinement":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }

    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S32_A0_RECEIPT="+json.dumps(result,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    main()
