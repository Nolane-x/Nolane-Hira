from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_ffn_lora import (
    FFN_ONLY_LORA_PARAMETER_COUNT,
    a13_ffn_only_lora_state_dict,
    iter_a13_ffn_only_lora_modules,
    load_a13_ffn_only_lora_state_dict,
)
from nmd.v1_a13_lora import (
    a13_lora_state_dict,
    iter_a13_lora_modules,
    load_a13_lora_state_dict,
)
from nmd.v1_norm_balanced_gradient import norm_balanced_gradient_update
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s30_semantic_core import (
    S30_ATTENTION_LORA_PARAMETER_COUNT,
    S30_ATTENTION_TOTAL_PARAMETER_COUNT,
    S30_FFN_TOTAL_PARAMETER_COUNT,
    S30_PROJECTION_PARAMETER_COUNT,
    build_s30_attention_arm,
    build_s30_ffn_arm,
)
from hira_v1_s17_a0_identity import _collect_decisions, _collect_fusion, _text_bank
from hira_v1_s17_train_dev import _losses as _s17_losses

SCHEMA_VERSION="hira-v1-s30-a0-matched-adaptation-v1"
OUTCOME="HIRA_V1_S30_A0_MATCHED_ADAPTATION_READY"


@dataclass(frozen=True)
class Case:
    case_id: str
    noun: str
    field_a: str
    first: str
    field_b: str
    second: str
    wrong_a: str
    wrong_b: str
    seed: int

    @property
    def state_a(self):
        return (
            f"S30-A0 {self.noun} record {self.case_id}: "
            f"{self.field_a} = {self.first}; {self.field_b} = {self.second}."
        )

    @property
    def state_b(self):
        return (
            f"Record {self.case_id} stores {self.second} under {self.field_b}. "
            f"The same S30-A0 {self.noun} lists {self.first} for {self.field_a}."
        )

    @property
    def qa1(self):
        return f"For S30-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which value is entered for {self.field_a} in S30-A0 record {self.case_id}?"

    @property
    def qb1(self):
        return f"For S30-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which value is entered for {self.field_b} in S30-A0 record {self.case_id}?"

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


def cases() -> tuple[Case,...]:
    return (
        Case("MA11","optical phase meter","reference arm","short","phase span","240 deg","long","90 deg",55101),
        Case("MA22","vacuum spectrometer","grating","1200 l/mm","slit width","40 um","600 l/mm","120 um",55102),
        Case("MA33","piezo controller","drive mode","bipolar","slew rate","18 V/us","unipolar","5 V/us",55103),
        Case("MA44","photon counter","detector","Si SPAD","dead time","35 ns","PMT","120 ns",55104),
        Case("MA55","magnetic trap supply","coil mode","anti-Helmholtz","current","42 A","Helmholtz","15 A",55105),
        Case("MA66","microbalance","sensor","QCM","sample period","200 ms","strain gauge","1 s",55106),
        Case("MA77","optical chopper","blade","30-slot","rotation","7200 rpm","10-slot","1800 rpm",55107),
        Case("MA88","plasma chamber","electrode","showerhead","pressure","12 Pa","plate","4 Pa",55108),
        Case("MB11","RF power meter","sensor","thermocouple","range","20 dBm","diode","5 dBm",55109),
        Case("MB22","nanopositioner","axis drive","piezo stack","travel","180 um","voice coil","60 um",55110),
        Case("MB33","cryogenic amplifier","device","HEMT","bias current","8 mA","BJT","2 mA",55111),
        Case("MB44","laser cavity lock","actuator","PZT","bandwidth","40 kHz","heater","2 kHz",55112),
        Case("MB55","ion gauge","collector","Faraday cup","emission","1.2 mA","channeltron","0.3 mA",55113),
        Case("MB66","spectral filter","filter type","etalon","FSR","18 GHz","grating","4 GHz",55114),
        Case("MB77","ultrafast detector","photodiode","GaAs","rise time","25 ps","silicon","200 ps",55115),
        Case("MB88","beam profiler","sensor","CMOS","pixel pitch","3.45 um","CCD","9 um",55116),
    )


def _rows(suite):
    rows=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        rows.append(S17FusionCase(
            case_id=f"a0-s30-{case.case_id}",
            split="train",
            domain="s30_a0_only",
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


def _build_frozen(bundle,manifest,arm):
    fresh=load_hira_v0_m4_bundle(bundle)
    builder=build_s30_attention_arm if arm=="attention" else build_s30_ffn_arm
    runtime=builder(
        fresh.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    del fresh
    runtime.clear_schema_cache()
    runtime.eval()
    return runtime


def _loss_identity(attention,ffn,rows):
    with torch.inference_mode():
        a=_s17_losses(attention,rows)
        b=_s17_losses(ffn,rows)
    names=("fused_c","fused_p","raw_c","raw_p","relation_c","relation_p","signature_c","signature_p")
    out={}
    for name,index in zip(names,range(4,12)):
        diff=float((a[index]-b[index]).abs().max().cpu())
        out[name+"_max_abs"]=diff
        if diff!=0.0:
            raise RuntimeError(f"S30-A0 arm identity changed: {name}")
    ea,eb=a[12],b[12]
    for key in (
        "state_a_tokens","state_b_tokens",
        "question_canonical_tokens","question_paraphrase_tokens",
        "option_tokens","option_pooled",
    ):
        diff=float((ea[key]-eb[key]).abs().max().cpu())
        out["encoded_"+key+"_max_abs"]=diff
        if diff!=0.0:
            raise RuntimeError(f"S30-A0 encoded arm identity changed: {key}")
    return out


def _gradient_probe(bundle,manifest,rows,arm):
    with torch.inference_mode(False),torch.enable_grad():
        fresh=load_hira_v0_m4_bundle(bundle)
        builder=build_s30_attention_arm if arm=="attention" else build_s30_ffn_arm
        runtime=builder(
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
        expected=S30_ATTENTION_TOTAL_PARAMETER_COUNT if arm=="attention" else S30_FFN_TOTAL_PARAMETER_COUNT
        if sum(p.numel() for p in trainable)!=expected:
            raise RuntimeError(f"S30-A0 {arm} trainable surface changed")
        _total,primary_block,relation_block,*_rest=_s17_losses(runtime,rows)
        pg=torch.autograd.grad(primary_block,trainable,retain_graph=True,allow_unused=True)
        rg=torch.autograd.grad(relation_block,trainable,allow_unused=True)
        primary=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,pg)]
        relation=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,rg)]
        combined,diag=norm_balanced_gradient_update(primary,relation,epsilon=1e-12)
        index={id(p):i for i,p in enumerate(trainable)}
        modules=(
            iter_a13_lora_modules(runtime.encoder)
            if arm=="attention"
            else iter_a13_ffn_only_lora_modules(runtime.encoder)
        )
        b_l1=[
            float(combined[index[id(module.lora_b)]].abs().sum().cpu())
            for module in modules
        ]
        scorer=runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S30-A0 projection scorer missing")
        projection_l1=float(
            combined[index[id(scorer.projection.weight)]].abs().sum().cpu()
        )
        if any(v<=0.0 for v in b_l1):
            raise RuntimeError(f"S30-A0 {arm} LoRA B gradient vanished")
        if projection_l1<=0.0:
            raise RuntimeError(f"S30-A0 {arm} projection gradient vanished")
        if diag.primary_norm<=0.0 or diag.relation_norm<=0.0:
            raise RuntimeError(f"S30-A0 {arm} gradient partition vanished")
        return {
            "arm":arm,
            "lora_b_gradient_l1":b_l1,
            "projection_gradient_l1":projection_l1,
            "primary_block_gradient_norm":diag.primary_norm,
            "relation_block_gradient_norm":diag.relation_norm,
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
        raise RuntimeError("S30-A0 suite size changed")
    rows=_rows(suite)
    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    attention=_build_frozen(bundle,manifest,"attention")
    ffn=_build_frozen(bundle,manifest,"ffn")

    bank=_text_bank(suite)
    ab=attention.encoder.encode_texts(bank)
    fb=ffn.encoder.encode_texts(bank)
    token_error=float((ab.token_embeddings-fb.token_embeddings).abs().max().cpu())
    pooled_error=float((ab.pooled_embeddings-fb.pooled_embeddings).abs().max().cpu())
    if token_error!=0.0 or pooled_error!=0.0:
        raise RuntimeError("S30-A0 arm encoder identity changed")

    identity=_loss_identity(attention,ffn,rows)

    adec=_collect_decisions(attention,suite,"projection_triadic")
    fdec=_collect_decisions(ffn,suite,"projection_triadic")
    exact_logits=exact_choices=0
    for key,(alogits,achoice) in adec["records"].items():
        flogits,fchoice=fdec["records"][key]
        same_logits=torch.equal(alogits,flogits)
        same_choice=achoice==fchoice
        exact_logits+=int(same_logits)
        exact_choices+=int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S30-A0 decision arm identity changed: {key}")

    afusion=_collect_fusion(attention,suite)
    ffusion=_collect_fusion(ffn,suite)
    for key in afusion:
        av=afusion[key]
        fv=ffusion[key]
        if isinstance(av,(int,float)) and float(av)!=float(fv):
            raise RuntimeError(f"S30-A0 fusion arm identity changed: {key}")

    amods=iter_a13_lora_modules(attention.encoder)
    fmods=iter_a13_ffn_only_lora_modules(ffn.encoder)
    a_lora=sum(p.numel() for m in amods for p in (m.lora_a,m.lora_b))
    f_lora=sum(p.numel() for m in fmods for p in (m.lora_a,m.lora_b))
    ascorer=attention.projection_triadic_scorer
    fscorer=ffn.projection_triadic_scorer
    if ascorer is None or fscorer is None:
        raise RuntimeError("S30-A0 projection scorer missing")
    if a_lora!=S30_ATTENTION_LORA_PARAMETER_COUNT:
        raise RuntimeError("S30-A0 attention capacity changed")
    if f_lora!=FFN_ONLY_LORA_PARAMETER_COUNT:
        raise RuntimeError("S30-A0 FFN capacity changed")
    if ascorer.projection_parameter_count!=S30_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S30-A0 attention projection capacity changed")
    if fscorer.projection_parameter_count!=S30_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S30-A0 FFN projection capacity changed")

    a_trainable=sum(p.numel() for p in attention.parameters() if p.requires_grad)
    f_trainable=sum(p.numel() for p in ffn.parameters() if p.requires_grad)
    if a_trainable!=0 or f_trainable!=0:
        raise RuntimeError("S30-A0 frozen runtimes became trainable")
    a_original=sum(
        p.numel() for name,p in attention.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    f_original=sum(
        p.numel() for name,p in ffn.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if a_original!=0 or f_original!=0:
        raise RuntimeError("S30-A0 original A13 became trainable")
    if any(p.requires_grad for p in attention.hira.parameters()):
        raise RuntimeError("S30-A0 attention HIRACore became trainable")
    if any(p.requires_grad for p in ffn.hira.parameters()):
        raise RuntimeError("S30-A0 FFN HIRACore became trainable")

    astate=a13_lora_state_dict(attention.encoder)
    areplay=_build_frozen(bundle,manifest,"attention")
    load_a13_lora_state_dict(areplay.encoder,astate,freeze=True)
    a_roundtrip=all(
        torch.equal(astate[k],a13_lora_state_dict(areplay.encoder)[k])
        for k in astate
    )
    fstate=a13_ffn_only_lora_state_dict(ffn.encoder)
    freplay=_build_frozen(bundle,manifest,"ffn")
    load_a13_ffn_only_lora_state_dict(freplay.encoder,fstate,freeze=True)
    f_roundtrip=all(
        torch.equal(fstate[k],a13_ffn_only_lora_state_dict(freplay.encoder)[k])
        for k in fstate
    )
    if not a_roundtrip or not f_roundtrip:
        raise RuntimeError("S30-A0 checkpoint roundtrip changed")

    agrad=_gradient_probe(bundle,manifest,rows,"attention")
    fgrad=_gradient_probe(bundle,manifest,rows,"ffn")

    for label,dec in (("attention",adec),("ffn",fdec)):
        if dec["state_encode_calls"]!=len(suite)*2:
            raise RuntimeError(f"S30-A0 {label} state-once changed")
        if dec["option_order_flip_rate"]!=0.0:
            raise RuntimeError(f"S30-A0 {label} option order changed")
        if dec["max_probability_mass_error"]>1e-6:
            raise RuntimeError(f"S30-A0 {label} probability mass changed")

    total_decisions=len(suite)*4
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S30_A0_MATCHED_ATTENTION_VS_FFN_ONLY",
        "semantic_case_count":len(suite),
        "decision_count":total_decisions,
        "state_view_count":len(suite)*2,
        "language":"en",
        "k":4,
        "views_per_option":2,
        "arm_token_output_identity_max_abs":token_error,
        "arm_pooled_output_identity_max_abs":pooled_error,
        "exact_logit_identity_rate":exact_logits/total_decisions,
        "exact_choice_identity_rate":exact_choices/total_decisions,
        **identity,
        "attention_arm":{
            "lora_parameter_count":a_lora,
            "projection_parameter_count":ascorer.projection_parameter_count,
            "physical_parameter_count":a_lora+ascorer.projection_parameter_count,
            "runtime_trainable_parameter_count":a_trainable,
            "original_a13_trainable_parameter_count":a_original,
            "hira_core_trainable_parameter_count":sum(p.numel() for p in attention.hira.parameters() if p.requires_grad),
            "checkpoint_key_count":len(astate),
            "checkpoint_roundtrip_exact":a_roundtrip,
            "state_encode_calls":adec["state_encode_calls"],
            "option_order_flip_rate":adec["option_order_flip_rate"],
            "max_probability_mass_error":adec["max_probability_mass_error"],
            "fusion":afusion,
            "gradient":agrad,
        },
        "ffn_arm":{
            "lora_parameter_count":f_lora,
            "projection_parameter_count":fscorer.projection_parameter_count,
            "physical_parameter_count":f_lora+fscorer.projection_parameter_count,
            "runtime_trainable_parameter_count":f_trainable,
            "original_a13_trainable_parameter_count":f_original,
            "hira_core_trainable_parameter_count":sum(p.numel() for p in ffn.hira.parameters() if p.requires_grad),
            "checkpoint_key_count":len(fstate),
            "checkpoint_roundtrip_exact":f_roundtrip,
            "state_encode_calls":fdec["state_encode_calls"],
            "option_order_flip_rate":fdec["option_order_flip_rate"],
            "max_probability_mass_error":fdec["max_probability_mass_error"],
            "fusion":ffusion,
            "gradient":fgrad,
        },
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
    print("HIRA_V1_S30_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
