from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_fullblock_lora import (
    ATTENTION_LORA_PARAMETER_COUNT,
    FFN_INTERMEDIATE_LORA_PARAMETER_COUNT,
    FFN_OUTPUT_LORA_PARAMETER_COUNT,
    FULL_BLOCK_LORA_PARAMETER_COUNT,
    a13_full_block_lora_state_dict,
    iter_a13_full_block_lora_modules,
    load_a13_full_block_lora_state_dict,
)
from nmd.v1_norm_balanced_gradient import norm_balanced_gradient_update
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import build_hira_v1_s17_norm_balanced_core
from nmd.v1_s29_semantic_core import (
    HIRA_V1_S29_LORA_PARAMETER_COUNT,
    HIRA_V1_S29_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S29_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s29_full_block_lora_core,
)
from hira_v1_s17_a0_identity import (
    _collect_decisions,
    _collect_fusion,
    _text_bank,
)
from hira_v1_s17_train_dev import _losses as _s17_losses

SCHEMA_VERSION = "hira-v1-s29-a0-full-block-lora-v1"
OUTCOME = "HIRA_V1_S29_A0_FULL_BLOCK_LORA_READY"


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
    def state_a(self) -> str:
        return (
            f"S29-A0 {self.noun} record {self.case_id}: "
            f"{self.field_a} = {self.first}; {self.field_b} = {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"Record {self.case_id} stores {self.second} under {self.field_b}. "
            f"The same S29-A0 {self.noun} lists {self.first} for {self.field_a}."
        )

    @property
    def qa1(self) -> str:
        return f"For S29-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self) -> str:
        return f"Which value is entered for {self.field_a} in record {self.case_id}?"

    @property
    def qb1(self) -> str:
        return f"For S29-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self) -> str:
        return f"Which value is entered for {self.field_b} in record {self.case_id}?"

    def option_pack(self):
        rows = [
            ("a", self.field_a, self.first),
            ("b", self.field_b, self.second),
            ("x", self.field_a, self.wrong_a),
            ("y", self.field_b, self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        options = tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"for this {self.noun}, {field} is {value}",
                aliases=(
                    f"{value} is the recorded {field} value for this {self.noun}",
                ),
            )
            for i, (_kind, field, value) in enumerate(rows)
        )
        ga = next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb = next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options, ga, gb


def cases() -> tuple[Case, ...]:
    return (
        Case("FB11","ion trap controller","RF amplitude","210 V","secular frequency","1.8 MHz","80 V","0.6 MHz",50101),
        Case("FB22","Raman spectrometer","laser power","42 mW","integration","6 s","12 mW","1 s",50102),
        Case("FB33","vacuum gauge","filament mode","hot","emission current","0.8 mA","cold","0.2 mA",50103),
        Case("FB44","optical cavity","mirror spacing","48 mm","lock bandwidth","24 kHz","20 mm","5 kHz",50104),
        Case("FB55","microheater array","drive voltage","3.6 V","duty cycle","72 percent","1.2 V","20 percent",50105),
        Case("FB66","laser vibrometer","velocity range","20 mm/s","filter cutoff","8 kHz","5 mm/s","1 kHz",50106),
        Case("FB77","mass comparator","capacity","2 kg","resolution","1 ug","10 kg","50 ug",50107),
        Case("FB88","electron multiplier","bias","1.9 kV","gain mode","high","0.7 kV","low",50108),
        Case("FC11","thermal chamber","ramp","4 C/min","soak","35 min","1 C/min","10 min",50109),
        Case("FC22","photon source","pump wavelength","405 nm","pulse rate","80 MHz","532 nm","10 MHz",50110),
        Case("FC33","micropositioner","travel","25 mm","step size","50 nm","5 mm","500 nm",50111),
        Case("FC44","plasma source","gas","argon","forward power","320 W","helium","90 W",50112),
        Case("FC55","lock-in amplifier","time constant","300 ms","sensitivity","20 uV","30 ms","2 mV",50113),
        Case("FC66","optical isolator","center wavelength","1550 nm","isolation","42 dB","1064 nm","18 dB",50114),
        Case("FC77","ultrasonic pulser","pulse voltage","180 V","PRF","4 kHz","60 V","500 Hz",50115),
        Case("FC88","cryostat controller","sensor","RuO2","setpoint","1.8 K","Pt100","20 K",50116),
    )


def _rows(suite: tuple[Case, ...]) -> list[S17FusionCase]:
    rows=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        rows.append(S17FusionCase(
            case_id=f"a0-s29-{case.case_id}",
            split="train",
            domain="s29_a0_only",
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


def _frozen_runtime(bundle: Path, manifest: dict, *, s29: bool):
    fresh=load_hira_v0_m4_bundle(bundle)
    builder=(
        build_hira_v1_s29_full_block_lora_core
        if s29 else build_hira_v1_s17_norm_balanced_core
    )
    runtime=builder(
        fresh.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    del fresh
    runtime.clear_schema_cache()
    runtime.eval()
    return runtime


def _tensor_identity(baseline, candidate, rows):
    with torch.inference_mode():
        b=_s17_losses(baseline, rows)
        c=_s17_losses(candidate, rows)
    names=("fused_c","fused_p","raw_c","raw_p","relation_c","relation_p","signature_c","signature_p")
    identity={}
    for name,index in zip(names,range(4,12)):
        diff=float((b[index]-c[index]).abs().max().cpu())
        identity[name+"_max_abs"]=diff
        if diff != 0.0:
            raise RuntimeError(f"S29-A0 S17 identity changed: {name}")
    encoded_b=b[12]
    encoded_c=c[12]
    for key in (
        "state_a_tokens","state_b_tokens",
        "question_canonical_tokens","question_paraphrase_tokens",
        "option_tokens","option_pooled",
    ):
        diff=float((encoded_b[key]-encoded_c[key]).abs().max().cpu())
        identity["encoded_"+key+"_max_abs"]=diff
        if diff != 0.0:
            raise RuntimeError(f"S29-A0 encoded identity changed: {key}")
    return identity


def _gradient_probe(bundle: Path, manifest: dict, rows):
    with torch.inference_mode(False), torch.enable_grad():
        fresh=load_hira_v0_m4_bundle(bundle)
        runtime=build_hira_v1_s29_full_block_lora_core(
            fresh.runtime.encoder,
            bundle / str(manifest["t0_checkpoint"]),
            expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
            train_lora=True,
            train_projection=True,
        )
        del fresh
        runtime.clear_schema_cache()
        runtime.eval()
        trainable=[p for p in runtime.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable) != HIRA_V1_S29_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S29-A0 trainable surface changed")
        _total,primary_block,relation_block,*_rest=_s17_losses(runtime,rows)
        pg=torch.autograd.grad(primary_block,trainable,retain_graph=True,allow_unused=True)
        rg=torch.autograd.grad(relation_block,trainable,allow_unused=True)
        primary=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,pg)]
        relation=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,rg)]
        combined,diag=norm_balanced_gradient_update(primary,relation,epsilon=1e-12)
        index={id(p):i for i,p in enumerate(trainable)}
        modules=iter_a13_full_block_lora_modules(runtime.encoder)
        b_l1=[]
        for module in modules:
            i=index[id(module.lora_b)]
            b_l1.append(float(combined[i].abs().sum().cpu()))
        scorer=runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S29-A0 projection scorer missing")
        projection_l1=float(combined[index[id(scorer.projection.weight)]].abs().sum().cpu())
        if max(b_l1[:4]) <= 0.0:
            raise RuntimeError("S29-A0 attention LoRA B gradient vanished")
        if b_l1[4] <= 0.0:
            raise RuntimeError("S29-A0 FFN intermediate LoRA B gradient vanished")
        if b_l1[5] <= 0.0:
            raise RuntimeError("S29-A0 FFN output LoRA B gradient vanished")
        if projection_l1 <= 0.0:
            raise RuntimeError("S29-A0 projection gradient vanished")
        if diag.primary_norm <= 0.0 or diag.relation_norm <= 0.0:
            raise RuntimeError("S29-A0 S17 gradient partition vanished")
        payload=diag.to_dict()
        return {
            "attention_lora_b_gradient_l1_max": max(b_l1[:4]),
            "attention_lora_b_gradient_l1": b_l1[:4],
            "ffn_intermediate_lora_b_gradient_l1": b_l1[4],
            "ffn_output_lora_b_gradient_l1": b_l1[5],
            "projection_gradient_l1": projection_l1,
            "primary_block_gradient_norm": diag.primary_norm,
            "relation_block_gradient_norm": diag.relation_norm,
            "normalized_pre_dot": diag.normalized_pre_dot,
            "normalized_post_dot": diag.normalized_post_dot,
            "projection_coefficient": diag.projection_coefficient,
            "combined_norm": diag.combined_norm,
            "balance_diagnostics": payload,
        }


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    suite=cases()
    if len(suite)!=16:
        raise RuntimeError("S29-A0 suite size changed")
    rows=_rows(suite)
    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    baseline=_frozen_runtime(bundle,manifest,s29=False)
    candidate=_frozen_runtime(bundle,manifest,s29=True)

    bank=_text_bank(suite)
    bbatch=baseline.encoder.encode_texts(bank)
    cbatch=candidate.encoder.encode_texts(bank)
    token_error=float((bbatch.token_embeddings-cbatch.token_embeddings).abs().max().cpu())
    pooled_error=float((bbatch.pooled_embeddings-cbatch.pooled_embeddings).abs().max().cpu())
    if token_error!=0.0 or pooled_error!=0.0:
        raise RuntimeError("S29-A0 A13 output identity changed")

    identity=_tensor_identity(baseline,candidate,rows)
    base_dec=_collect_decisions(baseline,suite,"projection_triadic")
    cand_dec=_collect_decisions(candidate,suite,"projection_triadic")
    exact_logits=exact_choices=0
    for key,(b_logits,b_choice) in base_dec["records"].items():
        c_logits,c_choice=cand_dec["records"][key]
        same_logits=torch.equal(b_logits,c_logits)
        same_choice=b_choice==c_choice
        exact_logits+=int(same_logits)
        exact_choices+=int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S29-A0 decision identity changed: {key}")

    fusion=_collect_fusion(candidate,suite)
    modules=iter_a13_full_block_lora_modules(candidate.encoder)
    lora_params=sum(
        p.numel() for m in modules for p in (m.lora_a,m.lora_b)
    )
    scorer=candidate.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S29-A0 projection scorer missing")
    projection_params=scorer.projection_parameter_count
    physical=lora_params+projection_params
    trainable=sum(p.numel() for p in candidate.parameters() if p.requires_grad)
    original_a13=sum(
        p.numel() for name,p in candidate.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    hira_trainable=sum(p.numel() for p in candidate.hira.parameters() if p.requires_grad)
    if lora_params!=HIRA_V1_S29_LORA_PARAMETER_COUNT:
        raise RuntimeError("S29-A0 LoRA capacity changed")
    if projection_params!=HIRA_V1_S29_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S29-A0 projection capacity changed")
    if physical!=HIRA_V1_S29_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S29-A0 total capacity changed")
    if trainable!=0 or original_a13!=0 or hira_trainable!=0:
        raise RuntimeError("S29-A0 frozen surface changed")

    module_counts=[m.lora_parameter_count for m in modules]
    expected_counts=[4096,4096,4096,4096,FFN_INTERMEDIATE_LORA_PARAMETER_COUNT,FFN_OUTPUT_LORA_PARAMETER_COUNT]
    if module_counts!=expected_counts:
        raise RuntimeError("S29-A0 six-module parameter layout changed")

    state=a13_full_block_lora_state_dict(candidate.encoder)
    if len(state)!=12:
        raise RuntimeError("S29-A0 full-block checkpoint key count changed")
    replay=_frozen_runtime(bundle,manifest,s29=True)
    load_a13_full_block_lora_state_dict(replay.encoder,state,freeze=True)
    replay_state=a13_full_block_lora_state_dict(replay.encoder)
    checkpoint_roundtrip=all(torch.equal(state[k],replay_state[k]) for k in state)
    if not checkpoint_roundtrip:
        raise RuntimeError("S29-A0 six-module checkpoint roundtrip changed")

    gradient=_gradient_probe(bundle,manifest,rows)
    if cand_dec["state_encode_calls"]!=len(suite)*2:
        raise RuntimeError("S29-A0 state-once changed")
    if cand_dec["option_order_flip_rate"]!=0.0:
        raise RuntimeError("S29-A0 primary option order changed")
    if cand_dec["max_probability_mass_error"]>1e-6:
        raise RuntimeError("S29-A0 probability mass changed")
    if fusion["fused_option_order_flip_rate"]!=0.0:
        raise RuntimeError("S29-A0 fused option order changed")
    if fusion["fused_max_probability_mass_error"]>1e-6:
        raise RuntimeError("S29-A0 fused probability mass changed")

    total_decisions=len(suite)*4
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S29_A0_FINAL_BLOCK_FFN_LORA_ONLY",
        "semantic_case_count":len(suite),
        "decision_count":total_decisions,
        "state_view_count":len(suite)*2,
        "language":"en",
        "k":4,
        "views_per_option":2,
        "a13_token_output_identity_max_abs":token_error,
        "a13_pooled_output_identity_max_abs":pooled_error,
        "exact_logit_identity_rate":exact_logits/total_decisions,
        "exact_choice_identity_rate":exact_choices/total_decisions,
        **identity,
        "attention_lora_parameter_count":ATTENTION_LORA_PARAMETER_COUNT,
        "ffn_intermediate_lora_parameter_count":FFN_INTERMEDIATE_LORA_PARAMETER_COUNT,
        "ffn_output_lora_parameter_count":FFN_OUTPUT_LORA_PARAMETER_COUNT,
        "ffn_lora_parameter_count":FFN_INTERMEDIATE_LORA_PARAMETER_COUNT+FFN_OUTPUT_LORA_PARAMETER_COUNT,
        "lora_parameter_count":lora_params,
        "projection_parameter_count":projection_params,
        "candidate_parameter_count":physical,
        "runtime_trainable_parameter_count":trainable,
        "original_a13_trainable_parameter_count":original_a13,
        "hira_core_trainable_parameter_count":hira_trainable,
        "six_module_parameter_counts":module_counts,
        "checkpoint_key_count":len(state),
        "checkpoint_roundtrip_exact":checkpoint_roundtrip,
        "state_encode_calls":cand_dec["state_encode_calls"],
        "option_order_flip_rate":cand_dec["option_order_flip_rate"],
        "max_probability_mass_error":cand_dec["max_probability_mass_error"],
        **fusion,
        **gradient,
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
    print("HIRA_V1_S29_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
