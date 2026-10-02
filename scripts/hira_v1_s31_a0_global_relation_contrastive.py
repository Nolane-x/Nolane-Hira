from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle,read_runtime_bundle_manifest
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
from nmd.v1_s31_training import (
    S31_GLOBAL_CANONICALIZATION_COEFFICIENT,
    S31_GLOBAL_TEMPERATURE,
    s31_global_relation_block,
)
from hira_v1_s17_a0_identity import _collect_decisions,_collect_fusion,_text_bank
from hira_v1_s17_train_dev import (
    BINDING_COEFFICIENT,
    CANONICALIZATION_COEFFICIENT,
    _gold_tensors,
    _losses,
)

SCHEMA_VERSION="hira-v1-s31-a0-global-relation-contrastive-v1"
OUTCOME="HIRA_V1_S31_A0_GLOBAL_RELATION_CONTRASTIVE_READY"
TEMPERATURE=S31_GLOBAL_TEMPERATURE


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
        return f"S31-A0 {self.noun} record {self.case_id}: {self.field_a} = {self.first}; {self.field_b} = {self.second}."

    @property
    def state_b(self):
        return f"Record {self.case_id} stores {self.second} under {self.field_b}. The same S31-A0 {self.noun} lists {self.first} for {self.field_a}."

    @property
    def qa1(self): return f"For S31-A0 record {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which value is entered for {self.field_a} in S31-A0 record {self.case_id}?"
    @property
    def qb1(self): return f"For S31-A0 record {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which value is entered for {self.field_b} in S31-A0 record {self.case_id}?"

    def option_pack(self):
        rows=[("a",self.field_a,self.first),("b",self.field_b,self.second),("x",self.field_a,self.wrong_a),("y",self.field_b,self.wrong_b)]
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
        Case("GA11","optical frequency divider","divider mode","harmonic","output ratio","1:8","subharmonic","1:2",58101),
        Case("GA22","microplasma probe","probe material","tungsten","sweep rate","40 V/s","molybdenum","10 V/s",58102),
        Case("GA33","magnetic encoder","sensor","AMR","pole pitch","2 mm","Hall","8 mm",58103),
        Case("GA44","photon correlator","channel mode","cross","bin width","64 ps","auto","1 ns",58104),
        Case("GA55","laser attenuator","element","waveplate","extinction","45 dB","ND filter","12 dB",58105),
        Case("GA66","cryogenic bridge","excitation","AC","frequency","17 Hz","DC","1 Hz",58106),
        Case("GA77","ion optics supply","lens mode","Einzel","voltage","1.6 kV","quadrupole","0.4 kV",58107),
        Case("GA88","acoustic array","element","MEMS","spacing","12 mm","piezo","40 mm",58108),
        Case("GB11","beam stabilizer","sensor","quadrant diode","loop rate","18 kHz","camera","1 kHz",58109),
        Case("GB22","thermal stage","heater","ceramic","ramp","6 C/min","foil","2 C/min",58110),
        Case("GB33","microwave detector","diode","zero-bias","bandwidth","18 GHz","Schottky","6 GHz",58111),
        Case("GB44","vacuum valve","actuator","piezo","stroke","1.2 mm","solenoid","5 mm",58112),
        Case("GB55","spectrometer shutter","blade","carbon","latency","4 ms","steel","20 ms",58113),
        Case("GB66","optical delay line","drive","linear motor","resolution","20 nm","stepper","1 um",58114),
        Case("GB77","particle counter","illumination","405 nm","flow","2 L/min","650 nm","0.5 L/min",58115),
        Case("GB88","RF phase shifter","topology","vector","range","360 deg","loaded line","90 deg",58116),
    )


def _rows(suite):
    rows=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        rows.append(S17FusionCase(
            case_id=f"a0-s31-{case.case_id}",
            split="train",domain="s31_a0_only",language="en",
            state_a=case.state_a,state_b=case.state_b,
            question_a1=case.qa1,question_a2=case.qa2,
            question_b1=case.qb1,question_b2=case.qb2,
            option_texts=tuple(o.criterion_text for o in options),
            option_aliases=tuple(o.aliases[0] for o in options),
            option_ids=tuple(o.option_id for o in options),
            gold_a=ga,gold_b=gb,
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
    relation_block,global_pieces=s31_global_relation_block(
        relation_c,relation_p,signature_c,signature_p,gold
    )
    global_loss=global_pieces["global_contrastive"]
    c2p=global_pieces["global_c2p"]
    p2c=global_pieces["global_p2c"]
    return {
        "primary_block":primary_block,
        "relation_block":relation_block,
        "global_loss":global_loss,
        "global_c2p":c2p,
        "global_p2c":p2c,
        "gold":gold,
        "signature_c":signature_c,
        "signature_p":signature_p,
        "fused_c":fused_c,"fused_p":fused_p,
        "raw_c":raw_c,"raw_p":raw_p,
        "relation_c":relation_c,"relation_p":relation_p,
        "encoded":encoded,
    }


def _gradient_probe(bundle,manifest,rows):
    with torch.inference_mode(False),torch.enable_grad():
        runtime=_runtime(bundle,manifest,True)
        trainable=[p for p in runtime.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable)!=HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S31-A0 treatment surface changed")
        parts=_treatment_parts(runtime,rows)
        pg=torch.autograd.grad(parts["primary_block"],trainable,retain_graph=True,allow_unused=True)
        rg=torch.autograd.grad(parts["relation_block"],trainable,allow_unused=True)
        primary=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,pg)]
        relation=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,rg)]
        combined,diag=norm_balanced_gradient_update(primary,relation,epsilon=1e-12)
        idx={id(p):i for i,p in enumerate(trainable)}
        modules=iter_a13_lora_modules(runtime.encoder)
        b_l1=[float(combined[idx[id(m.lora_b)]].abs().sum().cpu()) for m in modules]
        scorer=runtime.projection_triadic_scorer
        if scorer is None: raise RuntimeError("S31-A0 scorer missing")
        projection_l1=float(combined[idx[id(scorer.projection.weight)]].abs().sum().cpu())
        if any(v<=0 for v in b_l1) or projection_l1<=0:
            raise RuntimeError("S31-A0 intended gradients vanished")
        if diag.primary_norm<=0 or diag.relation_norm<=0:
            raise RuntimeError("S31-A0 gradient partition vanished")
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
    if len(suite)!=16: raise RuntimeError("S31-A0 suite size changed")
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
    if token_error!=0 or pooled_error!=0:
        raise RuntimeError("S31-A0 encoder identity changed")

    with torch.inference_mode():
        c=_losses(control,rows)
        t=_losses(treatment,rows)
    names=("fused_c","fused_p","raw_c","raw_p","relation_c","relation_p","signature_c","signature_p")
    identity={}
    for name,index in zip(names,range(4,12)):
        diff=float((c[index]-t[index]).abs().max().cpu())
        identity[name+"_max_abs"]=diff
        if diff!=0:
            raise RuntimeError(f"S31-A0 inference identity changed: {name}")

    cdec=_collect_decisions(control,suite,"projection_triadic")
    tdec=_collect_decisions(treatment,suite,"projection_triadic")
    exact_logits=exact_choices=0
    for key,(clogits,cchoice) in cdec["records"].items():
        tlogits,tchoice=tdec["records"][key]
        exact_logits+=int(torch.equal(clogits,tlogits))
        exact_choices+=int(cchoice==tchoice)
        if not torch.equal(clogits,tlogits) or cchoice!=tchoice:
            raise RuntimeError(f"S31-A0 choice identity changed: {key}")

    cfusion=_collect_fusion(control,suite)
    tfusion=_collect_fusion(treatment,suite)
    if cfusion!=tfusion:
        raise RuntimeError("S31-A0 fusion identity changed")

    # Real semantic global-loss diagnostic.
    real=_treatment_parts(treatment,rows)
    real_global=float(real["global_loss"].cpu())

    # Controlled operator diagnostic.
    q,d=8,16
    canonical=torch.zeros(q,4,d)
    paraphrase=torch.zeros(q,4,d)
    gold=torch.arange(q,dtype=torch.long)%4
    for i in range(q):
        canonical[i,gold[i],i]=1
        paraphrase[i,gold[i],i]=1
    good,_,_=global_cross_case_relation_contrastive_loss(canonical,paraphrase,gold,temperature=TEMPERATURE)
    bad,_,_=global_cross_case_relation_contrastive_loss(canonical,paraphrase.roll(1,0),gold,temperature=TEMPERATURE)
    perm=torch.tensor([3,0,6,2,7,1,5,4])
    permuted,_,_=global_cross_case_relation_contrastive_loss(canonical[perm],paraphrase[perm],gold[perm],temperature=TEMPERATURE)
    swapped,_,_=global_cross_case_relation_contrastive_loss(paraphrase,canonical,gold,temperature=TEMPERATURE)
    if not float(bad)>float(good)+1.0:
        raise RuntimeError("S31-A0 shuffled positives are not materially worse")
    if not torch.allclose(good,permuted,atol=1e-7,rtol=0):
        raise RuntimeError("S31-A0 matched permutation invariance changed")
    if not torch.allclose(good,swapped,atol=1e-7,rtol=0):
        raise RuntimeError("S31-A0 view-swap symmetry changed")

    # Direct operator gradient court. main() is inference-mode by design for
    # the identity/mechanics checks, so explicitly re-enable autograd only
    # inside this diagnostic block.
    with torch.inference_mode(False), torch.enable_grad():
        torch.manual_seed(991)
        gc=torch.randn(8,4,32,requires_grad=True)
        gp=torch.randn(8,4,32,requires_grad=True)
        gg=torch.arange(8,dtype=torch.long)%4
        gl,_,_=global_cross_case_relation_contrastive_loss(
            gc,gp,gg,temperature=TEMPERATURE
        )
        gl.backward()
        gc_grad=float(gc.grad.abs().sum()) if gc.grad is not None else 0.0
        gp_grad=float(gp.grad.abs().sum()) if gp.grad is not None else 0.0
        if gc_grad<=0 or gp_grad<=0:
            raise RuntimeError("S31-A0 operator gradients vanished")

    # Checkpoint roundtrip.
    state=a13_lora_state_dict(control.encoder)
    replay=_runtime(bundle,manifest,False)
    load_a13_lora_state_dict(replay.encoder,state,freeze=True)
    replay_state=a13_lora_state_dict(replay.encoder)
    checkpoint_roundtrip=all(torch.equal(state[k],replay_state[k]) for k in state)
    if not checkpoint_roundtrip:
        raise RuntimeError("S31-A0 checkpoint roundtrip changed")

    grad=_gradient_probe(bundle,manifest,rows)

    total_decisions=len(suite)*4
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S31_A0_MATCHED_LOCAL_VS_GLOBAL_RELATION",
        "semantic_case_count":len(suite),
        "decision_count":total_decisions,
        "state_view_count":len(suite)*2,
        "k":4,
        "views_per_option":2,
        "temperature":TEMPERATURE,
        "outer_coefficient":S31_GLOBAL_CANONICALIZATION_COEFFICIENT,
        "operator_parameter_count":0,
        "arm_token_output_identity_max_abs":token_error,
        "arm_pooled_output_identity_max_abs":pooled_error,
        "exact_logit_identity_rate":exact_logits/total_decisions,
        "exact_choice_identity_rate":exact_choices/total_decisions,
        **identity,
        "control_physical_parameter_count":HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        "treatment_physical_parameter_count":HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        "runtime_trainable_parameter_count":sum(p.numel() for p in control.parameters() if p.requires_grad),
        "original_a13_trainable_parameter_count":sum(p.numel() for name,p in control.encoder.model.named_parameters() if p.requires_grad and ".lora_" not in name),
        "hira_core_trainable_parameter_count":sum(p.numel() for p in control.hira.parameters() if p.requires_grad),
        "checkpoint_key_count":len(state),
        "checkpoint_roundtrip_exact":checkpoint_roundtrip,
        "controlled_good_loss":float(good),
        "controlled_shuffled_loss":float(bad),
        "controlled_shuffled_minus_good":float(bad-good),
        "controlled_matched_permutation_error":float((good-permuted).abs()),
        "controlled_view_swap_error":float((good-swapped).abs()),
        "controlled_canonical_gradient_l1":gc_grad,
        "controlled_paraphrase_gradient_l1":gp_grad,
        "real_semantic_global_loss":real_global,
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
    (args.out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("HIRA_V1_S31_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
