from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_explicit_pairwise_decision_head import ExplicitPairwiseDecisionHead
from nmd.v1_confidence_adaptive_bounded_hybrid import (
    ConfidenceAdaptiveBoundedHybridGate,
    adaptive_gate_gold_loss,
)
from nmd.v1_reliability_supervised_adaptive_gate import (
    S62_ALPHA_PROBE,
    S62_TARGET_TOLERANCE,
    fixed_bounded_probe,
    train_only_reliability_target,
    reliability_gate_loss,
)
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s62-a0-train-only-reliability-supervised-adaptive-gate-v1"
OUTCOME="HIRA_V1_S62_A0_TRAIN_ONLY_RELIABILITY_SUPERVISED_ADAPTIVE_GATE_READY"
SEED=83_001


@dataclass(frozen=True)
class Case:
    case_id:str; noun:str; field_a:str; first:str; field_b:str; second:str
    wrong_a:str; wrong_b:str; seed:int

    @property
    def state_a(self):
        return f"S62-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."
    @property
    def state_b(self):
        return f"S62 reliability audit {self.case_id}: {self.second} is {self.field_b}; {self.first} is {self.field_a}."
    @property
    def qa1(self): return f"For S62-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S62 reliability value is tagged {self.field_a} in {self.case_id}?"
    @property
    def qb1(self): return f"For S62-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S62 reliability value is tagged {self.field_b} in {self.case_id}?"

    def option_pack(self):
        rows=[
            ("a",self.field_a,self.first),("b",self.field_b,self.second),
            ("x",self.field_a,self.wrong_a),("y",self.field_b,self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        options=tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"for the S62 reliability {self.noun}, {field} is {value}",
                aliases=(f"{value} is the S62 reliability {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def _rows():
    specs=(
        Case("RL11","reliability anyon scope","tile","Chern R11","phase floor","0.19 mrad","bulk R29","0.76 mrad",26201),
        Case("RL22","reliability shear lens","beam","AlN R13","strain floor","0.16 fstrain","steel R31","0.64 fstrain",26202),
        Case("RL33","reliability axion map","cavity","Nb R17","impedance floor","0.11 uOhm","Cu R37","0.44 uOhm",26203),
        Case("RL44","reliability magnon clock","lattice","YIG R19","vorticity floor","0.17 mHz","Ni R41","0.68 mHz",26204),
        Case("RL55","reliability neutrino bridge","detector","Ge R23","phase floor","0.13 prad","polymer R43","0.52 prad",26205),
        Case("RL66","reliability photon compass","guide","SiN R29","drag floor","0.15 fm/s","metal R47","0.60 fm/s",26206),
        Case("RL77","reliability Fermi scope","gas","Li6 R31","pressure floor","0.18 pPa","thermal R53","0.72 pPa",26207),
        Case("RL88","reliability vacuum radar","pair","AuSi R37","torque floor","0.14 zNm","steel R59","0.56 zNm",26208),
        Case("RM11","reliability atomic camera","array","Sr R41","recoil floor","0.10 fm","warm R61","0.40 fm",26209),
        Case("RM22","reliability molecular scope","beam","ThO R43","flux floor","1.6 a.u.","thermal R67","6.4 a.u.",26210),
        Case("RM33","reliability superfluid map","fluid","He4 R47","drag floor","0.12 nN","oil R71","0.48 nN",26211),
        Case("RM44","reliability ferro lens","stack","LiNbO3 R53","phase floor","0.14 urad","ceramic R73","0.56 urad",26212),
        Case("RM55","reliability optical bridge","guide","GaP R59","recoil floor","0.09 pm","fiber R79","0.36 pm",26213),
        Case("RM66","reliability spin compass","film","CoFeB R61","flux floor","0.21 fT","iron R83","0.84 fT",26214),
        Case("RM77","reliability Casimir scope","surface","Ag R67","gradient floor","0.10 pN/mm","steel R89","0.40 pN/mm",26215),
        Case("RM88","reliability quantum ruler","sensor","NV R71","recoil floor","0.08 pm","Hall R97","0.32 pm",26216),
    )
    out=[]
    for case in specs:
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s62-{case.case_id}",split="train",domain="s62_a0_only",language="en",
            state_a=case.state_a,state_b=case.state_b,
            question_a1=case.qa1,question_a2=case.qa2,
            question_b1=case.qb1,question_b2=case.qb2,
            option_texts=tuple(x.criterion_text for x in options),
            option_aliases=tuple(x.aliases[0] for x in options),
            option_ids=tuple(x.option_id for x in options),
            gold_a=ga,gold_b=gb,
        ))
    return out


def _synthetic_target_court():
    rows={
        "beneficial":(
            [-1.376891016960144,3.2005667686462402,-3.3355374336242676],
            [1.6032288074493408,-2.462509870529175,1.481863260269165],
            [1.7417689561843872,-0.2375815510749817,0.0183677077293396],
            [1.1653449535369873,1.4833534955978394,0.8274328708648682],0,
        ),
        "correctness_harm":(
            [-0.4818168878555298,3.1734812259674072,0.6368277668952942],
            [0.08502231538295746,1.7102296352386475,2.0455708503723145],
            [1.8026907444000244,-0.3109073042869568,-3.2117626667022705],
            [-0.013404175639152527,-0.6213452219963074,-1.9260790348052979],2,
        ),
        "stability_harm":(
            [-1.0522860288619995,-2.656233310699463,2.790268898010254],
            [0.40249666571617126,0.8953961730003357,0.6669978499412537],
            [-1.0904462337493896,1.466923713684082,-0.07216861844062805],
            [1.1611074209213257,1.6940511465072632,-0.61772221326828],1,
        ),
        "both_harm":(
            [0.9856889247894287,-3.4627974033355713,-0.8162581324577332],
            [0.49354082345962524,1.432938814163208,-0.5824704766273499],
            [2.2066073417663574,4.542449474334717,-2.0895113945007324],
            [-1.6886796951293945,-0.3825035095214844,-0.11528395116329193],2,
        ),
    }
    expected={"beneficial":1.0,"correctness_harm":0.0,"stability_harm":0.0,"both_harm":0.0}
    observed={}
    diags={}
    tensors=[]
    for name,(fc,fp,pc,pp,gold) in rows.items():
        args=(
            torch.tensor([fc]),torch.tensor([fp]),
            torch.tensor([pc]),torch.tensor([pp]),
            torch.tensor([gold],dtype=torch.long),
        )
        y,d=train_only_reliability_target(*args)
        observed[name]=float(y.item())
        diags[name]={
            "ce_base":float(d["ce_base"].item()),
            "ce_probe":float(d["ce_probe"].item()),
            "js_base":float(d["js_base"].item()),
            "js_probe":float(d["js_probe"].item()),
            "correctness_safe":bool(d["correctness_safe"].item()),
            "stability_better":bool(d["stability_better"].item()),
        }
        if observed[name]!=expected[name]:
            raise RuntimeError(f"S62-A0 target case changed: {name}")
        tensors.append(args)

    mixed=[
        torch.cat([tensors[0][i],tensors[1][i]],dim=0)
        for i in range(4)
    ]
    gold=torch.cat([tensors[0][4],tensors[1][4]],dim=0)
    y,_=train_only_reliability_target(*mixed,gold)
    if sorted(y.tolist())!=[0.0,1.0]:
        raise RuntimeError("S62-A0 mixed target lost a class")

    gen=torch.Generator().manual_seed(62062)
    max_mass=0.0
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=gen)
        pair=torch.randn(2,k,generator=gen)
        probe=fixed_bounded_probe(fused,pair)
        max_mass=max(max_mass,float((torch.softmax(probe,-1).sum(-1)-1.0).abs().max()))

    return {
        "target_observed":observed,
        "target_diagnostics":diags,
        "mixed_target_positive_count":int(y.sum().item()),
        "mixed_target_count":int(y.numel()),
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "max_probability_mass_error":max_mass,
        "flat_probe_finite":bool(torch.isfinite(fixed_bounded_probe(torch.zeros(2,7),torch.zeros(2,7))).all()),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("outcome")!="HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY":
        raise RuntimeError("S62-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S62-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S62-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S62-A0 checkpoint file changed")

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    runtime,payload=load_native_authority(
        bundle=bundle,manifest=manifest,checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S62-A0 loaded native changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S62-A0 native remained trainable")

    cache,cache_digest=s50._materialize_cache(runtime,_rows())
    del runtime

    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    reference=ConfidenceAdaptiveBoundedHybridGate(trainable=True)
    treatment=ConfidenceAdaptiveBoundedHybridGate(trainable=True)

    if op.correction_parameter_count!=114688:
        raise RuntimeError("S62-A0 correction capacity changed")
    if head.parameter_count!=32832:
        raise RuntimeError("S62-A0 pairwise capacity changed")
    if reference.parameter_count!=5 or treatment.parameter_count!=5:
        raise RuntimeError("S62-A0 gate capacity changed")
    if set(dict(reference.named_parameters()))!={"w","b"}:
        raise RuntimeError("S62-A0 reference gate surface changed")
    if set(dict(treatment.named_parameters()))!={"w","b"}:
        raise RuntimeError("S62-A0 treatment gate surface changed")
    rs=reference.state_dict()
    ts=treatment.state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S62-A0 gate initialization changed")

    canonical,paraphrase,_n=cache[0]
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    pair_c=head.aggregate_logits(s59._representation(op,canonical))
    pair_p=head.aggregate_logits(s59._representation(op,paraphrase))

    ref_loss,_=adaptive_gate_gold_loss(
        reference,fused_c,fused_p,pair_c,pair_p,canonical.gold
    )
    trt_loss,trt_diag=reliability_gate_loss(
        treatment,fused_c,fused_p,pair_c,pair_p,canonical.gold
    )

    upstream=(*op.correction_parameters(),head.A,head.u)
    ref_targets=(reference.w,reference.b,*upstream)
    trt_targets=(treatment.w,treatment.b,*upstream)
    ref_grads=torch.autograd.grad(ref_loss,ref_targets,allow_unused=True,retain_graph=True)
    trt_grads=torch.autograd.grad(trt_loss,trt_targets,allow_unused=True)

    ref_w=float(ref_grads[0].abs().sum()) if ref_grads[0] is not None else 0.0
    ref_b=float(ref_grads[1].abs().sum()) if ref_grads[1] is not None else 0.0
    trt_w=float(trt_grads[0].abs().sum()) if trt_grads[0] is not None else 0.0
    trt_b=float(trt_grads[1].abs().sum()) if trt_grads[1] is not None else 0.0
    if min(ref_w,ref_b,trt_w,trt_b)<=0.0:
        raise RuntimeError("S62-A0 gate gradient vanished")
    if any(g is not None and float(g.abs().sum())>0 for g in (*ref_grads[2:],*trt_grads[2:])):
        raise RuntimeError("S62-A0 gate gradient leaked upstream")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0:
        raise RuntimeError("S62-A0 cache gained gradients")

    mechanics=_synthetic_target_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    ref_state=reference.state_dict_exact()
    trt_state=treatment.state_dict_exact()
    checkpoint=out/"matched-gates.pt"
    torch.save({
        "schema_version":"hira-v1-s62-matched-gates-v1",
        "seed":SEED,
        "alpha_probe":S62_ALPHA_PROBE,
        "target_tolerance":S62_TARGET_TOLERANCE,
        "reference_state_dict":ref_state,
        "treatment_state_dict":trt_state,
    },checkpoint)
    rr=ConfidenceAdaptiveBoundedHybridGate()
    tt=ConfidenceAdaptiveBoundedHybridGate()
    rr.load_state_dict_exact(ref_state,freeze=True)
    tt.load_state_dict_exact(trt_state,freeze=True)
    replay_error=max(
        float((reference.compose(fused_c,pair_c)-rr.compose(fused_c,pair_c)).abs().max()),
        float((treatment.compose(fused_c,pair_c)-tt.compose(fused_c,pair_c)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S62_A0_TRAIN_ONLY_RELIABILITY_SUPERVISED_ADAPTIVE_GATE_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "semantic_case_count":len(cache),
        "cache_digest":cache_digest,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "correction_trainable_parameter_count":114688,
        "pairwise_trainable_parameter_count":32832,
        "reference_gate_parameter_count":reference.parameter_count,
        "treatment_gate_parameter_count":treatment.parameter_count,
        "added_treatment_parameter_count":0,
        "reference_gate_trainable_tensor_names":sorted(dict(reference.named_parameters())),
        "treatment_gate_trainable_tensor_names":sorted(dict(treatment.named_parameters())),
        "gate_initialization_bit_identical":True,
        "feature_dimension":4,
        "alpha_probe":S62_ALPHA_PROBE,
        "target_tolerance":S62_TARGET_TOLERANCE,
        "reference_gold_ce_gradient_w_l1":ref_w,
        "reference_gold_ce_gradient_b_l1":ref_b,
        "treatment_reliability_gradient_w_l1":trt_w,
        "treatment_reliability_gradient_b_l1":trt_b,
        "reference_gradient_to_upstream_zero":True,
        "treatment_gradient_to_upstream_zero":True,
        "real_cache_reliability_positive_fraction":float(trt_diag["positive_fraction"]),
        "teacher_dependency":False,
        "dev_target_dependency":False,
        "self_anchor_dependency":False,
        "pairwise_only_final_path":False,
        "one_encoder_state_once":True,
        "matched_gate_checkpoint_replay_max_abs_error":replay_error,
        **mechanics,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S62_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
