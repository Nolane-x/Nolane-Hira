from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import read_runtime_bundle_manifest
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_joint_state_query_option_interaction import JointStateQueryOptionPrivateCorrectionFork
from nmd.v1_teacher_consensus_ranking import (
    teacher_consensus_targets,
    teacher_consensus_pairwise_loss,
    weighted_teacher_consensus_auxiliary,
)
from hira_v1_s50_train_dev import _materialize_cache, _branch_logits


SCHEMA_VERSION="hira-v1-s58-a0-consensus-teacher-pairwise-ranking-v1"
OUTCOME="HIRA_V1_S58_A0_CONSENSUS_TEACHER_PAIRWISE_RANKING_READY"
SEED=79_001
TEACHER_RUN=37271509208
TEACHER_ARTIFACT_ID=11327849211
TEACHER_CHECKPOINT_SHA="804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa"
TEACHER_SELECTED_EPOCH=19


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
        return f"S58-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."
    @property
    def state_b(self):
        return f"S58 teacher audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."
    @property
    def qa1(self): return f"For S58-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S58-A0 entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For S58-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S58-A0 entry is tagged {self.field_b} for {self.case_id}?"

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
                aliases=(f"{value} is the S58-A0 {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("TC11","muon phase array","sensor","diamond cavity","phase floor","0.05 rad","steel cell","0.20 rad",23101),
        Case("TC22","spin pressure lens","film","PtYIG layer","pressure floor","1.1 nPa","bulk Ni","4.4 nPa",23102),
        Case("TC33","phonon torque map","guide","soft AlN beam","torque floor","0.03 zNm","Cu strip","0.12 zNm",23103),
        Case("TC44","Rydberg gradient scope","ensemble","dressed Rb 72D","gradient floor","0.04 V/mm","hot vapor","0.16 V/mm",23104),
        Case("TC55","moire recoil clock","stack","aligned MoTe2-WS2","recoil floor","0.05 pm","bulk Si","0.20 pm",23105),
        Case("TC66","neutron curvature camera","target","perfect Ge","curvature blur","0.03 mm-1","polymer tile","0.12 mm-1",23106),
        Case("TC77","topological torque compass","channel","Chern edge","torque leak","-50 dB","bulk bar","-22 dB",23107),
        Case("TC88","molecular phase radar","beam","selected HfF","phase drift","0.03 rad","thermal SO2","0.12 rad",23108),
        Case("TD11","vacuum recoil bridge","surface","template Ag","recoil floor","0.06 pm","rough steel","0.24 pm",23109),
        Case("TD22","atomic gradient lens","species","clocked Sr","gradient floor","0.03 V/mm","thermal K","0.12 V/mm",23110),
        Case("TD33","magnon pressure scope","guide","low-loss YIG","pressure floor","1.0 nPa","Ni strip","4.0 nPa",23111),
        Case("TD44","optical torque ruler","interface","NV cavity","torque loss","0.03 zNm","free-space link","0.12 zNm",23112),
        Case("TD55","superfluid curvature map","fluid","He4 film","curvature noise","0.04 mm-1","oil film","0.16 mm-1",23113),
        Case("TD66","Casimir gradient camera","surface","Ag-Si pair","gradient drift","0.04 pN/mm","steel pair","0.16 pN/mm",23114),
        Case("TD77","quantum recoil compass","sensor","SiV resonator","recoil floor","0.04 pm","Hall bar","0.16 pm",23115),
        Case("TD88","ferroelectric torque radar","crystal","poled LiNbO3","torque jitter","0.03 zNm","ceramic slab","0.12 zNm",23116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s58-{case.case_id}",
            split="train",
            domain="s58_a0_only",
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


def _load_teacher(path:Path):
    if file_sha256(path)!=TEACHER_CHECKPOINT_SHA:
        raise RuntimeError("S58-A0 teacher checkpoint SHA changed")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("schema_version")!="hira-v1-s57-private-candidate-v1":
        raise RuntimeError("S58-A0 teacher schema changed")
    if payload.get("branch")!="reference":
        raise RuntimeError("S58-A0 teacher branch changed")
    if int(payload.get("seed",-1))!=78001:
        raise RuntimeError("S58-A0 teacher seed changed")
    if int(payload.get("selected_dev_epoch",-1))!=TEACHER_SELECTED_EPOCH:
        raise RuntimeError("S58-A0 teacher selected epoch changed")
    if int(payload.get("correction_parameter_count",-1))!=114688:
        raise RuntimeError("S58-A0 teacher correction surface changed")
    if float(payload.get("ordinal_consistency_coefficient",-1.0))!=0.0:
        raise RuntimeError("S58-A0 teacher is not S57 reference")
    teacher=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    teacher.load_correction_state_dict(payload["correction_state_dict"],freeze=True)
    if any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError("S58-A0 teacher remained trainable")
    return teacher,payload


def _synthetic_mechanics():
    gold=torch.tensor([0],dtype=torch.long)
    strong_a=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    strong_b=torch.tensor([[3.5,0.8,-0.8,-2.5]])

    active,sign,strong_diag=teacher_consensus_targets(strong_a,strong_b,gold)
    if int(active.sum())<=0:
        raise RuntimeError("S58-A0 strong consensus missing")
    if sign.requires_grad:
        raise RuntimeError("S58-A0 consensus sign not detached")

    low=torch.zeros(1,4)
    low_active,_s,low_diag=teacher_consensus_targets(low,low,gold)
    if int(low_active.sum())!=0:
        raise RuntimeError("S58-A0 low confidence teacher became active")

    disagree_active,_s,disagree_diag=teacher_consensus_targets(strong_a,-strong_b,gold)
    if int(disagree_active.sum())!=0:
        raise RuntimeError("S58-A0 teacher sign disagreement became active")

    wrong_a=torch.tensor([[0.0,4.0,2.0,-2.0]])
    wrong_b=torch.tensor([[0.0,3.5,1.5,-2.5]])
    wrong_active,_s,wrong_diag=teacher_consensus_targets(wrong_a,wrong_b,gold)
    if int(wrong_diag["teacher_wrong_gold_filtered_count"])<=0:
        raise RuntimeError("S58-A0 wrong gold teacher pair not filtered")
    if float(wrong_diag["teacher_non_gold_active_fraction"])<=0.0:
        raise RuntimeError("S58-A0 non-gold teacher consensus disappeared")

    good_student_a=strong_a.clone().requires_grad_(True)
    good_student_b=strong_b.clone().requires_grad_(True)
    bad_student_a=(-strong_a).clone().requires_grad_(True)
    bad_student_b=(-strong_b).clone().requires_grad_(True)
    good_loss,_=teacher_consensus_pairwise_loss(
        good_student_a,good_student_b,strong_a,strong_b,gold
    )
    bad_loss,_=teacher_consensus_pairwise_loss(
        bad_student_a,bad_student_b,strong_a,strong_b,gold
    )
    if not float(good_loss)<float(bad_loss):
        raise RuntimeError("S58-A0 student target margin ordering failed")

    offset_active,offset_sign,_=teacher_consensus_targets(
        strong_a+11.0,strong_b-7.0,gold
    )
    scale_active,scale_sign,_=teacher_consensus_targets(
        strong_a*4.0,strong_b*0.5,gold
    )
    if not torch.equal(active,offset_active) or not torch.equal(sign,offset_sign):
        raise RuntimeError("S58-A0 teacher offset invariance failed")
    if not torch.equal(active,scale_active) or not torch.equal(sign,scale_sign):
        raise RuntimeError("S58-A0 teacher scale invariance failed")

    flat_student_a=torch.zeros(1,4,requires_grad=True)
    flat_student_b=torch.zeros(1,4,requires_grad=True)
    flat_student_loss,flat_student_diag=teacher_consensus_pairwise_loss(
        flat_student_a,flat_student_b,strong_a,strong_b,gold
    )
    if float(flat_student_loss)<=0.0 or float(flat_student_diag["student_violation_fraction"])<=0.0:
        raise RuntimeError("S58-A0 flat student anti-collapse failed")

    return {
        "strong_active_consensus_fraction":float(strong_diag["teacher_active_consensus_fraction"]),
        "low_teacher_active_consensus_count":int(low_active.sum()),
        "teacher_disagreement_active_consensus_count":int(disagree_active.sum()),
        "teacher_sign_disagreement_fraction":float(disagree_diag["teacher_sign_disagreement_fraction"]),
        "wrong_gold_filtered_count":int(wrong_diag["teacher_wrong_gold_filtered_count"]),
        "non_gold_active_fraction":float(wrong_diag["teacher_non_gold_active_fraction"]),
        "satisfied_student_loss":float(good_loss.detach()),
        "violated_student_loss":float(bad_loss.detach()),
        "flat_student_loss":float(flat_student_loss.detach()),
        "flat_student_violation_fraction":float(flat_student_diag["student_violation_fraction"]),
        "teacher_offset_invariant":True,
        "teacher_positive_scale_invariant":True,
        "consensus_sign_detached":True,
    }


def _k_court():
    g=torch.Generator().manual_seed(79500)
    max_mass=0.0
    for k in (3,7,255):
        b=2
        teacher_a=torch.randn(b,k,generator=g)
        teacher_b=teacher_a+0.15*torch.randn(b,k,generator=g)
        student_a=torch.randn(b,k,generator=g,requires_grad=True)
        student_b=torch.randn(b,k,generator=g,requires_grad=True)
        gold=torch.tensor([0,k-1],dtype=torch.long)
        loss,_=teacher_consensus_pairwise_loss(
            student_a,student_b,teacher_a,teacher_b,gold
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("S58-A0 arbitrary-K loss non-finite")
        mass=float((torch.softmax(student_a,dim=-1).sum(-1)-1.0).abs().max())
        max_mass=max(max_mass,mass)
    if max_mass>1e-6:
        raise RuntimeError("S58-A0 probability mass changed")
    return max_mass


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--authority-checkpoint",type=Path,required=True)
    parser.add_argument("--authority-receipt",type=Path,required=True)
    parser.add_argument("--teacher-checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    authority=json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    expected_runtime="ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628"
    expected_checkpoint="19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916"
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S58-A0 parent native runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S58-A0 parent native checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S58-A0 native checkpoint file changed")

    teacher,teacher_payload=_load_teacher(args.teacher_checkpoint)

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    runtime,payload=load_native_authority(
        bundle=bundle,
        manifest=manifest,
        checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,
        expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S58-A0 loaded native runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S58-A0 native runtime remained trainable")

    rows=_rows()
    cache_pairs,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    reference=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    treatment=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S58-A0 student initialization changed")
    if reference.correction_parameter_count!=114688 or treatment.correction_parameter_count!=114688:
        raise RuntimeError("S58-A0 student private capacity changed")

    mechanics=_synthetic_mechanics()

    # Teacher deterministic replay and real-cache gradient ownership.
    canonical,paraphrase,_n=cache_pairs[0]
    with torch.inference_mode():
        _tca,_tsa,teacher_c1=_branch_logits(teacher,"treatment",canonical)
        _tpa,_tsp,teacher_p1=_branch_logits(teacher,"treatment",paraphrase)
        _tca2,_tsa2,teacher_c2=_branch_logits(teacher,"treatment",canonical)
        _tpa2,_tsp2,teacher_p2=_branch_logits(teacher,"treatment",paraphrase)
    teacher_replay_error=max(
        float((teacher_c1-teacher_c2).abs().max()),
        float((teacher_p1-teacher_p2).abs().max()),
    )
    if teacher_replay_error!=0.0:
        raise RuntimeError("S58-A0 teacher deterministic replay changed")
    if teacher_c1.requires_grad or teacher_p1.requires_grad:
        raise RuntimeError("S58-A0 teacher logits retained gradient")

    _rc,_rsc,ref_c=_branch_logits(reference,"treatment",canonical)
    _rp,_rsp,ref_p=_branch_logits(reference,"treatment",paraphrase)
    ref_aux,_=weighted_teacher_consensus_auxiliary(
        ref_c,ref_p,teacher_c1,teacher_p1,canonical.gold,coefficient=0.0
    )
    if float(ref_aux)!=0.0:
        raise RuntimeError("S58-A0 reference auxiliary nonzero")

    _tc,_tsc,trt_c=_branch_logits(treatment,"treatment",canonical)
    _tp,_tsp,trt_p=_branch_logits(treatment,"treatment",paraphrase)
    trt_aux,trt_diag=weighted_teacher_consensus_auxiliary(
        trt_c,trt_p,teacher_c1,teacher_p1,canonical.gold,coefficient=0.05
    )
    grads=torch.autograd.grad(
        trt_aux,treatment.correction_parameters(),allow_unused=True
    )
    treatment_gradient_l1=sum(float(g.abs().sum()) for g in grads if g is not None)
    if treatment_gradient_l1<=0.0:
        raise RuntimeError("S58-A0 treatment consensus gradient vanished")

    if any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError("S58-A0 teacher gained trainable parameters")

    cache_inference=cache_requires_grad=0
    for c,p,_n in cache_pairs:
        for evidence in (c,p):
            for tensor in evidence.tensors():
                cache_inference+=int(torch.is_inference(tensor))
                cache_requires_grad+=int(tensor.requires_grad)
    if cache_inference!=0 or cache_requires_grad!=0:
        raise RuntimeError("S58-A0 cache ownership changed")

    max_mass=_k_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S58_A0_CONSENSUS_TEACHER_PAIRWISE_RANKING_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "teacher_run":TEACHER_RUN,
        "teacher_artifact_id":TEACHER_ARTIFACT_ID,
        "teacher_checkpoint_sha256":TEACHER_CHECKPOINT_SHA,
        "teacher_selected_epoch":TEACHER_SELECTED_EPOCH,
        "teacher_branch":teacher_payload["branch"],
        "teacher_private_parameter_count":teacher_payload["private_trainable_parameter_count"],
        "teacher_correction_parameter_count":teacher_payload["correction_parameter_count"],
        "teacher_ordinal_consistency_coefficient":teacher_payload["ordinal_consistency_coefficient"],
        "teacher_trainable_parameter_count":sum(p.numel() for p in teacher.parameters() if p.requires_grad),
        "teacher_deterministic_replay_max_abs_error":teacher_replay_error,
        "teacher_logits_require_grad":False,
        "semantic_case_count":len(rows),
        "cache_digest":cache_digest,
        "cache_inference_tensor_count":cache_inference,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "reference_private_trainable_parameter_count":114688,
        "treatment_private_trainable_parameter_count":114688,
        "added_trainable_parameter_count":0,
        "identity_parameter_count":0,
        "initialization_bit_identical":True,
        "reference_auxiliary_exact_zero":float(ref_aux)==0.0,
        "treatment_auxiliary_value":float(trt_aux.detach()),
        "treatment_auxiliary_gradient_l1":treatment_gradient_l1,
        "real_cache_teacher_active_consensus_fraction":float(trt_diag["teacher_active_consensus_fraction"]),
        "real_cache_teacher_wrong_gold_filtered_fraction":float(trt_diag["teacher_wrong_gold_filtered_fraction"]),
        "real_cache_student_violation_fraction":float(trt_diag["student_violation_fraction"]),
        **mechanics,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "max_probability_mass_error":max_mass,
        "second_encoder_pass":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S58_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
