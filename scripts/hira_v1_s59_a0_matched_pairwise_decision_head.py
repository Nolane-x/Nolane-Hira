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
from nmd.v1_persisted_native_authority import file_sha256, load_native_authority
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_learned_pairwise_decision_head import (
    MatchedLearnedDecisionHeadFork,
    matched_head_initialization_exact,
)
from hira_v1_s50_train_dev import _materialize_cache, _branch_logits


SCHEMA_VERSION="hira-v1-s59-a0-matched-antisymmetric-pairwise-head-v1"
OUTCOME="HIRA_V1_S59_A0_MATCHED_ANTISYMMETRIC_PAIRWISE_HEAD_READY"
SEED=80_001


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
        return f"S59-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."
    @property
    def state_b(self):
        return f"S59 mechanical audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."
    @property
    def qa1(self): return f"For S59-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S59-A0 entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For S59-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S59-A0 entry is tagged {self.field_b} for {self.case_id}?"

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
                aliases=(f"{value} is the S59-A0 {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("PA11","muon shear array","sensor","diamond Q11","shear floor","0.11 nstrain","steel Q11","0.44 nstrain",25101),
        Case("PA22","spin torsion lens","film","PtYIG Q13","torsion floor","0.13 urad","Ni Q13","0.52 urad",25102),
        Case("PA33","phonon flux map","guide","AlN Q17","flux floor","1.7 fT","Cu Q17","6.8 fT",25103),
        Case("PA44","Rydberg impedance scope","ensemble","Rb Q19","impedance floor","1.9 mOhm","vapor Q19","7.6 mOhm",25104),
        Case("PA55","moire vorticity clock","stack","MoTe Q23","vorticity floor","0.23 mHz","Si Q23","0.92 mHz",25105),
        Case("PA66","neutron susceptibility camera","target","Ge Q29","susceptibility floor","2.9 ppt","polymer Q29","11.6 ppt",25106),
        Case("PA77","topological shear compass","channel","Chern Q31","shear leak","0.31 nstrain","bulk Q31","1.24 nstrain",25107),
        Case("PA88","molecular torsion radar","beam","HfF Q37","torsion drift","0.37 prad","SO2 Q37","1.48 prad",25108),
        Case("PB11","vacuum flux bridge","surface","Ag Q41","flux floor","4.1 fT","steel Q41","16.4 fT",25109),
        Case("PB22","atomic vorticity lens","species","Sr Q43","vorticity floor","0.43 mHz","K Q43","1.72 mHz",25110),
        Case("PB33","magnon impedance scope","guide","YIG Q47","impedance floor","4.7 mOhm","Ni Q47","18.8 mOhm",25111),
        Case("PB44","optical shear ruler","interface","NV Q53","shear loss","0.53 nstrain","free Q53","2.12 nstrain",25112),
        Case("PB55","superfluid torsion map","fluid","He4 Q59","torsion noise","0.59 urad","oil Q59","2.36 urad",25113),
        Case("PB66","Casimir flux camera","surface","AgSi Q61","flux drift","6.1 fT","steel Q61","24.4 fT",25114),
        Case("PB77","quantum vorticity compass","sensor","SiV Q67","vorticity floor","0.67 mHz","Hall Q67","2.68 mHz",25115),
        Case("PB88","ferroelectric impedance radar","crystal","LiNb Q71","impedance jitter","7.1 mOhm","ceramic Q71","28.4 mOhm",25116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s59-{case.case_id}",
            split="train",
            domain="s59_a0_only",
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


def _synthetic_pairwise_mechanics():
    op=MatchedLearnedDecisionHeadFork(
        mode="pairwise",train_correction=True,train_head=True
    )
    g=torch.Generator().manual_seed(80599)
    with torch.no_grad():
        op.decision_b.copy_(torch.randn(op.decision_b.shape,generator=g)*0.05)

    max_antisym=max_diag=max_permutation=max_mass=0.0
    for k in (3,7,255):
        identity=F.normalize(torch.randn(2,k,256,generator=g),dim=-1)
        context=F.normalize(torch.randn(2,k,256,generator=g),dim=-1)
        pair=op.pairwise_matrix(identity,context)
        max_antisym=max(max_antisym,float((pair+pair.transpose(-1,-2)).abs().max()))
        max_diag=max(max_diag,float(torch.diagonal(pair,dim1=-2,dim2=-1).abs().max()))

        perm=torch.randperm(k,generator=g)
        base=op.pairwise_residual(identity,context)
        got=op.pairwise_residual(identity[:,perm],context[:,perm])
        max_permutation=max(max_permutation,float((got-base[:,perm]).abs().max()))
        max_mass=max(
            max_mass,
            float((torch.softmax(base,dim=-1).sum(-1)-1.0).abs().max()),
        )

    zero_context=torch.zeros(2,7,256)
    identity=F.normalize(torch.randn(2,7,256,generator=g),dim=-1)
    degenerate=op.pairwise_residual(identity,zero_context)
    if not bool(torch.isfinite(degenerate).all()):
        raise RuntimeError("S59-A0 degenerate pair context non-finite")

    if max_antisym>1e-7:
        raise RuntimeError("S59-A0 pairwise antisymmetry failed")
    if max_diag!=0.0:
        raise RuntimeError("S59-A0 pairwise diagonal changed")
    if max_permutation>1e-6:
        raise RuntimeError("S59-A0 permutation equivariance failed")
    if max_mass>1e-6:
        raise RuntimeError("S59-A0 probability mass changed")

    return {
        "pairwise_max_antisymmetry_error":max_antisym,
        "pairwise_max_diagonal_abs":max_diag,
        "pairwise_max_permutation_error":max_permutation,
        "pairwise_max_probability_mass_error":max_mass,
        "finite_degenerate_context":True,
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
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
        raise RuntimeError("S59-A0 parent native authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S59-A0 parent native runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S59-A0 parent native checkpoint changed")
    if authority.get("dev_encoded") is not False or authority.get("dev_scored") is not False:
        raise RuntimeError("S59-A0 parent native authority exposed DEV")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S59-A0 native checkpoint file changed")

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
        raise RuntimeError("S59-A0 loaded native runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S59-A0 native runtime remained trainable")

    rows=_rows()
    cache_pairs,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    reference=MatchedLearnedDecisionHeadFork(
        mode="pointwise",train_correction=True,train_head=True
    )
    treatment=MatchedLearnedDecisionHeadFork(
        mode="pairwise",train_correction=True,train_head=True
    )
    if not matched_head_initialization_exact(reference,treatment):
        raise RuntimeError("S59-A0 matched initialization changed")
    for op in (reference,treatment):
        if op.correction_parameter_count!=114688:
            raise RuntimeError("S59-A0 correction parameter count changed")
        if op.decision_head_parameter_count!=16384:
            raise RuntimeError("S59-A0 decision head parameter count changed")
        if op.total_private_parameter_count!=131072:
            raise RuntimeError("S59-A0 total private parameter count changed")

    canonical,paraphrase,_n=cache_pairs[0]
    rr_c,_rs_c,rf_c=_branch_logits(reference,"treatment",canonical)
    tr_c,_ts_c,tf_c=_branch_logits(treatment,"treatment",canonical)
    rr_p,_rs_p,rf_p=_branch_logits(reference,"treatment",paraphrase)
    tr_p,_ts_p,tf_p=_branch_logits(treatment,"treatment",paraphrase)

    initial_relation_error=max(
        float((rr_c-tr_c).abs().max()),
        float((rr_p-tr_p).abs().max()),
    )
    initial_fused_error=max(
        float((rf_c-tf_c).abs().max()),
        float((rf_p-tf_p).abs().max()),
    )
    if initial_relation_error!=0.0 or initial_fused_error!=0.0:
        raise RuntimeError("S59-A0 initial matched outputs changed")

    # Zero-B warm start: B must learn first; A must become live after one B update.
    loss=F.cross_entropy(tr_c,canonical.gold)
    ga0,gb0=torch.autograd.grad(
        loss,treatment.decision_head_parameters(),retain_graph=True
    )
    a_grad_initial=float(ga0.abs().sum())
    b_grad_initial=float(gb0.abs().sum())
    if a_grad_initial!=0.0:
        raise RuntimeError("S59-A0 A unexpectedly live before B warm-start")
    if b_grad_initial<=0.0:
        raise RuntimeError("S59-A0 B warm-start gradient vanished")

    with torch.no_grad():
        treatment.decision_b.add_(-0.10*gb0)
    tr2_c,_sig2,_fused2=_branch_logits(treatment,"treatment",canonical)
    loss2=F.cross_entropy(tr2_c,canonical.gold)
    ga1,gb1=torch.autograd.grad(loss2,treatment.decision_head_parameters())
    a_grad_after=float(ga1.abs().sum())
    b_grad_after=float(gb1.abs().sum())
    if a_grad_after<=0.0 or b_grad_after<=0.0:
        raise RuntimeError("S59-A0 A/B learning path not live after B update")

    # Existing correction path must remain trainable independently.
    ref_loss=F.cross_entropy(rr_c,canonical.gold)
    correction_grads=torch.autograd.grad(
        ref_loss,reference.correction_parameters(),allow_unused=True
    )
    correction_gradient_l1=sum(
        float(g.abs().sum()) for g in correction_grads if g is not None
    )
    if correction_gradient_l1<=0.0:
        raise RuntimeError("S59-A0 legacy correction gradient vanished")

    cache_inference=cache_requires_grad=0
    for c,p,_n in cache_pairs:
        for evidence in (c,p):
            for tensor in evidence.tensors():
                cache_inference+=int(torch.is_inference(tensor))
                cache_requires_grad+=int(tensor.requires_grad)
    if cache_inference!=0 or cache_requires_grad!=0:
        raise RuntimeError("S59-A0 cache ownership changed")

    mechanics=_synthetic_pairwise_mechanics()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S59_A0_MATCHED_PAIRWISE_MECHANICS_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "reference_mode":"pointwise",
        "treatment_mode":"pairwise",
        "correction_parameter_count_per_arm":114688,
        "decision_head_parameter_count_per_arm":16384,
        "total_private_parameter_count_per_arm":131072,
        "decision_head_rank":32,
        "decision_head_seed":80590,
        "decision_head_scale":1.0,
        "matched_initialization_exact":True,
        "decision_b_initial_exact_zero":True,
        "initial_relation_max_abs_error":initial_relation_error,
        "initial_fused_max_abs_error":initial_fused_error,
        "initial_a_gradient_l1":a_grad_initial,
        "initial_b_gradient_l1":b_grad_initial,
        "post_b_update_a_gradient_l1":a_grad_after,
        "post_b_update_b_gradient_l1":b_grad_after,
        "legacy_correction_gradient_l1":correction_gradient_l1,
        "semantic_case_count":len(rows),
        "cache_digest":cache_digest,
        "cache_inference_tensor_count":cache_inference,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        **mechanics,
        "teacher_artifact_dependency":False,
        "pseudo_target_dependency":False,
        "second_encoder_pass":False,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S59_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
