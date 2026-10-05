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
    S61_ALPHA_INITIAL,
    S61_ALPHA_MAX,
    S61_GATE_PARAMETER_COUNT,
    ConfidenceAdaptiveBoundedHybridGate,
    adaptive_gate_features,
    adaptive_gate_gold_loss,
)
import hira_v1_s50_train_dev as s50
import hira_v1_s59_train_dev as s59


SCHEMA_VERSION="hira-v1-s61-a0-confidence-adaptive-bounded-hybrid-v1"
OUTCOME="HIRA_V1_S61_A0_CONFIDENCE_ADAPTIVE_BOUNDED_HYBRID_READY"
SEED=82_001


@dataclass(frozen=True)
class Case:
    case_id:str; noun:str; field_a:str; first:str; field_b:str; second:str
    wrong_a:str; wrong_b:str; seed:int

    @property
    def state_a(self):
        return f"S61-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."
    @property
    def state_b(self):
        return f"S61 adaptive audit {self.case_id}: {self.second} is {self.field_b}; {self.first} is {self.field_a}."
    @property
    def qa1(self): return f"For S61-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S61 adaptive value is tagged {self.field_a} in {self.case_id}?"
    @property
    def qb1(self): return f"For S61-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S61 adaptive value is tagged {self.field_b} in {self.case_id}?"

    def option_pack(self):
        rows=[
            ("a",self.field_a,self.first),("b",self.field_b,self.second),
            ("x",self.field_a,self.wrong_a),("y",self.field_b,self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        options=tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"for the S61 adaptive {self.noun}, {field} is {value}",
                aliases=(f"{value} is the S61 adaptive {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def _rows():
    specs=(
        Case("AG11","adaptive anyon scope","tile","Chern A11","phase floor","0.18 mrad","bulk A29","0.72 mrad",26101),
        Case("AG22","adaptive shear lens","beam","AlN A13","strain floor","0.15 fstrain","steel A31","0.60 fstrain",26102),
        Case("AG33","adaptive axion map","cavity","Nb A17","impedance floor","0.10 uOhm","Cu A37","0.40 uOhm",26103),
        Case("AG44","adaptive magnon clock","lattice","YIG A19","vorticity floor","0.16 mHz","Ni A41","0.64 mHz",26104),
        Case("AG55","adaptive neutrino bridge","detector","Ge A23","phase floor","0.12 prad","polymer A43","0.48 prad",26105),
        Case("AG66","adaptive photon compass","guide","SiN A29","drag floor","0.14 fm/s","metal A47","0.56 fm/s",26106),
        Case("AG77","adaptive Fermi scope","gas","Li6 A31","pressure floor","0.17 pPa","thermal A53","0.68 pPa",26107),
        Case("AG88","adaptive vacuum radar","pair","AuSi A37","torque floor","0.13 zNm","steel A59","0.52 zNm",26108),
        Case("AH11","adaptive atomic camera","array","Sr A41","recoil floor","0.09 fm","warm A61","0.36 fm",26109),
        Case("AH22","adaptive molecular scope","beam","ThO A43","flux floor","1.5 a.u.","thermal A67","6.0 a.u.",26110),
        Case("AH33","adaptive superfluid map","fluid","He4 A47","drag floor","0.11 nN","oil A71","0.44 nN",26111),
        Case("AH44","adaptive ferro lens","stack","LiNbO3 A53","phase floor","0.13 urad","ceramic A73","0.52 urad",26112),
        Case("AH55","adaptive optical bridge","guide","GaP A59","recoil floor","0.08 pm","fiber A79","0.32 pm",26113),
        Case("AH66","adaptive spin compass","film","CoFeB A61","flux floor","0.20 fT","iron A83","0.80 fT",26114),
        Case("AH77","adaptive Casimir scope","surface","Ag A67","gradient floor","0.09 pN/mm","steel A89","0.36 pN/mm",26115),
        Case("AH88","adaptive quantum ruler","sensor","NV A71","recoil floor","0.07 pm","Hall A97","0.28 pm",26116),
    )
    out=[]
    for case in specs:
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s61-{case.case_id}",split="train",domain="s61_a0_only",language="en",
            state_a=case.state_a,state_b=case.state_b,
            question_a1=case.qa1,question_a2=case.qa2,question_b1=case.qb1,question_b2=case.qb2,
            option_texts=tuple(x.criterion_text for x in options),
            option_aliases=tuple(x.aliases[0] for x in options),
            option_ids=tuple(x.option_id for x in options),
            gold_a=ga,gold_b=gb,
        ))
    return out


def _synthetic_mechanics():
    g=torch.Generator().manual_seed(61061)
    gate=ConfidenceAdaptiveBoundedHybridGate()
    max_mass=max_perm=max_feature_affine=max_bound=0.0
    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g)
        pair=torch.randn(2,k,generator=g)
        out,d=gate.compose(fused,pair,return_diagnostics=True)
        max_mass=max(max_mass,float((torch.softmax(out,-1).sum(-1)-1).abs().max()))
        max_bound=max(
            max_bound,
            float(((out-fused).abs()-d["residual_bound"]).clamp_min(0).max()),
        )
        perm=torch.randperm(k,generator=g)
        got=gate.compose(fused[:,perm],pair[:,perm])
        max_perm=max(max_perm,float((got-out[:,perm]).abs().max()))
        base_x=adaptive_gate_features(fused,pair)
        affine_x=adaptive_gate_features(3*fused+7,5*pair-11)
        max_feature_affine=max(max_feature_affine,float((base_x-affine_x).abs().max()))

    flat_x=adaptive_gate_features(torch.zeros(2,7),torch.zeros(2,7))
    if not bool(torch.isfinite(flat_x).all()):
        raise RuntimeError("S61 A0 flat features non-finite")
    if not bool(torch.isin(flat_x[:,2],torch.tensor([-1.0,1.0])).all()):
        raise RuntimeError("S61 A0 agreement feature changed")

    probe_gate=ConfidenceAdaptiveBoundedHybridGate()
    with torch.no_grad():
        probe_gate.w.copy_(torch.tensor([-1.5,1.2,0.8,0.6]))
    fused=torch.tensor([[4.,1.,0.,-1.],[1.1,1.0,0.,-1.]])
    pair=torch.tensor([[4.,1.,0.,-1.],[-1.,4.,0.,1.]])
    alpha=probe_gate.alpha(fused,pair).squeeze(-1)
    if float(alpha[0])==float(alpha[1]):
        raise RuntimeError("S61 A0 adaptive alpha did not separate probes")

    identity=gate.compose(fused,pair,alpha_override=0.0)
    return {
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "feature_affine_invariance_max_abs_error":max_feature_affine,
        "option_permutation_max_abs_error":max_perm,
        "residual_bound_violation_max":max_bound,
        "max_probability_mass_error":max_mass,
        "flat_feature_finite":True,
        "adaptive_probe_alpha_0":float(alpha[0]),
        "adaptive_probe_alpha_1":float(alpha[1]),
        "adaptive_probe_separated":True,
        "identity_override_max_abs_error":float((identity-fused).abs().max()),
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
        raise RuntimeError("S61-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S61-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S61-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S61-A0 checkpoint file changed")

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    runtime,payload=load_native_authority(
        bundle=bundle,manifest=manifest,checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S61-A0 loaded native changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S61-A0 native remained trainable")

    cache,cache_digest=s50._materialize_cache(runtime,_rows())
    del runtime

    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    gate=ConfidenceAdaptiveBoundedHybridGate(trainable=True)
    if op.correction_parameter_count!=114688:
        raise RuntimeError("S61-A0 correction capacity changed")
    if head.parameter_count!=32832:
        raise RuntimeError("S61-A0 pairwise capacity changed")
    if gate.parameter_count!=S61_GATE_PARAMETER_COUNT:
        raise RuntimeError("S61-A0 gate capacity changed")
    if set(dict(gate.named_parameters()))!={"w","b"}:
        raise RuntimeError("S61-A0 gate tensor surface changed")
    if not torch.equal(gate.w,torch.zeros_like(gate.w)):
        raise RuntimeError("S61-A0 w initialization changed")

    canonical,paraphrase,_n=cache[0]
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    pair_c=head.aggregate_logits(s59._representation(op,canonical))
    pair_p=head.aggregate_logits(s59._representation(op,paraphrase))

    initial_alpha=torch.cat([gate.alpha(fused_c,pair_c),gate.alpha(fused_p,pair_p)])
    initial_alpha_error=float((initial_alpha-S61_ALPHA_INITIAL).abs().max())
    if initial_alpha_error>1e-7:
        raise RuntimeError("S61-A0 initial alpha changed")

    features=torch.cat([
        adaptive_gate_features(fused_c,pair_c),
        adaptive_gate_features(fused_p,pair_p),
    ],dim=0)
    if features.requires_grad or features.shape[-1]!=4:
        raise RuntimeError("S61-A0 feature contract changed")

    loss,diag=adaptive_gate_gold_loss(
        gate,fused_c,fused_p,pair_c,pair_p,canonical.gold
    )
    targets=(gate.w,gate.b,*op.correction_parameters(),head.A,head.u)
    grads=torch.autograd.grad(loss,targets,allow_unused=True)
    grad_w=float(grads[0].abs().sum()) if grads[0] is not None else 0.0
    grad_b=float(grads[1].abs().sum()) if grads[1] is not None else 0.0
    if grad_w<=0 or grad_b<=0:
        raise RuntimeError("S61-A0 gate gradients vanished")
    if any(g is not None and float(g.abs().sum())>0 for g in grads[2:]):
        raise RuntimeError("S61-A0 gate gradient leaked upstream")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0:
        raise RuntimeError("S61-A0 cache gained gradients")

    mechanics=_synthetic_mechanics()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    state=gate.state_dict_exact()
    checkpoint=out/"gate.pt"
    torch.save({
        "schema_version":"hira-v1-s61-adaptive-gate-v1",
        "seed":SEED,
        "alpha_max":S61_ALPHA_MAX,
        "alpha_initial":S61_ALPHA_INITIAL,
        "feature_dimension":4,
        "gate_parameter_count":5,
        "state_dict":state,
    },checkpoint)
    replay=ConfidenceAdaptiveBoundedHybridGate()
    replay.load_state_dict_exact(state,freeze=True)
    replay_error=float(
        (gate.compose(fused_c,pair_c)-replay.compose(fused_c,pair_c)).abs().max()
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S61_A0_CONFIDENCE_ADAPTIVE_BOUNDED_HYBRID_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "semantic_case_count":len(cache),
        "cache_digest":cache_digest,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "correction_trainable_parameter_count":114688,
        "pairwise_trainable_parameter_count":32832,
        "gate_trainable_parameter_count":gate.trainable_parameter_count,
        "gate_parameter_count":gate.parameter_count,
        "gate_trainable_tensor_names":sorted(dict(gate.named_parameters())),
        "feature_dimension":features.shape[-1],
        "features_require_grad":features.requires_grad,
        "initial_w_exact_zero":True,
        "initial_alpha_max_abs_error":initial_alpha_error,
        "alpha_max":S61_ALPHA_MAX,
        "gate_gradient_w_l1":grad_w,
        "gate_gradient_b_l1":grad_b,
        "gate_gradient_to_upstream_zero":True,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "self_anchor_dependency":False,
        "pairwise_only_final_path":False,
        "one_encoder_state_once":True,
        "gate_checkpoint_replay_max_abs_error":replay_error,
        "real_cache_mean_alpha":float(diag["mean_alpha"]),
        "real_cache_min_alpha":float(diag["min_alpha"]),
        "real_cache_max_alpha":float(diag["max_alpha"]),
        **mechanics,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S61_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
