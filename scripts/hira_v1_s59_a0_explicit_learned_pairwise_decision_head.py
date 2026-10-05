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
from nmd.v1_explicit_pairwise_decision_head import (
    S59_PAIRWISE_PARAMETER_COUNT,
    ExplicitPairwiseDecisionHead,
    build_pairwise_representation,
    gold_pairwise_loss,
    pairwise_head_loss,
)
from hira_v1_s50_train_dev import _materialize_cache


SCHEMA_VERSION="hira-v1-s59-a0-explicit-learned-pairwise-decision-head-v1"
OUTCOME="HIRA_V1_S59_A0_EXPLICIT_LEARNED_PAIRWISE_DECISION_HEAD_READY"
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
        return f"S59 pairwise audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."
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
        Case("PH11","pairwise Hall scope","sensor","graphene Q11","Hall floor","0.13 nV","copper Q19","0.52 nV",25901),
        Case("PH22","pairwise axion lens","cavity","Nb A13","phase floor","0.07 mrad","steel A29","0.28 mrad",25902),
        Case("PH33","pairwise phonon map","ridge","AlN P17","shear floor","0.12 nstrain","Cu P31","0.48 nstrain",25903),
        Case("PH44","pairwise Rydberg clock","ensemble","Rb 151D","vorticity floor","1.3 nrad/mm2","warm cell","5.2 nrad/mm2",25904),
        Case("PH55","pairwise moire compass","stack","WSe2-X1","strain floor","0.21 ppm","bulk Si","0.84 ppm",25905),
        Case("PH66","pairwise neutron radar","analyzer","Ge N23","torsion floor","0.09 urad","polymer N41","0.36 urad",25906),
        Case("PH77","pairwise topological bridge","channel","Chern T17","impedance floor","1.2 mOhm","bulk T47","4.8 mOhm",25907),
        Case("PH88","pairwise molecular scope","beam","HfF M19","torque floor","0.08 prad","thermal M53","0.32 prad",25908),
        Case("PJ11","pairwise vacuum clock","surface","Ag V23","susceptibility floor","0.7 ppt","steel V59","2.8 ppt",25909),
        Case("PJ22","pairwise atomic lens","species","Sr A31","shear floor","1.1 nstrain","thermal A61","4.4 nstrain",25910),
        Case("PJ33","pairwise magnon map","guide","YIG G37","pressure floor","0.8 nPa","Ni G67","3.2 nPa",25911),
        Case("PJ44","pairwise optical ruler","interface","NV O41","recoil floor","0.05 pm","free O71","0.20 pm",25912),
        Case("PJ55","pairwise superfluid scope","fluid","He4 S43","vorticity floor","0.11 mHz","oil S73","0.44 mHz",25913),
        Case("PJ66","pairwise Casimir map","surface","AgSi C47","gradient floor","0.06 pN/mm","steel C79","0.24 pN/mm",25914),
        Case("PJ77","pairwise quantum compass","sensor","SiV Q53","recoil floor","0.07 pm","Hall Q83","0.28 pm",25915),
        Case("PJ88","pairwise ferro radar","crystal","LiNbO3 F59","torque floor","0.09 zNm","ceramic F89","0.36 zNm",25916),
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


def _representation(op,evidence):
    identity=op.identity_signatures(
        state_tokens=evidence.state_tokens,
        state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
    )
    context=op.joint_query_context(
        state_tokens=evidence.state_tokens,
        state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
        question_tokens=evidence.question_tokens,
        question_mask=evidence.question_mask,
    )
    return build_pairwise_representation(identity,context)


def _synthetic_court():
    g=torch.Generator().manual_seed(59059)
    head=ExplicitPairwiseDecisionHead()
    max_antisym=max_diag=max_mass=0.0
    for k in (3,7,255):
        rep=torch.randn(1,k,512,generator=g)
        pair=head.pairwise_logits(rep)
        score=head.aggregate_logits(rep)
        max_antisym=max(max_antisym,float((pair+pair.transpose(-1,-2)).abs().max()))
        max_diag=max(max_diag,float(torch.diagonal(pair,dim1=-2,dim2=-1).abs().max()))
        max_mass=max(max_mass,float((torch.softmax(score,-1).sum(-1)-1.0).abs().max()))

        perm=torch.randperm(k,generator=g)
        pp=head.pairwise_logits(rep[:,perm])
        ps=head.aggregate_logits(rep[:,perm])
        if not torch.allclose(pp,pair[:,perm][:,:,perm],atol=1e-6,rtol=0):
            raise RuntimeError(f"S59-A0 K={k} pair permutation failed")
        if not torch.allclose(ps,score[:,perm],atol=1e-6,rtol=0):
            raise RuntimeError(f"S59-A0 K={k} aggregate permutation failed")

    good=torch.tensor([[
        [0.0,2.0,1.0,3.0],
        [-2.0,0.0,0.2,-0.3],
        [-1.0,-0.2,0.0,0.5],
        [-3.0,0.3,-0.5,0.0],
    ]])
    gold=torch.tensor([0],dtype=torch.long)
    good_loss,diag=gold_pairwise_loss(good,gold)
    bad_loss,_=gold_pairwise_loss(-good,gold)
    if not float(good_loss)<float(bad_loss):
        raise RuntimeError("S59-A0 gold sign sensitivity failed")
    if int(diag["supervised_gold_pair_count"])!=3:
        raise RuntimeError("S59-A0 gold pair count changed")
    if int(diag["supervised_distractor_pair_count"])!=0:
        raise RuntimeError("S59-A0 distractor pair was supervised")

    tie_head=ExplicitPairwiseDecisionHead()
    with torch.no_grad():
        tie_head.A.zero_(); tie_head.u.zero_()
    rep=torch.eye(4,512).reshape(1,4,512)
    w1,d1=tie_head.select_with_tiebreak(rep)
    w2,d2=tie_head.select_with_tiebreak(rep)
    if not torch.equal(w1,w2):
        raise RuntimeError("S59-A0 tie break nondeterministic")

    return {
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "antisymmetry_max_abs_error":max_antisym,
        "diagonal_max_abs_error":max_diag,
        "max_probability_mass_error":max_mass,
        "correct_sign_loss":float(good_loss),
        "flipped_sign_loss":float(bad_loss),
        "supervised_gold_pair_count":int(diag["supervised_gold_pair_count"]),
        "supervised_distractor_pair_count":int(diag["supervised_distractor_pair_count"]),
        "deterministic_tie_winner":int(w1.item()),
        "top_score_tie_count":int(d1["top_score_tie_count"].item()),
        "representation_key_tie_count":int(d1["representation_key_tie_count"].item()),
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
        raise RuntimeError("S59-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S59-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S59-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S59-A0 checkpoint file changed")

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
    cache,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    correction=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    correction_before={
        k:v.detach().clone()
        for k,v in correction.correction_state_dict().items()
    }
    if correction.correction_parameter_count!=114688:
        raise RuntimeError("S59-A0 correction capacity changed")

    head1=ExplicitPairwiseDecisionHead()
    head2=ExplicitPairwiseDecisionHead()
    if head1.parameter_count!=S59_PAIRWISE_PARAMETER_COUNT:
        raise RuntimeError("S59-A0 pairwise capacity changed")
    if head1.state_dict().keys()!=head2.state_dict().keys() or not all(
        torch.equal(head1.state_dict()[k],head2.state_dict()[k])
        for k in head1.state_dict()
    ):
        raise RuntimeError("S59-A0 pairwise initialization changed")

    canonical,paraphrase,_n=cache[0]
    rep_c=_representation(correction,canonical)
    rep_p=_representation(correction,paraphrase)
    if rep_c.requires_grad or rep_p.requires_grad:
        raise RuntimeError("S59-A0 representation retained gradients")

    loss_c,diag_c=pairwise_head_loss(head1,rep_c,canonical.gold)
    loss_p,diag_p=pairwise_head_loss(head1,rep_p,paraphrase.gold)
    loss=0.5*(loss_c+loss_p)
    grads=torch.autograd.grad(
        loss,(head1.A,head1.u),allow_unused=True
    )
    grad_A=float(grads[0].abs().sum()) if grads[0] is not None else 0.0
    grad_u=float(grads[1].abs().sum()) if grads[1] is not None else 0.0
    if grad_A<=0.0 or grad_u<=0.0:
        raise RuntimeError("S59-A0 head gradients vanished")

    correction_after=correction.correction_state_dict()
    if correction_before.keys()!=correction_after.keys() or not all(
        torch.equal(correction_before[k],correction_after[k])
        for k in correction_before
    ):
        raise RuntimeError("S59-A0 pairwise loss altered correction trajectory")
    if any(p.grad is not None for p in correction.correction_parameters()):
        raise RuntimeError("S59-A0 correction received pairwise gradient")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0:
        raise RuntimeError("S59-A0 cache gained gradients")

    mechanics=_synthetic_court()

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    state=head1.state_dict_exact()
    checkpoint=out/"pairwise-head.pt"
    torch.save({
        "schema_version":"hira-v1-s59-pairwise-head-v1",
        "seed":SEED,
        "head_seed":80059,
        "pairwise_parameter_count":S59_PAIRWISE_PARAMETER_COUNT,
        "state_dict":state,
    },checkpoint)
    replay=ExplicitPairwiseDecisionHead()
    replay.load_state_dict_exact(state,freeze=True)
    replay_error=max(
        float((head1.pairwise_logits(rep_c)-replay.pairwise_logits(rep_c)).abs().max()),
        float((head1.pairwise_logits(rep_p)-replay.pairwise_logits(rep_p)).abs().max()),
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S59_A0_EXPLICIT_LEARNED_PAIRWISE_DECISION_HEAD_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "semantic_case_count":len(rows),
        "cache_digest":cache_digest,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "correction_trainable_parameter_count":114688,
        "pairwise_trainable_parameter_count":head1.trainable_parameter_count,
        "pairwise_parameter_count":head1.parameter_count,
        "pairwise_trainable_tensor_names":sorted(dict(head1.named_parameters())),
        "pairwise_has_bias":False,
        "pairwise_head_initialization_bit_identical":True,
        "pairwise_representation_requires_grad":False,
        "pairwise_gradient_A_l1":grad_A,
        "pairwise_gradient_u_l1":grad_u,
        "correction_received_pairwise_gradient":False,
        "correction_state_unchanged_after_pairwise_backward":True,
        "native_trainable_parameter_count":0,
        "teacher_dependency":False,
        "raw_native_logit_input_to_head":False,
        "raw_fused_logit_input_to_head":False,
        "raw_corrected_logit_input_to_head":False,
        "one_encoder_state_once":True,
        "head_checkpoint_replay_max_abs_error":replay_error,
        "real_cache_gold_pair_accuracy_canonical":float(diag_c["gold_pair_accuracy"]),
        "real_cache_gold_pair_accuracy_paraphrase":float(diag_p["gold_pair_accuracy"]),
        **mechanics,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S59_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
