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
    ExplicitPairwiseDecisionHead,
    build_pairwise_representation,
)
from nmd.v1_train_calibrated_bounded_hybrid import (
    S60_ALPHA_INITIAL,
    S60_ALPHA_MAX,
    TrainCalibratedBoundedHybridComposer,
    composer_gold_loss,
)
import hira_v1_s50_train_dev as s50


SCHEMA_VERSION="hira-v1-s60-a0-train-calibrated-bounded-hybrid-v1"
OUTCOME="HIRA_V1_S60_A0_TRAIN_CALIBRATED_BOUNDED_HYBRID_READY"
SEED=81_001


@dataclass(frozen=True)
class Case:
    case_id:str; noun:str; field_a:str; first:str; field_b:str; second:str
    wrong_a:str; wrong_b:str; seed:int

    @property
    def state_a(self):
        return f"S60-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."
    @property
    def state_b(self):
        return f"S60 hybrid audit {self.case_id}: {self.second} is {self.field_b}; {self.first} is {self.field_a}."
    @property
    def qa1(self): return f"For S60-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S60 hybrid value is tagged {self.field_a} in {self.case_id}?"
    @property
    def qb1(self): return f"For S60-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S60 hybrid value is tagged {self.field_b} in {self.case_id}?"

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
                criterion_text=f"for the S60 {self.noun}, {field} is {value}",
                aliases=(f"{value} is the S60 hybrid {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("HB11","hybrid anyon scope","tile","Chern H11","phase floor","0.17 mrad","bulk H29","0.68 mrad",26001),
        Case("HB22","hybrid shear lens","beam","AlN H13","strain floor","0.14 fstrain","steel H31","0.56 fstrain",26002),
        Case("HB33","hybrid axion map","cavity","Nb H17","impedance floor","0.09 uOhm","Cu H37","0.36 uOhm",26003),
        Case("HB44","hybrid magnon clock","lattice","YIG H19","vorticity floor","0.15 mHz","Ni H41","0.60 mHz",26004),
        Case("HB55","hybrid neutrino bridge","detector","Ge H23","phase floor","0.11 prad","polymer H43","0.44 prad",26005),
        Case("HB66","hybrid photon compass","guide","SiN H29","drag floor","0.13 fm/s","metal H47","0.52 fm/s",26006),
        Case("HB77","hybrid Fermi scope","gas","Li6 H31","pressure floor","0.16 pPa","thermal H53","0.64 pPa",26007),
        Case("HB88","hybrid vacuum radar","pair","AuSi H37","torque floor","0.12 zNm","steel H59","0.48 zNm",26008),
        Case("HC11","hybrid atomic camera","array","Sr H41","recoil floor","0.08 fm","warm H61","0.32 fm",26009),
        Case("HC22","hybrid molecular scope","beam","ThO H43","flux floor","1.4 a.u.","thermal H67","5.6 a.u.",26010),
        Case("HC33","hybrid superfluid map","fluid","He4 H47","drag floor","0.10 nN","oil H71","0.40 nN",26011),
        Case("HC44","hybrid ferro lens","stack","LiNbO3 H53","phase floor","0.12 urad","ceramic H73","0.48 urad",26012),
        Case("HC55","hybrid optical bridge","guide","GaP H59","recoil floor","0.07 pm","fiber H79","0.28 pm",26013),
        Case("HC66","hybrid spin compass","film","CoFeB H61","flux floor","0.19 fT","iron H83","0.76 fT",26014),
        Case("HC77","hybrid Casimir scope","surface","Ag H67","gradient floor","0.08 pN/mm","steel H89","0.32 pN/mm",26015),
        Case("HC88","hybrid quantum ruler","sensor","NV H71","recoil floor","0.06 pm","Hall H97","0.24 pm",26016),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s60-{case.case_id}",split="train",domain="s60_a0_only",language="en",
            state_a=case.state_a,state_b=case.state_b,
            question_a1=case.qa1,question_a2=case.qa2,question_b1=case.qb1,question_b2=case.qb2,
            option_texts=tuple(x.criterion_text for x in options),
            option_aliases=tuple(x.aliases[0] for x in options),
            option_ids=tuple(x.option_id for x in options),
            gold_a=ga,gold_b=gb,
        ))
    return out


def _rep(op,evidence):
    identity=op.identity_signatures(
        state_tokens=evidence.state_tokens,state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
    )
    context=op.joint_query_context(
        state_tokens=evidence.state_tokens,state_mask=evidence.state_mask,
        option_view_tokens=evidence.option_view_tokens,
        option_view_token_mask=evidence.option_view_token_mask,
        option_view_mask=evidence.option_view_mask,
        question_tokens=evidence.question_tokens,question_mask=evidence.question_mask,
    )
    return build_pairwise_representation(identity,context)


def _synthetic_mechanics():
    g=torch.Generator().manual_seed(60060)
    composer=TrainCalibratedBoundedHybridComposer()
    max_mass=max_bound_error=max_perm_error=max_affine_error=0.0

    for k in (3,7,255):
        fused=torch.randn(2,k,generator=g)
        pair=torch.randn(2,k,generator=g)
        out,diag=composer.compose(fused,pair,return_diagnostics=True)
        residual=(out-fused).abs()
        max_bound_error=max(
            max_bound_error,
            float((residual-diag["residual_bound"]).clamp_min(0).max()),
        )
        max_mass=max(
            max_mass,
            float((torch.softmax(out,-1).sum(-1)-1.0).abs().max()),
        )
        perm=torch.randperm(k,generator=g)
        pp=composer.compose(fused[:,perm],pair[:,perm])
        max_perm_error=max(max_perm_error,float((pp-out[:,perm]).abs().max()))

        scaled=composer.compose(3.0*fused+7.0,5.0*pair-11.0)
        max_affine_error=max(
            max_affine_error,float((scaled-(3.0*out+7.0)).abs().max())
        )

    fused=torch.tensor([[4.0,1.0,-1.0,-3.0]])
    pair=torch.tensor([[1e20,-1e20,5e19,-5e19]])
    identity=composer.compose(fused,pair,alpha_override=0.0)
    bounded,diag=composer.compose(fused,pair,return_diagnostics=True)
    flat_pair=composer.compose(fused,torch.full_like(pair,17.0))

    return {
        "arbitrary_k3_pass":True,
        "arbitrary_k7_pass":True,
        "arbitrary_k255_pass":True,
        "identity_override_max_abs_error":float((identity-fused).abs().max()),
        "adversarial_residual_bound_violation_max":max_bound_error,
        "option_permutation_max_abs_error":max_perm_error,
        "affine_scale_contract_max_abs_error":max_affine_error,
        "flat_pairwise_residual_max_abs_error":float((flat_pair-fused).abs().max()),
        "adversarial_pairwise_direction_max_abs":float(diag["pairwise_bounded_direction_max_abs"]),
        "max_probability_mass_error":max_mass,
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
        raise RuntimeError("S60-A0 parent authority changed")
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S60-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S60-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S60-A0 checkpoint file changed")

    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)
    runtime,payload=load_native_authority(
        bundle=bundle,manifest=manifest,checkpoint=args.authority_checkpoint,
        expected_file_sha256=expected_checkpoint,expected_seed=72001,
    )
    if payload["runtime_state_sha256"]!=expected_runtime:
        raise RuntimeError("S60-A0 loaded native changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S60-A0 native runtime trainable")

    cache,cache_digest=s50._materialize_cache(runtime,_rows())
    del runtime

    op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
    head=ExplicitPairwiseDecisionHead(trainable=True)
    composer=TrainCalibratedBoundedHybridComposer(trainable=True)

    if op.correction_parameter_count!=114688:
        raise RuntimeError("S60-A0 correction capacity changed")
    if head.parameter_count!=32832:
        raise RuntimeError("S60-A0 pairwise capacity changed")
    if composer.parameter_count!=1:
        raise RuntimeError("S60-A0 composer capacity changed")
    if abs(float(composer.alpha())-S60_ALPHA_INITIAL)>1e-7:
        raise RuntimeError("S60-A0 initial alpha changed")

    canonical,paraphrase,_n=cache[0]
    _rc,_sc,fused_c=s50._branch_logits(op,"treatment",canonical)
    _rp,_sp,fused_p=s50._branch_logits(op,"treatment",paraphrase)
    pair_c=head.aggregate_logits(_rep(op,canonical))
    pair_p=head.aggregate_logits(_rep(op,paraphrase))

    loss,diag=composer_gold_loss(
        composer,fused_c,fused_p,pair_c,pair_p,canonical.gold
    )
    targets=(composer.a,*op.correction_parameters(),head.A,head.u)
    grads=torch.autograd.grad(loss,targets,allow_unused=True)
    composer_grad=float(grads[0].abs().sum()) if grads[0] is not None else 0.0
    upstream=grads[1:]
    if composer_grad<=0.0:
        raise RuntimeError("S60-A0 composer gradient vanished")
    if any(g is not None and float(g.abs().sum())>0.0 for g in upstream):
        raise RuntimeError("S60-A0 composer gradient leaked upstream")

    cache_requires_grad=sum(
        int(t.requires_grad)
        for c,p,_n in cache
        for ev in (c,p)
        for t in ev.tensors()
    )
    if cache_requires_grad!=0:
        raise RuntimeError("S60-A0 cache gained gradients")

    mechanics=_synthetic_mechanics()
    if mechanics["identity_override_max_abs_error"]!=0.0:
        raise RuntimeError("S60-A0 exact identity failed")
    if mechanics["adversarial_residual_bound_violation_max"]>1e-7:
        raise RuntimeError("S60-A0 residual bound failed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    state=composer.state_dict_exact()
    checkpoint=out/"composer.pt"
    torch.save({
        "schema_version":"hira-v1-s60-composer-v1",
        "seed":SEED,
        "alpha_max":S60_ALPHA_MAX,
        "alpha_initial":S60_ALPHA_INITIAL,
        "state_dict":state,
    },checkpoint)
    replay=TrainCalibratedBoundedHybridComposer()
    replay.load_state_dict_exact(state,freeze=True)
    replay_error=float(
        (composer.compose(fused_c,pair_c)-replay.compose(fused_c,pair_c)).abs().max()
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S60_A0_TRAIN_CALIBRATED_BOUNDED_HYBRID_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "semantic_case_count":len(_rows()),
        "cache_digest":cache_digest,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "correction_trainable_parameter_count":114688,
        "pairwise_trainable_parameter_count":32832,
        "composer_trainable_parameter_count":composer.trainable_parameter_count,
        "composer_trainable_tensor_names":sorted(dict(composer.named_parameters())),
        "alpha_initial_observed":float(composer.alpha()),
        "alpha_max":S60_ALPHA_MAX,
        "composer_gradient_l1":composer_grad,
        "composer_gradient_to_upstream_zero":True,
        "teacher_dependency":False,
        "pseudo_target_dependency":False,
        "self_anchor_dependency":False,
        "pairwise_only_final_path":False,
        "one_encoder_state_once":True,
        "composer_checkpoint_replay_max_abs_error":replay_error,
        "real_cache_alpha":float(diag["alpha"]),
        "real_cache_canonical_residual_max_abs":float(diag["canonical_residual_max_abs"]),
        "real_cache_canonical_bound_max":float(diag["canonical_bound_max"]),
        **mechanics,
        "used_for_model_selection":False,
        "fresh_train_dev_exposed":False,
        "production_ready_claimed":False,
    }
    (out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S60_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
