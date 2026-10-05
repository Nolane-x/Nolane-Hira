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
from nmd.v1_pairwise_ranking_consistency import (
    detached_anchor_sign,
    decision_discrimination_diagnostics,
    pairwise_ordinal_consistency,
    weighted_pairwise_ordinal_auxiliary,
)
from hira_v1_s50_train_dev import _materialize_cache, _branch_logits


SCHEMA_VERSION="hira-v1-s57-a0-discrete-pairwise-ranking-consistency-v1"
OUTCOME="HIRA_V1_S57_A0_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_READY"
SEED=78_001


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
        return f"S57-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."
    @property
    def state_b(self):
        return f"S57 ordinal audit {self.case_id}: {self.second} belongs to {self.field_b}; {self.first} belongs to {self.field_a}."
    @property
    def qa1(self): return f"For S57-A0 {self.case_id}, identify {self.field_a}."
    @property
    def qa2(self): return f"Which S57-A0 entry is tagged {self.field_a} for {self.case_id}?"
    @property
    def qb1(self): return f"For S57-A0 {self.case_id}, identify {self.field_b}."
    @property
    def qb2(self): return f"Which S57-A0 entry is tagged {self.field_b} for {self.case_id}?"

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
                aliases=(f"{value} is the S57-A0 {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("OR11","muon gradient meter","sensor","diamond cavity","gradient floor","0.06 V/mm","steel cell","0.24 V/mm",22101),
        Case("OR22","spin recoil lens","film","PtYIG layer","recoil floor","0.04 um","bulk Ni","0.16 um",22102),
        Case("OR33","phonon pressure map","guide","soft AlN beam","pressure floor","1.2 nPa","Cu strip","4.8 nPa",22103),
        Case("OR44","Rydberg torque scope","ensemble","dressed Rb 70D","torque spread","0.03 zNm","hot vapor","0.12 zNm",22104),
        Case("OR55","moire phase clock","stack","aligned MoTe2-WS2","phase floor","0.04 rad","bulk Si","0.16 rad",22105),
        Case("OR66","neutron recoil camera","target","perfect Ge","recoil blur","0.10 eV","polymer tile","0.40 eV",22106),
        Case("OR77","topological pressure compass","channel","Chern edge","pressure leak","-48 dB","bulk bar","-20 dB",22107),
        Case("OR88","molecular curvature radar","beam","selected HfF","curvature drift","0.04 mm-1","thermal SO2","0.16 mm-1",22108),
        Case("OS11","vacuum gradient bridge","surface","template Ag","gradient floor","0.07 pN/mm","rough steel","0.28 pN/mm",22109),
        Case("OS22","atomic recoil lens","species","clocked Sr","recoil floor","0.02 mm/s","thermal K","0.08 mm/s",22110),
        Case("OS33","magnon torque scope","guide","low-loss YIG","torque floor","0.02 zNm","Ni strip","0.08 zNm",22111),
        Case("OS44","optical curvature ruler","interface","NV cavity","curvature loss","0.03 mm-1","free-space link","0.12 mm-1",22112),
        Case("OS55","superfluid gradient map","fluid","He4 film","gradient noise","0.05 V/mm","oil film","0.20 V/mm",22113),
        Case("OS66","Casimir recoil camera","surface","Ag-Si pair","recoil drift","0.09 pm","steel pair","0.36 pm",22114),
        Case("OS77","quantum torque compass","sensor","SiV resonator","torque floor","0.04 zNm","Hall bar","0.16 zNm",22115),
        Case("OS88","ferroelectric phase radar","crystal","poled LiNbO3","phase jitter","0.04 rad","ceramic slab","0.16 rad",22116),
    )


def _rows():
    out=[]
    for case in cases():
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s57-{case.case_id}",
            split="train",
            domain="s57_a0_only",
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


def _arms():
    return (
        JointStateQueryOptionPrivateCorrectionFork(train_correction=True),
        JointStateQueryOptionPrivateCorrectionFork(train_correction=True),
    )


def _synthetic_k_court():
    g=torch.Generator().manual_seed(78500)
    max_mass=0.0
    for k in (3,7,255):
        b=2
        op=JointStateQueryOptionPrivateCorrectionFork(train_correction=True)
        with torch.no_grad():
            op.adapter_b.normal_(generator=torch.Generator().manual_seed(78510+k),std=0.01)
            op.bilinear_weight.normal_(generator=torch.Generator().manual_seed(78520+k),std=0.01)
        data=dict(
            native_logits=torch.randn(b,k,generator=g),
            state_tokens=torch.randn(b,5,256,generator=g),
            state_mask=torch.ones(b,5,dtype=torch.bool),
            option_view_tokens=torch.randn(b,k,2,4,256,generator=g),
            option_view_token_mask=torch.ones(b,k,2,4,dtype=torch.bool),
            option_view_mask=torch.ones(b,k,2,dtype=torch.bool),
            question_tokens=torch.randn(b,5,256,generator=g),
            question_mask=torch.ones(b,5,dtype=torch.bool),
        )
        logits,_=op.correction_logits_from_state_option(**data)
        if tuple(logits.shape)!=(b,k) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("S57-A0 arbitrary-K failed")
        gold=torch.tensor([0,k-1],dtype=torch.long)
        aux,diag=weighted_pairwise_ordinal_auxiliary(
            logits,torch.flip(logits,dims=[-1]),gold,coefficient=0.05
        )
        if not bool(torch.isfinite(aux)) or not bool(torch.isfinite(diag["ordinal_loss"])):
            raise RuntimeError("S57-A0 arbitrary-K ordinal failed")
        mass=float((torch.softmax(logits,dim=-1).sum(-1)-1.0).abs().max())
        max_mass=max(max_mass,mass)
    if max_mass>1e-6:
        raise RuntimeError("S57-A0 probability mass changed")
    return max_mass


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
    if authority.get("runtime_state_sha256")!=expected_runtime:
        raise RuntimeError("S57-A0 parent runtime changed")
    if authority.get("checkpoint_file_sha256")!=expected_checkpoint:
        raise RuntimeError("S57-A0 parent checkpoint changed")
    if file_sha256(args.authority_checkpoint)!=expected_checkpoint:
        raise RuntimeError("S57-A0 checkpoint file changed")

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
        raise RuntimeError("S57-A0 loaded runtime changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S57-A0 native runtime remained trainable")

    rows=_rows()
    cache_pairs,cache_digest=_materialize_cache(runtime,rows)
    del runtime

    reference,treatment=_arms()
    if reference.correction_parameter_count!=114688 or treatment.correction_parameter_count!=114688:
        raise RuntimeError("S57-A0 private capacity changed")
    rs=reference.correction_state_dict()
    ts=treatment.correction_state_dict()
    if rs.keys()!=ts.keys() or not all(torch.equal(rs[k],ts[k]) for k in rs):
        raise RuntimeError("S57-A0 initialization differs")
    if reference.identity_parameter_count!=0 or treatment.identity_parameter_count!=0:
        raise RuntimeError("S57-A0 identity gained parameters")

    # Pure detached-anchor mechanics.
    probe=torch.tensor([[1.0,-2.0]],requires_grad=True)
    signs=detached_anchor_sign(probe)
    anchor_sign_detached=(signs.requires_grad is False)

    matching_a=torch.tensor([[4.0,1.0,-1.0,-3.0]],requires_grad=True)
    matching_b=torch.tensor([[3.5,0.8,-0.8,-2.5]],requires_grad=True)
    probe_gold=torch.tensor([0],dtype=torch.long)
    matching_loss,matching_diag=pairwise_ordinal_consistency(
        matching_a,matching_b,probe_gold
    )
    flip_loss,flip_diag=pairwise_ordinal_consistency(
        matching_a,-matching_b,probe_gold
    )

    offset_loss,_=pairwise_ordinal_consistency(
        matching_a+11.0,matching_b-7.0,probe_gold
    )
    scale_loss,_=pairwise_ordinal_consistency(
        matching_a*4.0,matching_b*0.5,probe_gold
    )
    offset_error=abs(float(offset_loss)-float(matching_loss))
    scale_error=abs(float(scale_loss)-float(matching_loss))

    wrong_gold_a=torch.tensor([[0.0,4.0,2.0,-2.0]])
    wrong_gold_b=torch.tensor([[3.0,2.0,-1.0,-3.0]])
    _wrong_loss,wrong_diag=pairwise_ordinal_consistency(
        wrong_gold_a,wrong_gold_b,probe_gold
    )
    correct_gold_a=torch.tensor([[4.0,1.0,0.0,-2.0]])
    correct_gold_b=torch.tensor([[3.0,0.5,-0.5,-2.5]])
    _correct_loss,correct_diag=pairwise_ordinal_consistency(
        correct_gold_a,correct_gold_b,probe_gold
    )

    flat_a=torch.zeros(1,4,requires_grad=True)
    flat_b=torch.zeros(1,4,requires_grad=True)
    flat_aux,flat_diag=weighted_pairwise_ordinal_auxiliary(
        flat_a,flat_b,probe_gold,coefficient=0.05
    )
    uniform_diag=decision_discrimination_diagnostics(torch.zeros(2,7))

    # Pure option permutation court with remapped gold.
    perm=torch.tensor([2,0,3,1])
    remapped=(perm==probe_gold.item()).nonzero(as_tuple=False).flatten().to(torch.long)
    permuted_loss,_=pairwise_ordinal_consistency(
        matching_a[:,perm],matching_b[:,perm],remapped
    )
    permutation_error=abs(float(permuted_loss)-float(matching_loss))

    # Real cache: reference exact zero, treatment gradient live.
    canonical,paraphrase,_n=cache_pairs[0]
    if not torch.equal(canonical.gold,paraphrase.gold):
        raise RuntimeError("S57-A0 paired gold changed")

    _rc,_src,fused_rc=_branch_logits(reference,"treatment",canonical)
    _rp,_srp,fused_rp=_branch_logits(reference,"treatment",paraphrase)
    ref_aux,_=weighted_pairwise_ordinal_auxiliary(
        fused_rc,fused_rp,canonical.gold,coefficient=0.0
    )
    if float(ref_aux)!=0.0:
        raise RuntimeError("S57-A0 reference auxiliary nonzero")

    _tc,_stc,fused_tc=_branch_logits(treatment,"treatment",canonical)
    _tp,_stp,fused_tp=_branch_logits(treatment,"treatment",paraphrase)
    trt_aux,trt_diag=weighted_pairwise_ordinal_auxiliary(
        fused_tc,fused_tp,canonical.gold,coefficient=0.05
    )
    grads=torch.autograd.grad(
        trt_aux,treatment.correction_parameters(),allow_unused=True
    )
    treatment_gradient_l1=sum(float(g.abs().sum()) for g in grads if g is not None)
    if treatment_gradient_l1<=0.0:
        raise RuntimeError("S57-A0 treatment ordinal gradient vanished")

    cache_inference=cache_requires_grad=0
    for c,p,_n in cache_pairs:
        for evidence in (c,p):
            for tensor in evidence.tensors():
                cache_inference+=int(torch.is_inference(tensor))
                cache_requires_grad+=int(tensor.requires_grad)
    if cache_inference!=0 or cache_requires_grad!=0:
        raise RuntimeError("S57-A0 cache ownership changed")

    max_mass=_synthetic_k_court()

    if not anchor_sign_detached:
        raise RuntimeError("S57-A0 anchor sign has gradient path")
    if not float(matching_loss)<float(flip_loss):
        raise RuntimeError("S57-A0 sign-flip separation failed")
    if offset_error>1e-6 or scale_error>3e-5:
        raise RuntimeError("S57-A0 standardization invariance failed")
    if int(wrong_diag["gold_filtered_direction_count"])<=0:
        raise RuntimeError("S57-A0 wrong gold anchor not filtered")
    if int(correct_diag["active_directional_anchor_count"])<=0:
        raise RuntimeError("S57-A0 correct gold anchor not retained")
    if float(wrong_diag["non_gold_active_direction_fraction"])<=0.0:
        raise RuntimeError("S57-A0 non-gold anchors disappeared")
    if float(flat_aux)!=0.0 or float(flat_diag["active_directional_anchor_fraction"])!=0.0:
        raise RuntimeError("S57-A0 flat anti-collapse failed")
    if permutation_error>1e-7:
        raise RuntimeError("S57-A0 option permutation failed")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S57_A0_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_ONLY",
        "seed":SEED,
        "parent_native_runtime_sha256":expected_runtime,
        "parent_native_checkpoint_sha256":expected_checkpoint,
        "native_trainable_parameter_count":0,
        "semantic_case_count":len(rows),
        "cache_digest":cache_digest,
        "cache_inference_tensor_count":cache_inference,
        "cache_requires_grad_tensor_count":cache_requires_grad,
        "reference_private_trainable_parameter_count":114688,
        "treatment_private_trainable_parameter_count":114688,
        "added_trainable_parameter_count":0,
        "identity_parameter_count":0,
        "initialization_bit_identical":True,
        "anchor_sign_detached":anchor_sign_detached,
        "matching_ordinal_loss":float(matching_loss.detach()),
        "sign_flip_ordinal_loss":float(flip_loss.detach()),
        "sign_flip_disagreement_fraction":float(flip_diag["sign_disagreement_fraction"]),
        "shared_offset_invariance_error":offset_error,
        "positive_scale_invariance_error":scale_error,
        "wrong_gold_filtered_direction_count":int(wrong_diag["gold_filtered_direction_count"]),
        "wrong_gold_filtered_direction_fraction":float(wrong_diag["gold_filtered_direction_fraction"]),
        "correct_gold_active_direction_count":int(correct_diag["active_directional_anchor_count"]),
        "non_gold_active_direction_fraction":float(wrong_diag["non_gold_active_direction_fraction"]),
        "flat_weighted_ordinal_auxiliary":float(flat_aux.detach()),
        "flat_active_directional_anchor_fraction":float(flat_diag["active_directional_anchor_fraction"]),
        "uniform_top1_top2_probability_gap":float(uniform_diag["mean_top1_top2_probability_gap"]),
        "uniform_logit_rms":float(uniform_diag["mean_unregularized_logit_rms"]),
        "option_permutation_loss_error":permutation_error,
        "reference_auxiliary_exact_zero":float(ref_aux)==0.0,
        "treatment_auxiliary_value":float(trt_aux.detach()),
        "treatment_auxiliary_gradient_l1":treatment_gradient_l1,
        "treatment_active_directional_anchor_fraction":float(trt_diag["active_directional_anchor_fraction"]),
        "treatment_gold_filtered_direction_fraction":float(trt_diag["gold_filtered_direction_fraction"]),
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
    print("HIRA_V1_S57_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
