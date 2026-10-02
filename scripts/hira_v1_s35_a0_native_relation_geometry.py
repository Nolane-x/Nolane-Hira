from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle,read_runtime_bundle_manifest
from nmd.v1_a13_lora import (
    a13_lora_state_dict,
    iter_a13_lora_modules,
    load_a13_lora_state_dict,
)
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_native_relation_geometry import NativeA13RelationCanonicalizer
from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    cross_view_relation_signature_loss,
)
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
)
from hira_v1_s17_train_dev import (
    BINDING_COEFFICIENT,
    BINDING_CONTRASTIVE_TEMPERATURE,
    CANONICALIZATION_COEFFICIENT,
    FUSION_EPSILON,
    PAIR_TEMPERATURE,
    ROLE_TEMPERATURE,
    SIGNATURE_SEPARATION_MARGIN,
    _decision_logits,
    _encode_batch,
    _gold_tensors,
    _losses,
)

SCHEMA_VERSION="hira-v1-s35-a0-native-relation-geometry-v1"
OUTCOME="HIRA_V1_S35_A0_NATIVE_RELATION_GEOMETRY_READY"


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
        return f"S35-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Record {self.case_id} stores {self.second} as {self.field_b}. The same S35-A0 {self.noun} stores {self.first} as {self.field_a}."

    @property
    def qa1(self):
        return f"For S35-A0 {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry is {self.field_a} in S35-A0 record {self.case_id}?"

    @property
    def qb1(self):
        return f"For S35-A0 {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry is {self.field_b} in S35-A0 record {self.case_id}?"

    def option_pack(self):
        rows=[
            ("a",self.field_a,self.first),("b",self.field_b,self.second),
            ("x",self.field_a,self.wrong_a),("y",self.field_b,self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        options=tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"for this {self.noun}, {field} is {value}",
                aliases=(f"{value} is the archived {field} entry for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("NG11","polarimeter","retarder","quartz","rotation","32 deg","polymer","8 deg",67101),
        Case("NG22","ion microscope","objective","electrostatic","energy","18 keV","magnetic","6 keV",67102),
        Case("NG33","nanocalorimeter","absorber","gold","decay","14 us","silicon","4 us",67103),
        Case("NG44","fiber gyroscope","coil","PM fiber","length","420 m","SMF","120 m",67104),
        Case("NG55","optical tweezer","trap","dual-beam","power","240 mW","single-beam","60 mW",67105),
        Case("NG66","mass analyzer","filter","quadrupole","scan","36 kHz","sector","9 kHz",67106),
        Case("NG77","microwave resonator","cavity","sapphire","Q factor","2e6","copper","5e5",67107),
        Case("NG88","ultrafast camera","sensor","streak","gate","8 ps","CMOS","2 ns",67108),
        Case("NH11","magnetometer","sensor","SERF","bandwidth","160 Hz","fluxgate","40 Hz",67109),
        Case("NH22","XUV spectrometer","grating","variable-line","range","90 eV","ruled","25 eV",67110),
        Case("NH33","microreactor","heater","Pt film","setpoint","480 C","nichrome","120 C",67111),
        Case("NH44","laser tracker","interferometer","heterodyne","range","40 m","homodyne","10 m",67112),
        Case("NH55","neutron counter","converter","B10","efficiency","72 pct","He3","18 pct",67113),
        Case("NH66","plasma probe","tip","tungsten","bias","48 V","molybdenum","12 V",67114),
        Case("NH77","photonic crystal sensor","lattice","hexagonal","pitch","640 nm","square","160 nm",67115),
        Case("NH88","quantum voltage source","junction","NbN","clock","80 GHz","Al","20 GHz",67116),
    )


def _rows(suite):
    out=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s35-{case.case_id}",split="train",domain="s35_a0_only",language="en",
            state_a=case.state_a,state_b=case.state_b,
            question_a1=case.qa1,question_a2=case.qa2,
            question_b1=case.qb1,question_b2=case.qb2,
            option_texts=tuple(x.criterion_text for x in options),
            option_aliases=tuple(x.aliases[0] for x in options),
            option_ids=tuple(x.option_id for x in options),
            gold_a=ga,gold_b=gb,
        ))
    return out


def _modules():
    control=CrossViewRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=BINDING_CONTRASTIVE_TEMPERATURE,
    )
    native=NativeA13RelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=BINDING_CONTRASTIVE_TEMPERATURE,
        native_dimension=256,
    )
    return control,native


def _views(runtime,rows):
    encoded=_encode_batch(runtime,rows)
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S35-A0 projection scorer missing")
    control,native=_modules()

    def one(state_prefix,question_prefix):
        st=encoded[f"state_{state_prefix}_tokens"].repeat_interleave(2,dim=0)
        sm=encoded[f"state_{state_prefix}_mask"].repeat_interleave(2,dim=0)
        qt=encoded[f"question_{question_prefix}_tokens"]
        qm=encoded[f"question_{question_prefix}_mask"]
        ov=encoded["option_tokens"].repeat_interleave(2,dim=0)
        ovtm=encoded["option_mask"].repeat_interleave(2,dim=0)
        ovm=encoded["option_view_mask"].repeat_interleave(2,dim=0)
        c_logits,c_sig,c_diag=control(
            projection=scorer.projection,
            state_tokens=st,state_mask=sm,
            question_tokens=qt,question_mask=qm,
            option_view_tokens=ov,option_view_token_mask=ovtm,option_view_mask=ovm,
        )
        n_logits,n_sig,n_diag=native(
            state_tokens=st,state_mask=sm,
            question_tokens=qt,question_mask=qm,
            option_view_tokens=ov,option_view_token_mask=ovtm,option_view_mask=ovm,
        )
        return c_logits,c_sig,c_diag,n_logits,n_sig,n_diag

    return encoded,one("a","canonical"),one("b","paraphrase")


def _identity_algorithm_court():
    g=torch.Generator().manual_seed(73501)
    state=torch.randn(2,4,256,generator=g)
    question=torch.randn(2,3,256,generator=g)
    options=torch.randn(2,4,2,3,256,generator=g)
    sm=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
    qm=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
    otm=torch.ones(2,4,2,3,dtype=torch.bool)
    otm[:,:,1,-1]=False
    ovm=torch.ones(2,4,2,dtype=torch.bool)
    identity=nn.Linear(256,256,bias=False)
    with torch.no_grad():
        identity.weight.copy_(torch.eye(256))
    control,native=_modules()
    cl,cs,_=control(
        projection=identity,state_tokens=state,state_mask=sm,
        question_tokens=question,question_mask=qm,
        option_view_tokens=options,option_view_token_mask=otm,option_view_mask=ovm,
    )
    nl,ns,_=native(
        state_tokens=state,state_mask=sm,
        question_tokens=question,question_mask=qm,
        option_view_tokens=options,option_view_token_mask=otm,option_view_mask=ovm,
    )
    logit_error=float((cl-nl).abs().max())
    signature_error=float((cs-ns).abs().max())
    if logit_error!=0.0 or signature_error!=0.0:
        raise RuntimeError("S35-A0 native algorithm differs from identity-projected S13")

    def call(st=state,stm=sm,q=question,qm_=qm,opt=options,optm=otm,viewm=ovm):
        return native(
            state_tokens=st,state_mask=stm,
            question_tokens=q,question_mask=qm_,
            option_view_tokens=opt,option_view_token_mask=optm,option_view_mask=viewm,
        )

    base_l,base_s,_=call()
    sp=torch.tensor([2,0,3,1])
    qp=torch.tensor([2,0,1])
    tp=torch.tensor([2,0,1])
    kp=torch.tensor([2,0,3,1])

    sl,ss,_=call(st=state[:,sp],stm=sm[:,sp])
    ql,qs,_=call(q=question[:,qp],qm_=qm[:,qp])
    tl,ts,_=call(opt=options[:,:,:,tp],optm=otm[:,:,:,tp])
    kl,ks,_=call(opt=options[:,kp],optm=otm[:,kp],viewm=ovm[:,kp])

    padded_state=torch.cat([state,torch.randn(2,2,256,generator=g)],dim=1)
    padded_sm=torch.cat([sm,torch.zeros(2,2,dtype=torch.bool)],dim=1)
    padded_q=torch.cat([question,torch.randn(2,2,256,generator=g)],dim=1)
    padded_qm=torch.cat([qm,torch.zeros(2,2,dtype=torch.bool)],dim=1)
    padded_o=torch.cat([options,torch.randn(2,4,2,2,256,generator=g)],dim=3)
    padded_om=torch.cat([otm,torch.zeros(2,4,2,2,dtype=torch.bool)],dim=3)
    pl,ps,_=call(st=padded_state,stm=padded_sm,q=padded_q,qm_=padded_qm,opt=padded_o,optm=padded_om)

    one=torch.ones(1,1,256)
    dl,ds,_=native(
        state_tokens=one,state_mask=torch.ones(1,1,dtype=torch.bool),
        question_tokens=one,question_mask=torch.ones(1,1,dtype=torch.bool),
        option_view_tokens=torch.ones(1,2,1,1,256),
        option_view_token_mask=torch.ones(1,2,1,1,dtype=torch.bool),
        option_view_mask=torch.ones(1,2,1,dtype=torch.bool),
    )

    checks={
        "state_token_permutation_logit_error":float((sl-base_l).abs().max()),
        "state_token_permutation_signature_error":float((ss-base_s).abs().max()),
        "question_token_permutation_logit_error":float((ql-base_l).abs().max()),
        "question_token_permutation_signature_error":float((qs-base_s).abs().max()),
        "option_token_permutation_logit_error":float((tl-base_l).abs().max()),
        "option_token_permutation_signature_error":float((ts-base_s).abs().max()),
        "logical_option_permutation_logit_error":float((kl-base_l[:,kp]).abs().max()),
        "logical_option_permutation_signature_error":float((ks-base_s[:,kp]).abs().max()),
        "masked_padding_logit_error":float((pl-base_l).abs().max()),
        "masked_padding_signature_error":float((ps-base_s).abs().max()),
    }
    if any(v>1e-6 for v in checks.values()):
        raise RuntimeError(f"S35-A0 native equivariance failed: {checks}")
    if not bool(torch.isfinite(dl).all() and torch.isfinite(ds).all()):
        raise RuntimeError("S35-A0 degenerate native geometry non-finite")
    return {
        "identity_projection_logit_max_abs":logit_error,
        "identity_projection_signature_max_abs":signature_error,
        **checks,
        "degenerate_finite":True,
    }


def _projection_perturbation_court(runtime,rows):
    encoded,c,p=_views(runtime,rows)
    cc,cs,_cd,nc,ns,_nd=c
    pc,ps,_pd,np_,nsp,_npd=p
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S35-A0 projection scorer missing")

    raw_c=_decision_logits(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    raw_p=_decision_logits(
        runtime,
        state_tokens=encoded["state_b_tokens"],
        state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )
    primary_identity=0.0

    original=scorer.projection.weight.detach().clone()
    with torch.no_grad():
        rowscale=torch.linspace(0.2,3.0,steps=original.shape[0],device=original.device,dtype=original.dtype)[:,None]
        perturb=0.07*torch.sin(
            torch.arange(original.numel(),device=original.device,dtype=original.dtype)
        ).reshape_as(original)
        scorer.projection.weight.copy_(original*rowscale+perturb)

    _encoded2,c2,p2=_views(runtime,rows)
    cc2,cs2,_cd2,nc2,ns2,_nd2=c2
    pc2,ps2,_pd2,np2,nsp2,_npd2=p2
    raw_c2=_decision_logits(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    raw_p2=_decision_logits(
        runtime,
        state_tokens=encoded["state_b_tokens"],
        state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )

    with torch.no_grad():
        scorer.projection.weight.copy_(original)

    native_logit=max(float((nc2-nc).abs().max()),float((np2-np_).abs().max()))
    native_sig=max(float((ns2-ns).abs().max()),float((nsp2-nsp).abs().max()))
    control_logit=max(float((cc2-cc).abs().max()),float((pc2-pc).abs().max()))
    control_sig=max(float((cs2-cs).abs().max()),float((ps2-ps).abs().max()))
    primary_change=max(float((raw_c2-raw_c).abs().max()),float((raw_p2-raw_p).abs().max()))

    if native_logit!=0.0 or native_sig!=0.0:
        raise RuntimeError("S35-A0 native relation depends on shared projection")
    if control_logit<=1e-6 or control_sig<=1e-6:
        raise RuntimeError("S35-A0 control failed projection sensitivity court")
    if primary_change<=1e-6:
        raise RuntimeError("S35-A0 primary failed projection sensitivity court")

    return {
        "primary_logit_identity_max_abs":primary_identity,
        "control_projection_perturbation_logit_max_abs":control_logit,
        "control_projection_perturbation_signature_max_abs":control_sig,
        "native_projection_perturbation_logit_max_abs":native_logit,
        "native_projection_perturbation_signature_max_abs":native_sig,
        "primary_projection_perturbation_logit_max_abs":primary_change,
        "control_signature_dimension":int(cs.shape[-1]),
        "native_signature_dimension":int(ns.shape[-1]),
    }


def _gradient_probe(bundle,manifest,rows):
    with torch.inference_mode(False),torch.enable_grad():
        fresh=load_hira_v0_m4_bundle(bundle)
        runtime=build_hira_v1_s17_norm_balanced_core(
            fresh.runtime.encoder,
            bundle/str(manifest["t0_checkpoint"]),
            expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
            train_lora=True,train_projection=True,
        )
        del fresh
        runtime.clear_schema_cache()
        runtime.eval()
        trainable=[p for p in runtime.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable)!=HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S35-A0 trainable surface changed")

        encoded=_encode_batch(runtime,rows)
        native=_modules()[1]

        def relation_one(state_prefix,question_prefix):
            return native(
                state_tokens=encoded[f"state_{state_prefix}_tokens"].repeat_interleave(2,dim=0),
                state_mask=encoded[f"state_{state_prefix}_mask"].repeat_interleave(2,dim=0),
                question_tokens=encoded[f"question_{question_prefix}_tokens"],
                question_mask=encoded[f"question_{question_prefix}_mask"],
                option_view_tokens=encoded["option_tokens"].repeat_interleave(2,dim=0),
                option_view_token_mask=encoded["option_mask"].repeat_interleave(2,dim=0),
                option_view_mask=encoded["option_view_mask"].repeat_interleave(2,dim=0),
            )

        rc,sc,_=relation_one("a","canonical")
        rp,sp,_=relation_one("b","paraphrase")
        gold,_other=_gold_tensors(rows,device=rc.device)
        relation_ce=0.5*(F.cross_entropy(rc,gold)+F.cross_entropy(rp,gold))
        canonicalization,_align,_sep=cross_view_relation_signature_loss(
            sc,sp,separation_margin=SIGNATURE_SEPARATION_MARGIN
        )
        relation_block=(
            BINDING_COEFFICIENT*relation_ce
            +CANONICALIZATION_COEFFICIENT*canonicalization
        )

        relation_raw=torch.autograd.grad(
            relation_block,trainable,retain_graph=True,allow_unused=True
        )
        relation_grads=[
            torch.zeros_like(p) if g is None else g
            for p,g in zip(trainable,relation_raw)
        ]
        index={id(p):i for i,p in enumerate(trainable)}
        modules=iter_a13_lora_modules(runtime.encoder)
        relation_lora=sum(
            float(relation_grads[index[id(m.lora_b)]].abs().sum().cpu())
            for m in modules
        )
        scorer=runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S35-A0 projection scorer missing")
        relation_projection=float(
            relation_grads[index[id(scorer.projection.weight)]].abs().sum().cpu()
        )

        # Restore the standard S17 relation hook for the primary-block court.
        import hira_v1_s17_train_dev as s17mod
        total,primary_block,_rb,*_rest=_losses(runtime,rows)
        primary_grad=torch.autograd.grad(
            primary_block,scorer.projection.weight,allow_unused=False
        )[0]
        primary_projection=float(primary_grad.abs().sum().cpu())

        if relation_projection!=0.0:
            raise RuntimeError("S35-A0 native relation leaked direct gradient into projection")
        if relation_lora<=0.0:
            raise RuntimeError("S35-A0 native relation LoRA gradient vanished")
        if primary_projection<=0.0:
            raise RuntimeError("S35-A0 primary projection gradient vanished")

        return {
            "relation_block":float(relation_block.detach().cpu()),
            "relation_ce":float(relation_ce.detach().cpu()),
            "canonicalization":float(canonicalization.detach().cpu()),
            "relation_to_projection_gradient_l1":relation_projection,
            "relation_to_lora_gradient_l1":relation_lora,
            "primary_to_projection_gradient_l1":primary_projection,
        }


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    suite=cases()
    if len(suite)!=16:
        raise RuntimeError("S35-A0 suite size changed")
    rows=_rows(suite)
    bundle=args.bundle.resolve()
    manifest=read_runtime_bundle_manifest(bundle)

    fresh=load_hira_v0_m4_bundle(bundle)
    runtime=build_hira_v1_s17_norm_balanced_core(
        fresh.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,train_projection=False,
    )
    del fresh
    runtime.clear_schema_cache()
    runtime.eval()

    encoded,c,p=_views(runtime,rows)
    cc,cs,_cd,nc,ns,_nd=c
    pc,ps,_pd,np_,nsp,_npd=p
    if cs.shape[-1]!=128 or ns.shape[-1]!=256:
        raise RuntimeError("S35-A0 relation signature width changed")

    fusion=GradientIsolatedFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    raw_c=_decision_logits(
        runtime,state_tokens=encoded["state_a_tokens"],state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    raw_p=_decision_logits(
        runtime,state_tokens=encoded["state_b_tokens"],state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )
    fused_c,_=fusion(raw_c,nc)
    fused_p,_=fusion(raw_p,np_)
    mass=max(
        float((torch.softmax(fused_c,dim=-1).sum(-1)-1).abs().max().cpu()),
        float((torch.softmax(fused_p,dim=-1).sum(-1)-1).abs().max().cpu()),
    )
    if mass>1e-6:
        raise RuntimeError("S35-A0 probability mass changed")

    operator=_modules()[1]
    physical=HIRA_V1_S17_TOTAL_PARAMETER_COUNT
    if operator.parameter_count!=0 or physical!=49_152:
        raise RuntimeError("S35-A0 capacity changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S35-A0 frozen runtime became trainable")
    original=sum(
        p.numel() for name,p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original!=0:
        raise RuntimeError("S35-A0 original A13 became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S35-A0 HIRACore became trainable")

    state=a13_lora_state_dict(runtime.encoder)
    replay_fresh=load_hira_v0_m4_bundle(bundle)
    replay=build_hira_v1_s17_norm_balanced_core(
        replay_fresh.runtime.encoder,
        bundle/str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,train_projection=False,
    )
    del replay_fresh
    load_a13_lora_state_dict(replay.encoder,state,freeze=True)
    replay_state=a13_lora_state_dict(replay.encoder)
    checkpoint_roundtrip=all(torch.equal(state[k],replay_state[k]) for k in state)
    if not checkpoint_roundtrip:
        raise RuntimeError("S35-A0 checkpoint roundtrip changed")

    identity=_identity_algorithm_court()
    perturb=_projection_perturbation_court(runtime,rows)
    gradient=_gradient_probe(bundle,manifest,rows)

    gold,_=_gold_tensors(rows,device=nc.device)
    native_accuracy=0.5*(
        float((nc.argmax(-1)==gold).float().mean().cpu())
        +float((np_.argmax(-1)==gold).float().mean().cpu())
    )

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S35_A0_NATIVE_A13_RELATION_GEOMETRY_ONLY",
        "semantic_case_count":len(suite),
        "state_view_count":2*len(suite),
        "k":4,
        "views_per_option":2,
        "native_dimension":256,
        "role_temperature":ROLE_TEMPERATURE,
        "pair_temperature":PAIR_TEMPERATURE,
        "contrastive_temperature":BINDING_CONTRASTIVE_TEMPERATURE,
        "operator_added_parameter_count":operator.parameter_count,
        "physical_trainable_parameter_count":physical,
        "lora_parameter_count":16_384,
        "projection_parameter_count":HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
        "runtime_trainable_parameter_count":0,
        "original_a13_trainable_parameter_count":original,
        "hira_core_trainable_parameter_count":0,
        "checkpoint_key_count":len(state),
        "checkpoint_roundtrip_exact":checkpoint_roundtrip,
        "native_probability_mass_error":mass,
        "full_k":fused_c.shape[-1]==4 and fused_p.shape[-1]==4,
        "state_once_expected_view_count":2*len(suite),
        "native_a0_relation_accuracy":native_accuracy,
        **identity,
        **perturb,
        **gradient,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S35_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
