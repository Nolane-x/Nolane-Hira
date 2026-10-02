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
from nmd.v1_a13_lora import a13_lora_state_dict,iter_a13_lora_modules,load_a13_lora_state_dict
from nmd.v1_entropic_relation_transport import QueryConditionedEntropicRelationTransport
from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    cross_view_relation_signature_loss,
    relation_signature_same_option_cosine,
)
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
)
from hira_v1_s17_train_dev import (
    BINDING_COEFFICIENT,
    CANONICALIZATION_COEFFICIENT,
    BINDING_CONTRASTIVE_TEMPERATURE,
    PAIR_TEMPERATURE,
    ROLE_TEMPERATURE,
    SIGNATURE_SEPARATION_MARGIN,
    _encode_batch,
    _gold_tensors,
)

SCHEMA_VERSION="hira-v1-s34-a0-entropic-relation-transport-v1"
OUTCOME="HIRA_V1_S34_A0_ENTROPIC_RELATION_TRANSPORT_READY"


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
        return f"S34-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Record {self.case_id} assigns {self.second} to {self.field_b}. The S34-A0 {self.noun} separately assigns {self.first} to {self.field_a}."

    @property
    def qa1(self):
        return f"For S34-A0 {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which S34-A0 entry belongs to {self.field_a} in {self.case_id}?"

    @property
    def qb1(self):
        return f"For S34-A0 {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which S34-A0 entry belongs to {self.field_b} in {self.case_id}?"

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
                aliases=(f"{value} is the archived {field} value for this {self.noun}",),
            )
            for i,(_kind,field,value) in enumerate(rows)
        )
        ga=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="a")
        gb=next(i for i,(kind,_f,_v) in enumerate(rows) if kind=="b")
        return options,ga,gb


def cases():
    return (
        Case("ET11","muon telescope","absorber","lead","coincidence","18 ns","iron","6 ns",65101),
        Case("ET22","acoustic levitator","array","phased","drive","42 kHz","single","12 kHz",65102),
        Case("ET33","cold-atom camera","sensor","EMCCD","exposure","24 ms","CMOS","6 ms",65103),
        Case("ET44","terahertz imager","detector","bolometer","band","0.9 THz","Schottky","0.3 THz",65104),
        Case("ET55","microfluidic sorter","actuator","dielectrophoretic","rate","36 kHz","pressure","9 kHz",65105),
        Case("ET66","quantum magnetometer","center","NV ensemble","bias","28 mT","alkali vapor","7 mT",65106),
        Case("ET77","electron spectrometer","analyzer","hemispherical","pass energy","16 eV","TOF","4 eV",65107),
        Case("ET88","optical comb","reference","ULE cavity","spacing","250 MHz","atomic line","50 MHz",65108),
        Case("EU11","gamma camera","scintillator","LaBr3","frame","600 fps","NaI","150 fps",65109),
        Case("EU22","micropositioner","drive","piezo","travel","80 um","voice coil","20 um",65110),
        Case("EU33","vacuum gauge","sensor","Bayard-Alpert","range","1e-9 mbar","Pirani","1e-4 mbar",65111),
        Case("EU44","Raman probe","laser","785 nm","power","160 mW","532 nm","40 mW",65112),
        Case("EU55","ion trap","geometry","surface","depth","320 meV","linear","80 meV",65113),
        Case("EU66","thermal imager","array","InSb","rate","240 Hz","VOx","60 Hz",65114),
        Case("EU77","frequency counter","reference","rubidium","gate","4 s","TCXO","1 s",65115),
        Case("EU88","spin qubit reader","sensor","rf-SET","bandwidth","8 MHz","QPC","2 MHz",65116),
    )


def _rows(suite):
    out=[]
    for case in suite:
        options,ga,gb=case.option_pack()
        out.append(S17FusionCase(
            case_id=f"a0-s34-{case.case_id}",split="train",domain="s34_a0_only",language="en",
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
    treatment=QueryConditionedEntropicRelationTransport(
        state_relevance_temperature=0.10,
        option_relevance_temperature=0.10,
        kernel_temperature=0.10,
        logit_temperature=0.10,
        sinkhorn_iterations=8,
        epsilon=1e-12,
    )
    return control,treatment


def _outputs(runtime,rows):
    encoded=_encode_batch(runtime,rows)
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S34-A0 projection scorer missing")
    control,treatment=_modules()

    def one(state_prefix,question_prefix):
        st=encoded[f"state_{state_prefix}_tokens"].repeat_interleave(2,dim=0)
        sm=encoded[f"state_{state_prefix}_mask"].repeat_interleave(2,dim=0)
        qt=encoded[f"question_{question_prefix}_tokens"]
        qm=encoded[f"question_{question_prefix}_mask"]
        ov=encoded["option_tokens"].repeat_interleave(2,dim=0)
        ovtm=encoded["option_mask"].repeat_interleave(2,dim=0)
        ovm=encoded["option_view_mask"].repeat_interleave(2,dim=0)
        c_logits,c_sig,_=control(
            projection=scorer.projection,state_tokens=st,state_mask=sm,
            question_tokens=qt,question_mask=qm,
            option_view_tokens=ov,option_view_token_mask=ovtm,option_view_mask=ovm,
        )
        t_logits,t_sig,t_diag,parts=treatment.forward_with_components(
            projection=scorer.projection,state_tokens=st,state_mask=sm,
            question_tokens=qt,question_mask=qm,
            option_view_tokens=ov,option_view_token_mask=ovtm,option_view_mask=ovm,
        )
        return c_logits,c_sig,t_logits,t_sig,t_diag,parts

    return encoded,one("a","canonical"),one("b","paraphrase")


def _controlled_transport_court():
    projection=nn.Linear(4,4,bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(4))
    module=QueryConditionedEntropicRelationTransport()
    state=torch.tensor([[[1.,0.,0.,0.],[0.,1.,0.,0.]]])
    sm=torch.tensor([[True,True]])
    qa=torch.tensor([[[1.,0.,0.,0.]]])
    qb=torch.tensor([[[0.,1.,0.,0.]]])
    qm=torch.tensor([[True]])
    options=torch.tensor([[
        [[[1.,0.,0.,0.],[0.,0.,1.,0.]]],
        [[[0.,1.,0.,0.],[0.,0.,0.,1.]]],
    ]])
    otm=torch.ones(1,2,1,2,dtype=torch.bool)
    ovm=torch.ones(1,2,1,dtype=torch.bool)

    def call(q,st=state,stm=sm,opt=options,optm=otm,viewm=ovm):
        return module.forward_with_components(
            projection=projection,state_tokens=st,state_mask=stm,
            question_tokens=q,question_mask=qm,
            option_view_tokens=opt,option_view_token_mask=optm,option_view_mask=viewm,
        )

    la,sa,da,a=call(qa)
    lb,sb,db,b=call(qb)
    if int(la.argmax(-1).item())!=0 or int(lb.argmax(-1).item())!=1:
        raise RuntimeError("S34-A0 controlled query switch failed")

    sp=torch.tensor([1,0])
    lsp,ssp,_dsp,_=call(qa,st=state[:,sp],stm=sm[:,sp])
    state_perm_logit=float((lsp-la).abs().max())
    state_perm_sig=float((ssp-sa).abs().max())

    tp=torch.tensor([1,0])
    ltp,stp,_dtp,_=call(qa,opt=options[:,:,:,tp],optm=otm[:,:,:,tp])
    token_perm_logit=float((ltp-la).abs().max())
    token_perm_sig=float((stp-sa).abs().max())

    kp=torch.tensor([1,0])
    lkp,skp,_dkp,_=call(qa,opt=options[:,kp],optm=otm[:,kp],viewm=ovm[:,kp])
    logical_perm_logit=float((lkp-la[:,kp]).abs().max())
    logical_perm_sig=float((skp-sa[:,kp]).abs().max())

    ps=torch.cat([state,torch.randn(1,2,4)],dim=1)
    psm=torch.cat([sm,torch.zeros(1,2,dtype=torch.bool)],dim=1)
    po=torch.cat([options,torch.randn(1,2,1,2,4)],dim=3)
    pom=torch.cat([otm,torch.zeros(1,2,1,2,dtype=torch.bool)],dim=3)
    lpad,spad,_dpad,_=call(qa,st=ps,stm=psm,opt=po,optm=pom)
    padding_logit=float((lpad-la).abs().max())
    padding_sig=float((spad-sa).abs().max())

    one_state=state[:,:1]
    one_sm=sm[:,:1]
    dup=options.clone()
    dup[:,:,:,:,]=dup[:,:,:,0:1,:]
    ld,sd,dd,_=call(qa,st=one_state,stm=one_sm,opt=dup)
    degenerate_finite=bool(torch.isfinite(ld).all() and torch.isfinite(sd).all())

    for name,value in (
        ("state permutation logit",state_perm_logit),
        ("state permutation signature",state_perm_sig),
        ("option-token permutation logit",token_perm_logit),
        ("option-token permutation signature",token_perm_sig),
        ("logical-option permutation logit",logical_perm_logit),
        ("logical-option permutation signature",logical_perm_sig),
        ("padding logit",padding_logit),
        ("padding signature",padding_sig),
    ):
        if value>1e-6:
            raise RuntimeError(f"S34-A0 {name} changed: {value}")
    if not degenerate_finite:
        raise RuntimeError("S34-A0 degenerate transport became non-finite")

    return {
        "controlled_q_a_selected_option":int(la.argmax(-1).item()),
        "controlled_q_b_selected_option":int(lb.argmax(-1).item()),
        "controlled_state_marginal_intervention_max_abs":float(
            (a["state_marginal"]-b["state_marginal"]).abs().max()
        ),
        "controlled_option_marginal_intervention_max_abs":float(
            (a["option_marginal"]-b["option_marginal"]).abs().max()
        ),
        "controlled_transport_plan_intervention_max_abs":float(
            (a["transport_plan"]-b["transport_plan"]).abs().max()
        ),
        "controlled_state_token_permutation_logit_error":state_perm_logit,
        "controlled_state_token_permutation_signature_error":state_perm_sig,
        "controlled_option_token_permutation_logit_error":token_perm_logit,
        "controlled_option_token_permutation_signature_error":token_perm_sig,
        "controlled_logical_option_permutation_logit_error":logical_perm_logit,
        "controlled_logical_option_permutation_signature_error":logical_perm_sig,
        "controlled_padding_logit_error":padding_logit,
        "controlled_padding_signature_error":padding_sig,
        "controlled_max_row_marginal_residual":max(
            float(da.max_row_marginal_residual),float(db.max_row_marginal_residual)
        ),
        "controlled_max_column_marginal_residual":max(
            float(da.max_column_marginal_residual),float(db.max_column_marginal_residual)
        ),
        "controlled_degenerate_finite":degenerate_finite,
        "controlled_degenerate_row_residual":float(dd.max_row_marginal_residual),
        "controlled_degenerate_column_residual":float(dd.max_column_marginal_residual),
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
            raise RuntimeError("S34-A0 trainable surface changed")

        _encoded,c,p=_outputs(runtime,rows)
        _cc,_cs,tc,tsc,_tdc,_cparts=c
        _pc,_ps,tp,tsp,_tdp,_pparts=p
        gold,_other=_gold_tensors(rows,device=tc.device)
        relation_loss=0.5*(F.cross_entropy(tc,gold)+F.cross_entropy(tp,gold))
        canonicalization,_align,_sep=cross_view_relation_signature_loss(
            tsc,tsp,separation_margin=SIGNATURE_SEPARATION_MARGIN
        )
        relation_block=(
            BINDING_COEFFICIENT*relation_loss
            +CANONICALIZATION_COEFFICIENT*canonicalization
        )
        grads=torch.autograd.grad(relation_block,trainable,allow_unused=True)
        grads=[torch.zeros_like(p) if g is None else g for p,g in zip(trainable,grads)]
        index={id(p):i for i,p in enumerate(trainable)}
        modules=iter_a13_lora_modules(runtime.encoder)
        b_l1=[float(grads[index[id(m.lora_b)]].abs().sum().cpu()) for m in modules]
        scorer=runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S34-A0 projection scorer missing")
        projection_l1=float(grads[index[id(scorer.projection.weight)]].abs().sum().cpu())
        if any(v<=0.0 for v in b_l1) or projection_l1<=0.0:
            raise RuntimeError("S34-A0 intended gradients vanished")
        return {
            "relation_block":float(relation_block.detach().cpu()),
            "relation_ce":float(relation_loss.detach().cpu()),
            "canonicalization":float(canonicalization.detach().cpu()),
            "lora_b_gradient_l1":b_l1,
            "projection_gradient_l1":projection_l1,
        }


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()

    suite=cases()
    if len(suite)!=16:
        raise RuntimeError("S34-A0 suite size changed")
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

    _encoded,c,p=_outputs(runtime,rows)
    _cc,_cs,tc,tsc,tdc,cparts=c
    _pc,_ps,tp,tsp,tdp,pparts=p

    max_row=max(float(tdc.max_row_marginal_residual),float(tdp.max_row_marginal_residual))
    max_col=max(float(tdc.max_column_marginal_residual),float(tdp.max_column_marginal_residual))
    if max_row>1e-4 or max_col>1e-4:
        raise RuntimeError(f"S34-A0 Sinkhorn residual too high: row={max_row} col={max_col}")

    mass=max(
        float((torch.softmax(tc,dim=-1).sum(-1)-1).abs().max().cpu()),
        float((torch.softmax(tp,dim=-1).sum(-1)-1).abs().max().cpu()),
    )
    if mass>1e-6:
        raise RuntimeError("S34-A0 treatment probability mass changed")

    gold,_=_gold_tensors(rows,device=tc.device)
    accuracy=0.5*(
        float((tc.argmax(-1)==gold).float().mean().cpu())
        +float((tp.argmax(-1)==gold).float().mean().cpu())
    )
    same_cos=float(relation_signature_same_option_cosine(tsc,tsp).mean().cpu())

    physical=HIRA_V1_S17_TOTAL_PARAMETER_COUNT
    operator=_modules()[1]
    if operator.parameter_count!=0 or physical!=49_152:
        raise RuntimeError("S34-A0 capacity changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S34-A0 frozen runtime became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S34-A0 HIRACore became trainable")
    original=sum(
        p.numel() for name,p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original!=0:
        raise RuntimeError("S34-A0 original A13 became trainable")

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
        raise RuntimeError("S34-A0 checkpoint roundtrip changed")

    controlled=_controlled_transport_court()
    if (
        controlled["controlled_max_row_marginal_residual"]>1e-4
        or controlled["controlled_max_column_marginal_residual"]>1e-4
    ):
        raise RuntimeError("S34-A0 controlled marginal court failed")
    gradient=_gradient_probe(bundle,manifest,rows)

    result={
        "schema_version":SCHEMA_VERSION,
        "status":"PASS",
        "outcome":OUTCOME,
        "scientific_authority":"S34_A0_QUERY_CONDITIONED_ENTROPIC_TRANSPORT_ONLY",
        "semantic_case_count":len(suite),
        "state_view_count":2*len(suite),
        "k":4,
        "views_per_option":2,
        "state_relevance_temperature":0.10,
        "option_relevance_temperature":0.10,
        "kernel_temperature":0.10,
        "logit_temperature":0.10,
        "sinkhorn_iterations":8,
        "epsilon":1e-12,
        "operator_added_parameter_count":operator.parameter_count,
        "physical_trainable_parameter_count":physical,
        "lora_parameter_count":16_384,
        "projection_parameter_count":HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
        "runtime_trainable_parameter_count":0,
        "original_a13_trainable_parameter_count":original,
        "hira_core_trainable_parameter_count":0,
        "checkpoint_key_count":len(state),
        "checkpoint_roundtrip_exact":checkpoint_roundtrip,
        "treatment_probability_mass_error":mass,
        "full_k":tc.shape[-1]==4 and tp.shape[-1]==4,
        "state_once_expected_view_count":2*len(suite),
        "real_max_row_marginal_residual":max_row,
        "real_max_column_marginal_residual":max_col,
        "treatment_a0_relation_accuracy":accuracy,
        "treatment_a0_same_option_signature_cosine":same_cos,
        **controlled,
        "gradient":gradient,
        "used_for_model_selection":False,
        "production_ready_claimed":False,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S34_A0_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
