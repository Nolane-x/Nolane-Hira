from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_norm_balanced_gradient import norm_balanced_gradient_update
from nmd.v1_optimizer_step_anchor import (
    adamw_candidate_deltas,
    apply_parameter_deltas,
    initialize_adamw_state,
    project_adamw_runtime_delta_against_anchor,
)
from nmd.v1_reference_anchor_projection import signature_anchor_loss
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
from hira_v1_s17_train_dev import FUSION_EPSILON, _decision_logits, _encode_batch, _gold_tensors
import hira_v1_s17_train_dev as s17mod
import hira_v1_s35_train_dev as s35
import hira_v1_s38_train_dev as s38
from hira_v1_s38_a0_full_bilinear_readout import (
    _base,
    _mechanics_court,
    _projection_independence,
    _relation_inputs,
    _treatment,
)

SCHEMA_VERSION = "hira-v1-s41-a0-optimizer-step-anchored-coadaptation-v1"
OUTCOME = "HIRA_V1_S41_A0_OPTIMIZER_STEP_ANCHORED_COADAPTATION_READY"

LR = 2e-4
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PROJECTION_EPS = 1e-12


@dataclass(frozen=True)
class Case:
    case_id: str
    noun: str
    field_a: str
    first: str
    field_b: str
    second: str
    wrong_a: str
    wrong_b: str
    seed: int

    @property
    def state_a(self):
        return f"S41-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Audit {self.case_id} records {self.second} for {self.field_b}; the same S41-A0 {self.noun} records {self.first} for {self.field_a}."

    @property
    def qa1(self):
        return f"For S41-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry is tagged {self.field_a} in S41-A0 {self.case_id}?"

    @property
    def qb1(self):
        return f"For S41-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry is tagged {self.field_b} in S41-A0 {self.case_id}?"

    def option_pack(self):
        rows = [
            ("a", self.field_a, self.first),
            ("b", self.field_b, self.second),
            ("x", self.field_a, self.wrong_a),
            ("y", self.field_b, self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        options = tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"for this {self.noun}, {field} is {value}",
                aliases=(f"{value} is the audited {field} entry for this {self.noun}",),
            )
            for i, (_kind, field, value) in enumerate(rows)
        )
        ga = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "a")
        gb = next(i for i, (kind, _f, _v) in enumerate(rows) if kind == "b")
        return options, ga, gb


def cases():
    return (
        Case("SC11","cryogenic photon mixer","junction","SNAIL","noise","0.7 quanta","diode","2.8 quanta",74101),
        Case("SC22","quantum phase camera","sensor","NV array","precision","9 mrad","CMOS","36 mrad",74102),
        Case("SC33","nanophonon counter","absorber","graphene","rate","7 Mcps","silicon","1.8 Mcps",74103),
        Case("SC44","ultrafast spin imager","probe","MOKE","jitter","16 fs","Hall","64 fs",74104),
        Case("SC55","integrated ion spectrometer","filter","Wien","resolution","13 meV","grid","52 meV",74105),
        Case("SC66","microwave quantum router","switch","JPC","isolation","46 dB","PIN","11 dB",74106),
        Case("SC77","coherent neutron imager","converter","B10","efficiency","0.91","plastic","0.63",74107),
        Case("SC88","atom cavity thermometer","species","Sr","resolution","8 nK","Rb","32 nK",74108),
        Case("SD11","xray timing camera","detector","MCP","jitter","11 ps","CCD","44 ps",74109),
        Case("SD22","spinwave spectrum camera","medium","YIG","span","18 GHz","NiFe","4.5 GHz",74110),
        Case("SD33","quantum pressure resonator","membrane","hBN","floor","0.8 Pa","polymer","3.2 Pa",74111),
        Case("SD44","photonic delay analyzer","guide","SiN","delay","12 us","polymer","3 us",74112),
        Case("SD55","electron phase microscope","optic","biprism","precision","5 mrad","aperture","20 mrad",74113),
        Case("SD66","cryogenic charge camera","sensor","RF-SET","noise","0.15 ue","MOS","0.6 ue",74114),
        Case("SD77","quantum acoustic router","transducer","IDT","loss","0.3 dB","bulk","1.2 dB",74115),
        Case("SD88","nanoscale current imager","probe","SQUID","floor","4 nA","GMR","16 nA",74116),
    )


def _rows(suite):
    out = []
    for case in suite:
        options, ga, gb = case.option_pack()
        out.append(
            S17FusionCase(
                case_id=f"a0-s41-{case.case_id}",
                split="train",
                domain="s41_a0_only",
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
            )
        )
    return out


def _runtime(*, bundle, manifest, seed, train):
    torch.manual_seed(seed)
    fresh = load_hira_v0_m4_bundle(bundle)
    runtime = build_hira_v1_s17_norm_balanced_core(
        fresh.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=train,
        train_projection=train,
    )
    del fresh
    runtime.clear_schema_cache()
    enforce_s17_eval(runtime)
    return runtime


def _native_losses(runtime, rows):
    s17mod._relation_outputs = s35._native_relation_outputs
    return s35._losses(runtime, rows)


def _treatment_losses(runtime, rows, readout):
    s38._ACTIVE_READOUT = readout
    s17mod._relation_outputs = s38._treatment_relation_outputs
    return s35._losses(runtime, rows)


def _native_outputs(runtime, rows):
    (
        _total, _primary, _relation, _pieces, _fc, _fp,
        raw_c, raw_p, relation_c, relation_p, signature_c, signature_p, encoded,
    ) = _native_losses(runtime, rows)
    return raw_c, raw_p, relation_c, relation_p, signature_c, signature_p, encoded


def _anchor_pair(treatment, reference, rows):
    _a, _b, _c, _d, tsc, tsp, _e = _native_outputs(treatment, rows)
    with torch.no_grad():
        _f, _g, _h, _i, rsc, rsp, _j = _native_outputs(reference, rows)
    anchor = 0.5 * (
        signature_anchor_loss(tsc, rsc)
        + signature_anchor_loss(tsp, rsp)
    )
    return anchor, tsc, tsp, rsc, rsp


def _zeros(params, grads):
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def _clip_gradient_list(params, grads):
    for p in params:
        p.grad = None
    for p, g in zip(params, grads):
        p.grad = g.detach().clone()
    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
    clipped = [p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
    return clipped


def _standard_adamw_probe(params, grads):
    probe = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt = torch.optim.AdamW(
        probe,
        lr=LR,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
        amsgrad=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )
    for p, g in zip(probe, grads):
        p.grad = g.detach().clone()
    opt.step()
    return probe, opt


def _optimizer_step_probe(bundle, manifest, rows):
    with torch.inference_mode(False), torch.enable_grad():
        reference = _runtime(bundle=bundle, manifest=manifest, seed=62141, train=True)
        treatment = _runtime(bundle=bundle, manifest=manifest, seed=62141, train=True)
        readout = _treatment(train_readout=True)

        reference_params = [p for p in reference.parameters() if p.requires_grad]
        treatment_runtime = [p for p in treatment.parameters() if p.requires_grad]
        treatment_all = [*treatment_runtime, readout.bilinear_weight]

        if sum(p.numel() for p in reference_params) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S41-A0 reference surface changed")
        if sum(p.numel() for p in treatment_runtime) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S41-A0 treatment runtime surface changed")
        if readout.added_parameter_count != 65_536:
            raise RuntimeError("S41-A0 W capacity changed")
        if any(
            not torch.equal(a.detach(), b.detach())
            for a, b in zip(reference_params, treatment_runtime)
        ):
            raise RuntimeError("S41-A0 matched runtime initialization changed")

        _ra, _rb, rrelc, rrelp, rsc, rsp, _re = _native_outputs(reference, rows)
        _ta, _tb, trelc, trelp, tsc, tsp, _te = _native_outputs(treatment, rows)
        relation_identity = max(
            float((rrelc-trelc).abs().max().detach().cpu()),
            float((rrelp-trelp).abs().max().detach().cpu()),
        )
        signature_identity = max(
            float((rsc-tsc).abs().max().detach().cpu()),
            float((rsp-tsp).abs().max().detach().cpu()),
        )
        if relation_identity != 0.0 or signature_identity != 0.0:
            raise RuntimeError("S41-A0 matched native initialization changed")

        modules = iter_a13_lora_modules(treatment.encoder)
        if not modules:
            raise RuntimeError("S41-A0 treatment LoRA missing")
        with torch.no_grad():
            for i, module in enumerate(modules):
                target = module.lora_b
                perturb = torch.linspace(
                    -2e-2, 2e-2, target.numel(), device=target.device, dtype=target.dtype
                ).reshape_as(target)
                if i % 2:
                    perturb = perturb.flip(0)
                target.add_(perturb)

        anchor, tsc_drift, tsp_drift, rsc_drift, rsp_drift = _anchor_pair(
            treatment, reference, rows
        )
        anchor_value = float(anchor.detach().cpu())
        drift = max(
            float((tsc_drift-rsc_drift).abs().max().detach().cpu()),
            float((tsp_drift-rsp_drift).abs().max().detach().cpu()),
        )
        if drift <= 1e-7 or anchor_value <= 1e-9:
            raise RuntimeError("S41-A0 synthetic signature drift failed")

        anchor_raw = torch.autograd.grad(
            anchor, treatment_runtime, retain_graph=True, allow_unused=True
        )
        anchor_grad = _zeros(treatment_runtime, anchor_raw)
        anchor_l1 = sum(float(g.abs().sum().detach().cpu()) for g in anchor_grad)

        reference_anchor_raw = torch.autograd.grad(
            anchor, reference_params, retain_graph=True, allow_unused=True
        )
        reference_anchor_l1 = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for g in reference_anchor_raw
        )
        w_anchor_raw = torch.autograd.grad(
            anchor, readout.bilinear_weight, retain_graph=True, allow_unused=True
        )[0]
        w_anchor_l1 = 0.0 if w_anchor_raw is None else float(w_anchor_raw.abs().sum().cpu())

        lora_ids = {id(m.lora_b) for m in modules}
        anchor_lora_l1 = sum(
            float(g.abs().sum().detach().cpu())
            for p, g in zip(treatment_runtime, anchor_grad)
            if id(p) in lora_ids
        )
        if anchor_lora_l1 <= 0.0 or reference_anchor_l1 != 0.0 or w_anchor_l1 != 0.0:
            raise RuntimeError("S41-A0 anchor ownership failed")

        (
            _total, primary, relation, _pieces, *_rest,
        ) = _treatment_losses(treatment, rows, readout)
        pg_raw = torch.autograd.grad(primary, treatment_all, retain_graph=True, allow_unused=True)
        rg_raw = torch.autograd.grad(relation, treatment_all, allow_unused=True)
        pg = _zeros(treatment_all, pg_raw)
        rg = _zeros(treatment_all, rg_raw)
        joint, _diag = norm_balanced_gradient_update(pg, rg, epsilon=s35.BALANCE_EPSILON)

        w_joint = joint[-1]
        w_l1 = float(w_joint.abs().sum().detach().cpu())
        w_offdiag = float(
            (w_joint.abs().sum() - torch.diagonal(w_joint).abs().sum()).detach().cpu()
        )
        lora_joint_l1 = sum(
            float(g.abs().sum().detach().cpu())
            for p, g in zip(treatment_runtime, joint[:-1])
            if id(p) in lora_ids
        )
        if w_l1 <= 0.0 or w_offdiag <= 0.0 or lora_joint_l1 <= 0.0:
            raise RuntimeError("S41-A0 joint correctness gradient vanished")

        clipped_joint = _clip_gradient_list(treatment_all, joint)
        states = initialize_adamw_state(treatment_all)
        candidate, next_states = adamw_candidate_deltas(
            treatment_all, clipped_joint, states,
            lr=LR, betas=BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
        )

        standard_params, standard_opt = _standard_adamw_probe(treatment_all, clipped_joint)
        # Compare the actual AdamW parameter *movement*. Reconstructing
        # candidate parameters as old + delta can introduce a second float32
        # rounding even when delta == (standard_candidate - old) exactly.
        candidate_param_error = max(
            float((d-(q.detach()-p.detach())).abs().max().cpu())
            for p, d, q in zip(treatment_all, candidate, standard_params)
        )
        candidate_reconstruction_error = max(
            float(((p.detach()+d)-q.detach()).abs().max().cpu())
            for p, d, q in zip(treatment_all, candidate, standard_params)
        )
        exp_avg_error = 0.0
        exp_avg_sq_error = 0.0
        step_error = 0
        for state, p in zip(next_states, standard_params):
            ref_state = standard_opt.state[p]
            exp_avg_error = max(
                exp_avg_error,
                float((state.exp_avg-ref_state["exp_avg"]).abs().max().cpu()),
            )
            exp_avg_sq_error = max(
                exp_avg_sq_error,
                float((state.exp_avg_sq-ref_state["exp_avg_sq"]).abs().max().cpu()),
            )
            step_error = max(step_error, abs(state.step-int(ref_state["step"].item())))
        if candidate_param_error != 0.0 or exp_avg_error != 0.0 or exp_avg_sq_error != 0.0 or step_error != 0:
            raise RuntimeError("S41-A0 AdamW candidate/state parity failed")

        real_runtime_delta = candidate[:-1]
        real_w_delta = candidate[-1]
        _real_projected, real_diag = project_adamw_runtime_delta_against_anchor(
            real_runtime_delta, anchor_grad, epsilon=PROJECTION_EPS
        )
        w_copy = real_w_delta.detach().clone()
        w_projection_error = float((real_w_delta-w_copy).abs().max().cpu())
        if w_projection_error != 0.0:
            raise RuntimeError("S41-A0 projection changed real W candidate delta")

        # Force an actual AdamW candidate whose parameter movement conflicts
        # with the live model anchor gradient. This validates the projection
        # after AdamW transformation rather than on the raw gradient.
        conflict_raw = [(-1000.0*g).detach() for g in anchor_grad]
        conflict_raw.append(torch.zeros_like(readout.bilinear_weight))
        conflict_clipped = _clip_gradient_list(treatment_all, conflict_raw)
        conflict_states = initialize_adamw_state(treatment_all)
        conflict_candidate, conflict_next = adamw_candidate_deltas(
            treatment_all, conflict_clipped, conflict_states,
            lr=LR, betas=BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
        )
        projected_runtime, pdiag = project_adamw_runtime_delta_against_anchor(
            conflict_candidate[:-1], anchor_grad, epsilon=PROJECTION_EPS
        )
        if not pdiag.projected or pdiag.pre_dot <= 0.0 or abs(pdiag.post_dot) > 2e-6:
            raise RuntimeError("S41-A0 actual AdamW conflict projection failed")

        safe_candidate = [(-x).detach().clone() for x in anchor_grad]
        safe_projected, safe_diag = project_adamw_runtime_delta_against_anchor(
            safe_candidate, anchor_grad, epsilon=PROJECTION_EPS
        )
        safe_identity_error = max(
            float((a-b).abs().max().cpu())
            for a, b in zip(safe_projected, safe_candidate)
        )
        if safe_diag.projected or safe_identity_error != 0.0:
            raise RuntimeError("S41-A0 safe actual-step identity failed")

        before_runtime = [p.detach().clone() for p in treatment_runtime]
        before_w = readout.bilinear_weight.detach().clone()
        anchor_before = float(anchor.detach().cpu())

        apply_parameter_deltas(treatment_runtime, projected_runtime)
        apply_parameter_deltas([readout.bilinear_weight], [conflict_candidate[-1]])

        runtime_movement_error = max(
            float(((p.detach()-old)-d).abs().max().cpu())
            for p, old, d in zip(treatment_runtime, before_runtime, projected_runtime)
        )
        w_movement_error = float(
            ((readout.bilinear_weight.detach()-before_w)-conflict_candidate[-1]).abs().max().cpu()
        )
        anchor_after, *_ = _anchor_pair(treatment, reference, rows)
        anchor_after_value = float(anchor_after.detach().cpu())

        with torch.no_grad():
            for p, old in zip(treatment_runtime, before_runtime):
                p.copy_(old)
            readout.bilinear_weight.copy_(before_w)

        if runtime_movement_error != 0.0 or w_movement_error != 0.0:
            raise RuntimeError("S41-A0 applied movement differs from projected AdamW candidate")
        if anchor_after_value > anchor_before + 2e-7:
            raise RuntimeError("S41-A0 optimizer-faithful projected step increased anchor")

        # A nonzero W candidate from the live joint correctness path remains
        # exactly outside the runtime projection.
        if float(real_w_delta.abs().sum().cpu()) <= 0.0:
            raise RuntimeError("S41-A0 live W AdamW candidate delta vanished")

        return {
            "zero_init_reference_treatment_relation_max_abs": relation_identity,
            "zero_init_reference_treatment_signature_max_abs": signature_identity,
            "synthetic_signature_drift_max_abs": drift,
            "synthetic_signature_anchor": anchor_value,
            "anchor_treatment_runtime_gradient_l1": anchor_l1,
            "anchor_treatment_lora_gradient_l1": anchor_lora_l1,
            "anchor_reference_runtime_gradient_l1": reference_anchor_l1,
            "anchor_w_gradient_l1": w_anchor_l1,
            "joint_correctness_w_gradient_l1": w_l1,
            "joint_correctness_w_offdiagonal_gradient_l1": w_offdiag,
            "joint_correctness_lora_gradient_l1": lora_joint_l1,
            "adamw_candidate_parameter_max_abs_error": candidate_param_error,
            "adamw_candidate_reconstruction_max_abs_error": candidate_reconstruction_error,
            "adamw_exp_avg_max_abs_error": exp_avg_error,
            "adamw_exp_avg_sq_max_abs_error": exp_avg_sq_error,
            "adamw_step_counter_max_abs_error": step_error,
            "live_candidate_runtime_anchor_dot": real_diag.pre_dot,
            "live_candidate_runtime_projected": real_diag.projected,
            "live_w_candidate_delta_l1": float(real_w_delta.abs().sum().cpu()),
            "w_projection_max_abs": w_projection_error,
            "conflict_actual_step_pre_dot": pdiag.pre_dot,
            "conflict_actual_step_post_dot": pdiag.post_dot,
            "conflict_actual_step_projected": pdiag.projected,
            "safe_actual_step_projected": safe_diag.projected,
            "safe_actual_step_identity_max_abs": safe_identity_error,
            "applied_runtime_delta_max_abs_error": runtime_movement_error,
            "applied_w_delta_max_abs_error": w_movement_error,
            "synthetic_anchor_before_applied_step": anchor_before,
            "synthetic_anchor_after_applied_step": anchor_after_value,
            "conflict_adamw_next_step": conflict_next[0].step,
        }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S41-A0 suite size changed")
    rows = _rows(suite)
    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)

    frozen = _runtime(bundle=bundle, manifest=manifest, seed=62001, train=False)
    encoded = _encode_batch(frozen, rows)
    base = _base()
    treatment = _treatment(train_readout=True)

    bc, bs, _ = base(**_relation_inputs(encoded, "a", "canonical"))
    bp, bsp, _ = base(**_relation_inputs(encoded, "b", "paraphrase"))
    tc, ts, _ = treatment(**_relation_inputs(encoded, "a", "canonical"))
    tp, tsp, _ = treatment(**_relation_inputs(encoded, "b", "paraphrase"))

    relation_identity = max(float((tc-bc).abs().max()), float((tp-bp).abs().max()))
    signature_identity = max(float((ts-bs).abs().max()), float((tsp-bsp).abs().max()))
    if relation_identity != 0.0 or signature_identity != 0.0:
        raise RuntimeError("S41-A0 zero-init treatment/base identity failed")

    raw_c = _decision_logits(
        frozen,
        state_tokens=encoded["state_a_tokens"], state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"], encoded=encoded,
    )
    raw_p = _decision_logits(
        frozen,
        state_tokens=encoded["state_b_tokens"], state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"], encoded=encoded,
    )
    fusion = GradientIsolatedFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    cb_c, _ = fusion(raw_c, bc)
    cb_p, _ = fusion(raw_p, bp)
    tr_c, _ = fusion(raw_c, tc)
    tr_p, _ = fusion(raw_p, tp)
    fused_identity = max(float((cb_c-tr_c).abs().max()), float((cb_p-tr_p).abs().max()))
    if fused_identity != 0.0:
        raise RuntimeError("S41-A0 zero-init fused identity failed")
    choice_identity = 0.5 * (
        float((cb_c.argmax(-1)==tr_c.argmax(-1)).float().mean().cpu())
        + float((cb_p.argmax(-1)==tr_p.argmax(-1)).float().mean().cpu())
    )
    if choice_identity != 1.0:
        raise RuntimeError("S41-A0 selected choice identity failed")

    mass = max(
        float((torch.softmax(tr_c,dim=-1).sum(-1)-1).abs().max().cpu()),
        float((torch.softmax(tr_p,dim=-1).sum(-1)-1).abs().max().cpu()),
    )
    if mass > 1e-6:
        raise RuntimeError("S41-A0 probability mass changed")

    if treatment.added_parameter_count != 65_536:
        raise RuntimeError("S41-A0 W capacity changed")
    reference_surface = HIRA_V1_S17_TOTAL_PARAMETER_COUNT
    treatment_surface = reference_surface + treatment.added_parameter_count
    if reference_surface != 49_152 or treatment_surface != 114_688:
        raise RuntimeError("S41-A0 parameter surface changed")

    mechanics = _mechanics_court()
    projection = _projection_independence(frozen, rows)
    optimizer_step = _optimizer_step_probe(bundle, manifest, rows)

    gold, _ = _gold_tensors(rows, device=tc.device)
    relation_acc = 0.5 * (
        float((tc.argmax(-1)==gold).float().mean().cpu())
        + float((tp.argmax(-1)==gold).float().mean().cpu())
    )
    fused_acc = 0.5 * (
        float((tr_c.argmax(-1)==gold).float().mean().cpu())
        + float((tr_p.argmax(-1)==gold).float().mean().cpu())
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S41_A0_OPTIMIZER_STEP_ANCHORED_FULL_BILINEAR_ONLY",
        "semantic_case_count": len(suite),
        "state_view_count": 2*len(suite),
        "k": 4,
        "views_per_option": 2,
        "native_dimension": 256,
        "operator_added_parameter_count": treatment.added_parameter_count,
        "reference_trainable_parameter_count": 49_152,
        "treatment_trainable_parameter_count": 114_688,
        "query_norm_epsilon": 1e-12,
        "residual_scale": 1.0,
        "anchor": "mean_one_minus_cosine_native_signature_to_detached_matched_reference",
        "anchor_coefficient": None,
        "projection_epsilon": PROJECTION_EPS,
        "optimizer": {
            "name": "AdamW",
            "lr": LR,
            "betas": list(BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "foreach": False,
            "fused": False,
        },
        "zero_init_relation_logit_max_abs": relation_identity,
        "zero_init_signature_max_abs": signature_identity,
        "zero_init_fused_logit_max_abs": fused_identity,
        "zero_init_selected_choice_identity_rate": choice_identity,
        "native_probability_mass_error": mass,
        "full_k": tr_c.shape[-1] == 4 and tr_p.shape[-1] == 4,
        "state_once_expected_view_count": 2*len(suite),
        "a0_relation_accuracy": relation_acc,
        "a0_fused_accuracy": fused_acc,
        **mechanics,
        **projection,
        **optimizer_step,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )
    print("HIRA_V1_S41_A0_RECEIPT="+json.dumps(result,sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
