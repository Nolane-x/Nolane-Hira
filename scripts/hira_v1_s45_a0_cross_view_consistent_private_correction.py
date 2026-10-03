from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_norm_balanced_gradient import apply_gradient_update, norm_balanced_gradient_update
from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
from hira_v1_s17_train_dev import FUSION_EPSILON, _decision_logits, _encode_batch, _gold_tensors
import hira_v1_s17_train_dev as s17mod
import hira_v1_s35_train_dev as s35
from hira_v1_s38_a0_full_bilinear_readout import _projection_independence

SCHEMA_VERSION = "hira-v1-s45-a0-cross-view-consistent-private-correction-v1"
OUTCOME = "HIRA_V1_S45_A0_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_READY"

ADAPTER_A_COUNT = 32_768
ADAPTER_B_COUNT = 16_384
PRIVATE_ADAPTER_COUNT = 49_152
BILINEAR_COUNT = 65_536
CORRECTION_COUNT = 114_688
TREATMENT_TOTAL = 163_840
ADAPTER_SEED = 65_044

LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0


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
        return f"S45-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Audit {self.case_id} stores {self.second} for {self.field_b}; the same S45-A0 {self.noun} stores {self.first} for {self.field_a}."

    @property
    def qa1(self):
        return f"For S45-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry is tagged {self.field_a} in S45-A0 {self.case_id}?"

    @property
    def qb1(self):
        return f"For S45-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry is tagged {self.field_b} in S45-A0 {self.case_id}?"

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
        Case("CV11","quantum Hall velocimeter","channel","graphene edge","drift floor","0.13 mm/s","copper trace","0.52 mm/s",88101),
        Case("CV22","femtosecond magnetostriction camera","probe","xray nanobeam","strain noise","6 peps","optical fiber","24 peps",88102),
        Case("CV33","exciton dipole compass","sensor","moire WSe2 pair","angle floor","0.11 deg","silicon diode","0.44 deg",88103),
        Case("CV44","cryogenic phonon gyroscope","resonator","quartz whispering ring","bias drift","3 nrad/s","MEMS disk","12 nrad/s",88104),
        Case("CV55","Rydberg microwave holograph","cell","Cs vapor lattice","field noise","9 nV/cm","metal antenna","36 nV/cm",88105),
        Case("CV66","neutrino recoil chronograph","target","cryogenic Ge array","timing spread","7 ns","plastic scintillator","28 ns",88106),
        Case("CV77","spin-wave curvature mapper","guide","CoFeB nanoribbon","curvature error","0.08 1/um","Ni wire","0.32 1/um",88107),
        Case("CV88","Casimir torque microscope","plate","patterned Au rotor","torque floor","5 zNm","polymer vane","20 zNm",88108),
        Case("CW11","polariton lifetime raster","cavity","perovskite microcavity","lifetime jitter","14 fs","glass etalon","56 fs",88109),
        Case("CW22","topological acoustic compass","waveguide","valley-Hall Si membrane","heading noise","0.16 deg","bulk ceramic","0.64 deg",88110),
        Case("CW33","molecular Stark tomography","species","trapped CaF packet","field resolution","4 mV/cm","thermal gas cell","16 mV/cm",88111),
        Case("CW44","superfluid vortex chronometer","fluid","He3 microchannel","period noise","8 us","water channel","32 us",88112),
        Case("CW55","plasma wake phase camera","probe","electron witness bunch","phase jitter","0.6 mrad","photodiode pulse","2.4 mrad",88113),
        Case("CW66","xray orbital compass","optic","diamond Laue lens","axis error","0.09 mrad","polymer lens","0.36 mrad",88114),
        Case("CW77","atomic parity vectormeter","ensemble","Yb optical lattice","vector floor","2 fT","Hall sensor","8 fT",88115),
        Case("CW88","nanophotonic recoil balance","mirror","SiN metasurface","force floor","7 aN","glass plate","28 aN",88116),
    )


def _rows(suite):
    out = []
    for case in suite:
        options, ga, gb = case.option_pack()
        out.append(
            S17FusionCase(
                case_id=f"a0-s45-{case.case_id}",
                split="train",
                domain="s45_a0_only",
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


def _native_outputs(runtime, rows):
    (
        _total, _primary, _relation, _pieces, _fc, _fp,
        raw_c, raw_p, relation_c, relation_p, signature_c, signature_p, encoded,
    ) = _native_losses(runtime, rows)
    return raw_c, raw_p, relation_c, relation_p, signature_c, signature_p, encoded


def _js_divergence(logits_a, logits_b):
    pa = F.softmax(logits_a, dim=-1)
    pb = F.softmax(logits_b, dim=-1)
    midpoint = 0.5 * (pa + pb)
    log_pa = torch.log(pa.clamp_min(1e-12))
    log_pb = torch.log(pb.clamp_min(1e-12))
    log_mid = torch.log(midpoint.clamp_min(1e-12))
    kl_a = (pa * (log_pa - log_mid)).sum(dim=-1)
    kl_b = (pb * (log_pb - log_mid)).sum(dim=-1)
    js = 0.5 * (kl_a + kl_b).mean()
    if not bool(torch.isfinite(js)):
        raise RuntimeError("S45-A0 correction JS became non-finite")
    return js


def _correction_loss(op, relation_c, relation_p, signature_c, signature_p, encoded, rows):
    gold, _ = _gold_tensors(rows, device=relation_c.device)
    cc = op.correction_logits(
        native_logits=relation_c,
        signatures=signature_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    cp = op.correction_logits(
        native_logits=relation_p,
        signatures=signature_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    ce = 0.5 * (F.cross_entropy(cc, gold) + F.cross_entropy(cp, gold))
    js = _js_divergence(cc, cp)
    block = s35.BINDING_COEFFICIENT * ce + s35.INVARIANCE_COEFFICIENT * js
    return block, ce, js, cc, cp


def _zeros(params, grads):
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def _native_gradient(runtime, rows):
    params = [p for p in runtime.parameters() if p.requires_grad]
    (
        _total, primary, relation, _pieces, *_rest,
    ) = _native_losses(runtime, rows)
    pg_raw = torch.autograd.grad(primary, params, retain_graph=True, allow_unused=True)
    rg_raw = torch.autograd.grad(relation, params, allow_unused=True)
    pg = _zeros(params, pg_raw)
    rg = _zeros(params, rg_raw)
    combined, diag = norm_balanced_gradient_update(pg, rg, epsilon=s35.BALANCE_EPSILON)
    return params, combined, diag


def _state_max_abs(reference_params, treatment_params):
    return max(
        float((a.detach()-b.detach()).abs().max().cpu())
        for a, b in zip(reference_params, treatment_params)
    )


def _private_mechanics_court():
    g = torch.Generator().manual_seed(65444)
    op = PrivateCorrectionRepresentationFork()

    results = {}
    for k in (3, 7):
        native = torch.randn(3, k, generator=g)
        sig = F.normalize(torch.randn(3, k, 256, generator=g), dim=-1)
        q = torch.randn(3, 5, 256, generator=g)
        qm = torch.ones(3, 5, dtype=torch.bool)
        logits = op.correction_logits(
            native_logits=native,
            signatures=sig,
            question_tokens=q,
            question_mask=qm,
        )
        if tuple(logits.shape) != (3, k):
            raise RuntimeError("S45-A0 arbitrary-K changed")
        results[f"k{k}_pass"] = True

    with torch.no_grad():
        g2 = torch.Generator().manual_seed(65445)
        op.adapter_b.copy_(torch.randn(256,64,generator=g2)*0.01)
        op.bilinear_weight.copy_(torch.randn(256,256,generator=g2)*0.01)

    native = torch.randn(2, 7, generator=g)
    sig = F.normalize(torch.randn(2, 7, 256, generator=g), dim=-1)
    q = torch.randn(2, 4, 256, generator=g)
    qm = torch.ones(2, 4, dtype=torch.bool)
    base = op.correction_logits(
        native_logits=native, signatures=sig,
        question_tokens=q, question_mask=qm,
    )
    perm = torch.tensor([4,0,6,2,1,5,3])
    moved = op.correction_logits(
        native_logits=native[:,perm], signatures=sig[:,perm],
        question_tokens=q, question_mask=qm,
    )
    option_err = float((moved-base[:,perm]).abs().max().cpu())

    qperm = torch.tensor([2,0,3,1])
    q_moved = op.correction_logits(
        native_logits=native, signatures=sig,
        question_tokens=q[:,qperm], question_mask=qm[:,qperm],
    )
    question_err = float((q_moved-base).abs().max().cpu())

    padded_q = torch.cat([q, torch.randn(2,3,256,generator=g)], dim=1)
    padded_m = torch.cat([qm, torch.zeros(2,3,dtype=torch.bool)], dim=1)
    padded = op.correction_logits(
        native_logits=native, signatures=sig,
        question_tokens=padded_q, question_mask=padded_m,
    )
    padding_err = float((padded-base).abs().max().cpu())

    state = op.correction_state_dict()
    replay = PrivateCorrectionRepresentationFork()
    replay.load_correction_state_dict(state, freeze=True)
    replay_state = replay.correction_state_dict()
    checkpoint_exact = all(torch.equal(state[k], replay_state[k]) for k in state)

    if option_err > 2e-6 or question_err > 2e-6 or padding_err > 2e-6:
        raise RuntimeError("S45-A0 private mechanics invariance failed")
    if not checkpoint_exact:
        raise RuntimeError("S45-A0 private checkpoint roundtrip failed")

    return {
        "arbitrary_k3_pass": results["k3_pass"],
        "arbitrary_k7_pass": results["k7_pass"],
        "logical_option_permutation_max_abs_error": option_err,
        "question_permutation_max_abs_error": question_err,
        "masked_padding_max_abs_error": padding_err,
        "checkpoint_roundtrip_exact": checkpoint_exact,
    }


def _ownership_warmstart_court(bundle, manifest, rows):
    with torch.inference_mode(False), torch.enable_grad():
        reference = _runtime(bundle=bundle, manifest=manifest, seed=66141, train=True)
        treatment = _runtime(bundle=bundle, manifest=manifest, seed=65141, train=True)
        op = PrivateCorrectionRepresentationFork(train_correction=True)

        reference_params = [p for p in reference.parameters() if p.requires_grad]
        treatment_params = [p for p in treatment.parameters() if p.requires_grad]
        correction_params = op.correction_parameters()

        if sum(p.numel() for p in reference_params) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S45-A0 reference native surface changed")
        if sum(p.numel() for p in treatment_params) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S45-A0 treatment native surface changed")
        if HIRA_V1_S17_TOTAL_PARAMETER_COUNT != 49_152:
            raise RuntimeError("S45-A0 native surface constant changed")
        if op.adapter_a_parameter_count != ADAPTER_A_COUNT:
            raise RuntimeError("S45-A0 A capacity changed")
        if op.adapter_b_parameter_count != ADAPTER_B_COUNT:
            raise RuntimeError("S45-A0 B capacity changed")
        if op.private_adapter_parameter_count != PRIVATE_ADAPTER_COUNT:
            raise RuntimeError("S45-A0 private adapter capacity changed")
        if op.bilinear_parameter_count != BILINEAR_COUNT:
            raise RuntimeError("S45-A0 W capacity changed")
        if op.correction_parameter_count != CORRECTION_COUNT:
            raise RuntimeError("S45-A0 correction capacity changed")
        if HIRA_V1_S17_TOTAL_PARAMETER_COUNT + CORRECTION_COUNT != TREATMENT_TOTAL:
            raise RuntimeError("S45-A0 treatment total changed")

        init_param_diff = _state_max_abs(reference_params, treatment_params)
        if init_param_diff != 0.0:
            raise RuntimeError("S45-A0 matched native initialization changed")

        (
            rrawc, rrawp, rrelc, rrelp, rsc, rsp, _re
        ) = _native_outputs(reference, rows)
        (
            trawc, trawp, trelc, trelp, tsc, tsp, te
        ) = _native_outputs(treatment, rows)

        native_relation_identity = max(
            float((rrelc-trelc).abs().max().detach().cpu()),
            float((rrelp-trelp).abs().max().detach().cpu()),
        )
        native_signature_identity = max(
            float((rsc-tsc).abs().max().detach().cpu()),
            float((rsp-tsp).abs().max().detach().cpu()),
        )
        native_primary_identity = max(
            float((rrawc-trawc).abs().max().detach().cpu()),
            float((rrawp-trawp).abs().max().detach().cpu()),
        )
        if max(native_relation_identity,native_signature_identity,native_primary_identity) != 0.0:
            raise RuntimeError("S45-A0 matched native output identity changed")

        corr_block, corr_ce, corr_js, cc, cp = _correction_loss(
            op, trelc, trelp, tsc, tsp, te, rows
        )
        correction_native_raw = torch.autograd.grad(
            corr_block, treatment_params, retain_graph=True, allow_unused=True
        )
        correction_native_l1 = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for g in correction_native_raw
        )
        if correction_native_l1 != 0.0:
            raise RuntimeError("S45-A0 correction gradient leaked into native runtime")

        reference_corr_raw = torch.autograd.grad(
            corr_block, reference_params, retain_graph=True, allow_unused=True
        )
        reference_corr_l1 = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for g in reference_corr_raw
        )
        if reference_corr_l1 != 0.0:
            raise RuntimeError("S45-A0 correction gradient leaked into reference runtime")

        corr_grads0 = torch.autograd.grad(
            corr_block, correction_params, retain_graph=True, allow_unused=True
        )
        a0 = 0.0 if corr_grads0[0] is None else float(corr_grads0[0].abs().sum().cpu())
        b0 = 0.0 if corr_grads0[1] is None else float(corr_grads0[1].abs().sum().cpu())
        w0 = 0.0 if corr_grads0[2] is None else float(corr_grads0[2].abs().sum().cpu())
        if a0 != 0.0 or b0 != 0.0 or w0 <= 0.0:
            raise RuntimeError("S45-A0 zero-init warm-start ownership changed")

        # Native objective must be disconnected from correction parameters.
        _ntotal, nprimary, nrelation, _npieces, *_nrest = _native_losses(treatment, rows)
        native_to_corr = torch.autograd.grad(
            nprimary+nrelation, correction_params, retain_graph=True, allow_unused=True
        )
        native_to_corr_l1 = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for g in native_to_corr
        )
        if native_to_corr_l1 != 0.0:
            raise RuntimeError("S45-A0 native objective leaked into correction params")

        # Matched one-step native update.
        ref_params, ref_grad, _rd = _native_gradient(reference, rows)
        tr_params, tr_grad, _td = _native_gradient(treatment, rows)
        grad_diff = max(
            float((a-b).abs().max().detach().cpu())
            for a,b in zip(ref_grad,tr_grad)
        )
        if grad_diff != 0.0:
            raise RuntimeError("S45-A0 matched native gradients changed")

        ref_opt = torch.optim.AdamW(
            ref_params, lr=s35.LR, weight_decay=s35.WEIGHT_DECAY
        )
        tr_opt = torch.optim.AdamW(
            tr_params, lr=s35.LR, weight_decay=s35.WEIGHT_DECAY
        )
        apply_gradient_update(ref_params, ref_grad)
        apply_gradient_update(tr_params, tr_grad)
        torch.nn.utils.clip_grad_norm_(ref_params, s35.GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(tr_params, s35.GRAD_CLIP)
        ref_opt.step()
        tr_opt.step()
        enforce_s17_eval(reference)
        enforce_s17_eval(treatment)

        native_step_param_diff = _state_max_abs(ref_params,tr_params)
        if native_step_param_diff != 0.0:
            raise RuntimeError("S45-A0 matched native one-step trajectory changed")

        (
            rrawc2, rrawp2, rrelc2, rrelp2, rsc2, rsp2, _re2
        ) = _native_outputs(reference, rows)
        (
            trawc2, trawp2, trelc2, trelp2, tsc2, tsp2, te2
        ) = _native_outputs(treatment, rows)
        native_step_output_diff = max(
            float((rrawc2-trawc2).abs().max().detach().cpu()),
            float((rrawp2-trawp2).abs().max().detach().cpu()),
            float((rrelc2-trelc2).abs().max().detach().cpu()),
            float((rrelp2-trelp2).abs().max().detach().cpu()),
            float((rsc2-tsc2).abs().max().detach().cpu()),
            float((rsp2-tsp2).abs().max().detach().cpu()),
        )
        if native_step_output_diff != 0.0:
            raise RuntimeError("S45-A0 matched native one-step outputs changed")

        # Three deterministic correction-only steps from true zero B/W.
        corr_opt = torch.optim.AdamW(
            correction_params,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        warm = []
        for step in range(1,4):
            corr_opt.zero_grad(set_to_none=True)
            block, ce, js, _cc, _cp = _correction_loss(
                op, trelc2, trelp2, tsc2, tsp2, te2, rows
            )
            grads = torch.autograd.grad(
                block, correction_params, retain_graph=False, allow_unused=True
            )
            l1 = [
                0.0 if g is None else float(g.abs().sum().detach().cpu())
                for g in grads
            ]
            for p,g in zip(correction_params,grads):
                p.grad = None if g is None else g.detach().clone()
            torch.nn.utils.clip_grad_norm_(correction_params, GRAD_CLIP)
            corr_opt.step()
            warm.append({
                "step":step,
                "adapter_a_gradient_l1":l1[0],
                "adapter_b_gradient_l1":l1[1],
                "w_gradient_l1":l1[2],
                "correction_block":float(block.detach().cpu()),
                "correction_ce":float(ce.detach().cpu()),
                "correction_js":float(js.detach().cpu()),
                "adapter_a_norm":float(op.adapter_a.detach().norm().cpu()),
                "adapter_b_norm":float(op.adapter_b.detach().norm().cpu()),
                "w_norm":float(op.bilinear_weight.detach().norm().cpu()),
            })

        if not (warm[0]["w_gradient_l1"] > 0.0 and warm[0]["adapter_b_gradient_l1"] == 0.0 and warm[0]["adapter_a_gradient_l1"] == 0.0):
            raise RuntimeError("S45-A0 warm-start step1 changed")
        if not (warm[1]["adapter_b_gradient_l1"] > 0.0 and warm[1]["w_gradient_l1"] > 0.0):
            raise RuntimeError("S45-A0 warm-start step2 B liveness failed")
        if not (warm[2]["adapter_a_gradient_l1"] > 0.0 and warm[2]["adapter_b_gradient_l1"] > 0.0 and warm[2]["w_gradient_l1"] > 0.0):
            raise RuntimeError("S45-A0 warm-start step3 A/B/W liveness failed")

        # Correction remains detached after warm start.
        block3, _ce3, _js3, _cc3, _cp3 = _correction_loss(
            op, trelc2, trelp2, tsc2, tsp2, te2, rows
        )
        leak3 = torch.autograd.grad(
            block3, tr_params, retain_graph=True, allow_unused=True
        )
        leak3_l1 = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for g in leak3
        )
        if leak3_l1 != 0.0:
            raise RuntimeError("S45-A0 warm correction leaked into native runtime")

        # Nonlinear private branch differs from W-only using the same learned W.
        private_residual = op.correction_residual(
            signatures=tsc2,
            question_tokens=te2["question_canonical_tokens"],
            question_mask=te2["question_canonical_mask"],
        )
        qsum = op.query_summary(
            question_tokens=te2["question_canonical_tokens"],
            question_mask=te2["question_canonical_mask"],
        )
        w_only = torch.einsum(
            "bkd,de,be->bk",
            tsc2.detach(),
            op.bilinear_weight.to(dtype=tsc2.dtype),
            qsum,
        )
        nonlinear_delta = float(
            (private_residual-w_only).abs().max().detach().cpu()
        )
        if nonlinear_delta <= 1e-7:
            raise RuntimeError("S45-A0 private representation collapsed to W-only")

        # S45-specific cross-view consistency court.
        identical_probe = torch.tensor(
            [[2.0, 0.5, -1.0, 0.0], [0.0, 1.0, 2.0, -0.5]],
            dtype=trelc2.dtype,
            device=trelc2.device,
        )
        identical_js = float(
            _js_divergence(identical_probe, identical_probe).detach().cpu()
        )
        shifted_probe = identical_probe.roll(shifts=1, dims=-1)
        positive_js = float(
            _js_divergence(identical_probe, shifted_probe).detach().cpu()
        )
        if identical_js > 1e-12:
            raise RuntimeError("S45-A0 identical-logit JS must be zero")
        if not math.isfinite(positive_js) or positive_js <= 0.0:
            raise RuntimeError("S45-A0 non-identical-logit JS must be finite positive")

        cc_js = op.correction_logits(
            native_logits=trelc2,
            signatures=tsc2,
            question_tokens=te2["question_canonical_tokens"],
            question_mask=te2["question_canonical_mask"],
        )
        cp_js = op.correction_logits(
            native_logits=trelp2,
            signatures=tsp2,
            question_tokens=te2["question_paraphrase_tokens"],
            question_mask=te2["question_paraphrase_mask"],
        )
        js_only = _js_divergence(cc_js, cp_js)
        js_grads = torch.autograd.grad(
            js_only, correction_params, retain_graph=True, allow_unused=True
        )
        js_grad_l1 = [
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for g in js_grads
        ]
        if not all(value > 0.0 and math.isfinite(value) for value in js_grad_l1):
            raise RuntimeError("S45-A0 JS-only A/B/W liveness failed")

        js_native = torch.autograd.grad(
            js_only, tr_params, retain_graph=True, allow_unused=True
        )
        js_native_l1 = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for g in js_native
        )
        if js_native_l1 != 0.0:
            raise RuntimeError("S45-A0 JS gradient leaked into native runtime")

        lora_ids = {id(m.lora_b) for m in iter_a13_lora_modules(treatment.encoder)}
        correction_lora_leak = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for p,g in zip(tr_params, leak3)
            if id(p) in lora_ids
        )
        scorer = treatment.projection_triadic_scorer
        projection_ids = set()
        if scorer is not None:
            projection_ids = {id(scorer.projection.weight)}
        correction_projection_leak = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for p,g in zip(tr_params, leak3)
            if id(p) in projection_ids
        )

        return {
            "zero_init_native_parameter_max_abs": init_param_diff,
            "zero_init_native_relation_max_abs": native_relation_identity,
            "zero_init_native_signature_max_abs": native_signature_identity,
            "zero_init_native_primary_max_abs": native_primary_identity,
            "zero_init_correction_native_runtime_gradient_l1": correction_native_l1,
            "zero_init_correction_reference_runtime_gradient_l1": reference_corr_l1,
            "zero_init_adapter_a_gradient_l1": a0,
            "zero_init_adapter_b_gradient_l1": b0,
            "zero_init_w_gradient_l1": w0,
            "native_objective_correction_gradient_l1": native_to_corr_l1,
            "matched_native_gradient_max_abs": grad_diff,
            "matched_native_one_step_parameter_max_abs": native_step_param_diff,
            "matched_native_one_step_output_max_abs": native_step_output_diff,
            "warm_start": warm,
            "warm_correction_native_runtime_gradient_l1": leak3_l1,
            "warm_correction_lora_gradient_l1": correction_lora_leak,
            "warm_correction_projection_gradient_l1": correction_projection_leak,
            "identical_corrected_logit_js": identical_js,
            "nonidentical_probe_js": positive_js,
            "js_only_adapter_a_gradient_l1": js_grad_l1[0],
            "js_only_adapter_b_gradient_l1": js_grad_l1[1],
            "js_only_w_gradient_l1": js_grad_l1[2],
            "js_only_native_runtime_gradient_l1": js_native_l1,
            "private_vs_w_only_residual_max_abs": nonlinear_delta,
        }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S45-A0 suite size changed")
    rows = _rows(suite)
    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)

    frozen = _runtime(bundle=bundle, manifest=manifest, seed=66001, train=False)
    (
        raw_c, raw_p, relation_c, relation_p, signature_c, signature_p, encoded
    ) = _native_outputs(frozen, rows)
    op = PrivateCorrectionRepresentationFork(train_correction=True)

    corrected_c = op.correction_logits(
        native_logits=relation_c,
        signatures=signature_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    corrected_p = op.correction_logits(
        native_logits=relation_p,
        signatures=signature_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    relation_identity = max(
        float((corrected_c-relation_c).abs().max().cpu()),
        float((corrected_p-relation_p).abs().max().cpu()),
    )
    if relation_identity != 0.0:
        raise RuntimeError("S45-A0 zero-init corrected relation identity failed")

    private_c, _ = op.private_signatures(
        signatures=signature_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    private_p, _ = op.private_signatures(
        signatures=signature_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    private_signature_identity = max(
        float((private_c-signature_c).abs().max().cpu()),
        float((private_p-signature_p).abs().max().cpu()),
    )
    if private_signature_identity > 2e-6:
        raise RuntimeError("S45-A0 zero-B private signature identity changed")

    private_residual_c, _ = op.private_residual(
        signatures=signature_c,
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    private_residual_p, _ = op.private_residual(
        signatures=signature_p,
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    zero_private_residual = max(
        float(private_residual_c.abs().max().cpu()),
        float(private_residual_p.abs().max().cpu()),
    )
    if zero_private_residual != 0.0:
        raise RuntimeError("S45-A0 zero-B private residual changed")

    fusion = GradientIsolatedFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    native_fused_c, _ = fusion(raw_c, relation_c)
    native_fused_p, _ = fusion(raw_p, relation_p)
    corrected_fused_c, _ = fusion(raw_c, corrected_c)
    corrected_fused_p, _ = fusion(raw_p, corrected_p)
    fused_identity = max(
        float((native_fused_c-corrected_fused_c).abs().max().cpu()),
        float((native_fused_p-corrected_fused_p).abs().max().cpu()),
    )
    if fused_identity != 0.0:
        raise RuntimeError("S45-A0 zero-init fused identity failed")

    choice_identity = 0.5 * (
        float((native_fused_c.argmax(-1)==corrected_fused_c.argmax(-1)).float().mean().cpu())
        + float((native_fused_p.argmax(-1)==corrected_fused_p.argmax(-1)).float().mean().cpu())
    )
    if choice_identity != 1.0:
        raise RuntimeError("S45-A0 zero-init choice identity failed")

    mass = max(
        float((torch.softmax(corrected_fused_c,dim=-1).sum(-1)-1).abs().max().cpu()),
        float((torch.softmax(corrected_fused_p,dim=-1).sum(-1)-1).abs().max().cpu()),
    )
    if mass > 1e-6:
        raise RuntimeError("S45-A0 probability mass changed")

    if op.adapter_a_parameter_count != ADAPTER_A_COUNT:
        raise RuntimeError("S45-A0 A count changed")
    if op.adapter_b_parameter_count != ADAPTER_B_COUNT:
        raise RuntimeError("S45-A0 B count changed")
    if op.private_adapter_parameter_count != PRIVATE_ADAPTER_COUNT:
        raise RuntimeError("S45-A0 private adapter count changed")
    if op.bilinear_parameter_count != BILINEAR_COUNT:
        raise RuntimeError("S45-A0 W count changed")
    if op.correction_parameter_count != CORRECTION_COUNT:
        raise RuntimeError("S45-A0 correction count changed")

    mechanics = _private_mechanics_court()
    ownership = _ownership_warmstart_court(bundle, manifest, rows)
    projection = _projection_independence(frozen, rows)

    gold, _ = _gold_tensors(rows, device=relation_c.device)
    relation_acc = 0.5 * (
        float((corrected_c.argmax(-1)==gold).float().mean().cpu())
        + float((corrected_p.argmax(-1)==gold).float().mean().cpu())
    )
    fused_acc = 0.5 * (
        float((corrected_fused_c.argmax(-1)==gold).float().mean().cpu())
        + float((corrected_fused_p.argmax(-1)==gold).float().mean().cpu())
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S45_A0_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_ONLY",
        "semantic_case_count": len(suite),
        "state_view_count": 2*len(suite),
        "k": 4,
        "views_per_option": 2,
        "native_dimension": 256,
        "hidden_dimension": 64,
        "adapter_seed": ADAPTER_SEED,
        "adapter_a_parameter_count": op.adapter_a_parameter_count,
        "adapter_b_parameter_count": op.adapter_b_parameter_count,
        "private_adapter_parameter_count": op.private_adapter_parameter_count,
        "bilinear_parameter_count": op.bilinear_parameter_count,
        "correction_parameter_count": op.correction_parameter_count,
        "native_trainable_parameter_count": HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        "treatment_total_trainable_parameter_count": TREATMENT_TOTAL,
        "query_norm_epsilon": 1e-12,
        "private_norm_epsilon": 1e-12,
        "residual_scale": 1.0,
        "correction_objective": "s44_binding_ce_plus_cross_view_js",
        "cross_view_js_coefficient": s35.INVARIANCE_COEFFICIENT,
        "binding_coefficient": s35.BINDING_COEFFICIENT,
        "correction_optimizer": {
            "name":"AdamW",
            "lr":LR,
            "weight_decay":WEIGHT_DECAY,
            "grad_clip":GRAD_CLIP,
        },
        "zero_init_corrected_relation_max_abs": relation_identity,
        "zero_init_private_signature_max_abs": private_signature_identity,
        "zero_init_private_residual_max_abs": zero_private_residual,
        "zero_init_fused_logit_max_abs": fused_identity,
        "zero_init_selected_choice_identity_rate": choice_identity,
        "native_probability_mass_error": mass,
        "full_k": corrected_fused_c.shape[-1] == 4 and corrected_fused_p.shape[-1] == 4,
        "state_once_expected_view_count": 2*len(suite),
        "a0_relation_accuracy": relation_acc,
        "a0_fused_accuracy": fused_acc,
        **mechanics,
        **projection,
        **ownership,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("HIRA_V1_S45_A0_CROSS_VIEW_RECEIPT="+json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
