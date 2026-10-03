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
from nmd.v1_reference_anchor_projection import (
    project_runtime_gradient_against_reference_anchor,
    signature_anchor_loss,
)
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
from hira_v1_s17_train_dev import (
    FUSION_EPSILON,
    _decision_logits,
    _encode_batch,
    _gold_tensors,
)
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

SCHEMA_VERSION = "hira-v1-s40-a0-reference-anchored-projected-coadaptation-v1"
OUTCOME = "HIRA_V1_S40_A0_REFERENCE_ANCHORED_PROJECTED_COADAPTATION_READY"


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
        return f"S40-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Audit {self.case_id} records {self.second} for {self.field_b}; the same S40-A0 {self.noun} records {self.first} for {self.field_a}."

    @property
    def qa1(self):
        return f"For S40-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry is tagged {self.field_a} in S40-A0 {self.case_id}?"

    @property
    def qb1(self):
        return f"For S40-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry is tagged {self.field_b} in S40-A0 {self.case_id}?"

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
        Case("SA11","photon correlation camera","sensor","SPAD array","timing","18 ps","CCD","72 ps",73101),
        Case("SA22","cryogenic nanophotonic router","switch","Pockels","loss","0.4 dB","thermal","1.6 dB",73102),
        Case("SA33","quantum microwave microscope","probe","SQUID","resolution","14 nm","Hall","56 nm",73103),
        Case("SA44","ultrafast ion detector","multiplier","MCP","jitter","24 ps","dynode","96 ps",73104),
        Case("SA55","phononic strain imager","transducer","AlN","bandwidth","12 GHz","PZT","3 GHz",73105),
        Case("SA66","integrated atomic clock","cell","microcell","stability","2e-13","lamp","8e-13",73106),
        Case("SA77","single spin spectrometer","readout","reflectometry","fidelity","0.94","DC","0.71",73107),
        Case("SA88","nano-optic displacement sensor","cavity","slot","noise","6 fm","ring","24 fm",73108),
        Case("SB11","xray photon counter","sensor","CdTe","rate","8 Mcps","Si","2 Mcps",73109),
        Case("SB22","terahertz electrooptic sampler","crystal","GaP","window","5 THz","ZnTe","1.2 THz",73110),
        Case("SB33","quantum dot charge sensor","electrometer","RF-QPC","noise","0.3 ue","SET","1.2 ue",73111),
        Case("SB44","nanocalorimetric spectrometer","absorber","Au","resolution","7 eV","Al","28 eV",73112),
        Case("SB55","cold atom magnetometer","ensemble","Rb","sensitivity","4 fT","Cs","16 fT",73113),
        Case("SB66","photonic tensor sensor","material","SiN","channels","96","polymer","24",73114),
        Case("SB77","coherent electron camera","detector","Timepix4","rate","12 kHz","CCD","3 kHz",73115),
        Case("SB88","quantum acoustic delay line","substrate","LiNbO3","delay","8 us","quartz","2 us",73116),
    )


def _rows(suite):
    out = []
    for case in suite:
        options, ga, gb = case.option_pack()
        out.append(
            S17FusionCase(
                case_id=f"a0-s40-{case.case_id}",
                split="train",
                domain="s40_a0_only",
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
        _total,
        _primary,
        _relation,
        _pieces,
        _fc,
        _fp,
        raw_c,
        raw_p,
        relation_c,
        relation_p,
        signature_c,
        signature_p,
        encoded,
    ) = _native_losses(runtime, rows)
    return raw_c, raw_p, relation_c, relation_p, signature_c, signature_p, encoded


def _anchor_pair(treatment, reference, rows):
    (
        _trawc,
        _trawp,
        _trc,
        _trp,
        tsc,
        tsp,
        _tencoded,
    ) = _native_outputs(treatment, rows)
    with torch.no_grad():
        (
            _rrawc,
            _rrawp,
            _rrc,
            _rrp,
            rsc,
            rsp,
            _rencoded,
        ) = _native_outputs(reference, rows)
    anchor_c = signature_anchor_loss(tsc, rsc)
    anchor_p = signature_anchor_loss(tsp, rsp)
    return 0.5 * (anchor_c + anchor_p), tsc, tsp, rsc, rsp


def _gradient_projection_probe(bundle, manifest, rows):
    with torch.inference_mode(False), torch.enable_grad():
        reference = _runtime(bundle=bundle, manifest=manifest, seed=61040, train=True)
        treatment = _runtime(bundle=bundle, manifest=manifest, seed=61040, train=True)
        readout = _treatment(train_readout=True)

        reference_params = [p for p in reference.parameters() if p.requires_grad]
        treatment_runtime_params = [p for p in treatment.parameters() if p.requires_grad]
        if sum(p.numel() for p in reference_params) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S40-A0 reference surface changed")
        if sum(p.numel() for p in treatment_runtime_params) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S40-A0 treatment runtime surface changed")

        if any(
            not torch.equal(a.detach(), b.detach())
            for a, b in zip(reference_params, treatment_runtime_params)
        ):
            raise RuntimeError("S40-A0 matched runtime initialization changed")

        (
            _rrc0,
            _rrp0,
            rrelc0,
            rrelp0,
            rsc0,
            rsp0,
            _,
        ) = _native_outputs(reference, rows)
        (
            _trc0,
            _trp0,
            trelc0,
            trelp0,
            tsc0,
            tsp0,
            _,
        ) = _native_outputs(treatment, rows)
        native_relation_identity = max(
            float((rrelc0 - trelc0).abs().max().detach().cpu()),
            float((rrelp0 - trelp0).abs().max().detach().cpu()),
        )
        native_signature_identity = max(
            float((rsc0 - tsc0).abs().max().detach().cpu()),
            float((rsp0 - tsp0).abs().max().detach().cpu()),
        )
        if native_relation_identity != 0.0 or native_signature_identity != 0.0:
            raise RuntimeError("S40-A0 reference/treatment zero-init native identity failed")

        modules = iter_a13_lora_modules(treatment.encoder)
        if not modules:
            raise RuntimeError("S40-A0 treatment LoRA missing")
        with torch.no_grad():
            target = modules[0].lora_b
            perturb = torch.linspace(
                -2e-3, 2e-3, target.numel(), device=target.device, dtype=target.dtype
            ).reshape_as(target)
            target.add_(perturb)

        anchor, _tsc, _tsp, _rsc, _rsp = _anchor_pair(treatment, reference, rows)
        anchor_value = float(anchor.detach().cpu())
        if anchor_value <= 0.0:
            raise RuntimeError("S40-A0 synthetic signature drift failed")

        anchor_t_raw = torch.autograd.grad(
            anchor, treatment_runtime_params, retain_graph=True, allow_unused=True
        )
        anchor_t = [
            torch.zeros_like(p) if g is None else g
            for p, g in zip(treatment_runtime_params, anchor_t_raw)
        ]
        anchor_l1 = sum(float(g.abs().sum().detach().cpu()) for g in anchor_t)

        anchor_ref_raw = torch.autograd.grad(
            anchor, reference_params, retain_graph=True, allow_unused=True
        )
        anchor_ref_l1 = sum(
            0.0 if g is None else float(g.abs().sum().detach().cpu())
            for g in anchor_ref_raw
        )
        anchor_w_raw = torch.autograd.grad(
            anchor, readout.bilinear_weight, retain_graph=True, allow_unused=True
        )[0]
        anchor_w_l1 = 0.0 if anchor_w_raw is None else float(anchor_w_raw.abs().sum().cpu())

        lora_params = [m.lora_b for m in modules]
        lora_ids = {id(p) for p in lora_params}
        anchor_lora_l1 = sum(
            float(g.abs().sum().detach().cpu())
            for p, g in zip(treatment_runtime_params, anchor_t)
            if id(p) in lora_ids
        )
        if anchor_lora_l1 <= 0.0:
            raise RuntimeError("S40-A0 anchor LoRA gradient vanished")
        if anchor_ref_l1 != 0.0 or anchor_w_l1 != 0.0:
            raise RuntimeError("S40-A0 anchor leaked into reference or W")

        (
            _total,
            primary_block,
            relation_block,
            _pieces,
            *_rest,
        ) = _treatment_losses(treatment, rows, readout)
        treatment_all = [*treatment_runtime_params, readout.bilinear_weight]
        pg_raw = torch.autograd.grad(
            primary_block, treatment_all, retain_graph=True, allow_unused=True
        )
        rg_raw = torch.autograd.grad(
            relation_block, treatment_all, allow_unused=True
        )
        pg = [torch.zeros_like(p) if g is None else g for p, g in zip(treatment_all, pg_raw)]
        rg = [torch.zeros_like(p) if g is None else g for p, g in zip(treatment_all, rg_raw)]
        joint, _diag = norm_balanced_gradient_update(pg, rg, epsilon=s35.BALANCE_EPSILON)

        runtime_joint = joint[:-1]
        w_joint = joint[-1]
        correctness_w_l1 = float(w_joint.abs().sum().detach().cpu())
        correctness_w_offdiag_l1 = float(
            (w_joint.abs().sum() - torch.diagonal(w_joint).abs().sum()).detach().cpu()
        )
        correctness_lora_l1 = sum(
            float(g.abs().sum().detach().cpu())
            for p, g in zip(treatment_runtime_params, runtime_joint)
            if id(p) in lora_ids
        )
        if correctness_w_l1 <= 0.0 or correctness_w_offdiag_l1 <= 0.0:
            raise RuntimeError("S40-A0 joint correctness W gradient vanished")
        if correctness_lora_l1 <= 0.0:
            raise RuntimeError("S40-A0 joint correctness LoRA gradient vanished")

        conflict = [-g for g in anchor_t]
        projected, pdiag = project_runtime_gradient_against_reference_anchor(
            conflict, anchor_t, epsilon=1e-12
        )
        if not pdiag.projected or pdiag.pre_dot >= 0.0 or abs(pdiag.post_dot) > 2e-6:
            raise RuntimeError("S40-A0 conflicting projection contract failed")

        nonconflict, ndiag = project_runtime_gradient_against_reference_anchor(
            anchor_t, anchor_t, epsilon=1e-12
        )
        identity_err = max(
            float((a - b).abs().max().detach().cpu())
            for a, b in zip(nonconflict, anchor_t)
        )
        if ndiag.projected or identity_err != 0.0:
            raise RuntimeError("S40-A0 nonconflicting projection changed gradient")

        w_copy = w_joint.detach().clone()
        w_after = w_joint.detach().clone()
        w_projection_error = float((w_after - w_copy).abs().max().cpu())
        if w_projection_error != 0.0:
            raise RuntimeError("S40-A0 projection changed W gradient")

        before = [p.detach().clone() for p in treatment_runtime_params]
        anchor_before = anchor_value
        with torch.no_grad():
            for p, g in zip(treatment_runtime_params, projected):
                p.add_(g, alpha=-1e-4)
        anchor_after, *_ = _anchor_pair(treatment, reference, rows)
        anchor_after_value = float(anchor_after.detach().cpu())
        with torch.no_grad():
            for p, old in zip(treatment_runtime_params, before):
                p.copy_(old)
        if anchor_after_value > anchor_before + 1e-7:
            raise RuntimeError("S40-A0 projected synthetic step increased anchor")

        return {
            "zero_init_reference_treatment_relation_max_abs": native_relation_identity,
            "zero_init_reference_treatment_signature_max_abs": native_signature_identity,
            "synthetic_signature_anchor": anchor_value,
            "anchor_treatment_runtime_gradient_l1": anchor_l1,
            "anchor_treatment_lora_gradient_l1": anchor_lora_l1,
            "anchor_reference_runtime_gradient_l1": anchor_ref_l1,
            "anchor_w_gradient_l1": anchor_w_l1,
            "joint_correctness_w_gradient_l1": correctness_w_l1,
            "joint_correctness_w_offdiagonal_gradient_l1": correctness_w_offdiag_l1,
            "joint_correctness_lora_gradient_l1": correctness_lora_l1,
            "conflict_pre_dot": pdiag.pre_dot,
            "conflict_post_dot": pdiag.post_dot,
            "conflict_projected": pdiag.projected,
            "nonconflict_projected": ndiag.projected,
            "nonconflict_identity_max_abs": identity_err,
            "w_projection_max_abs": w_projection_error,
            "synthetic_anchor_before_step": anchor_before,
            "synthetic_anchor_after_projected_step": anchor_after_value,
        }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S40-A0 suite size changed")
    rows = _rows(suite)
    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)

    frozen = _runtime(bundle=bundle, manifest=manifest, seed=61001, train=False)
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
        raise RuntimeError("S40-A0 zero-init treatment/base identity failed")

    raw_c = _decision_logits(
        frozen,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    raw_p = _decision_logits(
        frozen,
        state_tokens=encoded["state_b_tokens"],
        state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )
    fusion = GradientIsolatedFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    cb_c, _ = fusion(raw_c, bc)
    cb_p, _ = fusion(raw_p, bp)
    tr_c, _ = fusion(raw_c, tc)
    tr_p, _ = fusion(raw_p, tp)
    fused_identity = max(float((cb_c-tr_c).abs().max()), float((cb_p-tr_p).abs().max()))
    if fused_identity != 0.0:
        raise RuntimeError("S40-A0 zero-init fused identity failed")
    choice_identity = 0.5 * (
        float((cb_c.argmax(-1)==tr_c.argmax(-1)).float().mean().cpu())
        + float((cb_p.argmax(-1)==tr_p.argmax(-1)).float().mean().cpu())
    )
    if choice_identity != 1.0:
        raise RuntimeError("S40-A0 selected choice identity failed")

    mass = max(
        float((torch.softmax(tr_c,dim=-1).sum(-1)-1).abs().max().cpu()),
        float((torch.softmax(tr_p,dim=-1).sum(-1)-1).abs().max().cpu()),
    )
    if mass > 1e-6:
        raise RuntimeError("S40-A0 probability mass changed")

    if treatment.added_parameter_count != 65_536:
        raise RuntimeError("S40-A0 W capacity changed")
    control_surface = HIRA_V1_S17_TOTAL_PARAMETER_COUNT
    treatment_surface = control_surface + treatment.added_parameter_count
    if control_surface != 49_152 or treatment_surface != 114_688:
        raise RuntimeError("S40-A0 parameter surface changed")

    mechanics = _mechanics_court()
    projection = _projection_independence(frozen, rows)
    gradient_projection = _gradient_projection_probe(bundle, manifest, rows)

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
        "scientific_authority": "S40_A0_REFERENCE_ANCHORED_PROJECTED_FULL_BILINEAR_ONLY",
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
        "projection_epsilon": 1e-12,
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
        **gradient_projection,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S40_A0_RECEIPT="+json.dumps(result,sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
