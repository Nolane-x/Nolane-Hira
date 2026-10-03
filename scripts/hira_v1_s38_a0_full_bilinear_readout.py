from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_native_relation_geometry import NativeA13RelationCanonicalizer
from nmd.v1_bilinear_signature_readout import NativeBilinearSignatureCorrectnessReadout
from nmd.v1_relation_canonicalization import cross_view_relation_signature_loss
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

SCHEMA_VERSION = "hira-v1-s38-a0-full-bilinear-signature-readout-v1"
OUTCOME = "HIRA_V1_S38_A0_FULL_BILINEAR_READOUT_READY"


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
        return f"S38-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Audit {self.case_id} records {self.second} for {self.field_b}; the same S38-A0 {self.noun} records {self.first} for {self.field_a}."

    @property
    def qa1(self):
        return f"For S38-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry is tagged {self.field_a} in S38-A0 {self.case_id}?"

    @property
    def qb1(self):
        return f"For S38-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry is tagged {self.field_b} in S38-A0 {self.case_id}?"

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
        Case("RN11","quantum cascade detector","absorber","InGaAs","cutoff","11 um","HgCdTe","4 um",70101),
        Case("RN22","optical frequency divider","reference","ULE cavity","division","1000","quartz","250",70102),
        Case("RN33","helium ion microscope","source","GFIS","energy","35 keV","plasma","9 keV",70103),
        Case("RN44","terahertz near-field probe","antenna","bowtie","resolution","12 um","dipole","48 um",70104),
        Case("RN55","nanoscale NMR sensor","defect","NV center","depth","6 nm","P1 center","24 nm",70105),
        Case("RN66","femtosecond electron gun","cathode","Cu","pulse width","80 fs","LaB6","320 fs",70106),
        Case("RN77","integrated gyroscope","resonator","ring","bias stability","0.4 deg/h","disk","1.6 deg/h",70107),
        Case("RN88","superresolution spectrometer","decoder","compressed","channel width","2 pm","grating","8 pm",70108),
        Case("RP11","cryogenic photodiode","junction","Ge-on-Si","dark current","3 pA","InGaAs","12 pA",70109),
        Case("RP22","magnetic resonance force microscope","tip","FeCo","force noise","5 aN","Ni","20 aN",70110),
        Case("RP33","ultrafast streak camera","sweep","RF","resolution","180 fs","electrostatic","720 fs",70111),
        Case("RP44","photonic crystal accelerometer","cavity","L3","sensitivity","9 pm/g","ring","2 pm/g",70112),
        Case("RP55","neutral atom imager","objective","NA0.8","field","120 um","NA0.4","30 um",70113),
        Case("RP66","microresonator thermometer","material","SiN","resolution","6 uK","silica","24 uK",70114),
        Case("RP77","coherent xray detector","sensor","DEPFET","frame rate","8 kHz","CCD","2 kHz",70115),
        Case("RP88","optical neural probe","waveguide","SiN","loss","0.3 dB/cm","polymer","1.2 dB/cm",70116),
    )

def _rows(suite):
    out = []
    for case in suite:
        options, ga, gb = case.option_pack()
        out.append(
            S17FusionCase(
                case_id=f"a0-s38-{case.case_id}",
                split="train",
                domain="s38_a0_only",
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


def _base():
    return NativeA13RelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=BINDING_CONTRASTIVE_TEMPERATURE,
        native_dimension=256,
    )


def _treatment(*, train_readout=True):
    return NativeBilinearSignatureCorrectnessReadout(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=BINDING_CONTRASTIVE_TEMPERATURE,
        native_dimension=256,
        query_norm_epsilon=1e-12,
        residual_scale=1.0,
        train_readout=train_readout,
    )


def _relation_inputs(encoded, state_prefix, question_prefix):
    return dict(
        state_tokens=encoded[f"state_{state_prefix}_tokens"].repeat_interleave(2, dim=0),
        state_mask=encoded[f"state_{state_prefix}_mask"].repeat_interleave(2, dim=0),
        question_tokens=encoded[f"question_{question_prefix}_tokens"],
        question_mask=encoded[f"question_{question_prefix}_mask"],
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2, dim=0),
    )


def _mechanics_court():
    g = torch.Generator().manual_seed(73801)
    state = torch.randn(3, 5, 256, generator=g)
    question = torch.randn(3, 4, 256, generator=g)
    options = torch.randn(3, 7, 2, 4, 256, generator=g)
    sm = torch.ones(3, 5, dtype=torch.bool)
    qm = torch.ones(3, 4, dtype=torch.bool)
    otm = torch.ones(3, 7, 2, 4, dtype=torch.bool)
    ovm = torch.ones(3, 7, 2, dtype=torch.bool)

    op = _treatment()
    base = _base()
    kwargs = dict(
        state_tokens=state,
        state_mask=sm,
        question_tokens=question,
        question_mask=qm,
        option_view_tokens=options,
        option_view_token_mask=otm,
        option_view_mask=ovm,
    )
    bl, bs, _ = base(**kwargs)
    tl, ts, _ = op(**kwargs)
    if not torch.equal(bl, tl) or not torch.equal(bs, ts):
        raise RuntimeError("S38-A0 zero-init native identity failed")

    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    pk = dict(kwargs)
    pk["option_view_tokens"] = options[:, perm]
    pk["option_view_token_mask"] = otm[:, perm]
    pk["option_view_mask"] = ovm[:, perm]
    pl, ps, _ = op(**pk)
    logical_logit = float((pl - tl[:, perm]).abs().max())
    logical_sig = float((ps - ts[:, perm]).abs().max())
    if logical_logit > 2e-6 or logical_sig > 2e-6:
        raise RuntimeError("S38-A0 logical-option equivariance failed")

    arbitrary = {}
    for k in (3, 7):
        x = dict(kwargs)
        x["option_view_tokens"] = options[:, :k]
        x["option_view_token_mask"] = otm[:, :k]
        x["option_view_mask"] = ovm[:, :k]
        kl, ks, _ = op(**x)
        arbitrary[str(k)] = [list(kl.shape), list(ks.shape)]
        if tuple(kl.shape) != (3, k) or tuple(ks.shape) != (3, k, 256):
            raise RuntimeError(f"S38-A0 arbitrary K failed: {k}")

    probe = _treatment()
    with torch.no_grad():
        weight = torch.zeros(256, 256)
        idx = torch.arange(256)
        weight[idx, (idx + 17) % 256] = torch.linspace(-0.2, 0.2, 256)
        probe.bilinear_weight.copy_(weight)

    fixed_signatures = ts.detach()
    residual_a = probe.readout_residual(
        signatures=fixed_signatures,
        question_tokens=question,
        question_mask=qm,
    )
    changed_question = question.clone()
    changed_question[:, :, :128] *= -1.0
    residual_b = probe.readout_residual(
        signatures=fixed_signatures,
        question_tokens=changed_question,
        question_mask=qm,
    )
    query_intervention = float((residual_a - residual_b).abs().max())
    if query_intervention <= 1e-6:
        raise RuntimeError("S38-A0 query intervention discriminator failed")

    changed_signatures = fixed_signatures.clone()
    changed_signatures[:, :, 128:] *= -1.0
    residual_c = probe.readout_residual(
        signatures=changed_signatures,
        question_tokens=question,
        question_mask=qm,
    )
    signature_intervention = float((residual_a - residual_c).abs().max())
    if signature_intervention <= 1e-6:
        raise RuntimeError("S38-A0 signature intervention discriminator failed")

    probe_logits, probe_sig, _ = probe(**kwargs)
    qperm = torch.tensor([2, 0, 3, 1])
    qkwargs = dict(kwargs)
    qkwargs["question_tokens"] = question[:, qperm]
    qkwargs["question_mask"] = qm[:, qperm]
    qlog, qsig, _ = probe(**qkwargs)
    question_perm_logit = float((qlog - probe_logits).abs().max())
    question_perm_sig = float((qsig - probe_sig).abs().max())
    if question_perm_logit > 2e-6 or question_perm_sig > 2e-6:
        raise RuntimeError("S38-A0 question-token permutation invariance failed")

    padded_question = torch.cat(
        [question, torch.randn(3, 2, 256, generator=g)],
        dim=1,
    )
    padded_qm = torch.cat(
        [qm, torch.zeros(3, 2, dtype=torch.bool)],
        dim=1,
    )
    padkwargs = dict(kwargs)
    padkwargs["question_tokens"] = padded_question
    padkwargs["question_mask"] = padded_qm
    padlog, padsig, _ = probe(**padkwargs)
    padding_logit = float((padlog - probe_logits).abs().max())
    padding_sig = float((padsig - probe_sig).abs().max())
    if padding_logit > 2e-6 or padding_sig > 2e-6:
        raise RuntimeError("S38-A0 masked query-padding invariance failed")

    source = _treatment()
    with torch.no_grad():
        weight = torch.zeros(256, 256)
        idx = torch.arange(256)
        weight[idx, (idx + 31) % 256] = torch.linspace(-0.125, 0.125, 256)
        source.bilinear_weight.copy_(weight)
    state_dict = source.readout_state_dict()
    target = _treatment()
    target.load_readout_state_dict(state_dict, freeze=True)
    replay = target.readout_state_dict()
    roundtrip = torch.equal(state_dict["bilinear.weight"], replay["bilinear.weight"])
    if not roundtrip:
        raise RuntimeError("S38-A0 readout checkpoint roundtrip failed")

    return {
        "query_intervention_residual_max_abs": query_intervention,
        "signature_intervention_residual_max_abs": signature_intervention,
        "question_token_permutation_logit_error": question_perm_logit,
        "question_token_permutation_signature_error": question_perm_sig,
        "masked_query_padding_logit_error": padding_logit,
        "masked_query_padding_signature_error": padding_sig,
        "logical_option_permutation_logit_error": logical_logit,
        "logical_option_permutation_signature_error": logical_sig,
        "arbitrary_k_shapes": arbitrary,
        "checkpoint_key_count": len(state_dict),
        "checkpoint_roundtrip_exact": roundtrip,
    }


def _projection_independence(runtime, rows):
    encoded = _encode_batch(runtime, rows)
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S38-A0 projection scorer missing")

    kwargs_c = _relation_inputs(encoded, "a", "canonical")
    kwargs_p = _relation_inputs(encoded, "b", "paraphrase")
    base = _base()
    bc, sc, _ = base(**kwargs_c)
    bp, sp, _ = base(**kwargs_p)

    raw_c = _decision_logits(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    raw_p = _decision_logits(
        runtime,
        state_tokens=encoded["state_b_tokens"],
        state_mask=encoded["state_b_mask"],
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
        encoded=encoded,
    )

    original = scorer.projection.weight.detach().clone()
    with torch.no_grad():
        scale = torch.linspace(0.3, 2.7, original.shape[0], device=original.device, dtype=original.dtype)[:, None]
        perturb = 0.05 * torch.cos(
            torch.arange(original.numel(), device=original.device, dtype=original.dtype)
        ).reshape_as(original)
        scorer.projection.weight.copy_(original * scale + perturb)

    encoded2 = _encode_batch(runtime, rows)
    ac, asc, _ = base(**_relation_inputs(encoded2, "a", "canonical"))
    ap, asp, _ = base(**_relation_inputs(encoded2, "b", "paraphrase"))
    raw_c2 = _decision_logits(
        runtime,
        state_tokens=encoded2["state_a_tokens"],
        state_mask=encoded2["state_a_mask"],
        question_tokens=encoded2["question_canonical_tokens"],
        question_mask=encoded2["question_canonical_mask"],
        encoded=encoded2,
    )
    raw_p2 = _decision_logits(
        runtime,
        state_tokens=encoded2["state_b_tokens"],
        state_mask=encoded2["state_b_mask"],
        question_tokens=encoded2["question_paraphrase_tokens"],
        question_mask=encoded2["question_paraphrase_mask"],
        encoded=encoded2,
    )
    with torch.no_grad():
        scorer.projection.weight.copy_(original)

    relation_logit = max(float((ac-bc).abs().max()), float((ap-bp).abs().max()))
    relation_sig = max(float((asc-sc).abs().max()), float((asp-sp).abs().max()))
    primary = max(float((raw_c2-raw_c).abs().max()), float((raw_p2-raw_p).abs().max()))
    if relation_logit != 0.0 or relation_sig != 0.0:
        raise RuntimeError("S38-A0 native representation depends on shared projection")
    if primary <= 1e-6:
        raise RuntimeError("S38-A0 primary projection sensitivity vanished")
    return {
        "native_projection_perturbation_logit_max_abs": relation_logit,
        "native_projection_perturbation_signature_max_abs": relation_sig,
        "primary_projection_perturbation_logit_max_abs": primary,
    }


def _gradient_probe(bundle, manifest, rows):
    with torch.inference_mode(False), torch.enable_grad():
        fresh = load_hira_v0_m4_bundle(bundle)
        runtime = build_hira_v1_s17_norm_balanced_core(
            fresh.runtime.encoder,
            bundle / str(manifest["t0_checkpoint"]),
            expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
            train_lora=True,
            train_projection=True,
        )
        del fresh
        runtime.clear_schema_cache()
        runtime.eval()
        trainable = [p for p in runtime.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S38-A0 base trainable surface changed")

        encoded = _encode_batch(runtime, rows)
        op = _treatment(train_readout=True)
        rc, sc, _ = op(**_relation_inputs(encoded, "a", "canonical"))
        rp, sp, _ = op(**_relation_inputs(encoded, "b", "paraphrase"))
        gold, _ = _gold_tensors(rows, device=rc.device)
        relation_ce = 0.5 * (F.cross_entropy(rc, gold) + F.cross_entropy(rp, gold))
        canonicalization, _align, _sep = cross_view_relation_signature_loss(
            sc, sp, separation_margin=SIGNATURE_SEPARATION_MARGIN
        )
        relation_block = (
            BINDING_COEFFICIENT * relation_ce
            + CANONICALIZATION_COEFFICIENT * canonicalization
        )

        scorer = runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S38-A0 projection scorer missing")
        modules = iter_a13_lora_modules(runtime.encoder)
        lora_b = [m.lora_b for m in modules]

        targets = [op.bilinear_weight, scorer.projection.weight, *lora_b]
        grads = torch.autograd.grad(relation_block, targets, retain_graph=True, allow_unused=True)
        readout_relation = 0.0 if grads[0] is None else float(grads[0].abs().sum().cpu())
        if grads[0] is None:
            offdiag_relation = 0.0
        else:
            diag = torch.diagonal(grads[0])
            offdiag_relation = float((grads[0].abs().sum() - diag.abs().sum()).cpu())
        projection_relation = 0.0 if grads[1] is None else float(grads[1].abs().sum().cpu())
        lora_relation = sum(0.0 if g is None else float(g.abs().sum().cpu()) for g in grads[2:])

        _total, primary_block, _rb, *_rest = _losses(runtime, rows)
        pg = torch.autograd.grad(primary_block, scorer.projection.weight, retain_graph=True, allow_unused=False)[0]
        primary_projection = float(pg.abs().sum().cpu())
        rg = torch.autograd.grad(primary_block, op.bilinear_weight, allow_unused=True)[0]
        primary_readout = 0.0 if rg is None else float(rg.abs().sum().cpu())

        if readout_relation <= 0.0:
            raise RuntimeError("S38-A0 bilinear relation gradient vanished")
        if offdiag_relation <= 0.0:
            raise RuntimeError("S38-A0 off-diagonal bilinear gradient vanished")
        if primary_readout != 0.0:
            raise RuntimeError("S38-A0 primary block leaked into readout")
        if projection_relation != 0.0:
            raise RuntimeError("S38-A0 native relation leaked into projection")
        if primary_projection <= 0.0:
            raise RuntimeError("S38-A0 primary projection gradient vanished")
        if lora_relation <= 0.0:
            raise RuntimeError("S38-A0 relation LoRA gradient vanished")

        return {
            "relation_block": float(relation_block.detach().cpu()),
            "relation_ce": float(relation_ce.detach().cpu()),
            "canonicalization": float(canonicalization.detach().cpu()),
            "readout_relation_gradient_l1": readout_relation,
            "bilinear_off_diagonal_relation_gradient_l1": offdiag_relation,
            "readout_primary_gradient_l1": primary_readout,
            "relation_to_projection_gradient_l1": projection_relation,
            "primary_to_projection_gradient_l1": primary_projection,
            "relation_to_lora_gradient_l1": lora_relation,
        }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S38-A0 suite size changed")
    rows = _rows(suite)
    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)

    fresh = load_hira_v0_m4_bundle(bundle)
    runtime = build_hira_v1_s17_norm_balanced_core(
        fresh.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    del fresh
    runtime.clear_schema_cache()
    runtime.eval()

    encoded = _encode_batch(runtime, rows)
    base = _base()
    treatment = _treatment(train_readout=True)

    bc, bs, _ = base(**_relation_inputs(encoded, "a", "canonical"))
    bp, bsp, _ = base(**_relation_inputs(encoded, "b", "paraphrase"))
    tc, ts, _ = treatment(**_relation_inputs(encoded, "a", "canonical"))
    tp, tsp, _ = treatment(**_relation_inputs(encoded, "b", "paraphrase"))

    relation_identity = max(float((tc-bc).abs().max()), float((tp-bp).abs().max()))
    signature_identity = max(float((ts-bs).abs().max()), float((tsp-bsp).abs().max()))
    if relation_identity != 0.0 or signature_identity != 0.0:
        raise RuntimeError("S38-A0 zero-init relation identity failed")

    raw_c = _decision_logits(
        runtime,
        state_tokens=encoded["state_a_tokens"],
        state_mask=encoded["state_a_mask"],
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
        encoded=encoded,
    )
    raw_p = _decision_logits(
        runtime,
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
    primary_identity = 0.0
    if fused_identity != 0.0:
        raise RuntimeError("S38-A0 zero-init fused identity failed")

    choice_identity = 0.5 * (
        float((cb_c.argmax(-1)==tr_c.argmax(-1)).float().mean().cpu())
        + float((cb_p.argmax(-1)==tr_p.argmax(-1)).float().mean().cpu())
    )
    if choice_identity != 1.0:
        raise RuntimeError("S38-A0 selected choices changed at zero init")

    mass = max(
        float((torch.softmax(tr_c, dim=-1).sum(-1)-1).abs().max().cpu()),
        float((torch.softmax(tr_p, dim=-1).sum(-1)-1).abs().max().cpu()),
    )
    if mass > 1e-6:
        raise RuntimeError("S38-A0 probability mass changed")

    base_surface = HIRA_V1_S17_TOTAL_PARAMETER_COUNT
    treatment_surface = base_surface + treatment.added_parameter_count
    if treatment.added_parameter_count != 65_536 or base_surface != 49_152 or treatment_surface != 114_688:
        raise RuntimeError("S38-A0 parameter surface changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad) != 0:
        raise RuntimeError("S38-A0 frozen runtime became trainable")
    original = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original != 0:
        raise RuntimeError("S38-A0 original A13 became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S38-A0 HIRACore became trainable")

    mechanics = _mechanics_court()
    projection = _projection_independence(runtime, rows)
    gradient = _gradient_probe(bundle, manifest, rows)

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
        "scientific_authority": "S38_A0_FULL_BILINEAR_NATIVE_QUERY_SIGNATURE_READOUT_ONLY",
        "semantic_case_count": len(suite),
        "state_view_count": 2 * len(suite),
        "k": 4,
        "views_per_option": 2,
        "native_dimension": 256,
        "residual_scale": 1.0,
        "query_norm_epsilon": 1e-12,
        "operator_added_parameter_count": treatment.added_parameter_count,
        "control_trainable_parameter_count": base_surface,
        "treatment_trainable_parameter_count": treatment_surface,
        "lora_parameter_count": 16_384,
        "projection_parameter_count": HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
        "runtime_trainable_parameter_count": 0,
        "original_a13_trainable_parameter_count": original,
        "hira_core_trainable_parameter_count": 0,
        "zero_init_relation_logit_max_abs": relation_identity,
        "zero_init_signature_max_abs": signature_identity,
        "zero_init_primary_logit_max_abs": primary_identity,
        "zero_init_fused_logit_max_abs": fused_identity,
        "zero_init_selected_choice_identity_rate": choice_identity,
        "native_probability_mass_error": mass,
        "full_k": tr_c.shape[-1] == 4 and tr_p.shape[-1] == 4,
        "state_once_expected_view_count": 2 * len(suite),
        "a0_relation_accuracy": relation_acc,
        "a0_fused_accuracy": fused_acc,
        **mechanics,
        **projection,
        **gradient,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S38_A0_RECEIPT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
