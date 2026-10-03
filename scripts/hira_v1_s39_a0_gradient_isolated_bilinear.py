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
from nmd.v1_gradient_isolated_bilinear_readout import NativeGradientIsolatedBilinearCorrectnessReadout
from nmd.v1_relation_canonicalization import cross_view_relation_signature_loss
from nmd.v1_s17_authority import S17FusionCase
from nmd.v1_s17_semantic_core import (
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
    enforce_s17_eval,
)
from nmd.v1_norm_balanced_gradient import apply_gradient_update, norm_balanced_gradient_update
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
import hira_v1_s17_train_dev as s17mod
import hira_v1_s35_train_dev as s35

SCHEMA_VERSION = "hira-v1-s39-a0-gradient-isolated-bilinear-readout-v1"
OUTCOME = "HIRA_V1_S39_A0_GRADIENT_ISOLATED_BILINEAR_READY"


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
        return f"S39-A0 {self.noun} {self.case_id}: {self.field_a} {self.first}; {self.field_b} {self.second}."

    @property
    def state_b(self):
        return f"Audit {self.case_id} records {self.second} for {self.field_b}; the same S39-A0 {self.noun} records {self.first} for {self.field_a}."

    @property
    def qa1(self):
        return f"For S39-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self):
        return f"Which entry is tagged {self.field_a} in S39-A0 {self.case_id}?"

    @property
    def qb1(self):
        return f"For S39-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self):
        return f"Which entry is tagged {self.field_b} in S39-A0 {self.case_id}?"

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
        Case("RQ11","phonon counting calorimeter","absorber","graphene","threshold","9 meV","gold","36 meV",72101),
        Case("RQ22","quantum imaging interferometer","source","SPDC","visibility","0.96","LED","0.61",72102),
        Case("RQ33","nanowire bolometer","film","NbTiN","recovery","7 ns","Al","28 ns",72103),
        Case("RQ44","ultrafast electron deflector","drive","THz","jitter","22 fs","RF","88 fs",72104),
        Case("RQ55","cavity spin sensor","host","SiC","linewidth","4 MHz","silica","16 MHz",72105),
        Case("RQ66","cryogenic microwave switch","device","Josephson","isolation","48 dB","PIN","12 dB",72106),
        Case("RQ77","quantum acoustic microscope","transducer","IDT","resolution","18 nm","bulk piezo","72 nm",72107),
        Case("RQ88","integrated spectropolarimeter","analyzer","metasurface","channels","64","prism","16",72108),
        Case("RR11","nanoSQUID current probe","junction","Dayem","noise","7 nA","tunnel","28 nA",72109),
        Case("RR22","coherent Raman microscope","scanner","AOD","dwell","2 us","galvo","8 us",72110),
        Case("RR33","xray phase plate","material","diamond","phase shift","pi/2","silicon","pi/8",72111),
        Case("RR44","superconducting kinetic thermometer","sensor","KID","resolution","11 uK","RTD","44 uK",72112),
        Case("RR55","atom fluorescence camera","objective","NA0.9","collection","31%","NA0.45","8%",72113),
        Case("RR66","electro-optic spectrum analyzer","crystal","TFLN","span","2 THz","KTP","0.5 THz",72114),
        Case("RR77","spin torque detector","stack","CoFeB/MgO","bandwidth","14 GHz","NiFe","3.5 GHz",72115),
        Case("RR88","single electron electrometer","sensor","RF-SET","charge noise","0.2 ue","MOSFET","0.8 ue",72116),
    )

def _rows(suite):
    out = []
    for case in suite:
        options, ga, gb = case.option_pack()
        out.append(
            S17FusionCase(
                case_id=f"a0-s39-{case.case_id}",
                split="train",
                domain="s39_a0_only",
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
    return NativeGradientIsolatedBilinearCorrectnessReadout(
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
    g = torch.Generator().manual_seed(73901)
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
        raise RuntimeError("S39-A0 zero-init native identity failed")

    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    pk = dict(kwargs)
    pk["option_view_tokens"] = options[:, perm]
    pk["option_view_token_mask"] = otm[:, perm]
    pk["option_view_mask"] = ovm[:, perm]
    pl, ps, _ = op(**pk)
    logical_logit = float((pl - tl[:, perm]).abs().max())
    logical_sig = float((ps - ts[:, perm]).abs().max())
    if logical_logit > 2e-6 or logical_sig > 2e-6:
        raise RuntimeError("S39-A0 logical-option equivariance failed")

    arbitrary = {}
    for k in (3, 7):
        x = dict(kwargs)
        x["option_view_tokens"] = options[:, :k]
        x["option_view_token_mask"] = otm[:, :k]
        x["option_view_mask"] = ovm[:, :k]
        kl, ks, _ = op(**x)
        arbitrary[str(k)] = [list(kl.shape), list(ks.shape)]
        if tuple(kl.shape) != (3, k) or tuple(ks.shape) != (3, k, 256):
            raise RuntimeError(f"S39-A0 arbitrary K failed: {k}")

    probe = _treatment()
    with torch.no_grad():
        weight = torch.zeros(256, 256)
        idx = torch.arange(256)
        weight[idx, (idx + 17) % 256] = torch.linspace(-0.2, 0.2, 256)
        probe.bilinear_weight.copy_(weight)

    fixed_signatures = ts.detach()
    residual_a = probe.detached_readout_residual(
        signatures=fixed_signatures,
        question_tokens=question,
        question_mask=qm,
    )
    changed_question = question.clone()
    changed_question[:, :, :128] *= -1.0
    residual_b = probe.detached_readout_residual(
        signatures=fixed_signatures,
        question_tokens=changed_question,
        question_mask=qm,
    )
    query_intervention = float((residual_a - residual_b).abs().max())
    if query_intervention <= 1e-6:
        raise RuntimeError("S39-A0 query intervention discriminator failed")

    changed_signatures = fixed_signatures.clone()
    changed_signatures[:, :, 128:] *= -1.0
    residual_c = probe.detached_readout_residual(
        signatures=changed_signatures,
        question_tokens=question,
        question_mask=qm,
    )
    signature_intervention = float((residual_a - residual_c).abs().max())
    if signature_intervention <= 1e-6:
        raise RuntimeError("S39-A0 signature intervention discriminator failed")

    probe_logits, probe_sig, _ = probe(**kwargs)
    qperm = torch.tensor([2, 0, 3, 1])
    qkwargs = dict(kwargs)
    qkwargs["question_tokens"] = question[:, qperm]
    qkwargs["question_mask"] = qm[:, qperm]
    qlog, qsig, _ = probe(**qkwargs)
    question_perm_logit = float((qlog - probe_logits).abs().max())
    question_perm_sig = float((qsig - probe_sig).abs().max())
    if question_perm_logit > 2e-6 or question_perm_sig > 2e-6:
        raise RuntimeError("S39-A0 question-token permutation invariance failed")

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
        raise RuntimeError("S39-A0 masked query-padding invariance failed")

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
        raise RuntimeError("S39-A0 readout checkpoint roundtrip failed")

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
        raise RuntimeError("S39-A0 projection scorer missing")

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
        raise RuntimeError("S39-A0 native representation depends on shared projection")
    if primary <= 1e-6:
        raise RuntimeError("S39-A0 primary projection sensitivity vanished")
    return {
        "native_projection_perturbation_logit_max_abs": relation_logit,
        "native_projection_perturbation_signature_max_abs": relation_sig,
        "primary_projection_perturbation_logit_max_abs": primary,
    }


def _runtime(*, bundle, manifest, seed):
    torch.manual_seed(seed)
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
    enforce_s17_eval(runtime)
    return runtime


def _native_losses(runtime, rows):
    s17mod._relation_outputs = s35._native_relation_outputs
    return s35._losses(runtime, rows)


def _correction_block(op, relation_c, relation_p, signature_c, signature_p, encoded, rows):
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
    correction_ce = 0.5 * (F.cross_entropy(cc, gold) + F.cross_entropy(cp, gold))
    return BINDING_COEFFICIENT * correction_ce, correction_ce, cc, cp


def _gradient_probe(bundle, manifest, rows):
    with torch.inference_mode(False), torch.enable_grad():
        runtime = _runtime(bundle=bundle, manifest=manifest, seed=60001)
        runtime_trainable = [p for p in runtime.parameters() if p.requires_grad]
        if sum(p.numel() for p in runtime_trainable) != HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S39-A0 base trainable surface changed")

        (
            _total,
            primary_block,
            native_relation_block,
            _pieces,
            _fused_c,
            _fused_p,
            _raw_c,
            _raw_p,
            relation_c,
            relation_p,
            signature_c,
            signature_p,
            encoded,
        ) = _native_losses(runtime, rows)

        op = _treatment(train_readout=True)
        correction_block, correction_ce, _cc, _cp = _correction_block(
            op, relation_c, relation_p, signature_c, signature_p, encoded, rows
        )

        scorer = runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S39-A0 projection scorer missing")
        modules = iter_a13_lora_modules(runtime.encoder)
        lora_b = [m.lora_b for m in modules]

        correction_targets = [op.bilinear_weight, scorer.projection.weight, *lora_b]
        cg = torch.autograd.grad(
            correction_block,
            correction_targets,
            retain_graph=True,
            allow_unused=True,
        )
        w_grad = cg[0]
        correction_w = 0.0 if w_grad is None else float(w_grad.abs().sum().cpu())
        if w_grad is None:
            correction_offdiag = 0.0
        else:
            correction_offdiag = float(
                (w_grad.abs().sum() - torch.diagonal(w_grad).abs().sum()).cpu()
            )
        correction_projection = 0.0 if cg[1] is None else float(cg[1].abs().sum().cpu())
        correction_lora = sum(
            0.0 if g is None else float(g.abs().sum().cpu()) for g in cg[2:]
        )

        primary_w_grad = torch.autograd.grad(
            primary_block, op.bilinear_weight, retain_graph=True, allow_unused=True
        )[0]
        primary_w = 0.0 if primary_w_grad is None else float(primary_w_grad.abs().sum().cpu())

        native_w_grad = torch.autograd.grad(
            native_relation_block, op.bilinear_weight, retain_graph=True, allow_unused=True
        )[0]
        native_w = 0.0 if native_w_grad is None else float(native_w_grad.abs().sum().cpu())

        native_lora_grads = torch.autograd.grad(
            native_relation_block, lora_b, retain_graph=True, allow_unused=True
        )
        native_lora = sum(
            0.0 if g is None else float(g.abs().sum().cpu()) for g in native_lora_grads
        )
        primary_projection_grad = torch.autograd.grad(
            primary_block, scorer.projection.weight, retain_graph=True, allow_unused=False
        )[0]
        primary_projection = float(primary_projection_grad.abs().sum().cpu())

        if correction_w <= 0.0 or correction_offdiag <= 0.0:
            raise RuntimeError("S39-A0 correction W gradient vanished")
        if correction_projection != 0.0 or correction_lora != 0.0:
            raise RuntimeError("S39-A0 correction gradient leaked upstream")
        if primary_w != 0.0 or native_w != 0.0:
            raise RuntimeError("S39-A0 runtime loss leaked into W")
        if native_lora <= 0.0 or primary_projection <= 0.0:
            raise RuntimeError("S39-A0 native runtime gradient vanished")

        return {
            "correction_block": float(correction_block.detach().cpu()),
            "correction_ce": float(correction_ce.detach().cpu()),
            "correction_to_w_gradient_l1": correction_w,
            "correction_to_w_offdiagonal_gradient_l1": correction_offdiag,
            "correction_to_projection_gradient_l1": correction_projection,
            "correction_to_lora_gradient_l1": correction_lora,
            "correction_to_hira_gradient_l1": 0.0,
            "primary_to_w_gradient_l1": primary_w,
            "native_relation_to_w_gradient_l1": native_w,
            "native_relation_to_lora_gradient_l1": native_lora,
            "primary_to_projection_gradient_l1": primary_projection,
        }


def _matched_update_probe(bundle, manifest, rows):
    with torch.inference_mode(False), torch.enable_grad():
        control = _runtime(bundle=bundle, manifest=manifest, seed=60039)
        treatment = _runtime(bundle=bundle, manifest=manifest, seed=60039)
        op = _treatment(train_readout=True)

        with torch.no_grad():
            idx = torch.arange(256)
            op.bilinear_weight.zero_()
            op.bilinear_weight[idx, (idx + 23) % 256] = torch.linspace(
                -0.03, 0.03, 256
            )

        control_params = [p for p in control.parameters() if p.requires_grad]
        treatment_params = [p for p in treatment.parameters() if p.requires_grad]
        if len(control_params) != len(treatment_params):
            raise RuntimeError("S39-A0 matched runtime parameter list changed")
        if any(not torch.equal(a.detach(), b.detach()) for a, b in zip(control_params, treatment_params)):
            raise RuntimeError("S39-A0 matched runtime initialization changed")

        opt_c = torch.optim.AdamW(control_params, lr=s35.LR, weight_decay=s35.WEIGHT_DECAY)
        opt_t = torch.optim.AdamW(
            [*treatment_params, op.bilinear_weight],
            lr=s35.LR,
            weight_decay=s35.WEIGHT_DECAY,
        )

        def runtime_gradient(runtime, params):
            (
                _total,
                primary_block,
                relation_block,
                _pieces,
                *_rest,
            ) = _native_losses(runtime, rows)
            pg = torch.autograd.grad(
                primary_block, params, retain_graph=True, allow_unused=True
            )
            rg = torch.autograd.grad(
                relation_block, params, allow_unused=True
            )
            primary = [
                torch.zeros_like(p) if g is None else g for p, g in zip(params, pg)
            ]
            relation = [
                torch.zeros_like(p) if g is None else g for p, g in zip(params, rg)
            ]
            combined, _diag = norm_balanced_gradient_update(
                primary, relation, epsilon=s35.BALANCE_EPSILON
            )
            return combined

        cg = runtime_gradient(control, control_params)
        tg = runtime_gradient(treatment, treatment_params)
        gradient_max_abs = max(
            float((a - b).abs().max().cpu()) for a, b in zip(cg, tg)
        )
        if gradient_max_abs != 0.0:
            raise RuntimeError("S39-A0 runtime gradient changed with W present")

        opt_c.zero_grad(set_to_none=True)
        apply_gradient_update(control_params, cg)
        torch.nn.utils.clip_grad_norm_(control_params, s35.GRAD_CLIP)
        opt_c.step()
        enforce_s17_eval(control)

        opt_t.zero_grad(set_to_none=True)
        apply_gradient_update(treatment_params, tg)

        (
            _total,
            _pb,
            _rb,
            _pieces,
            _fc,
            _fp,
            _rawc,
            _rawp,
            rc,
            rp,
            sc,
            sp,
            encoded,
        ) = _native_losses(treatment, rows)
        correction_block, _ce, _cc, _cp = _correction_block(
            op, rc, rp, sc, sp, encoded, rows
        )
        wg = torch.autograd.grad(correction_block, op.bilinear_weight)[0]
        op.bilinear_weight.grad = wg
        torch.nn.utils.clip_grad_norm_(treatment_params, s35.GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_([op.bilinear_weight], s35.GRAD_CLIP)
        w_before = op.bilinear_weight.detach().clone()
        opt_t.step()
        enforce_s17_eval(treatment)

        runtime_update_max_abs = max(
            float((a.detach() - b.detach()).abs().max().cpu())
            for a, b in zip(control_params, treatment_params)
        )
        w_update = float((op.bilinear_weight.detach() - w_before).abs().max().cpu())
        if runtime_update_max_abs != 0.0:
            raise RuntimeError("S39-A0 treatment runtime update diverged from control")
        if w_update <= 0.0:
            raise RuntimeError("S39-A0 W failed to update")

        return {
            "matched_runtime_gradient_max_abs": gradient_max_abs,
            "matched_runtime_update_max_abs": runtime_update_max_abs,
            "matched_w_update_max_abs": w_update,
            "runtime_clip": float(s35.GRAD_CLIP),
            "w_clip": float(s35.GRAD_CLIP),
        }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S39-A0 suite size changed")
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
        raise RuntimeError("S39-A0 zero-init relation identity failed")

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
        raise RuntimeError("S39-A0 zero-init fused identity failed")

    choice_identity = 0.5 * (
        float((cb_c.argmax(-1)==tr_c.argmax(-1)).float().mean().cpu())
        + float((cb_p.argmax(-1)==tr_p.argmax(-1)).float().mean().cpu())
    )
    if choice_identity != 1.0:
        raise RuntimeError("S39-A0 selected choices changed at zero init")

    mass = max(
        float((torch.softmax(tr_c, dim=-1).sum(-1)-1).abs().max().cpu()),
        float((torch.softmax(tr_p, dim=-1).sum(-1)-1).abs().max().cpu()),
    )
    if mass > 1e-6:
        raise RuntimeError("S39-A0 probability mass changed")

    base_surface = HIRA_V1_S17_TOTAL_PARAMETER_COUNT
    treatment_surface = base_surface + treatment.added_parameter_count
    if treatment.added_parameter_count != 65_536 or base_surface != 49_152 or treatment_surface != 114_688:
        raise RuntimeError("S39-A0 parameter surface changed")
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad) != 0:
        raise RuntimeError("S39-A0 frozen runtime became trainable")
    original = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if original != 0:
        raise RuntimeError("S39-A0 original A13 became trainable")
    if any(p.requires_grad for p in runtime.hira.parameters()):
        raise RuntimeError("S39-A0 HIRACore became trainable")

    mechanics = _mechanics_court()
    projection = _projection_independence(runtime, rows)
    gradient = _gradient_probe(bundle, manifest, rows)
    matched_update = _matched_update_probe(bundle, manifest, rows)

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
        "scientific_authority": "S39_A0_GRADIENT_ISOLATED_FULL_BILINEAR_ONLY",
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
        **matched_update,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S39_A0_RECEIPT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
