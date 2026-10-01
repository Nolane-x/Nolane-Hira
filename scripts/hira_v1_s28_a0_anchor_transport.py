from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
from torch import Tensor
import torch.nn.functional as F

from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_anchor_factor_transport import (
    ANCHOR_SEPARATION_MARGIN,
    StateFactorAnchors,
    anchor_level_factor_transport_loss,
    anchor_transport_added_parameter_count,
    extract_state_factor_anchors,
)
from nmd.v1_evidence_fusion import SymmetricFullKEvidenceFusion
from nmd.v1_invariance import symmetric_js_divergence
from nmd.v1_s25_gradient_ownership import apply_s25_decoupled_gradient_update
from nmd.v1_s25_semantic_core import get_s25_relation_projection
from nmd.v1_s27_semantic_core import build_hira_v1_s27_blockwise_canonicalization_core
from nmd.v1_s28_semantic_core import (
    HIRA_V1_S28_ANCHOR_TRANSPORT_ADDED_PARAMETER_COUNT,
    HIRA_V1_S28_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S28_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S28_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S28_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s28_anchor_transport_core,
)
from hira_v1_s26_a0_factorized_relation import (
    _encode,
    _gold,
    _option_alignment,
    _primary,
    _relation,
    _supervised,
)

SCHEMA_VERSION = "hira-v1-s28-a0-anchor-factor-transport-v1"
OUTCOME = "HIRA_V1_S28_A0_ANCHOR_FACTOR_TRANSPORT_READY"
FUSION_EPSILON = 1e-6
BALANCE_EPSILON = 1e-12
OPTION_ALIGN_COEFFICIENT = 0.05
INVARIANCE_COEFFICIENT = 0.25
BINDING_COEFFICIENT = 0.10
ANCHOR_TRANSPORT_COEFFICIENT = 0.15


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
    def state_a(self) -> str:
        return (
            f"S28-A0 {self.noun} record {self.case_id}: "
            f"{self.field_a} = {self.first}; {self.field_b} = {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"Record {self.case_id} reports {self.second} under {self.field_b}. "
            f"The S28-A0 {self.noun} lists {self.first} for {self.field_a}."
        )

    @property
    def qa1(self) -> str:
        return f"For S28-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self) -> str:
        return f"Which value belongs to {self.field_a} in S28-A0 record {self.case_id}?"

    @property
    def qb1(self) -> str:
        return f"For S28-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self) -> str:
        return f"Which value belongs to {self.field_b} in S28-A0 record {self.case_id}?"

    def options(self):
        rows = [
            ("a", self.field_a, self.first),
            ("b", self.field_b, self.second),
            ("x", self.field_a, self.wrong_a),
            ("y", self.field_b, self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        texts = tuple(
            f"for this {self.noun}, {field} is {value}"
            for _kind, field, value in rows
        )
        aliases = tuple(
            f"{value} is logged as the {field} setting of this {self.noun}"
            for _kind, field, value in rows
        )
        ga = next(i for i, row in enumerate(rows) if row[0] == "a")
        gb = next(i for i, row in enumerate(rows) if row[0] == "b")
        return texts, aliases, ga, gb


def cases() -> tuple[Case, ...]:
    return (
        Case("AN11","magneto-optic trap","detuning","-12 MHz","gradient","18 G/cm","-3 MHz","5 G/cm",49101),
        Case("AN22","microbalance","drive mode","shear","sample rate","240 Hz","bending","60 Hz",49102),
        Case("AN33","optical spectrum analyzer","resolution","0.05 nm","span","80 nm","0.5 nm","20 nm",49103),
        Case("AN44","plasma probe","sweep start","-35 V","step","0.4 V","-5 V","2.0 V",49104),
        Case("AN55","cavity ringdown rig","pulse width","8 ns","repetition","12 kHz","40 ns","2 kHz",49105),
        Case("AN66","thermal imager","emissivity","0.92","frame rate","120 Hz","0.65","30 Hz",49106),
        Case("AN77","wavefront sensor","lenslet pitch","150 um","exposure","1.5 ms","300 um","8 ms",49107),
        Case("AN88","current preamplifier","gain","1e7 V/A","bandwidth","8 kHz","1e5 V/A","80 kHz",49108),
        Case("AP11","nanovoltmeter","range","10 mV","integration","20 PLC","1 V","1 PLC",49109),
        Case("AP22","vibration shaker","control mode","sine","acceleration","4 g","random","0.8 g",49110),
        Case("AP33","laser scanner","scan angle","24 deg","line rate","18 kHz","6 deg","4 kHz",49111),
        Case("AP44","pressure calibrator","medium","nitrogen","setpoint","6 bar","air","1.5 bar",49112),
        Case("AP55","thermal evaporator","boat current","42 A","deposition rate","0.8 nm/s","18 A","3.0 nm/s",49113),
        Case("AP66","optical attenuator","mode","continuous","attenuation","18 dB","step","3 dB",49114),
        Case("AP77","vector network analyzer","IF bandwidth","300 Hz","power","-12 dBm","10 kHz","2 dBm",49115),
        Case("AP88","particle sizer","refractive index","1.47","viscosity","0.89 cP","1.33","2.5 cP",49116),
    )


def _reference_extract(
    projection,
    state_tokens: Tensor,
    state_mask: Tensor,
    question_tokens: Tensor,
    question_mask: Tensor,
):
    state = F.normalize(projection(state_tokens), dim=-1)
    question = F.normalize(projection(question_tokens), dim=-1)
    state_q = torch.einsum("bqd,bsd->bqs", question, state)
    state_q = state_q.masked_fill(~question_mask[..., None], -1e4)
    score = state_q.amax(dim=1).masked_fill(~state_mask, -1e4)
    role_weights = torch.softmax(score / 0.10, dim=-1)
    role_weights = role_weights * state_mask.to(role_weights.dtype)
    role_weights = role_weights / role_weights.sum(
        -1, keepdim=True
    ).clamp_min(1e-12)
    role = F.normalize(
        torch.einsum("bs,bsd->bd", role_weights, state),
        dim=-1,
    )
    coefficient = torch.einsum("bsd,bd->bs", state, role)
    content = F.normalize(
        state - coefficient[..., None] * role[:, None, :],
        dim=-1,
    )
    value_weights = (1.0-role_weights) * state_mask.to(role_weights.dtype)
    value_weights = value_weights / value_weights.sum(
        -1, keepdim=True
    ).clamp_min(1e-12)
    value = F.normalize(
        torch.einsum("bs,bsd->bd", value_weights, content),
        dim=-1,
    )
    return role, value


def _synthetic_anchor_court() -> dict[str, float]:
    role = torch.eye(8)[:4]
    value = torch.eye(8)[4:8]
    ones = torch.ones(4, 1)
    canonical = StateFactorAnchors(
        role=role,
        value=value,
        role_weights=ones,
        value_weights=ones,
    )
    identical = anchor_level_factor_transport_loss(canonical, canonical)

    role_mismatch = canonical.role.clone()
    role_mismatch[0] = canonical.role[1]
    role_view = StateFactorAnchors(
        role=role_mismatch,
        value=canonical.value,
        role_weights=ones,
        value_weights=ones,
    )
    role_loss = anchor_level_factor_transport_loss(canonical, role_view)

    value_mismatch = canonical.value.clone()
    value_mismatch[0] = canonical.value[1]
    value_view = StateFactorAnchors(
        role=canonical.role,
        value=value_mismatch,
        role_weights=ones,
        value_weights=ones,
    )
    value_loss = anchor_level_factor_transport_loss(canonical, value_view)

    collapsed = StateFactorAnchors(
        role=torch.ones(4, 8),
        value=torch.ones(4, 8),
        role_weights=ones,
        value_weights=ones,
    )
    collapse_loss = anchor_level_factor_transport_loss(collapsed, collapsed)

    perm = torch.tensor([2,0,3,1])
    moved = anchor_level_factor_transport_loss(
        StateFactorAnchors(
            role=canonical.role[perm],
            value=canonical.value[perm],
            role_weights=ones,
            value_weights=ones,
        ),
        StateFactorAnchors(
            role=role_view.role[perm],
            value=role_view.value[perm],
            role_weights=ones,
            value_weights=ones,
        ),
    )
    return {
        "identical_total": float(identical.total),
        "identical_role_total": float(identical.role_total),
        "identical_value_total": float(identical.value_total),
        "role_only_role_total": float(role_loss.role_total),
        "role_only_value_total": float(role_loss.value_total),
        "value_only_role_total": float(value_loss.role_total),
        "value_only_value_total": float(value_loss.value_total),
        "collapsed_role_separation": float(collapse_loss.role_separation),
        "collapsed_value_separation": float(collapse_loss.value_separation),
        "collapsed_total": float(collapse_loss.total),
        "batch_permutation_total_abs": float((moved.total-role_loss.total).abs()),
        "batch_permutation_role_abs": float(
            (moved.role_total-role_loss.role_total).abs()
        ),
        "batch_permutation_value_abs": float(
            (moved.value_total-role_loss.value_total).abs()
        ),
    }


def _anchors_for_views(runtime, encoded: dict[str, Tensor]):
    projection = get_s25_relation_projection(runtime)
    canonical = extract_state_factor_anchors(
        projection=projection,
        state_tokens=encoded["state_a_tokens"].repeat_interleave(2, dim=0),
        state_mask=encoded["state_a_mask"].repeat_interleave(2, dim=0),
        question_tokens=encoded["question_canonical_tokens"],
        question_mask=encoded["question_canonical_mask"],
    )
    paraphrase = extract_state_factor_anchors(
        projection=projection,
        state_tokens=encoded["state_b_tokens"].repeat_interleave(2, dim=0),
        state_mask=encoded["state_b_mask"].repeat_interleave(2, dim=0),
        question_tokens=encoded["question_paraphrase_tokens"],
        question_mask=encoded["question_paraphrase_mask"],
    )
    return canonical, paraphrase


def _gradient_court(bundle: Path, manifest: dict, suite: tuple[Case, ...]) -> dict:
    with torch.inference_mode(False), torch.enable_grad():
        frozen = load_hira_v0_m4_bundle(bundle)
        runtime = build_hira_v1_s28_anchor_transport_core(
            frozen.runtime.encoder,
            bundle / str(manifest["t0_checkpoint"]),
            expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
            train_lora=True,
            train_primary_projection=True,
            train_relation_projection=True,
        )
        del frozen
        encoded = _encode(runtime, suite)
        raw_c = _primary(runtime, encoded, paraphrase=False)
        raw_p = _primary(runtime, encoded, paraphrase=True)
        relation_projection = get_s25_relation_projection(runtime)
        rel_c, _sig_c, _ = _relation(
            runtime, relation_projection, encoded, paraphrase=False, factorized=True
        )
        rel_p, _sig_p, _ = _relation(
            runtime, relation_projection, encoded, paraphrase=True, factorized=True
        )
        fusion = SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON)
        fused_c, _ = fusion(raw_c, rel_c.detach())
        fused_p, _ = fusion(raw_p, rel_p.detach())
        gold, other = _gold(suite, fused_c.device)

        decision = 0.5 * (
            _supervised(fused_c, gold, other)
            + _supervised(fused_p, gold, other)
        )
        primary_block = (
            decision
            + OPTION_ALIGN_COEFFICIENT * _option_alignment(encoded["option_pooled"])
            + INVARIANCE_COEFFICIENT * symmetric_js_divergence(fused_c, fused_p)
        )
        relation_loss = 0.5 * (
            F.cross_entropy(rel_c, gold) + F.cross_entropy(rel_p, gold)
        )
        canonical_anchors, paraphrase_anchors = _anchors_for_views(
            runtime,
            encoded,
        )
        anchor_loss = anchor_level_factor_transport_loss(
            canonical_anchors,
            paraphrase_anchors,
            separation_margin=ANCHOR_SEPARATION_MARGIN,
        )
        relation_block = (
            BINDING_COEFFICIENT * relation_loss
            + ANCHOR_TRANSPORT_COEFFICIENT * anchor_loss.total
        )

        shared = [
            p
            for module in iter_a13_lora_modules(runtime.encoder)
            for p in (module.lora_a, module.lora_b)
        ]
        scorer = runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S28-A0 primary scorer missing")
        d = apply_s25_decoupled_gradient_update(
            primary_block=primary_block,
            relation_block=relation_block,
            shared=shared,
            primary_private=[scorer.projection.weight],
            relation_private=[relation_projection.weight],
            epsilon=BALANCE_EPSILON,
        )
        return {
            "primary_to_relation_private_max_abs":
                d.primary_to_relation_private_max_abs,
            "relation_to_primary_private_max_abs":
                d.relation_to_primary_private_max_abs,
            "primary_private_gradient_l1": d.primary_private_l1,
            "relation_private_gradient_l1": d.relation_private_l1,
            "primary_shared_gradient_l1": d.primary_shared_l1,
            "relation_shared_gradient_l1": d.relation_shared_l1,
            "shared_projection_coefficient": d.shared.projection_coefficient,
            "real_anchor_transport_total": float(anchor_loss.total.detach().cpu()),
            "real_role_transport_total": float(
                anchor_loss.role_total.detach().cpu()
            ),
            "real_value_transport_total": float(
                anchor_loss.value_total.detach().cpu()
            ),
        }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S28-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    t0 = bundle / str(manifest["t0_checkpoint"])
    t0_sha = str(manifest["t0_checkpoint_sha256"])

    cand_bundle = load_hira_v0_m4_bundle(bundle)
    candidate = build_hira_v1_s28_anchor_transport_core(
        cand_bundle.runtime.encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_primary_projection=False,
        train_relation_projection=False,
    )
    del cand_bundle

    base_bundle = load_hira_v0_m4_bundle(bundle)
    baseline = build_hira_v1_s27_blockwise_canonicalization_core(
        base_bundle.runtime.encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_primary_projection=False,
        train_relation_projection=False,
    )
    del base_bundle

    cand_encoded = _encode(candidate, suite)
    base_encoded = _encode(baseline, suite)
    encoder_identity = all(
        torch.equal(cand_encoded[k].cpu(), base_encoded[k].cpu())
        for k in (
            "state_a_tokens",
            "state_b_tokens",
            "question_canonical_tokens",
            "question_paraphrase_tokens",
            "option_tokens",
            "option_pooled",
        )
    )
    if not encoder_identity:
        raise RuntimeError("S28-A0 shared encoder identity changed")

    cand_raw_c = _primary(candidate, cand_encoded, paraphrase=False)
    cand_raw_p = _primary(candidate, cand_encoded, paraphrase=True)
    base_raw_c = _primary(baseline, base_encoded, paraphrase=False)
    base_raw_p = _primary(baseline, base_encoded, paraphrase=True)
    primary_identity = max(
        float((cand_raw_c-base_raw_c).abs().max().cpu()),
        float((cand_raw_p-base_raw_p).abs().max().cpu()),
    )

    cand_projection = get_s25_relation_projection(candidate)
    base_projection = get_s25_relation_projection(baseline)
    cand_rel_c, cand_sig_c, _ = _relation(
        candidate, cand_projection, cand_encoded, paraphrase=False, factorized=True
    )
    cand_rel_p, cand_sig_p, _ = _relation(
        candidate, cand_projection, cand_encoded, paraphrase=True, factorized=True
    )
    base_rel_c, base_sig_c, _ = _relation(
        baseline, base_projection, base_encoded, paraphrase=False, factorized=True
    )
    base_rel_p, base_sig_p, _ = _relation(
        baseline, base_projection, base_encoded, paraphrase=True, factorized=True
    )
    relation_identity = max(
        float((cand_rel_c-base_rel_c).abs().max().cpu()),
        float((cand_rel_p-base_rel_p).abs().max().cpu()),
    )
    signature_identity = max(
        float((cand_sig_c-base_sig_c).abs().max().cpu()),
        float((cand_sig_p-base_sig_p).abs().max().cpu()),
    )

    fusion = SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    cand_fused_c, _ = fusion(cand_raw_c, cand_rel_c)
    cand_fused_p, _ = fusion(cand_raw_p, cand_rel_p)
    base_fused_c, _ = fusion(base_raw_c, base_rel_c)
    base_fused_p, _ = fusion(base_raw_p, base_rel_p)
    fused_identity = max(
        float((cand_fused_c-base_fused_c).abs().max().cpu()),
        float((cand_fused_p-base_fused_p).abs().max().cpu()),
    )
    if max(primary_identity, relation_identity, signature_identity, fused_identity) != 0.0:
        raise RuntimeError("S28-A0 production inference changed")

    lora_params = sum(
        p.numel()
        for module in iter_a13_lora_modules(candidate.encoder)
        for p in (module.lora_a, module.lora_b)
    )
    scorer = candidate.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S28-A0 primary scorer missing")
    primary_params = scorer.projection.weight.numel()
    relation_params = cand_projection.weight.numel()
    physical = lora_params + primary_params + relation_params
    trainable = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name,p in candidate.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    hira_trainable = sum(
        p.numel() for p in candidate.hira.parameters() if p.requires_grad
    )
    if (
        lora_params != HIRA_V1_S28_SHARED_LORA_PARAMETER_COUNT
        or primary_params != HIRA_V1_S28_PRIMARY_PROJECTION_PARAMETER_COUNT
        or relation_params != HIRA_V1_S28_RELATION_PROJECTION_PARAMETER_COUNT
        or physical != HIRA_V1_S28_TOTAL_PARAMETER_COUNT
    ):
        raise RuntimeError("S28-A0 capacity changed")
    if anchor_transport_added_parameter_count() != 0:
        raise RuntimeError("S28-A0 anchor objective added params")
    if HIRA_V1_S28_ANCHOR_TRANSPORT_ADDED_PARAMETER_COUNT != 0:
        raise RuntimeError("S28-A0 semantic-core anchor param contract changed")
    if trainable != 0 or original_a13 != 0 or hira_trainable != 0:
        raise RuntimeError("S28-A0 frozen runtime is not frozen")

    canonical_anchors, paraphrase_anchors = _anchors_for_views(
        candidate,
        cand_encoded,
    )
    ref_role_c, ref_value_c = _reference_extract(
        cand_projection,
        cand_encoded["state_a_tokens"].repeat_interleave(2, dim=0),
        cand_encoded["state_a_mask"].repeat_interleave(2, dim=0),
        cand_encoded["question_canonical_tokens"],
        cand_encoded["question_canonical_mask"],
    )
    ref_role_p, ref_value_p = _reference_extract(
        cand_projection,
        cand_encoded["state_b_tokens"].repeat_interleave(2, dim=0),
        cand_encoded["state_b_mask"].repeat_interleave(2, dim=0),
        cand_encoded["question_paraphrase_tokens"],
        cand_encoded["question_paraphrase_mask"],
    )
    role_reference_error = max(
        float((canonical_anchors.role-ref_role_c).abs().max().cpu()),
        float((paraphrase_anchors.role-ref_role_p).abs().max().cpu()),
    )
    value_reference_error = max(
        float((canonical_anchors.value-ref_value_c).abs().max().cpu()),
        float((paraphrase_anchors.value-ref_value_p).abs().max().cpu()),
    )
    if max(role_reference_error, value_reference_error) > 1e-7:
        raise RuntimeError("S28-A0 anchor extractor diverged from S26 formula")

    role_norm_error = max(
        float((canonical_anchors.role.norm(dim=-1)-1.0).abs().max().cpu()),
        float((paraphrase_anchors.role.norm(dim=-1)-1.0).abs().max().cpu()),
    )
    value_norm_error = max(
        float((canonical_anchors.value.norm(dim=-1)-1.0).abs().max().cpu()),
        float((paraphrase_anchors.value.norm(dim=-1)-1.0).abs().max().cpu()),
    )

    real_anchor_loss = anchor_level_factor_transport_loss(
        canonical_anchors,
        paraphrase_anchors,
        separation_margin=ANCHOR_SEPARATION_MARGIN,
    )
    synthetic = _synthetic_anchor_court()
    if max(
        synthetic["identical_total"],
        synthetic["identical_role_total"],
        synthetic["identical_value_total"],
    ) > 1e-7:
        raise RuntimeError("S28-A0 identical anchor loss changed")
    if synthetic["role_only_role_total"] <= 0.1 or synthetic["role_only_value_total"] > 1e-7:
        raise RuntimeError("S28-A0 role mismatch localization failed")
    if synthetic["value_only_value_total"] <= 0.1 or synthetic["value_only_role_total"] > 1e-7:
        raise RuntimeError("S28-A0 value mismatch localization failed")
    if synthetic["collapsed_role_separation"] < 0.19:
        raise RuntimeError("S28-A0 role collapse is not penalized")
    if synthetic["collapsed_value_separation"] < 0.19:
        raise RuntimeError("S28-A0 value collapse is not penalized")
    if max(
        synthetic["batch_permutation_total_abs"],
        synthetic["batch_permutation_role_abs"],
        synthetic["batch_permutation_value_abs"],
    ) > 1e-7:
        raise RuntimeError("S28-A0 anchor loss batch permutation changed")

    perm = torch.tensor([2,0,3,1], device=cand_fused_c.device)
    moved = {
        **cand_encoded,
        "option_tokens": cand_encoded["option_tokens"][:,perm],
        "option_mask": cand_encoded["option_mask"][:,perm],
        "option_view_mask": cand_encoded["option_view_mask"][:,perm],
        "option_pooled": cand_encoded["option_pooled"][:,perm],
    }
    moved_raw = _primary(candidate, moved, paraphrase=False)
    moved_rel, moved_sig, _ = _relation(
        candidate, cand_projection, moved, paraphrase=False, factorized=True
    )
    moved_fused, _ = fusion(moved_raw, moved_rel)
    primary_perm_error = float((moved_raw-cand_raw_c[:,perm]).abs().max().cpu())
    relation_perm_error = float((moved_rel-cand_rel_c[:,perm]).abs().max().cpu())
    signature_perm_error = float((moved_sig-cand_sig_c[:,perm]).abs().max().cpu())
    fused_flip = float(
        (moved_fused.argmax(-1) != cand_fused_c[:,perm].argmax(-1))
        .float().mean().cpu()
    )
    # State anchors do not consume options, so their objective must be exactly
    # unchanged by an option permutation.
    anchors_after_option_perm, paraphrase_after_option_perm = _anchors_for_views(
        candidate,
        moved,
    )
    moved_anchor_loss = anchor_level_factor_transport_loss(
        anchors_after_option_perm,
        paraphrase_after_option_perm,
        separation_margin=ANCHOR_SEPARATION_MARGIN,
    )
    option_independent_anchor_loss_abs = float(
        (moved_anchor_loss.total-real_anchor_loss.total).abs().cpu()
    )
    if max(primary_perm_error, relation_perm_error, signature_perm_error) > 1e-6:
        raise RuntimeError("S28-A0 inference permutation changed")
    if fused_flip != 0.0 or option_independent_anchor_loss_abs != 0.0:
        raise RuntimeError("S28-A0 option-independence court failed")

    gradient = _gradient_court(bundle, manifest, suite)
    for key in (
        "primary_private_gradient_l1",
        "relation_private_gradient_l1",
        "primary_shared_gradient_l1",
        "relation_shared_gradient_l1",
    ):
        if gradient[key] <= 0.0:
            raise RuntimeError(f"S28-A0 gradient vanished: {key}")
    if gradient["primary_to_relation_private_max_abs"] != 0.0:
        raise RuntimeError("S28-A0 primary leaked into relation-private")
    if gradient["relation_to_primary_private_max_abs"] != 0.0:
        raise RuntimeError("S28-A0 relation leaked into primary-private")

    gold, _ = _gold(suite, cand_fused_c.device)
    mass_error = max(
        float((torch.softmax(x,-1).sum(-1)-1.0).abs().max().cpu())
        for x in (cand_raw_c, cand_rel_c, cand_fused_c)
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S28_A0_ANCHOR_FACTOR_TRANSPORT_ONLY",
        "semantic_case_count": len(suite),
        "decision_count": len(suite)*4,
        "state_view_count": len(suite)*2,
        "k": 4,
        "views_per_option": 2,
        "shared_encoder_identity": encoder_identity,
        "primary_inference_identity_max_abs": primary_identity,
        "relation_inference_identity_max_abs": relation_identity,
        "signature_inference_identity_max_abs": signature_identity,
        "fused_inference_identity_max_abs": fused_identity,
        "shared_lora_parameter_count": lora_params,
        "primary_projection_parameter_count": primary_params,
        "relation_projection_parameter_count": relation_params,
        "candidate_parameter_count": physical,
        "anchor_transport_added_parameter_count":
            anchor_transport_added_parameter_count(),
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "hira_core_trainable_parameter_count": hira_trainable,
        "role_anchor_reference_max_abs": role_reference_error,
        "value_anchor_reference_max_abs": value_reference_error,
        "role_anchor_norm_error": role_norm_error,
        "value_anchor_norm_error": value_norm_error,
        "real_anchor_transport_total": float(real_anchor_loss.total.cpu()),
        "real_role_transport_total": float(real_anchor_loss.role_total.cpu()),
        "real_value_transport_total": float(real_anchor_loss.value_total.cpu()),
        "primary_option_permutation_max_abs": primary_perm_error,
        "relation_option_permutation_max_abs": relation_perm_error,
        "signature_option_permutation_max_abs": signature_perm_error,
        "fused_option_order_flip_rate": fused_flip,
        "option_independent_anchor_loss_abs": option_independent_anchor_loss_abs,
        "max_probability_mass_error": mass_error,
        "full_k": True,
        "state_once_view_count": len(suite)*2,
        "primary_canonical_accuracy": float(
            (cand_raw_c.argmax(-1)==gold).float().mean().cpu()
        ),
        "relation_canonical_accuracy": float(
            (cand_rel_c.argmax(-1)==gold).float().mean().cpu()
        ),
        "fused_canonical_accuracy": float(
            (cand_fused_c.argmax(-1)==gold).float().mean().cpu()
        ),
        **synthetic,
        **gradient,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S28_A0_ANCHOR_FACTOR_TRANSPORT_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
