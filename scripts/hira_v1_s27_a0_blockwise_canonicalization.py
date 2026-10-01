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
from nmd.v1_blockwise_factor_canonicalization import (
    blockwise_cross_view_factor_canonicalization_loss,
)
from nmd.v1_evidence_fusion import SymmetricFullKEvidenceFusion
from nmd.v1_invariance import symmetric_js_divergence
from nmd.v1_s25_gradient_ownership import apply_s25_decoupled_gradient_update
from nmd.v1_s25_semantic_core import get_s25_relation_projection
from nmd.v1_s26_semantic_core import build_hira_v1_s26_factorized_relation_core
from nmd.v1_s27_semantic_core import (
    HIRA_V1_S27_CANONICALIZATION_ADDED_PARAMETER_COUNT,
    HIRA_V1_S27_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S27_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S27_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S27_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s27_blockwise_canonicalization_core,
)
from scripts.hira_v1_s26_a0_factorized_relation import (
    _encode,
    _gold,
    _option_alignment,
    _primary,
    _relation,
    _supervised,
)

SCHEMA_VERSION = "hira-v1-s27-a0-blockwise-factor-canonicalization-v1"
OUTCOME = "HIRA_V1_S27_A0_BLOCKWISE_CANONICALIZATION_READY"
FUSION_EPSILON = 1e-6
BALANCE_EPSILON = 1e-12
OPTION_ALIGN_COEFFICIENT = 0.05
INVARIANCE_COEFFICIENT = 0.25
BINDING_COEFFICIENT = 0.10
CANONICALIZATION_COEFFICIENT = 0.15
SIGNATURE_SEPARATION_MARGIN = 0.20


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
            f"S27-A0 {self.noun} record {self.case_id}: "
            f"{self.field_a} = {self.first}; {self.field_b} = {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"Record {self.case_id} logs {self.second} under {self.field_b}. "
            f"The S27-A0 {self.noun} lists {self.first} for {self.field_a}."
        )

    @property
    def qa1(self) -> str:
        return f"For S27-A0 record {self.case_id}, identify {self.field_a}."

    @property
    def qa2(self) -> str:
        return f"Which value belongs to {self.field_a} in record {self.case_id}?"

    @property
    def qb1(self) -> str:
        return f"For S27-A0 record {self.case_id}, identify {self.field_b}."

    @property
    def qb2(self) -> str:
        return f"Which value belongs to {self.field_b} in record {self.case_id}?"

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
        Case("BW11","optical chopper","blade count","30","rotation rate","420 Hz","12","95 Hz",48101),
        Case("BW22","neutron detector","moderator","polyethylene","bias voltage","850 V","graphite","260 V",48102),
        Case("BW33","ellipsometer","angle","70 deg","wavelength","632.8 nm","45 deg","405 nm",48103),
        Case("BW44","cryogenic stage","sensor","Cernox","heater power","18 mW","silicon diode","75 mW",48104),
        Case("BW55","turbidity meter","source","860 nm","integration","24 ms","525 nm","6 ms",48105),
        Case("BW66","piezo driver","range","150 V","bandwidth","12 kHz","40 V","2 kHz",48106),
        Case("BW77","laser diode mount","control mode","constant current","limit","220 mA","constant power","70 mA",48107),
        Case("BW88","gas chromatograph","carrier","helium","oven ramp","15 C/min","nitrogen","3 C/min",48108),
        Case("BX11","spectrometer","slit","35 um","grating angle","18 deg","120 um","42 deg",48109),
        Case("BX22","beam profiler","exposure","2 ms","gain","6 dB","14 ms","24 dB",48110),
        Case("BX33","humidity chamber","humidity","72 percent","airflow","0.8 m/s","35 percent","2.4 m/s",48111),
        Case("BX44","pulse generator","amplitude","3.3 V","rise time","1.2 ns","0.8 V","8 ns",48112),
        Case("BX55","motor controller","microstep","1/32","current limit","1.4 A","1/4","0.5 A",48113),
        Case("BX66","camera trigger","polarity","rising","delay","18 us","falling","75 us",48114),
        Case("BX77","power meter","range","200 mW","averaging","16 samples","2 W","2 samples",48115),
        Case("BX88","frequency counter","gate","100 ms","coupling","AC","10 ms","DC",48116),
    )


def _synthetic_blockwise_court() -> dict[str, float]:
    role = torch.zeros(1, 4, 128)
    value = torch.zeros(1, 4, 128)
    for i in range(4):
        role[:, i, i] = 1.0
        value[:, i, 16 + i] = 1.0
    canonical = F.normalize(torch.cat((role, value), dim=-1), dim=-1)

    identical = blockwise_cross_view_factor_canonicalization_loss(
        canonical,
        canonical,
        separation_margin=SIGNATURE_SEPARATION_MARGIN,
    )

    role_mismatch = canonical.clone()
    role_mismatch[:, 0, :128] = canonical[:, 1, :128]
    role_loss = blockwise_cross_view_factor_canonicalization_loss(
        canonical,
        role_mismatch,
        separation_margin=SIGNATURE_SEPARATION_MARGIN,
    )

    value_mismatch = canonical.clone()
    value_mismatch[:, 0, 128:] = canonical[:, 1, 128:]
    value_loss = blockwise_cross_view_factor_canonicalization_loss(
        canonical,
        value_mismatch,
        separation_margin=SIGNATURE_SEPARATION_MARGIN,
    )

    both_mismatch = canonical.clone()
    both_mismatch[:, 0, :128] = canonical[:, 1, :128]
    both_mismatch[:, 0, 128:] = canonical[:, 2, 128:]
    both_loss = blockwise_cross_view_factor_canonicalization_loss(
        canonical,
        both_mismatch,
        separation_margin=SIGNATURE_SEPARATION_MARGIN,
    )

    perm = torch.tensor([2, 0, 3, 1])
    moved = blockwise_cross_view_factor_canonicalization_loss(
        canonical[:, perm],
        role_mismatch[:, perm],
        separation_margin=SIGNATURE_SEPARATION_MARGIN,
    )

    return {
        "identical_total": float(identical.total),
        "identical_role_total": float(identical.role_total),
        "identical_value_total": float(identical.value_total),
        "role_only_role_total": float(role_loss.role_total),
        "role_only_value_total": float(role_loss.value_total),
        "value_only_role_total": float(value_loss.role_total),
        "value_only_value_total": float(value_loss.value_total),
        "both_role_total": float(both_loss.role_total),
        "both_value_total": float(both_loss.value_total),
        "option_permutation_total_abs": float((moved.total-role_loss.total).abs()),
        "option_permutation_role_abs": float(
            (moved.role_total-role_loss.role_total).abs()
        ),
        "option_permutation_value_abs": float(
            (moved.value_total-role_loss.value_total).abs()
        ),
    }


def _gradient_court(bundle: Path, manifest: dict, suite: tuple[Case, ...]) -> dict:
    with torch.inference_mode(False), torch.enable_grad():
        frozen = load_hira_v0_m4_bundle(bundle)
        runtime = build_hira_v1_s27_blockwise_canonicalization_core(
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
        rel_c, sig_c, _ = _relation(
            runtime,
            relation_projection,
            encoded,
            paraphrase=False,
            factorized=True,
        )
        rel_p, sig_p, _ = _relation(
            runtime,
            relation_projection,
            encoded,
            paraphrase=True,
            factorized=True,
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
        blockwise = blockwise_cross_view_factor_canonicalization_loss(
            sig_c,
            sig_p,
            separation_margin=SIGNATURE_SEPARATION_MARGIN,
        )
        relation_block = (
            BINDING_COEFFICIENT * relation_loss
            + CANONICALIZATION_COEFFICIENT * blockwise.total
        )

        shared = [
            p
            for module in iter_a13_lora_modules(runtime.encoder)
            for p in (module.lora_a, module.lora_b)
        ]
        scorer = runtime.projection_triadic_scorer
        if scorer is None:
            raise RuntimeError("S27-A0 primary scorer missing")
        primary_private = [scorer.projection.weight]
        relation_private = [relation_projection.weight]
        diagnostics = apply_s25_decoupled_gradient_update(
            primary_block=primary_block,
            relation_block=relation_block,
            shared=shared,
            primary_private=primary_private,
            relation_private=relation_private,
            epsilon=BALANCE_EPSILON,
        )
        return {
            "primary_to_relation_private_max_abs":
                diagnostics.primary_to_relation_private_max_abs,
            "relation_to_primary_private_max_abs":
                diagnostics.relation_to_primary_private_max_abs,
            "primary_private_gradient_l1": diagnostics.primary_private_l1,
            "relation_private_gradient_l1": diagnostics.relation_private_l1,
            "primary_shared_gradient_l1": diagnostics.primary_shared_l1,
            "relation_shared_gradient_l1": diagnostics.relation_shared_l1,
            "shared_projection_coefficient":
                diagnostics.shared.projection_coefficient,
            "shared_combined_norm": diagnostics.shared.combined_norm,
            "real_blockwise_total": float(blockwise.total.detach().cpu()),
            "real_role_total": float(blockwise.role_total.detach().cpu()),
            "real_value_total": float(blockwise.value_total.detach().cpu()),
        }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S27-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    t0 = bundle / str(manifest["t0_checkpoint"])
    t0_sha = str(manifest["t0_checkpoint_sha256"])

    cand_bundle = load_hira_v0_m4_bundle(bundle)
    candidate = build_hira_v1_s27_blockwise_canonicalization_core(
        cand_bundle.runtime.encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_primary_projection=False,
        train_relation_projection=False,
    )
    del cand_bundle

    base_bundle = load_hira_v0_m4_bundle(bundle)
    baseline = build_hira_v1_s26_factorized_relation_core(
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
        raise RuntimeError("S27-A0 shared encoder identity changed")

    cand_raw_c = _primary(candidate, cand_encoded, paraphrase=False)
    cand_raw_p = _primary(candidate, cand_encoded, paraphrase=True)
    base_raw_c = _primary(baseline, base_encoded, paraphrase=False)
    base_raw_p = _primary(baseline, base_encoded, paraphrase=True)

    primary_identity_max_abs = max(
        float((cand_raw_c-base_raw_c).abs().max().cpu()),
        float((cand_raw_p-base_raw_p).abs().max().cpu()),
    )
    if primary_identity_max_abs != 0.0:
        raise RuntimeError("S27-A0 primary inference changed")

    cand_projection = get_s25_relation_projection(candidate)
    base_projection = get_s25_relation_projection(baseline)
    cand_rel_c, cand_sig_c, _ = _relation(
        candidate,
        cand_projection,
        cand_encoded,
        paraphrase=False,
        factorized=True,
    )
    cand_rel_p, cand_sig_p, _ = _relation(
        candidate,
        cand_projection,
        cand_encoded,
        paraphrase=True,
        factorized=True,
    )
    base_rel_c, base_sig_c, _ = _relation(
        baseline,
        base_projection,
        base_encoded,
        paraphrase=False,
        factorized=True,
    )
    base_rel_p, base_sig_p, _ = _relation(
        baseline,
        base_projection,
        base_encoded,
        paraphrase=True,
        factorized=True,
    )
    relation_identity_max_abs = max(
        float((cand_rel_c-base_rel_c).abs().max().cpu()),
        float((cand_rel_p-base_rel_p).abs().max().cpu()),
    )
    signature_identity_max_abs = max(
        float((cand_sig_c-base_sig_c).abs().max().cpu()),
        float((cand_sig_p-base_sig_p).abs().max().cpu()),
    )
    if relation_identity_max_abs != 0.0 or signature_identity_max_abs != 0.0:
        raise RuntimeError("S27-A0 relation/signature inference changed")

    fusion = SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    cand_fused_c, _ = fusion(cand_raw_c, cand_rel_c)
    cand_fused_p, _ = fusion(cand_raw_p, cand_rel_p)
    base_fused_c, _ = fusion(base_raw_c, base_rel_c)
    base_fused_p, _ = fusion(base_raw_p, base_rel_p)
    fused_identity_max_abs = max(
        float((cand_fused_c-base_fused_c).abs().max().cpu()),
        float((cand_fused_p-base_fused_p).abs().max().cpu()),
    )
    if fused_identity_max_abs != 0.0:
        raise RuntimeError("S27-A0 fused inference changed")

    lora_params = sum(
        p.numel()
        for module in iter_a13_lora_modules(candidate.encoder)
        for p in (module.lora_a, module.lora_b)
    )
    scorer = candidate.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S27-A0 primary scorer missing")
    primary_params = scorer.projection.weight.numel()
    relation_params = cand_projection.weight.numel()
    physical = lora_params + primary_params + relation_params
    trainable = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in candidate.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    hira_trainable = sum(
        p.numel() for p in candidate.hira.parameters() if p.requires_grad
    )
    if lora_params != HIRA_V1_S27_SHARED_LORA_PARAMETER_COUNT:
        raise RuntimeError("S27-A0 LoRA capacity changed")
    if primary_params != HIRA_V1_S27_PRIMARY_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S27-A0 primary projection capacity changed")
    if relation_params != HIRA_V1_S27_RELATION_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S27-A0 relation projection capacity changed")
    if physical != HIRA_V1_S27_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S27-A0 total capacity changed")
    if HIRA_V1_S27_CANONICALIZATION_ADDED_PARAMETER_COUNT != 0:
        raise RuntimeError("S27-A0 canonicalization added params")
    if trainable != 0 or original_a13 != 0 or hira_trainable != 0:
        raise RuntimeError("S27-A0 frozen candidate is not frozen")

    blockwise_real = blockwise_cross_view_factor_canonicalization_loss(
        cand_sig_c,
        cand_sig_p,
        separation_margin=SIGNATURE_SEPARATION_MARGIN,
    )
    if not bool(torch.isfinite(blockwise_real.total)):
        raise RuntimeError("S27-A0 real blockwise loss non-finite")

    synthetic = _synthetic_blockwise_court()
    if max(
        synthetic["identical_total"],
        synthetic["identical_role_total"],
        synthetic["identical_value_total"],
    ) > 1e-7:
        raise RuntimeError("S27-A0 identical-view blockwise loss changed")
    if synthetic["role_only_role_total"] <= 0.1:
        raise RuntimeError("S27-A0 role-only mismatch did not activate role block")
    if synthetic["role_only_value_total"] > 1e-7:
        raise RuntimeError("S27-A0 role-only mismatch leaked into value block")
    if synthetic["value_only_value_total"] <= 0.1:
        raise RuntimeError("S27-A0 value-only mismatch did not activate value block")
    if synthetic["value_only_role_total"] > 1e-7:
        raise RuntimeError("S27-A0 value-only mismatch leaked into role block")
    if min(
        synthetic["both_role_total"],
        synthetic["both_value_total"],
    ) <= 0.1:
        raise RuntimeError("S27-A0 both-factor mismatch did not activate both blocks")
    if max(
        synthetic["option_permutation_total_abs"],
        synthetic["option_permutation_role_abs"],
        synthetic["option_permutation_value_abs"],
    ) > 1e-7:
        raise RuntimeError("S27-A0 blockwise loss option permutation changed")

    perm = torch.tensor([2, 0, 3, 1], device=cand_fused_c.device)
    moved = {
        **cand_encoded,
        "option_tokens": cand_encoded["option_tokens"][:, perm],
        "option_mask": cand_encoded["option_mask"][:, perm],
        "option_view_mask": cand_encoded["option_view_mask"][:, perm],
        "option_pooled": cand_encoded["option_pooled"][:, perm],
    }
    moved_raw = _primary(candidate, moved, paraphrase=False)
    moved_rel, moved_sig, _ = _relation(
        candidate,
        cand_projection,
        moved,
        paraphrase=False,
        factorized=True,
    )
    moved_fused, _ = fusion(moved_raw, moved_rel)
    primary_perm_error = float(
        (moved_raw-cand_raw_c[:, perm]).abs().max().cpu()
    )
    relation_perm_error = float(
        (moved_rel-cand_rel_c[:, perm]).abs().max().cpu()
    )
    signature_perm_error = float(
        (moved_sig-cand_sig_c[:, perm]).abs().max().cpu()
    )
    fused_choice_flip = float(
        (
            moved_fused.argmax(-1)
            != cand_fused_c[:, perm].argmax(-1)
        ).float().mean().cpu()
    )
    if max(primary_perm_error, relation_perm_error, signature_perm_error) > 1e-6:
        raise RuntimeError("S27-A0 inference option permutation changed")
    if fused_choice_flip != 0.0:
        raise RuntimeError("S27-A0 fused option-order choice changed")

    gradient = _gradient_court(bundle, manifest, suite)
    for key in (
        "primary_private_gradient_l1",
        "relation_private_gradient_l1",
        "primary_shared_gradient_l1",
        "relation_shared_gradient_l1",
    ):
        if gradient[key] <= 0.0:
            raise RuntimeError(f"S27-A0 gradient vanished: {key}")
    if gradient["primary_to_relation_private_max_abs"] != 0.0:
        raise RuntimeError("S27-A0 primary leaked into relation-private surface")
    if gradient["relation_to_primary_private_max_abs"] != 0.0:
        raise RuntimeError("S27-A0 relation leaked into primary-private surface")

    gold, _ = _gold(suite, cand_fused_c.device)
    max_mass_error = max(
        float(
            (torch.softmax(x, -1).sum(-1)-1.0)
            .abs().max().cpu()
        )
        for x in (cand_raw_c, cand_rel_c, cand_fused_c)
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S27_A0_BLOCKWISE_CANONICALIZATION_ONLY",
        "semantic_case_count": len(suite),
        "decision_count": len(suite) * 4,
        "state_view_count": len(suite) * 2,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "shared_encoder_identity": encoder_identity,
        "primary_inference_identity_max_abs": primary_identity_max_abs,
        "relation_inference_identity_max_abs": relation_identity_max_abs,
        "signature_inference_identity_max_abs": signature_identity_max_abs,
        "fused_inference_identity_max_abs": fused_identity_max_abs,
        "shared_lora_parameter_count": lora_params,
        "primary_projection_parameter_count": primary_params,
        "relation_projection_parameter_count": relation_params,
        "candidate_parameter_count": physical,
        "canonicalization_added_parameter_count":
            HIRA_V1_S27_CANONICALIZATION_ADDED_PARAMETER_COUNT,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "hira_core_trainable_parameter_count": hira_trainable,
        "signature_dimension": int(cand_sig_c.shape[-1]),
        "role_block_dimension": 128,
        "value_block_dimension": 128,
        "real_blockwise_total": float(blockwise_real.total.cpu()),
        "real_role_total": float(blockwise_real.role_total.cpu()),
        "real_value_total": float(blockwise_real.value_total.cpu()),
        "primary_option_permutation_max_abs": primary_perm_error,
        "relation_option_permutation_max_abs": relation_perm_error,
        "signature_option_permutation_max_abs": signature_perm_error,
        "fused_option_order_flip_rate": fused_choice_flip,
        "max_probability_mass_error": max_mass_error,
        "full_k": True,
        "state_once_view_count": len(suite) * 2,
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
        "HIRA_V1_S27_A0_BLOCKWISE_CANONICALIZATION_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
