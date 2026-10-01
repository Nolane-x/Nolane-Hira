from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_evidence_fusion import SymmetricFullKEvidenceFusion
from nmd.v1_factorized_relation_signature import (
    FactorizedRoleValueRelationCanonicalizer,
)
from nmd.v1_invariance import symmetric_js_divergence
from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    cross_view_relation_signature_loss,
    relation_signature_same_option_cosine,
)
from nmd.v1_s25_gradient_ownership import apply_s25_decoupled_gradient_update
from nmd.v1_s25_semantic_core import (
    build_hira_v1_s25_decoupled_projection_core,
    get_s25_relation_projection,
)
from nmd.v1_s26_semantic_core import (
    HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S26_RELATION_ADDED_PARAMETER_COUNT,
    HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S26_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s26_factorized_relation_core,
)

SCHEMA_VERSION = "hira-v1-s26-a0-factorized-role-value-relation-v1"
OUTCOME = "HIRA_V1_S26_A0_FACTORIZED_RELATION_READY"
ROLE_TEMPERATURE = 0.10
PAIR_TEMPERATURE = 0.10
CONTRASTIVE_TEMPERATURE = 0.10
FUSION_EPSILON = 1e-6
BALANCE_EPSILON = 1e-12
SWAP_COEFFICIENT = 0.25
SWAP_MARGIN = 0.20
OPTION_ALIGN_COEFFICIENT = 0.05
OPTION_ALIGN_TEMPERATURE = 0.10
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
            f"S26-A0 {self.noun} record {self.case_id}: "
            f"{self.field_a} = {self.first}; {self.field_b} = {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"Record {self.case_id} gives {self.second} for {self.field_b}. "
            f"Its {self.field_a} entry is {self.first} on the S26-A0 {self.noun}."
        )

    @property
    def qa1(self) -> str:
        return f"On {self.case_id}, what is the {self.field_a}?"

    @property
    def qa2(self) -> str:
        return f"Read {self.case_id} and return the value assigned to {self.field_a}."

    @property
    def qb1(self) -> str:
        return f"On {self.case_id}, what is the {self.field_b}?"

    @property
    def qb2(self) -> str:
        return f"Read {self.case_id} and return the value assigned to {self.field_b}."

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
            f"{value} is recorded as the {field} value of this {self.noun}"
            for _kind, field, value in rows
        )
        ga = next(i for i, row in enumerate(rows) if row[0] == "a")
        gb = next(i for i, row in enumerate(rows) if row[0] == "b")
        return texts, aliases, ga, gb


def cases() -> tuple[Case, ...]:
    return (
        Case("FR11","laser vibrometer","decoder","heterodyne","bandwidth","80 kHz","homodyne","12 kHz",46101),
        Case("FR22","atomic force stage","cantilever","silicon","scan rate","2.5 Hz","diamond","14 Hz",46102),
        Case("FR33","fluorescence reader","excitation","470 nm","gain","42 dB","620 nm","18 dB",46103),
        Case("FR44","x-ray detector","sensor","CdTe","bias","450 V","silicon","90 V",46104),
        Case("FR55","microfluidic mixer","channel","serpentine","flow","18 uL/min","straight","90 uL/min",46105),
        Case("FR66","polarimetry head","retarder","quarter-wave","rotation","33 deg","half-wave","75 deg",46106),
        Case("FR77","acoustic camera","array","spiral","sample rate","192 kHz","linear","48 kHz",46107),
        Case("FR88","thermal chamber","coolant","glycol","setpoint","-25 C","water","15 C",46108),
        Case("FS11","mass analyzer","mode","reflectron","flight tube","1.8 m","linear","0.6 m",46109),
        Case("FS22","spectral imager","grating","600 l/mm","slit","20 um","1200 l/mm","90 um",46110),
        Case("FS33","magnetometer","cell","rubidium","pump power","3 mW","cesium","15 mW",46111),
        Case("FS44","plasma source","gas","argon","rf power","120 W","helium","35 W",46112),
        Case("FS55","optical trap","objective","1.2 NA","laser","1064 nm","0.65 NA","532 nm",46113),
        Case("FS66","ultrasound rig","probe","5 MHz","focus","35 mm","1 MHz","90 mm",46114),
        Case("FS77","electron column","aperture","40 um","lens current","1.6 A","150 um","0.4 A",46115),
        Case("FS88","raman stage","integration","8 s","accumulations","12","1 s","2",46116),
    )


def _content_mask(batch) -> Tensor:
    mask = batch.attention_mask.bool()
    if batch.special_token_mask is None:
        return mask
    return mask & ~batch.special_token_mask.bool()


def _encode(runtime, suite: tuple[Case, ...]) -> dict[str, Tensor]:
    runtime.eval()
    encoder = runtime.encoder
    n = len(suite)
    states = encoder.encode_texts(
        [c.state_a for c in suite] + [c.state_b for c in suite]
    )
    canonical_q = [q for c in suite for q in (c.qa1, c.qb1)]
    paraphrase_q = [q for c in suite for q in (c.qa2, c.qb2)]
    questions = encoder.encode_texts(canonical_q + paraphrase_q)

    option_texts = []
    for c in suite:
        texts, aliases, _ga, _gb = c.options()
        for text, alias in zip(texts, aliases):
            option_texts.extend((text, alias))
    options = encoder.encode_texts(option_texts)

    state_mask = _content_mask(states)
    question_mask = _content_mask(questions)
    option_mask = _content_mask(options)

    return {
        "state_a_tokens": states.token_embeddings[:n],
        "state_a_mask": state_mask[:n],
        "state_b_tokens": states.token_embeddings[n:],
        "state_b_mask": state_mask[n:],
        "question_canonical_tokens": questions.token_embeddings[: 2 * n],
        "question_canonical_mask": question_mask[: 2 * n],
        "question_paraphrase_tokens": questions.token_embeddings[2 * n :],
        "question_paraphrase_mask": question_mask[2 * n :],
        "option_tokens": options.token_embeddings.reshape(
            n, 4, 2, options.token_embeddings.shape[1], -1
        ),
        "option_mask": option_mask.reshape(
            n, 4, 2, options.attention_mask.shape[1]
        ),
        "option_view_mask": torch.ones(
            n, 4, 2, dtype=torch.bool, device=options.token_embeddings.device
        ),
        "option_pooled": options.pooled_embeddings.reshape(n, 4, 2, -1),
    }


def _primary(runtime, encoded: dict[str, Tensor], *, paraphrase: bool) -> Tensor:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S26-A0 primary scorer missing")
    prefix = "paraphrase" if paraphrase else "canonical"
    state_prefix = "b" if paraphrase else "a"
    return scorer(
        state_tokens=encoded[f"state_{state_prefix}_tokens"].repeat_interleave(2, dim=0),
        state_mask=encoded[f"state_{state_prefix}_mask"].repeat_interleave(2, dim=0),
        question_tokens=encoded[f"question_{prefix}_tokens"],
        question_mask=encoded[f"question_{prefix}_mask"],
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2, dim=0),
    )


def _relation(
    runtime,
    projection: nn.Linear,
    encoded: dict[str, Tensor],
    *,
    paraphrase: bool,
    factorized: bool,
):
    prefix = "paraphrase" if paraphrase else "canonical"
    state_prefix = "b" if paraphrase else "a"
    cls = (
        FactorizedRoleValueRelationCanonicalizer
        if factorized
        else CrossViewRelationCanonicalizer
    )
    op = cls(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )
    return op(
        projection=projection,
        state_tokens=encoded[f"state_{state_prefix}_tokens"].repeat_interleave(2, dim=0),
        state_mask=encoded[f"state_{state_prefix}_mask"].repeat_interleave(2, dim=0),
        question_tokens=encoded[f"question_{prefix}_tokens"],
        question_mask=encoded[f"question_{prefix}_mask"],
        option_view_tokens=encoded["option_tokens"].repeat_interleave(2, dim=0),
        option_view_token_mask=encoded["option_mask"].repeat_interleave(2, dim=0),
        option_view_mask=encoded["option_view_mask"].repeat_interleave(2, dim=0),
    )


def _gold(suite: tuple[Case, ...], device) -> tuple[Tensor, Tensor]:
    gold = []
    other = []
    for c in suite:
        _texts, _aliases, ga, gb = c.options()
        gold.extend((ga, gb))
        other.extend((gb, ga))
    return (
        torch.tensor(gold, dtype=torch.long, device=device),
        torch.tensor(other, dtype=torch.long, device=device),
    )


def _supervised(logits: Tensor, gold: Tensor, other: Tensor) -> Tensor:
    ce = F.cross_entropy(logits, gold)
    idx = torch.arange(logits.shape[0], device=logits.device)
    swap = F.relu(
        logits.new_tensor(SWAP_MARGIN)
        - (logits[idx, gold] - logits[idx, other])
    ).mean()
    return ce + SWAP_COEFFICIENT * swap


def _option_alignment(option_pooled: Tensor) -> Tensor:
    left = F.normalize(option_pooled[:, :, 0], dim=-1)
    right = F.normalize(option_pooled[:, :, 1], dim=-1)
    lr = torch.einsum("nkd,njd->nkj", left, right) / OPTION_ALIGN_TEMPERATURE
    rl = torch.einsum("nkd,njd->nkj", right, left) / OPTION_ALIGN_TEMPERATURE
    n, k, _ = lr.shape
    labels = torch.arange(k, device=lr.device).expand(n, k)
    return 0.5 * (
        F.cross_entropy(lr.reshape(n * k, k), labels.reshape(n * k))
        + F.cross_entropy(rl.reshape(n * k, k), labels.reshape(n * k))
    )


def _signature_margin(canonical: Tensor, paraphrase: Tensor) -> Tensor:
    c = F.normalize(canonical, dim=-1)
    p = F.normalize(paraphrase, dim=-1)
    cross = torch.einsum("bkd,bjd->bkj", c, p)
    same = cross.diagonal(dim1=-2, dim2=-1)
    eye = torch.eye(c.shape[1], dtype=torch.bool, device=c.device)[None]
    wrong = cross.masked_fill(eye, float("-inf")).amax(-1)
    return same - wrong


def _synthetic_quadrant_court() -> dict[str, float]:
    op = FactorizedRoleValueRelationCanonicalizer()
    projection = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(4))
    state = torch.tensor([[[1.,0.,0.,0.],[0.,1.,0.,0.]]])
    question = torch.tensor([[[1.,0.,0.,0.]]])
    options = torch.tensor(
        [[[
            [[1.,0.,0.,0.],[0.,1.,0.,0.]]
        ],[
            [[1.,0.,0.,0.],[0.,0.,0.,1.]]
        ],[
            [[0.,0.,1.,0.],[0.,1.,0.,0.]]
        ],[
            [[0.,0.,1.,0.],[0.,0.,0.,1.]]
        ]]]
    )
    logits, signatures, _ = op(
        projection=projection,
        state_tokens=state,
        state_mask=torch.ones(1,2,dtype=torch.bool),
        question_tokens=question,
        question_mask=torch.ones(1,1,dtype=torch.bool),
        option_view_tokens=options,
        option_view_token_mask=torch.ones(1,4,1,2,dtype=torch.bool),
        option_view_mask=torch.ones(1,4,1,dtype=torch.bool),
    )
    return {
        "correct_vs_same_role_wrong_value_margin": float((logits[0,0]-logits[0,1]).cpu()),
        "correct_vs_wrong_role_same_value_margin": float((logits[0,0]-logits[0,2]).cpu()),
        "correct_vs_wrong_role_wrong_value_margin": float((logits[0,0]-logits[0,3]).cpu()),
        "synthetic_projection_dimension": int(projection.out_features),
        "synthetic_signature_dimension": int(signatures.shape[-1]),
    }


def _gradient_court(bundle: Path, manifest: dict, suite: tuple[Case, ...]) -> dict:
    with torch.inference_mode(False), torch.enable_grad():
        fresh = load_hira_v0_m4_bundle(bundle)
        runtime = build_hira_v1_s26_factorized_relation_core(
            fresh.runtime.encoder,
            bundle / str(manifest["t0_checkpoint"]),
            expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
            train_lora=True,
            train_primary_projection=True,
            train_relation_projection=True,
        )
        del fresh
        encoded = _encode(runtime, suite)
        raw_c = _primary(runtime, encoded, paraphrase=False)
        raw_p = _primary(runtime, encoded, paraphrase=True)
        relation_projection = get_s25_relation_projection(runtime)
        rel_c, sig_c, _ = _relation(
            runtime, relation_projection, encoded, paraphrase=False, factorized=True
        )
        rel_p, sig_p, _ = _relation(
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
        canonicalization, _align, _sep = cross_view_relation_signature_loss(
            sig_c,
            sig_p,
            separation_margin=SIGNATURE_SEPARATION_MARGIN,
        )
        relation_block = (
            BINDING_COEFFICIENT * relation_loss
            + CANONICALIZATION_COEFFICIENT * canonicalization
        )

        shared = [
            p
            for module in iter_a13_lora_modules(runtime.encoder)
            for p in (module.lora_a, module.lora_b)
        ]
        primary_private = [runtime.projection_triadic_scorer.projection.weight]
        relation_private = [relation_projection.weight]
        d = apply_s25_decoupled_gradient_update(
            primary_block=primary_block,
            relation_block=relation_block,
            shared=shared,
            primary_private=primary_private,
            relation_private=relation_private,
            epsilon=BALANCE_EPSILON,
        )
        return {
            "primary_to_relation_private_max_abs": d.primary_to_relation_private_max_abs,
            "relation_to_primary_private_max_abs": d.relation_to_primary_private_max_abs,
            "primary_private_gradient_l1": d.primary_private_l1,
            "relation_private_gradient_l1": d.relation_private_l1,
            "primary_shared_gradient_l1": d.primary_shared_l1,
            "relation_shared_gradient_l1": d.relation_shared_l1,
            "shared_primary_norm": d.shared.primary_norm,
            "shared_relation_norm": d.shared.relation_norm,
            "shared_projection_coefficient": d.shared.projection_coefficient,
            "shared_combined_norm": d.shared.combined_norm,
        }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S26-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    t0 = bundle / str(manifest["t0_checkpoint"])
    t0_sha = str(manifest["t0_checkpoint_sha256"])

    candidate_bundle = load_hira_v0_m4_bundle(bundle)
    candidate = build_hira_v1_s26_factorized_relation_core(
        candidate_bundle.runtime.encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_primary_projection=False,
        train_relation_projection=False,
    )
    del candidate_bundle

    baseline_bundle = load_hira_v0_m4_bundle(bundle)
    baseline = build_hira_v1_s25_decoupled_projection_core(
        baseline_bundle.runtime.encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_primary_projection=False,
        train_relation_projection=False,
    )
    del baseline_bundle

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
        raise RuntimeError("S26-A0 encoder identity changed")

    cand_raw_c = _primary(candidate, cand_encoded, paraphrase=False)
    cand_raw_p = _primary(candidate, cand_encoded, paraphrase=True)
    base_raw_c = _primary(baseline, base_encoded, paraphrase=False)
    base_raw_p = _primary(baseline, base_encoded, paraphrase=True)
    primary_identity = bool(
        torch.equal(cand_raw_c, base_raw_c)
        and torch.equal(cand_raw_p, base_raw_p)
    )
    if not primary_identity:
        raise RuntimeError("S26-A0 primary path changed")

    relation_projection = get_s25_relation_projection(candidate)
    base_projection = get_s25_relation_projection(baseline)
    rel_c, sig_c, diag_c = _relation(
        candidate, relation_projection, cand_encoded, paraphrase=False, factorized=True
    )
    rel_p, sig_p, diag_p = _relation(
        candidate, relation_projection, cand_encoded, paraphrase=True, factorized=True
    )
    old_rel_c, _old_sig_c, _ = _relation(
        baseline, base_projection, base_encoded, paraphrase=False, factorized=False
    )
    old_rel_p, _old_sig_p, _ = _relation(
        baseline, base_projection, base_encoded, paraphrase=True, factorized=False
    )
    intervention_max_abs = max(
        float((rel_c-old_rel_c).abs().max().cpu()),
        float((rel_p-old_rel_p).abs().max().cpu()),
    )
    if intervention_max_abs <= 1e-7:
        raise RuntimeError("S26-A0 relation intervention is numerically inactive")

    lora_params = sum(
        p.numel()
        for module in iter_a13_lora_modules(candidate.encoder)
        for p in (module.lora_a, module.lora_b)
    )
    primary_projection = candidate.projection_triadic_scorer.projection
    primary_params = primary_projection.weight.numel()
    relation_params = relation_projection.weight.numel()
    physical = lora_params + primary_params + relation_params
    trainable = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in candidate.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    hira_trainable = sum(p.numel() for p in candidate.hira.parameters() if p.requires_grad)
    relation_operator = FactorizedRoleValueRelationCanonicalizer()
    if lora_params != HIRA_V1_S26_SHARED_LORA_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 LoRA capacity changed")
    if primary_params != HIRA_V1_S26_PRIMARY_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 primary projection capacity changed")
    if relation_params != HIRA_V1_S26_RELATION_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 relation projection capacity changed")
    if physical != HIRA_V1_S26_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 total capacity changed")
    if relation_operator.factorization_added_parameter_count != HIRA_V1_S26_RELATION_ADDED_PARAMETER_COUNT:
        raise RuntimeError("S26-A0 relation operator added learned state")
    if trainable != 0 or original_a13 != 0 or hira_trainable != 0:
        raise RuntimeError("S26-A0 frozen runtime is not frozen")
    if primary_projection.weight.data_ptr() == relation_projection.weight.data_ptr():
        raise RuntimeError("S26-A0 private projections share storage")

    fusion = SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    fused_c, _ = fusion(cand_raw_c, rel_c)
    fused_p, _ = fusion(cand_raw_p, rel_p)

    perm = torch.tensor([2,0,3,1], device=fused_c.device)
    moved = {
        **cand_encoded,
        "option_tokens": cand_encoded["option_tokens"][:, perm],
        "option_mask": cand_encoded["option_mask"][:, perm],
        "option_view_mask": cand_encoded["option_view_mask"][:, perm],
        "option_pooled": cand_encoded["option_pooled"][:, perm],
    }
    moved_raw = _primary(candidate, moved, paraphrase=False)
    moved_rel, moved_sig, _ = _relation(
        candidate, relation_projection, moved, paraphrase=False, factorized=True
    )
    moved_fused, _ = fusion(moved_raw, moved_rel)
    primary_perm_error = float((moved_raw-cand_raw_c[:,perm]).abs().max().cpu())
    relation_perm_error = float((moved_rel-rel_c[:,perm]).abs().max().cpu())
    signature_perm_error = float((moved_sig-sig_c[:,perm]).abs().max().cpu())
    fused_logit_perm_error = float((moved_fused-fused_c[:,perm]).abs().max().cpu())
    fused_choice_flip = float(
        (moved_fused.argmax(-1) != fused_c[:,perm].argmax(-1))
        .float().mean().cpu()
    )
    if max(primary_perm_error, relation_perm_error, signature_perm_error) > 1e-6:
        raise RuntimeError("S26-A0 expert/signature option permutation changed")
    if fused_choice_flip != 0.0:
        raise RuntimeError("S26-A0 fused option-order choice changed")

    gold, _other = _gold(suite, fused_c.device)
    max_mass_error = max(
        float((torch.softmax(x,-1).sum(-1)-1.0).abs().max().cpu())
        for x in (cand_raw_c,cand_raw_p,rel_c,rel_p,fused_c,fused_p)
    )
    if max_mass_error > 1e-6:
        raise RuntimeError("S26-A0 probability mass changed")

    same_cos = relation_signature_same_option_cosine(sig_c, sig_p)
    sig_margin = _signature_margin(sig_c, sig_p)
    synthetic = _synthetic_quadrant_court()
    if (
        synthetic["correct_vs_same_role_wrong_value_margin"] <= 0.0
        or synthetic["correct_vs_wrong_role_same_value_margin"] <= 0.0
        or synthetic["correct_vs_wrong_role_wrong_value_margin"] <= 0.0
    ):
        raise RuntimeError("S26-A0 synthetic hard-negative quadrant court failed")
    if synthetic["synthetic_signature_dimension"] != (
        2 * synthetic["synthetic_projection_dimension"]
    ):
        raise RuntimeError("S26-A0 synthetic factorized signature dimension changed")
    if int(sig_c.shape[-1]) != 256 or int(sig_p.shape[-1]) != 256:
        raise RuntimeError("S26-A0 real factorized signature dimension changed")

    gradient = _gradient_court(bundle, manifest, suite)
    for key in (
        "primary_private_gradient_l1",
        "relation_private_gradient_l1",
        "primary_shared_gradient_l1",
        "relation_shared_gradient_l1",
    ):
        if gradient[key] <= 0.0:
            raise RuntimeError(f"S26-A0 gradient vanished: {key}")
    if gradient["primary_to_relation_private_max_abs"] != 0.0:
        raise RuntimeError("S26-A0 primary leaked into relation-private surface")
    if gradient["relation_to_primary_private_max_abs"] != 0.0:
        raise RuntimeError("S26-A0 relation leaked into primary-private surface")
    if gradient["shared_projection_coefficient"] != 0.0:
        raise RuntimeError("S26-A0 neutral bisector unexpectedly projected")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S26_A0_FACTORIZED_ROLE_VALUE_ONLY",
        "semantic_case_count": len(suite),
        "decision_count": len(suite) * 4,
        "state_view_count": len(suite) * 2,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "shared_encoder_identity": encoder_identity,
        "primary_path_identity": primary_identity,
        "relation_intervention_max_abs": intervention_max_abs,
        "shared_lora_parameter_count": lora_params,
        "primary_projection_parameter_count": primary_params,
        "relation_projection_parameter_count": relation_params,
        "candidate_parameter_count": physical,
        "relation_operator_added_parameter_count": relation_operator.factorization_added_parameter_count,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "hira_core_trainable_parameter_count": hira_trainable,
        "projection_storage_distinct": primary_projection.weight.data_ptr() != relation_projection.weight.data_ptr(),
        "signature_dimension": int(sig_c.shape[-1]),
        "primary_option_permutation_max_abs": primary_perm_error,
        "relation_option_permutation_max_abs": relation_perm_error,
        "signature_option_permutation_max_abs": signature_perm_error,
        "fused_option_permutation_max_abs_diagnostic": fused_logit_perm_error,
        "fused_option_order_flip_rate": fused_choice_flip,
        "max_probability_mass_error": max_mass_error,
        "full_k": True,
        "state_once_view_count": len(suite) * 2,
        "primary_canonical_accuracy": float((cand_raw_c.argmax(-1)==gold).float().mean().cpu()),
        "primary_paraphrase_accuracy": float((cand_raw_p.argmax(-1)==gold).float().mean().cpu()),
        "relation_canonical_accuracy": float((rel_c.argmax(-1)==gold).float().mean().cpu()),
        "relation_paraphrase_accuracy": float((rel_p.argmax(-1)==gold).float().mean().cpu()),
        "fused_canonical_accuracy": float((fused_c.argmax(-1)==gold).float().mean().cpu()),
        "fused_paraphrase_accuracy": float((fused_p.argmax(-1)==gold).float().mean().cpu()),
        "mean_same_option_signature_cosine": float(same_cos.mean().cpu()),
        "mean_signature_same_vs_strongest_wrong_margin": float(sig_margin.mean().cpu()),
        "canonical_mean_role_compatibility": float(diag_c.mean_role_compatibility.cpu()),
        "canonical_mean_value_compatibility": float(diag_c.mean_value_compatibility.cpu()),
        "paraphrase_mean_role_compatibility": float(diag_p.mean_role_compatibility.cpu()),
        "paraphrase_mean_value_compatibility": float(diag_p.mean_value_compatibility.cpu()),
        **synthetic,
        **gradient,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "HIRA_V1_S26_A0_FACTORIZED_RELATION_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
