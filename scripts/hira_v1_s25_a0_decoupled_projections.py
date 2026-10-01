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
from nmd.semantic_core import load_rescued_projection_checkpoint
from nmd.v1_a13_lora import a13_lora_state_dict, iter_a13_lora_modules
from nmd.v1_evidence_fusion import SymmetricFullKEvidenceFusion
from nmd.v1_invariance import symmetric_js_divergence
from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    cross_view_relation_signature_loss,
)
from nmd.v1_s21_semantic_core import build_hira_v1_s21_role_content_core
from nmd.v1_s25_checkpoint import (
    S25_CHECKPOINT_KIND,
    S25_CHECKPOINT_SCHEMA,
    build_frozen_hira_v1_s25_candidate,
    load_hira_v1_s25_checkpoint,
)
from nmd.v1_s25_gradient_ownership import apply_s25_decoupled_gradient_update
from nmd.v1_s25_semantic_core import (
    HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S25_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s25_decoupled_projection_core,
    enforce_s25_eval,
    get_s25_relation_projection,
)

SCHEMA_VERSION = "hira-v1-s25-a0-decoupled-expert-projections-v1"
OUTCOME = "HIRA_V1_S25_A0_DECOUPLED_PROJECTIONS_READY"
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
            f"S25-A0 {self.noun} ledger {self.case_id}: "
            f"{self.field_a} = {self.first}; {self.field_b} = {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"Ledger {self.case_id} lists {self.second} for {self.field_b}. "
            f"The same S25-A0 {self.noun} entry assigns {self.first} to {self.field_a}."
        )

    @property
    def qa1(self) -> str:
        return f"For {self.case_id}, which value is the {self.field_a}?"

    @property
    def qa2(self) -> str:
        return f"Read ledger {self.case_id}: identify its {self.field_a} entry."

    @property
    def qb1(self) -> str:
        return f"For {self.case_id}, which value is the {self.field_b}?"

    @property
    def qb2(self) -> str:
        return f"Read ledger {self.case_id}: identify its {self.field_b} entry."

    def options(self):
        rows = [
            ("a", self.field_a, self.first),
            ("b", self.field_b, self.second),
            ("x", self.field_a, self.wrong_a),
            ("y", self.field_b, self.wrong_b),
        ]
        random.Random(self.seed).shuffle(rows)
        texts = tuple(
            f"for the {self.noun}, {field} is {value}"
            for _kind, field, value in rows
        )
        aliases = tuple(
            f"{value} is the logged {field} value for this {self.noun}"
            for _kind, field, value in rows
        )
        ga = next(i for i, row in enumerate(rows) if row[0] == "a")
        gb = next(i for i, row in enumerate(rows) if row[0] == "b")
        return texts, aliases, ga, gb


def cases() -> tuple[Case, ...]:
    return (
        Case("DP11","Raman mapper","grating","1800 l/mm","laser power","2.5 mW","600 l/mm","12 mW",45101),
        Case("DP22","ion mobility cell","drift gas","nitrogen","field strength","18 V/cm","helium","55 V/cm",45102),
        Case("DP33","photoelectron microscope","aperture","50 um","extractor bias","8 kV","200 um","2 kV",45103),
        Case("DP44","thermal wave imager","modulation","square","lock-in phase","37 deg","sine","82 deg",45104),
        Case("DP55","Brillouin spectrometer","etalon","VIPA","scan span","24 GHz","Fabry-Perot","8 GHz",45105),
        Case("DP66","magnetic force microscope","tip coating","CoCr","lift height","65 nm","PtIr","180 nm",45106),
        Case("DP77","gas chromatograph inlet","liner","splitless","inlet temp","275 C","packed","180 C",45107),
        Case("DP88","hyperspectral camera","slit width","25 um","integration","14 ms","100 um","60 ms",45108),
        Case("DQ11","electron backscatter rig","pattern binning","2x2","camera gain","high","8x8","low",45109),
        Case("DQ22","muon detector","scintillator","plastic","threshold","35 mV","crystal","120 mV",45110),
        Case("DQ33","neutron reflectometer","chopper mode","double-disk","wavelength","5.2 A","single-disk","2.1 A",45111),
        Case("DQ44","terahertz camera","sensor","microbolometer","frame rate","48 Hz","Schottky","12 Hz",45112),
        Case("DQ55","ellipsometry stage","incidence angle","70 deg","azimuth step","0.5 deg","55 deg","3 deg",45113),
        Case("DQ66","plasma probe","probe type","triple","sweep rate","40 Hz","single","5 Hz",45114),
        Case("DQ77","acoustic emission rig","couplant","silicone","sample rate","5 MHz","water","500 kHz",45115),
        Case("DQ88","spectral confocal head","pinhole","30 um","z step","120 nm","150 um","800 nm",45116),
    )


def _content_mask(batch) -> Tensor:
    mask = batch.attention_mask.bool()
    if batch.special_token_mask is None:
        return mask
    return mask & ~batch.special_token_mask.bool()


def _encode(runtime, suite: tuple[Case, ...]) -> dict[str, Tensor]:
    # Works for both the S25 candidate and the frozen S21 initialization
    # baseline.  S25-only ownership checks are performed separately.
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


def _primary(
    runtime,
    encoded: dict[str, Tensor],
    *,
    paraphrase: bool,
) -> Tensor:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S25-A0 primary scorer missing")
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
    projection,
    encoded: dict[str, Tensor],
    *,
    paraphrase: bool,
):
    prefix = "paraphrase" if paraphrase else "canonical"
    state_prefix = "b" if paraphrase else "a"
    canonicalizer = CrossViewRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )
    return canonicalizer(
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


def _supervised(logits: Tensor, gold: Tensor, other: Tensor):
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


def _semantic_metrics(raw_c, raw_p, rel_c, rel_p, fused_c, fused_p, gold):
    q = gold.numel()
    mass = max(
        float((torch.softmax(x, -1).sum(-1) - 1.0).abs().max().detach().cpu())
        for x in (raw_c, raw_p, rel_c, rel_p, fused_c, fused_p)
    )
    return {
        "primary_canonical_accuracy": float((raw_c.argmax(-1) == gold).float().mean().cpu()),
        "primary_paraphrase_accuracy": float((raw_p.argmax(-1) == gold).float().mean().cpu()),
        "relation_canonical_accuracy": float((rel_c.argmax(-1) == gold).float().mean().cpu()),
        "relation_paraphrase_accuracy": float((rel_p.argmax(-1) == gold).float().mean().cpu()),
        "fused_canonical_accuracy": float((fused_c.argmax(-1) == gold).float().mean().cpu()),
        "fused_paraphrase_accuracy": float((fused_p.argmax(-1) == gold).float().mean().cpu()),
        "decision_count_per_view": q,
        "max_probability_mass_error": mass,
    }


def _gradient_court(bundle: Path, manifest: dict, suite: tuple[Case, ...]) -> dict:
    with torch.inference_mode(False), torch.enable_grad():
        fresh = load_hira_v0_m4_bundle(bundle)
        runtime = build_hira_v1_s25_decoupled_projection_core(
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
            runtime, relation_projection, encoded, paraphrase=False
        )
        rel_p, sig_p, _ = _relation(
            runtime, relation_projection, encoded, paraphrase=True
        )
        fusion = SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON)
        fused_c, _ = fusion(raw_c, rel_c.detach())
        fused_p, _ = fusion(raw_p, rel_p.detach())
        gold, other = _gold(suite, fused_c.device)

        decision = 0.5 * (
            _supervised(fused_c, gold, other)
            + _supervised(fused_p, gold, other)
        )
        option_align = _option_alignment(encoded["option_pooled"])
        consistency = symmetric_js_divergence(fused_c, fused_p)
        primary_block = (
            decision
            + OPTION_ALIGN_COEFFICIENT * option_align
            + INVARIANCE_COEFFICIENT * consistency
        )

        relation_loss = 0.5 * (
            F.cross_entropy(rel_c, gold) + F.cross_entropy(rel_p, gold)
        )
        canonicalization, _alignment, _separation = (
            cross_view_relation_signature_loss(
                sig_c,
                sig_p,
                separation_margin=SIGNATURE_SEPARATION_MARGIN,
            )
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
            "shared_normalized_pre_dot": d.shared.normalized_pre_dot,
            "shared_normalized_post_dot": d.shared.normalized_post_dot,
            "shared_projection_coefficient": d.shared.projection_coefficient,
            "shared_combined_norm": d.shared.combined_norm,
            "shared_special_case": d.shared.special_case,
        }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S25-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    t0 = bundle / str(manifest["t0_checkpoint"])
    t0_sha = str(manifest["t0_checkpoint_sha256"])

    candidate_bundle = load_hira_v0_m4_bundle(bundle)
    candidate = build_hira_v1_s25_decoupled_projection_core(
        candidate_bundle.runtime.encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_primary_projection=False,
        train_relation_projection=False,
    )
    del candidate_bundle

    baseline_bundle = load_hira_v0_m4_bundle(bundle)
    baseline = build_hira_v1_s21_role_content_core(
        baseline_bundle.runtime.encoder,
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=False,
        train_projection=False,
    )
    del baseline_bundle

    candidate_encoded = _encode(candidate, suite)
    baseline_encoded = _encode(baseline, suite)

    encoder_identity = all(
        torch.equal(candidate_encoded[key].cpu(), baseline_encoded[key].cpu())
        for key in (
            "state_a_tokens", "state_b_tokens",
            "question_canonical_tokens", "question_paraphrase_tokens",
            "option_tokens", "option_pooled",
        )
    )
    if not encoder_identity:
        raise RuntimeError("S25-A0 shared A13 encoding identity changed")

    cand_raw_c = _primary(candidate, candidate_encoded, paraphrase=False)
    cand_raw_p = _primary(candidate, candidate_encoded, paraphrase=True)
    base_raw_c = _primary(baseline, baseline_encoded, paraphrase=False)
    base_raw_p = _primary(baseline, baseline_encoded, paraphrase=True)

    candidate_relation_projection = get_s25_relation_projection(candidate)
    baseline_projection = baseline.projection_triadic_scorer.projection
    cand_rel_c, _cand_sig_c, _ = _relation(
        candidate, candidate_relation_projection, candidate_encoded, paraphrase=False
    )
    cand_rel_p, _cand_sig_p, _ = _relation(
        candidate, candidate_relation_projection, candidate_encoded, paraphrase=True
    )
    base_rel_c, _base_sig_c, _ = _relation(
        baseline, baseline_projection, baseline_encoded, paraphrase=False
    )
    base_rel_p, _base_sig_p, _ = _relation(
        baseline, baseline_projection, baseline_encoded, paraphrase=True
    )

    primary_identity = bool(
        torch.equal(cand_raw_c, base_raw_c) and torch.equal(cand_raw_p, base_raw_p)
    )
    relation_identity = bool(
        torch.equal(cand_rel_c, base_rel_c) and torch.equal(cand_rel_p, base_rel_p)
    )
    if not primary_identity or not relation_identity:
        raise RuntimeError("S25-A0 initialization inference identity changed")

    fusion = SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON)
    cand_fused_c, _ = fusion(cand_raw_c, cand_rel_c)
    cand_fused_p, _ = fusion(cand_raw_p, cand_rel_p)
    base_fused_c, _ = fusion(base_raw_c, base_rel_c)
    base_fused_p, _ = fusion(base_raw_p, base_rel_p)
    fusion_identity = bool(
        torch.equal(cand_fused_c, base_fused_c)
        and torch.equal(cand_fused_p, base_fused_p)
    )
    if not fusion_identity:
        raise RuntimeError("S25-A0 S14 fusion identity changed")

    primary_projection = candidate.projection_triadic_scorer.projection
    relation_projection = get_s25_relation_projection(candidate)
    t0_weight = load_rescued_projection_checkpoint(
        t0,
        expected_sha256=t0_sha,
    ).to(device=primary_projection.weight.device, dtype=primary_projection.weight.dtype)
    projection_initialization_identity = bool(
        torch.equal(primary_projection.weight, relation_projection.weight)
    )
    t0_projection_identity = bool(
        torch.equal(primary_projection.weight, t0_weight)
        and torch.equal(relation_projection.weight, t0_weight)
    )
    projection_storage_distinct = (
        primary_projection.weight.data_ptr() != relation_projection.weight.data_ptr()
    )
    if (
        not projection_initialization_identity
        or not t0_projection_identity
        or not projection_storage_distinct
    ):
        raise RuntimeError("S25-A0 projection ownership initialization invalid")

    lora_params = sum(
        p.numel()
        for module in iter_a13_lora_modules(candidate.encoder)
        for p in (module.lora_a, module.lora_b)
    )
    primary_params = primary_projection.weight.numel()
    relation_params = relation_projection.weight.numel()
    physical = lora_params + primary_params + relation_params
    trainable = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in candidate.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    if lora_params != HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT:
        raise RuntimeError("S25-A0 shared LoRA capacity changed")
    if primary_params != HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S25-A0 primary projection capacity changed")
    if relation_params != HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S25-A0 relation projection capacity changed")
    if physical != HIRA_V1_S25_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S25-A0 total physical surface changed")
    if trainable != 0 or original_a13 != 0:
        raise RuntimeError("S25-A0 frozen inference runtime is not frozen")
    hira_core_trainable = sum(
        p.numel() for p in candidate.hira.parameters() if p.requires_grad
    )
    if hira_core_trainable != 0:
        raise RuntimeError("S25-A0 HIRACore became trainable")

    # A0 checkpoint ownership court: serialize the exact decoupled ownership
    # schema, reload it, and build a frozen replay from an independent encoder.
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_probe_path = args.out / "s25-a0-checkpoint-roundtrip.pt"
    torch.save(
        {
            "schema_version": S25_CHECKPOINT_SCHEMA,
            "kind": S25_CHECKPOINT_KIND,
            "lora_parameter_count": HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT,
            "primary_projection_parameter_count": HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT,
            "relation_projection_parameter_count": HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT,
            "total_parameter_count": HIRA_V1_S25_TOTAL_PARAMETER_COUNT,
            "lora_rank": 8,
            "selected_dev_epoch": 1,
            "semantic_revision": str(manifest["semantic_revision"]),
            "initialization_t0_sha256": t0_sha,
            "lora_state_dict": a13_lora_state_dict(candidate.encoder),
            "primary_projection_state_dict": {
                "projection.weight": primary_projection.weight.detach().cpu().clone(),
            },
            "relation_projection_state_dict": {
                "projection.weight": relation_projection.weight.detach().cpu().clone(),
            },
        },
        checkpoint_probe_path,
    )
    (
        checkpoint_lora,
        checkpoint_primary,
        checkpoint_relation,
        checkpoint_meta,
    ) = load_hira_v1_s25_checkpoint(
        checkpoint_probe_path,
        initialization_t0_sha256=t0_sha,
        semantic_revision=str(manifest["semantic_revision"]),
    )
    checkpoint_state_identity = (
        all(
            torch.equal(
                checkpoint_lora[key],
                a13_lora_state_dict(candidate.encoder)[key].detach().cpu(),
            )
            for key in checkpoint_lora
        )
        and torch.equal(
            checkpoint_primary["projection.weight"],
            primary_projection.weight.detach().cpu(),
        )
        and torch.equal(
            checkpoint_relation["projection.weight"],
            relation_projection.weight.detach().cpu(),
        )
    )
    if not checkpoint_state_identity:
        raise RuntimeError("S25-A0 checkpoint roundtrip changed owned state")

    replay_bundle = load_hira_v0_m4_bundle(bundle)
    replay, replay_meta = build_frozen_hira_v1_s25_candidate(
        replay_bundle.runtime.encoder,
        t0,
        checkpoint_probe_path,
        expected_t0_sha256=t0_sha,
        expected_candidate_sha256=str(checkpoint_meta["sha256"]),
        semantic_revision=str(manifest["semantic_revision"]),
    )
    del replay_bundle
    replay_primary = replay.projection_triadic_scorer.projection
    replay_relation = get_s25_relation_projection(replay)
    checkpoint_frozen_replay_passed = (
        sum(p.numel() for p in replay.parameters() if p.requires_grad) == 0
        and replay_primary.weight.data_ptr() != replay_relation.weight.data_ptr()
        and torch.equal(
            replay_primary.weight.detach().cpu(),
            checkpoint_primary["projection.weight"],
        )
        and torch.equal(
            replay_relation.weight.detach().cpu(),
            checkpoint_relation["projection.weight"],
        )
        and replay_meta["selected_dev_epoch"] == 1
    )
    if not checkpoint_frozen_replay_passed:
        raise RuntimeError("S25-A0 frozen checkpoint replay failed")
    checkpoint_probe_path.unlink()

    perm = torch.tensor([2, 0, 3, 1], device=cand_raw_c.device)
    moved = {
        **candidate_encoded,
        "option_tokens": candidate_encoded["option_tokens"][:, perm],
        "option_mask": candidate_encoded["option_mask"][:, perm],
        "option_view_mask": candidate_encoded["option_view_mask"][:, perm],
        "option_pooled": candidate_encoded["option_pooled"][:, perm],
    }
    moved_raw = _primary(candidate, moved, paraphrase=False)
    moved_rel, _moved_sig, _ = _relation(
        candidate, relation_projection, moved, paraphrase=False
    )
    moved_fused, _ = fusion(moved_raw, moved_rel)
    primary_perm_error = float((moved_raw - cand_raw_c[:, perm]).abs().max().cpu())
    relation_perm_error = float((moved_rel - cand_rel_c[:, perm]).abs().max().cpu())
    fused_perm_error = float((moved_fused - cand_fused_c[:, perm]).abs().max().cpu())
    print(
        "HIRA_V1_S25_A0_PERMUTATION_DIAGNOSTIC="
        + json.dumps(
            {
                "primary_option_permutation_max_abs": primary_perm_error,
                "relation_option_permutation_max_abs": relation_perm_error,
                "fused_option_permutation_max_abs": fused_perm_error,
                "frozen_tolerance": 1e-6,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if max(primary_perm_error, relation_perm_error, fused_perm_error) > 1e-6:
        raise RuntimeError("S25-A0 option permutation equivariance changed")

    gold, _other = _gold(suite, cand_fused_c.device)
    metrics = _semantic_metrics(
        cand_raw_c, cand_raw_p, cand_rel_c, cand_rel_p,
        cand_fused_c, cand_fused_p, gold
    )
    if metrics["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S25-A0 probability mass changed")

    gradient = _gradient_court(bundle, manifest, suite)
    for key in (
        "primary_private_gradient_l1",
        "relation_private_gradient_l1",
        "primary_shared_gradient_l1",
        "relation_shared_gradient_l1",
    ):
        if gradient[key] <= 0.0:
            raise RuntimeError(f"S25-A0 gradient vanished: {key}")
    if gradient["primary_to_relation_private_max_abs"] != 0.0:
        raise RuntimeError("S25-A0 primary leaked into relation-private projection")
    if gradient["relation_to_primary_private_max_abs"] != 0.0:
        raise RuntimeError("S25-A0 relation leaked into primary-private projection")
    if gradient["shared_projection_coefficient"] != 0.0:
        raise RuntimeError("S25-A0 shared neutral bisector unexpectedly projected")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S25_A0_DECOUPLED_EXPERT_PROJECTIONS_ONLY",
        "semantic_case_count": len(suite),
        "decision_count": len(suite) * 4,
        "state_view_count": len(suite) * 2,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "shared_encoder_identity": encoder_identity,
        "primary_initialization_identity": primary_identity,
        "relation_initialization_identity": relation_identity,
        "s14_equal_fusion_identity": fusion_identity,
        "projection_initialization_identity": projection_initialization_identity,
        "t0_projection_identity": t0_projection_identity,
        "projection_storage_distinct": projection_storage_distinct,
        "hira_core_trainable_parameter_count": hira_core_trainable,
        "checkpoint_state_identity": checkpoint_state_identity,
        "checkpoint_frozen_replay_passed": checkpoint_frozen_replay_passed,
        "shared_lora_parameter_count": lora_params,
        "primary_projection_parameter_count": primary_params,
        "relation_projection_parameter_count": relation_params,
        "candidate_parameter_count": physical,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "fusion_added_parameter_count": fusion.parameter_count,
        "primary_option_permutation_max_abs": primary_perm_error,
        "relation_option_permutation_max_abs": relation_perm_error,
        "fused_option_permutation_max_abs": fused_perm_error,
        "full_k": True,
        "state_once_view_count": len(suite) * 2,
        "fusion_epsilon": FUSION_EPSILON,
        "balance_epsilon": BALANCE_EPSILON,
        **metrics,
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
        "HIRA_V1_S25_A0_DECOUPLED_PROJECTIONS_RECEIPT="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
