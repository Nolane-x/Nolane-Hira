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
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_evidence_fusion import (
    SymmetricFullKEvidenceFusion,
    fused_gold_vs_max_wrong_margin,
)
from nmd.v1_gradient_isolated_fusion import GradientIsolatedFullKEvidenceFusion
from nmd.v1_norm_balanced_gradient import norm_balanced_gradient_update
from nmd.v1_standardized_triadic_consistency import standardized_triadic_consistency_loss
from nmd.v1_s20_authority import S20FusionCase
from hira_v1_s20_train_dev import _losses as _s20_losses
from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    relation_signature_same_option_cosine,
)
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core
from nmd.v1_s20_semantic_core import (
    HIRA_V1_S20_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S20_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s20_standardized_triadic_core,
)


SCHEMA_VERSION = "hira-v1-s20-a0-identity-v1"
OUTCOME = "HIRA_V1_S20_A0_IDENTITY_READY"
ROLE_TEMPERATURE = 0.10
PAIR_TEMPERATURE = 0.10
CONTRASTIVE_TEMPERATURE = 0.10
FUSION_EPSILON = 1e-6


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
            f"S20-A0 {self.noun} sheet {self.case_id} records "
            f"{self.field_a} as {self.first}, while {self.field_b} is {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"For {self.case_id}, {self.second} appears under {self.field_b}. "
            f"The same S20-A0 {self.noun} sheet places {self.first} under {self.field_a}."
        )

    @property
    def qa1(self) -> str:
        return f"Which {self.field_a} belongs to {self.case_id}?"

    @property
    def qa2(self) -> str:
        return f"Identify the value entered under {self.field_a} on sheet {self.case_id}."

    @property
    def qb1(self) -> str:
        return f"Which {self.field_b} belongs to {self.case_id}?"

    @property
    def qb2(self) -> str:
        return f"Identify the value entered under {self.field_b} on sheet {self.case_id}."

    def option_pack(self) -> tuple[tuple[LogicalOption, ...], int, int]:
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
                criterion_text=(
                    f"for this {self.noun}, {field} is {value}"
                ),
                aliases=(
                    f"{value} is the recorded {field} value for this {self.noun}",
                ),
            )
            for i, (_kind, field, value) in enumerate(rows)
        )
        ga = next(i for i, (kind, _field, _value) in enumerate(rows) if kind == "a")
        gb = next(i for i, (kind, _field, _value) in enumerate(rows) if kind == "b")
        return options, ga, gb


def cases() -> tuple[Case, ...]:
    return (
        Case("XE11","cryogenic quartz microbalance","crystal cut","AT-cut","drive frequency","5 MHz","BT-cut","2 MHz",36201),
        Case("XE22","Hall probe mapper","probe material","InSb","scan pitch","0.4 mm","silicon","1.5 mm",36202),
        Case("XE33","vacuum ultraviolet lamp","fill gas","deuterium","window material","MgF2","xenon","quartz",36203),
        Case("XE44","laser Doppler anemometer","beam geometry","two-component","fringe spacing","3.2 um","single-beam","9.5 um",36204),
        Case("XE55","thermal vacuum shroud","panel coating","black anodize","loop temperature","-145 C","bare aluminum","-40 C",36205),
        Case("XE66","powder XRD goniometer","anode target","copper","step size","0.015 deg","molybdenum","0.08 deg",36206),
        Case("XE77","Raman microscope stage","objective","100x long-working-distance","laser wavelength","532 nm","20x","785 nm",36207),
        Case("XE88","superconducting resonator tester","coupling mode","notch","probe power","-92 dBm","through","-55 dBm",36208),
        Case("XE99","microfluidic droplet sorter","actuation","dielectrophoretic","sort rate","1800 Hz","pneumatic","250 Hz",36209),
        Case("XF10","electron beam evaporator","crucible liner","pyrolytic BN","beam current","420 mA","graphite","130 mA",36210),
        Case("XF21","acoustic levitation rig","transducer array","phased ring","carrier frequency","40 kHz","single horn","18 kHz",36211),
        Case("XF32","optical tweezer array","trap pattern","holographic","site spacing","6.5 um","single Gaussian","18 um",36212),
        Case("XF43","plasma impedance probe","electrode material","tungsten","sweep amplitude","12 V","stainless steel","3 V",36213),
        Case("XF54","satellite star tracker","detector","sCMOS","exposure time","18 ms","CCD","75 ms",36214),
        Case("XF65","hyperspectral calibration lamp","reference source","integrating sphere","radiance setting","72 percent","bare LED","25 percent",36215),
        Case("XF76","ion trap controller","electrode drive","segmented RF","secular frequency","1.8 MHz","static DC","0.5 MHz",36216),
    )


def _text_bank(suite: tuple[Case, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for case in suite:
        values.extend((
            case.state_a,
            case.state_b,
            case.qa1,
            case.qa2,
            case.qb1,
            case.qb2,
        ))
        options, _ga, _gb = case.option_pack()
        for option in options:
            values.append(option.criterion_text)
            values.extend(option.aliases)
    return tuple(values)


@torch.inference_mode()
def _collect_decisions(runtime, suite: tuple[Case, ...], mode: str) -> dict:
    records = {}
    canonical_logits = []
    paraphrase_logits = []
    correct = 0
    canonical_pair_both = 0
    flips = 0
    max_mass = 0.0
    before = runtime.state_encode_calls

    for case in suite:
        options, ga, gb = case.option_pack()
        canonical_ok = []
        for view, state, qa, qb in (
            ("canonical", case.state_a, case.qa1, case.qb1),
            ("paraphrase", case.state_b, case.qa2, case.qb2),
        ):
            memory = runtime.compile_state(state)
            per_view = []
            for label, question, gold in (("a", qa, ga), ("b", qb, gb)):
                schema, _ = runtime.compile_schema(
                    primitive="choice",
                    question_text=question,
                    options=options,
                    include_token_artifacts=True,
                    use_cache=False,
                )
                out = runtime.forward_compiled(
                    memory,
                    schema,
                    coarse_mode=mode,
                    relation_refinement=False,
                )
                idx = int(out.probabilities.argmax())
                ok = idx == gold
                correct += int(ok)
                if view == "canonical":
                    canonical_ok.append(ok)
                records[(case.case_id, f"{view}-{label}")] = (
                    out.logits.detach().cpu().clone(),
                    out.selected_option_id,
                )
                per_view.append(out.logits.detach().cpu().clone())
                max_mass = max(max_mass, abs(float(out.probabilities.sum()) - 1.0))
                if int(out.hira.candidate_budget.item()) != 4:
                    raise RuntimeError("S20-A0 full-K changed")
                if not torch.equal(
                    out.hira.relation_delta,
                    torch.zeros_like(out.hira.relation_delta),
                ):
                    raise RuntimeError("S20-A0 relation delta changed")

                reverse_schema, _ = runtime.compile_schema(
                    primitive="choice",
                    question_text=question,
                    options=tuple(reversed(options)),
                    include_token_artifacts=True,
                    use_cache=False,
                )
                reverse = runtime.forward_compiled(
                    memory,
                    reverse_schema,
                    coarse_mode=mode,
                    relation_refinement=False,
                )
                flips += int(reverse.selected_option_id != out.selected_option_id)

            if view == "canonical":
                canonical_logits.extend(per_view)
            else:
                paraphrase_logits.extend(per_view)
        canonical_pair_both += int(all(canonical_ok))

    c = torch.stack(canonical_logits)
    p = torch.stack(paraphrase_logits)
    return {
        "records": records,
        "accuracy_all_views": correct / (len(suite) * 4),
        "canonical_paired_both_correct_rate": canonical_pair_both / len(suite),
        "cross_view_selected_choice_agreement": float(selected_choice_agreement(c, p)),
        "cross_view_mean_js": float(symmetric_js_divergence(c, p)),
        "option_order_flip_rate": flips / (len(suite) * 4),
        "state_encode_calls": runtime.state_encode_calls - before,
        "max_probability_mass_error": max_mass,
    }


@torch.inference_mode()
def _gold_margin(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    chosen = logits.gather(-1, gold[:, None]).squeeze(-1)
    indices = torch.arange(logits.shape[-1], device=logits.device)
    wrong = logits.masked_fill(indices[None, :].eq(gold[:, None]), float("-inf"))
    return chosen - wrong.max(-1).values


def _signature_margin(
    canonical: torch.Tensor,
    paraphrase: torch.Tensor,
) -> torch.Tensor:
    canonical = torch.nn.functional.normalize(canonical, dim=-1)
    paraphrase = torch.nn.functional.normalize(paraphrase, dim=-1)
    cross = torch.einsum("nkd,njd->nkj", canonical, paraphrase)
    same = cross.diagonal(dim1=-2, dim2=-1)
    k = canonical.shape[1]
    eye = torch.eye(k, dtype=torch.bool, device=cross.device)[None]
    wrong = cross.masked_fill(eye, float("-inf")).amax(-1)
    return same - wrong


@torch.inference_mode()
def _signature_margin(
    canonical: torch.Tensor,
    paraphrase: torch.Tensor,
) -> torch.Tensor:
    canonical = torch.nn.functional.normalize(canonical, dim=-1)
    paraphrase = torch.nn.functional.normalize(paraphrase, dim=-1)
    cross = torch.einsum("nkd,njd->nkj", canonical, paraphrase)
    same = cross.diagonal(dim1=-2, dim2=-1)
    k = canonical.shape[1]
    eye = torch.eye(k, dtype=torch.bool, device=cross.device)[None]
    wrong = cross.masked_fill(eye, float("-inf")).amax(-1)
    return same - wrong


@torch.inference_mode()
def _collect_fusion(runtime, suite: tuple[Case, ...]) -> dict:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S20-A0 projection scorer missing")

    canonicalizer = CrossViewRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )
    fusion = GradientIsolatedFullKEvidenceFusion(epsilon=FUSION_EPSILON)

    raw_c, raw_p = [], []
    relation_c, relation_p = [], []
    fused_c, fused_p = [], []
    sig_c, sig_p = [], []
    gold_c, gold_p = [], []
    fused_pair_both = 0
    fused_order_flips = 0
    fused_mass_error = 0.0
    swap_symmetry_max_abs = 0.0
    expert_agreement = []
    triadic_rms = []
    relation_rms = []

    for case in suite:
        options, ga, gb = case.option_pack()
        canonical_ok = []
        for view, state, qa, qb in (
            ("canonical", case.state_a, case.qa1, case.qb1),
            ("paraphrase", case.state_b, case.qa2, case.qb2),
        ):
            memory = runtime.compile_state(state)
            state_tokens = memory.content_token_embeddings
            if state_tokens is None:
                raise RuntimeError("S20-A0 state token artifacts missing")
            st = state_tokens.unsqueeze(0)
            sm = torch.ones(1, st.shape[1], dtype=torch.bool, device=st.device)

            for question, gold in ((qa, ga), (qb, gb)):
                schema, _ = runtime.compile_schema(
                    primitive="choice",
                    question_text=question,
                    options=options,
                    include_token_artifacts=True,
                    use_cache=False,
                )
                raw_out = runtime.forward_compiled(
                    memory,
                    schema,
                    coarse_mode="projection_triadic",
                    relation_refinement=False,
                )
                q = schema.question_token_embeddings
                qm = schema.question_content_token_mask
                ov = schema.option_view_token_embeddings
                ovtm = schema.option_view_token_mask
                ovm = schema.option_view_mask
                if any(x is None for x in (q, qm, ov, ovtm, ovm)):
                    raise RuntimeError("S20-A0 schema token artifacts missing")

                relation, signatures, _ = canonicalizer(
                    projection=scorer.projection,
                    state_tokens=st,
                    state_mask=sm,
                    question_tokens=q.unsqueeze(0),
                    question_mask=qm.unsqueeze(0),
                    option_view_tokens=ov.unsqueeze(0),
                    option_view_token_mask=ovtm.unsqueeze(0),
                    option_view_mask=ovm.unsqueeze(0),
                )
                raw = raw_out.logits.unsqueeze(0)
                fused, diag = fusion(raw, relation)
                swapped, _ = fusion(relation, raw)
                swap_symmetry_max_abs = max(
                    swap_symmetry_max_abs,
                    float((fused - swapped).abs().max().cpu()),
                )
                probs = torch.softmax(fused, dim=-1)
                fused_mass_error = max(
                    fused_mass_error,
                    float((probs.sum(-1) - 1.0).abs().max().cpu()),
                )
                pred = int(fused.argmax(-1).item())
                ok = pred == gold
                if view == "canonical":
                    canonical_ok.append(ok)
                    raw_c.append(raw[0].detach().cpu())
                    relation_c.append(relation[0].detach().cpu())
                    fused_c.append(fused[0].detach().cpu())
                    sig_c.append(signatures[0].detach().cpu())
                    gold_c.append(gold)
                else:
                    raw_p.append(raw[0].detach().cpu())
                    relation_p.append(relation[0].detach().cpu())
                    fused_p.append(fused[0].detach().cpu())
                    sig_p.append(signatures[0].detach().cpu())
                    gold_p.append(gold)

                expert_agreement.append(float(diag.expert_top1_agreement))
                triadic_rms.append(float(diag.triadic_rms))
                relation_rms.append(float(diag.relation_rms))

                reverse_schema, _ = runtime.compile_schema(
                    primitive="choice",
                    question_text=question,
                    options=tuple(reversed(options)),
                    include_token_artifacts=True,
                    use_cache=False,
                )
                reverse_raw_out = runtime.forward_compiled(
                    memory,
                    reverse_schema,
                    coarse_mode="projection_triadic",
                    relation_refinement=False,
                )
                rq = reverse_schema.question_token_embeddings
                rqm = reverse_schema.question_content_token_mask
                rov = reverse_schema.option_view_token_embeddings
                rovtm = reverse_schema.option_view_token_mask
                rovm = reverse_schema.option_view_mask
                if any(x is None for x in (rq, rqm, rov, rovtm, rovm)):
                    raise RuntimeError("S20-A0 reverse schema token artifacts missing")
                reverse_relation, _, _ = canonicalizer(
                    projection=scorer.projection,
                    state_tokens=st,
                    state_mask=sm,
                    question_tokens=rq.unsqueeze(0),
                    question_mask=rqm.unsqueeze(0),
                    option_view_tokens=rov.unsqueeze(0),
                    option_view_token_mask=rovtm.unsqueeze(0),
                    option_view_mask=rovm.unsqueeze(0),
                )
                reverse_fused, _ = fusion(
                    reverse_raw_out.logits.unsqueeze(0),
                    reverse_relation,
                )
                reverse_pred = int(reverse_fused.argmax(-1).item())
                mapped = len(options) - 1 - reverse_pred
                fused_order_flips += int(mapped != pred)

        fused_pair_both += int(all(canonical_ok))

    raw_c = torch.stack(raw_c)
    raw_p = torch.stack(raw_p)
    relation_c = torch.stack(relation_c)
    relation_p = torch.stack(relation_p)
    fused_c = torch.stack(fused_c)
    fused_p = torch.stack(fused_p)
    sig_c = torch.stack(sig_c)
    sig_p = torch.stack(sig_p)
    cg = torch.tensor(gold_c, dtype=torch.long)
    pg = torch.tensor(gold_p, dtype=torch.long)
    same_cos = relation_signature_same_option_cosine(sig_c, sig_p)
    sig_margin = _signature_margin(sig_c, sig_p)

    return {
        "raw_triadic_canonical_accuracy": float((raw_c.argmax(-1) == cg).float().mean()),
        "raw_triadic_paraphrase_accuracy": float((raw_p.argmax(-1) == pg).float().mean()),
        "raw_triadic_cross_view_agreement": float(selected_choice_agreement(raw_c, raw_p)),
        "raw_triadic_cross_view_mean_js": float(symmetric_js_divergence(raw_c, raw_p)),
        "raw_triadic_canonical_mean_gold_margin": float(fused_gold_vs_max_wrong_margin(raw_c, cg).mean()),
        "canonical_relation_binding_accuracy": float((relation_c.argmax(-1) == cg).float().mean()),
        "paraphrase_relation_binding_accuracy": float((relation_p.argmax(-1) == pg).float().mean()),
        "relation_cross_view_agreement": float(selected_choice_agreement(relation_c, relation_p)),
        "canonical_relation_mean_gold_margin": float(fused_gold_vs_max_wrong_margin(relation_c, cg).mean()),
        "fused_canonical_accuracy": float((fused_c.argmax(-1) == cg).float().mean()),
        "fused_paraphrase_accuracy": float((fused_p.argmax(-1) == pg).float().mean()),
        "fused_canonical_paired_both_correct_rate": fused_pair_both / len(suite),
        "fused_cross_view_selected_choice_agreement": float(selected_choice_agreement(fused_c, fused_p)),
        "fused_cross_view_mean_js": float(symmetric_js_divergence(fused_c, fused_p)),
        "fused_canonical_mean_gold_margin": float(fused_gold_vs_max_wrong_margin(fused_c, cg).mean()),
        "mean_same_option_signature_cosine": float(same_cos.mean()),
        "mean_signature_same_vs_strongest_wrong_margin": float(sig_margin.mean()),
        "expert_top1_agreement": sum(expert_agreement) / len(expert_agreement),
        "mean_triadic_rms": sum(triadic_rms) / len(triadic_rms),
        "mean_relation_rms": sum(relation_rms) / len(relation_rms),
        "fusion_expert_swap_max_abs": swap_symmetry_max_abs,
        "fused_option_order_flip_rate": fused_order_flips / (len(suite) * 4),
        "fused_max_probability_mass_error": fused_mass_error,
    }


def _gradient_route_probe() -> dict[str, float | bool]:
    with torch.inference_mode(False), torch.enable_grad():
        torch.manual_seed(24217)
        triadic = torch.randn(8, 4, requires_grad=True)
        relation = torch.randn(8, 4, requires_grad=True)
        gold = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

        protected = GradientIsolatedFullKEvidenceFusion(epsilon=FUSION_EPSILON)
        reference = SymmetricFullKEvidenceFusion(epsilon=FUSION_EPSILON)

        fused, _ = protected(triadic, relation)
        reference_fused, _ = reference(triadic.detach(), relation.detach())
        max_abs = float((fused.detach() - reference_fused).abs().max().cpu())

        primary_loss = torch.nn.functional.cross_entropy(fused, gold)
        primary_loss.backward(retain_graph=True)
        triadic_grad = 0.0 if triadic.grad is None else float(triadic.grad.abs().sum().cpu())
        relation_direct_grad = 0.0 if relation.grad is None else float(relation.grad.abs().sum().cpu())

        relation.grad = None
        relation_aux = torch.nn.functional.cross_entropy(relation, gold)
        relation_aux.backward()
        relation_aux_grad = 0.0 if relation.grad is None else float(relation.grad.abs().sum().cpu())

        return {
            "s20_vs_s14_forward_max_abs": max_abs,
            "primary_triadic_logit_gradient_l1": triadic_grad,
            "primary_relation_logit_direct_gradient_l1": relation_direct_grad,
            "relation_auxiliary_logit_gradient_l1": relation_aux_grad,
            "primary_triadic_gradient_nonzero": triadic_grad > 0.0,
            "primary_relation_direct_gradient_zero": relation_direct_grad == 0.0,
            "relation_auxiliary_gradient_nonzero": relation_aux_grad > 0.0,
        }


def _standardized_consistency_probe() -> dict[str, float | bool]:
    with torch.inference_mode(False), torch.enable_grad():
        a = torch.tensor(
            [[1.0, 2.0, 4.0, -1.0], [0.3, -0.5, 0.8, 1.7]],
            requires_grad=True,
        )
        b = torch.tensor(
            [[1.0, 4.0, 2.0, -1.0], [1.1, -0.2, 0.4, 2.3]],
            requires_grad=True,
        )

        identical, za, _, _ = standardized_triadic_consistency_loss(
            a, a, epsilon=FUSION_EPSILON
        )
        forward, *_ = standardized_triadic_consistency_loss(
            a, b, epsilon=FUSION_EPSILON
        )
        reverse, *_ = standardized_triadic_consistency_loss(
            b, a, epsilon=FUSION_EPSILON
        )
        permutation = torch.tensor([2, 0, 3, 1])
        permuted, *_ = standardized_triadic_consistency_loss(
            a[:, permutation],
            b[:, permutation],
            epsilon=FUSION_EPSILON,
        )

        transformed = 7.0 * a.detach() + 13.0
        _, z_transformed, _, _ = standardized_triadic_consistency_loss(
            transformed,
            transformed,
            epsilon=FUSION_EPSILON,
        )
        affine_max_abs = float((za.detach() - z_transformed.detach()).abs().max().cpu())

        tiny, *_ = standardized_triadic_consistency_loss(
            a * 1e-3,
            b * 1e-3,
            epsilon=FUSION_EPSILON,
        )

        flat_a = torch.full((3, 4), 7.0, requires_grad=True)
        flat_b = torch.full((3, 4), -2.0, requires_grad=True)
        flat, flat_za, flat_zb, flat_diag = standardized_triadic_consistency_loss(
            flat_a,
            flat_b,
            epsilon=FUSION_EPSILON,
        )

        forward.backward()
        a_grad = 0.0 if a.grad is None else float(a.grad.abs().sum().cpu())
        b_grad = 0.0 if b.grad is None else float(b.grad.abs().sum().cpu())

        return {
            "standardized_consistency_identical": float(identical.detach().cpu()),
            "standardized_consistency_mismatch": float(forward.detach().cpu()),
            "standardized_consistency_swap_max_abs": abs(float((forward.detach() - reverse.detach()).cpu())),
            "standardized_consistency_permutation_max_abs": abs(float((forward.detach() - permuted.detach()).cpu())),
            "standardized_consistency_affine_max_abs": affine_max_abs,
            "standardized_consistency_scale_max_abs": abs(float((forward.detach() - tiny.detach()).cpu())),
            "standardized_consistency_canonical_gradient_l1": a_grad,
            "standardized_consistency_paraphrase_gradient_l1": b_grad,
            "standardized_consistency_gradients_nonzero": a_grad > 0.0 and b_grad > 0.0,
            "standardized_consistency_flat_loss": float(flat.detach().cpu()),
            "standardized_consistency_flat_zero": bool(
                torch.equal(flat_za, torch.zeros_like(flat_za))
                and torch.equal(flat_zb, torch.zeros_like(flat_zb))
            ),
            "standardized_consistency_flat_finite": bool(torch.isfinite(flat)),
            "standardized_consistency_flat_rate": 0.5 * (
                flat_diag.canonical_flat_rate + flat_diag.paraphrase_flat_rate
            ),
            "standardized_consistency_parameter_count": 0,
        }


def _a0_rows(suite: tuple[Case, ...]) -> list[S20FusionCase]:
    rows: list[S20FusionCase] = []
    for case in suite:
        options, ga, gb = case.option_pack()
        rows.append(
            S20FusionCase(
                case_id=f"a0-s20-{case.case_id}",
                split="train",
                domain="s20_a0_only",
                language="en",
                state_a=case.state_a,
                state_b=case.state_b,
                question_a1=case.qa1,
                question_a2=case.qa2,
                question_b1=case.qb1,
                question_b2=case.qb2,
                option_texts=tuple(option.criterion_text for option in options),
                option_aliases=tuple(option.aliases[0] for option in options),
                option_ids=tuple(option.option_id for option in options),
                gold_a=ga,
                gold_b=gb,
            )
        )
    return rows


def _real_shared_gradient_probe(bundle: Path, manifest: dict, suite: tuple[Case, ...]) -> dict:
    with torch.inference_mode(False), torch.enable_grad():
        fresh = load_hira_v0_m4_bundle(bundle)
        runtime = build_hira_v1_s20_standardized_triadic_core(
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
        if sum(p.numel() for p in trainable) != HIRA_V1_S20_TOTAL_PARAMETER_COUNT:
            raise RuntimeError("S20-A0 real-gradient physical surface changed")

        (
            _total,
            primary_block,
            relation_block,
            _pieces,
            *_rest,
        ) = _s20_losses(runtime, _a0_rows(suite))

        primary_raw = torch.autograd.grad(
            primary_block,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        relation_raw = torch.autograd.grad(
            relation_block,
            trainable,
            allow_unused=True,
        )
        primary = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(trainable, primary_raw)
        ]
        relation = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(trainable, relation_raw)
        ]
        _combined, diag = norm_balanced_gradient_update(
            primary,
            relation,
            epsilon=1e-12,
        )
        payload = diag.to_dict()
        payload["gradient_cosine"] = diag.normalized_pre_dot
        payload["surface_parameter_count"] = sum(p.numel() for p in trainable)
        payload["semantic_case_count"] = len(suite)
        payload["balance_epsilon"] = 1e-12
        if diag.primary_norm > 0.0 and diag.relation_norm > 0.0:
            payload["raw_norm_ratio_primary_over_relation"] = (
                diag.primary_norm / diag.relation_norm
            )
        else:
            payload["raw_norm_ratio_primary_over_relation"] = None
        return payload


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S20-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder
    encoder.eval()

    bank = _text_bank(suite)
    before_batch = encoder.encode_texts(bank)
    token_before = before_batch.token_embeddings.detach().cpu().clone()
    pooled_before = before_batch.pooled_embeddings.detach().cpu().clone()

    baseline = build_hira_v1_s3_parameter_free_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
    )
    base = _collect_decisions(baseline, suite, "parameter_free_triadic")

    runtime = build_hira_v1_s20_standardized_triadic_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    runtime.encoder.eval()

    after_batch = runtime.encoder.encode_texts(bank)
    token_identity = torch.equal(
        after_batch.token_embeddings.detach().cpu(),
        token_before,
    )
    pooled_identity = torch.equal(
        after_batch.pooled_embeddings.detach().cpu(),
        pooled_before,
    )
    if not token_identity or not pooled_identity:
        raise RuntimeError("S20-A0 A13 identity changed")

    observed = _collect_decisions(runtime, suite, "projection_triadic")
    fusion_result = _collect_fusion(runtime, suite)
    shared_gradient_probe = _real_shared_gradient_probe(bundle, manifest, suite)

    exact_logits = 0
    exact_choices = 0
    for key, (base_logits, base_choice) in base["records"].items():
        logits, choice = observed["records"][key]
        same_logits = torch.equal(base_logits, logits)
        same_choice = base_choice == choice
        exact_logits += int(same_logits)
        exact_choices += int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S20-A0 primary decision identity changed: {key}")

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S20-A0 projection scorer missing")
    physical = lora_params + scorer.projection_parameter_count
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )

    canonicalizer_probe = CrossViewRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )
    fusion_probe = GradientIsolatedFullKEvidenceFusion(epsilon=FUSION_EPSILON)

    if lora_params != 16_384:
        raise RuntimeError("S20-A0 LoRA capacity changed")
    if scorer.projection_parameter_count != HIRA_V1_S20_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S20-A0 projection capacity changed")
    if physical != HIRA_V1_S20_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S20-A0 total capacity changed")
    if canonicalizer_probe.parameter_count != 0:
        raise RuntimeError("S20-A0 canonicalizer unexpectedly adds parameters")
    if fusion_probe.parameter_count != 0:
        raise RuntimeError("S20-A0 fusion unexpectedly adds parameters")
    if trainable != 0 or original_a13 != 0:
        raise RuntimeError("S20-A0 must be fully frozen")
    if observed["state_encode_calls"] != len(suite) * 2:
        raise RuntimeError("S20-A0 decision state-once changed")
    if observed["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S20-A0 primary option permutation changed")
    if observed["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S20-A0 probability mass changed")

    gradient_probe = _gradient_route_probe()
    if gradient_probe["s20_vs_s14_forward_max_abs"] != 0.0:
        raise RuntimeError("S20 inference changed from S14")
    if not gradient_probe["primary_triadic_gradient_nonzero"]:
        raise RuntimeError("S20 primary gradient no longer reaches triadic logits")
    if not gradient_probe["primary_relation_direct_gradient_zero"]:
        raise RuntimeError("S20 primary gradient leaked directly into relation logits")
    if not gradient_probe["relation_auxiliary_gradient_nonzero"]:
        raise RuntimeError("S20 relation auxiliary no longer trains relation logits")

    standardized_probe = _standardized_consistency_probe()
    if standardized_probe["standardized_consistency_identical"] != 0.0:
        raise RuntimeError("S20 standardized identical-view invariant failed")
    if standardized_probe["standardized_consistency_mismatch"] <= 0.0:
        raise RuntimeError("S20 standardized mismatch probe is not positive")
    if standardized_probe["standardized_consistency_swap_max_abs"] > 1e-7:
        raise RuntimeError("S20 standardized symmetry changed")
    if standardized_probe["standardized_consistency_permutation_max_abs"] > 1e-7:
        raise RuntimeError("S20 standardized option permutation changed")
    if standardized_probe["standardized_consistency_affine_max_abs"] > 2e-6:
        raise RuntimeError("S20 standardized affine invariance changed")
    if standardized_probe["standardized_consistency_scale_max_abs"] > 2e-6:
        raise RuntimeError("S20 standardized scaling invariance changed")
    if not standardized_probe["standardized_consistency_gradients_nonzero"]:
        raise RuntimeError("S20 standardized consistency no longer trains both views")
    if standardized_probe["standardized_consistency_flat_loss"] != 0.0:
        raise RuntimeError("S20 flat neutral loss changed")
    if not standardized_probe["standardized_consistency_flat_zero"]:
        raise RuntimeError("S20 flat neutral evidence changed")
    if not standardized_probe["standardized_consistency_flat_finite"]:
        raise RuntimeError("S20 flat neutral evidence became non-finite")

    total_decisions = len(suite) * 4
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S20_A0_STANDARDIZED_TRIADIC_CONSISTENCY_ONLY",
        "semantic_case_count": len(suite),
        "decision_count": total_decisions,
        "state_view_count": len(suite) * 2,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "a13_token_output_identity": token_identity,
        "a13_pooled_output_identity": pooled_identity,
        "exact_logit_identity_rate": exact_logits / total_decisions,
        "exact_choice_identity_rate": exact_choices / total_decisions,
        "accuracy_all_views": observed["accuracy_all_views"],
        "canonical_paired_both_correct_rate": observed["canonical_paired_both_correct_rate"],
        "cross_view_selected_choice_agreement": observed["cross_view_selected_choice_agreement"],
        "cross_view_mean_js": observed["cross_view_mean_js"],
        **fusion_result,
        "option_order_flip_rate": observed["option_order_flip_rate"],
        "state_encode_calls": observed["state_encode_calls"],
        "max_probability_mass_error": observed["max_probability_mass_error"],
        "lora_parameter_count": lora_params,
        "projection_parameter_count": scorer.projection_parameter_count,
        "candidate_parameter_count": physical,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "canonicalizer_added_parameter_count": canonicalizer_probe.parameter_count,
        "fusion_added_parameter_count": fusion_probe.parameter_count,
        "role_temperature": ROLE_TEMPERATURE,
        "pair_temperature": PAIR_TEMPERATURE,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        "fusion_epsilon": FUSION_EPSILON,
        **gradient_probe,
        "shared_surface_gradient_probe": shared_gradient_probe,
        **standardized_probe,
        "standardized_triadic_consistency_coefficient": 0.25,
        "standardized_triadic_consistency_epsilon": FUSION_EPSILON,
        "full_k": True,
        "relation_refinement": False,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S20_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
