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
from nmd.v1_relation_canonicalization import (
    CrossViewRelationCanonicalizer,
    relation_signature_same_option_cosine,
)
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core
from nmd.v1_s13_semantic_core import (
    HIRA_V1_S13_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S13_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s13_canonicalization_core,
)


SCHEMA_VERSION = "hira-v1-s13-a0-identity-v1"
OUTCOME = "HIRA_V1_S13_A0_IDENTITY_READY"
ROLE_TEMPERATURE = 0.10
PAIR_TEMPERATURE = 0.10
CONTRASTIVE_TEMPERATURE = 0.10


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
            f"S13-A0 {self.noun} sheet {self.case_id} records "
            f"{self.field_a} as {self.first}, while {self.field_b} is {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"For {self.case_id}, {self.second} appears under {self.field_b}. "
            f"The same S13-A0 {self.noun} sheet places {self.first} under {self.field_a}."
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
        Case("TA11", "cryogenic pump", "bearing type", "magnetic", "shaft speed", "7200 rpm", "ceramic", "3600 rpm", 20201),
        Case("TA22", "terahertz imager", "detector type", "bolometer", "frame rate", "48 fps", "Schottky", "18 fps", 20202),
        Case("TA33", "electrospinning rig", "polymer", "PCL", "collector speed", "1400 rpm", "PLA", "700 rpm", 20203),
        Case("TA44", "quantum optics rack", "source type", "SPDC", "coincidence window", "2 ns", "attenuated laser", "8 ns", 20204),
        Case("TA55", "wave tank", "waveform", "JONSWAP", "peak period", "7 s", "Pierson-Moskowitz", "4 s", 20205),
        Case("TA66", "gas chromatograph", "column phase", "DB-5ms", "oven ramp", "12 C/min", "wax", "5 C/min", 20206),
        Case("TA77", "laser welder", "beam mode", "single-mode", "travel speed", "22 mm/s", "multimode", "10 mm/s", 20207),
        Case("TA88", "neutron detector", "converter", "boron-10", "bias voltage", "680 V", "lithium-6", "320 V", 20208),
        Case("TA99", "district heat node", "heat exchanger", "plate", "supply setpoint", "82 C", "shell-tube", "64 C", 20209),
        Case("TB10", "microscope stage", "drive type", "piezo", "step size", "20 nm", "stepper", "200 nm", 20210),
        Case("TB21", "ocean glider", "buoyancy fluid", "silicone oil", "dive depth", "900 m", "mineral oil", "400 m", 20211),
        Case("TB32", "cell sorter", "laser line", "488 nm", "event rate", "28000/s", "405 nm", "9000/s", 20212),
        Case("TB43", "thermal vacuum rig", "shroud mode", "liquid nitrogen", "base pressure", "2e-6 mbar", "water cooling", "8e-4 mbar", 20213),
        Case("TB54", "satellite reaction wheel", "rotor material", "titanium", "momentum capacity", "18 Nms", "aluminum", "7 Nms", 20214),
        Case("TB65", "acoustic chamber", "absorber type", "melamine wedge", "cutoff frequency", "80 Hz", "foam panel", "250 Hz", 20215),
        Case("TB76", "microfluidic sorter", "channel coating", "PEG", "flow rate", "35 uL/min", "PDMS", "12 uL/min", 20216),
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
                    raise RuntimeError("S13-A0 full-K changed")
                if not torch.equal(
                    out.hira.relation_delta,
                    torch.zeros_like(out.hira.relation_delta),
                ):
                    raise RuntimeError("S13-A0 relation delta changed")

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
def _collect_canonicalization(runtime, suite: tuple[Case, ...]) -> dict:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S13-A0 projection scorer missing")

    canonicalizer = CrossViewRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )
    canonical_logits = []
    paraphrase_logits = []
    canonical_signatures = []
    paraphrase_signatures = []
    canonical_gold = []
    paraphrase_gold = []
    diagnostics = {
        "canonical_state_entropy": [],
        "paraphrase_state_entropy": [],
        "canonical_state_max": [],
        "paraphrase_state_max": [],
        "canonical_option_entropy": [],
        "paraphrase_option_entropy": [],
        "canonical_option_max": [],
        "paraphrase_option_max": [],
        "canonical_pair_entropy": [],
        "paraphrase_pair_entropy": [],
        "canonical_pair": [],
        "paraphrase_pair": [],
    }
    signature_permutation_max_abs = 0.0
    logit_permutation_max_abs = 0.0
    reverse_index = torch.tensor([3, 2, 1, 0], dtype=torch.long)

    for case in suite:
        options, ga, gb = case.option_pack()
        for view, state, qa, qb in (
            ("canonical", case.state_a, case.qa1, case.qb1),
            ("paraphrase", case.state_b, case.qa2, case.qb2),
        ):
            memory = runtime.compile_state(state)
            state_tokens = memory.content_token_embeddings
            if state_tokens is None:
                raise RuntimeError("S13-A0 state token artifacts missing")
            state_tokens = state_tokens.unsqueeze(0)
            state_mask = torch.ones(
                1,
                state_tokens.shape[1],
                dtype=torch.bool,
                device=state_tokens.device,
            )

            for question, gold in ((qa, ga), (qb, gb)):
                schema, _ = runtime.compile_schema(
                    primitive="choice",
                    question_text=question,
                    options=options,
                    include_token_artifacts=True,
                    use_cache=False,
                )
                q = schema.question_token_embeddings
                qm = schema.question_content_token_mask
                ov = schema.option_view_token_embeddings
                ovtm = schema.option_view_token_mask
                ovm = schema.option_view_mask
                if any(x is None for x in (q, qm, ov, ovtm, ovm)):
                    raise RuntimeError("S13-A0 schema token artifacts missing")

                logits, signatures, diag = canonicalizer(
                    projection=scorer.projection,
                    state_tokens=state_tokens,
                    state_mask=state_mask,
                    question_tokens=q.unsqueeze(0),
                    question_mask=qm.unsqueeze(0),
                    option_view_tokens=ov.unsqueeze(0),
                    option_view_token_mask=ovtm.unsqueeze(0),
                    option_view_mask=ovm.unsqueeze(0),
                )

                if view == "canonical":
                    canonical_logits.append(logits[0].detach().cpu())
                    canonical_signatures.append(signatures[0].detach().cpu())
                    canonical_gold.append(gold)
                else:
                    paraphrase_logits.append(logits[0].detach().cpu())
                    paraphrase_signatures.append(signatures[0].detach().cpu())
                    paraphrase_gold.append(gold)

                prefix = "canonical" if view == "canonical" else "paraphrase"
                diagnostics[f"{prefix}_state_entropy"].append(
                    float(diag.state_role_normalized_entropy)
                )
                diagnostics[f"{prefix}_state_max"].append(
                    float(diag.state_role_max_weight)
                )
                diagnostics[f"{prefix}_option_entropy"].append(
                    float(diag.option_role_normalized_entropy)
                )
                diagnostics[f"{prefix}_option_max"].append(
                    float(diag.option_role_max_weight)
                )
                diagnostics[f"{prefix}_pair_entropy"].append(
                    float(diag.mean_pair_entropy)
                )
                diagnostics[f"{prefix}_pair"].append(
                    float(diag.mean_best_pair_score)
                )

                # The relation signature must transform exactly with option order.
                reverse_schema, _ = runtime.compile_schema(
                    primitive="choice",
                    question_text=question,
                    options=tuple(reversed(options)),
                    include_token_artifacts=True,
                    use_cache=False,
                )
                rq = reverse_schema.question_token_embeddings
                rqm = reverse_schema.question_content_token_mask
                rov = reverse_schema.option_view_token_embeddings
                rovtm = reverse_schema.option_view_token_mask
                rovm = reverse_schema.option_view_mask
                if any(x is None for x in (rq, rqm, rov, rovtm, rovm)):
                    raise RuntimeError("S13-A0 reverse schema token artifacts missing")
                rlogits, rsignatures, _ = canonicalizer(
                    projection=scorer.projection,
                    state_tokens=state_tokens,
                    state_mask=state_mask,
                    question_tokens=rq.unsqueeze(0),
                    question_mask=rqm.unsqueeze(0),
                    option_view_tokens=rov.unsqueeze(0),
                    option_view_token_mask=rovtm.unsqueeze(0),
                    option_view_mask=rovm.unsqueeze(0),
                )
                idx = reverse_index.to(rlogits.device)
                logit_permutation_max_abs = max(
                    logit_permutation_max_abs,
                    float((rlogits[:, idx] - logits).abs().max().cpu()),
                )
                signature_permutation_max_abs = max(
                    signature_permutation_max_abs,
                    float((rsignatures[:, idx] - signatures).abs().max().cpu()),
                )

    c = torch.stack(canonical_logits)
    p = torch.stack(paraphrase_logits)
    cs = torch.stack(canonical_signatures)
    ps = torch.stack(paraphrase_signatures)
    cg = torch.tensor(canonical_gold, dtype=torch.long)
    pg = torch.tensor(paraphrase_gold, dtype=torch.long)

    same_cos = relation_signature_same_option_cosine(cs, ps)
    signature_margin = _signature_margin(cs, ps)

    def avg(key: str) -> float:
        values = diagnostics[key]
        return sum(values) / len(values)

    return {
        "canonical_relation_binding_accuracy": float((c.argmax(-1) == cg).float().mean()),
        "paraphrase_relation_binding_accuracy": float((p.argmax(-1) == pg).float().mean()),
        "relation_binding_cross_view_selected_choice_agreement": float(
            (c.argmax(-1) == p.argmax(-1)).float().mean()
        ),
        "canonical_relation_binding_mean_gold_margin": float(_gold_margin(c, cg).mean()),
        "paraphrase_relation_binding_mean_gold_margin": float(_gold_margin(p, pg).mean()),
        "mean_same_option_signature_cosine": float(same_cos.mean()),
        "mean_signature_same_vs_strongest_wrong_margin": float(signature_margin.mean()),
        "minimum_same_option_signature_cosine": float(same_cos.min()),
        "signature_option_permutation_max_abs": signature_permutation_max_abs,
        "canonicalizer_logit_permutation_max_abs": logit_permutation_max_abs,
        "canonical_state_role_normalized_entropy": avg("canonical_state_entropy"),
        "paraphrase_state_role_normalized_entropy": avg("paraphrase_state_entropy"),
        "canonical_state_role_max_weight": avg("canonical_state_max"),
        "paraphrase_state_role_max_weight": avg("paraphrase_state_max"),
        "canonical_option_role_normalized_entropy": avg("canonical_option_entropy"),
        "paraphrase_option_role_normalized_entropy": avg("paraphrase_option_entropy"),
        "canonical_option_role_max_weight": avg("canonical_option_max"),
        "paraphrase_option_role_max_weight": avg("paraphrase_option_max"),
        "canonical_mean_pair_entropy": avg("canonical_pair_entropy"),
        "paraphrase_mean_pair_entropy": avg("paraphrase_pair_entropy"),
        "canonical_mean_best_pair_score": avg("canonical_pair"),
        "paraphrase_mean_best_pair_score": avg("paraphrase_pair"),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S13-A0 suite size changed")

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

    runtime = build_hira_v1_s13_canonicalization_core(
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
        raise RuntimeError("S13-A0 A13 identity changed")

    observed = _collect_decisions(runtime, suite, "projection_triadic")
    canonicalization = _collect_canonicalization(runtime, suite)

    exact_logits = 0
    exact_choices = 0
    for key, (base_logits, base_choice) in base["records"].items():
        logits, choice = observed["records"][key]
        same_logits = torch.equal(base_logits, logits)
        same_choice = base_choice == choice
        exact_logits += int(same_logits)
        exact_choices += int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S13-A0 primary decision identity changed: {key}")

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S13-A0 projection scorer missing")
    physical = lora_params + scorer.projection_parameter_count
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )

    probe = CrossViewRelationCanonicalizer(
        role_temperature=ROLE_TEMPERATURE,
        pair_temperature=PAIR_TEMPERATURE,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )

    if lora_params != 16_384:
        raise RuntimeError("S13-A0 LoRA capacity changed")
    if scorer.projection_parameter_count != HIRA_V1_S13_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S13-A0 projection capacity changed")
    if physical != HIRA_V1_S13_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S13-A0 total capacity changed")
    if probe.parameter_count != 0:
        raise RuntimeError("S13-A0 canonicalizer unexpectedly adds parameters")
    if trainable != 0 or original_a13 != 0:
        raise RuntimeError("S13-A0 must be fully frozen")
    if observed["state_encode_calls"] != len(suite) * 2:
        raise RuntimeError("S13-A0 decision state-once changed")
    if observed["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S13-A0 primary option permutation changed")
    if observed["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S13-A0 probability mass changed")

    total_decisions = len(suite) * 4
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S13_A0_CROSS_VIEW_RELATION_CANONICALIZATION_ONLY",
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
        **canonicalization,
        "option_order_flip_rate": observed["option_order_flip_rate"],
        "state_encode_calls": observed["state_encode_calls"],
        "max_probability_mass_error": observed["max_probability_mass_error"],
        "lora_parameter_count": lora_params,
        "projection_parameter_count": scorer.projection_parameter_count,
        "candidate_parameter_count": physical,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "canonicalizer_added_parameter_count": probe.parameter_count,
        "role_temperature": ROLE_TEMPERATURE,
        "pair_temperature": PAIR_TEMPERATURE,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
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
    print("HIRA_V1_S13_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
