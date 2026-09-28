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
from nmd.v1_relation_binding import (
    RelationStructuredRoleValueBinding,
    relation_binding_gold_vs_max_wrong_margin,
)
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core
from nmd.v1_s12_semantic_core import (
    HIRA_V1_S12_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S12_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s12_relation_binding_core,
)


SCHEMA_VERSION = "hira-v1-s12-a0-identity-v1"
OUTCOME = "HIRA_V1_S12_A0_IDENTITY_READY"
ROLE_TEMPERATURE = 0.10
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
            f"S12-A0 {self.noun} sheet {self.case_id} records "
            f"{self.field_a} as {self.first}, while {self.field_b} is {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"For {self.case_id}, {self.second} appears under {self.field_b}. "
            f"The same S12-A0 {self.noun} sheet places {self.first} under {self.field_a}."
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
        Case("SA11", "wave-energy converter", "control mode", "latching", "capture width", "18 m", "declutching", "11 m", 19201),
        Case("SA22", "ion thruster", "propellant", "xenon", "beam current", "3.4 A", "krypton", "1.8 A", 19202),
        Case("SA33", "vacuum furnace", "process gas", "argon", "soak temperature", "1180 C", "nitrogen", "940 C", 19203),
        Case("SA44", "photonics bench", "laser source", "DFB", "modulation rate", "25 GHz", "VCSEL", "10 GHz", 19204),
        Case("SA55", "algae reactor", "strain", "Nannochloropsis", "light intensity", "220 umol/m2/s", "Chlorella", "90 umol/m2/s", 19205),
        Case("SA66", "rail grinder", "stone grade", "A24", "feed speed", "1.6 m/s", "C36", "0.8 m/s", 19206),
        Case("SA77", "radar front-end", "waveform", "FMCW", "sweep bandwidth", "800 MHz", "pulse-Doppler", "250 MHz", 19207),
        Case("SA88", "heat-pump loop", "refrigerant", "R290", "condensing pressure", "14 bar", "R134a", "8 bar", 19208),
        Case("SA99", "powder mixer", "binder type", "PVA", "mix time", "18 min", "PEG", "9 min", 19209),
        Case("SB10", "lidar head", "wavelength", "1550 nm", "pulse rate", "400 kHz", "905 nm", "120 kHz", 19210),
        Case("SB21", "hydrogen stack", "membrane", "Nafion 212", "current density", "1.4 A/cm2", "Nafion 115", "0.7 A/cm2", 19211),
        Case("SB32", "wafer prober", "probe type", "MEMS vertical", "contact force", "18 gf", "cantilever", "7 gf", 19212),
        Case("SB43", "waveguide line", "dielectric", "PTFE", "center frequency", "77 GHz", "alumina", "24 GHz", 19213),
        Case("SB54", "microgrid feeder", "control law", "droop", "voltage setpoint", "410 V", "PQ control", "380 V", 19214),
        Case("SB65", "fuel injector", "nozzle type", "sac", "rail pressure", "1800 bar", "VCO", "900 bar", 19215),
        Case("SB76", "spectral imager", "grating", "1200 lines/mm", "slit width", "40 um", "600 lines/mm", "90 um", 19216),
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
                    raise RuntimeError("S12-A0 full-K changed")
                if not torch.equal(
                    out.hira.relation_delta,
                    torch.zeros_like(out.hira.relation_delta),
                ):
                    raise RuntimeError("S12-A0 relation delta changed")

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
def _collect_binding(runtime, suite: tuple[Case, ...]) -> dict:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S12-A0 projection scorer missing")

    binding = RelationStructuredRoleValueBinding(
        role_temperature=ROLE_TEMPERATURE,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )
    canonical_logits = []
    paraphrase_logits = []
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
        "canonical_pair": [],
        "paraphrase_pair": [],
    }

    for case in suite:
        options, ga, gb = case.option_pack()
        for view, state, qa, qb in (
            ("canonical", case.state_a, case.qa1, case.qb1),
            ("paraphrase", case.state_b, case.qa2, case.qb2),
        ):
            memory = runtime.compile_state(state)
            state_tokens = memory.content_token_embeddings
            if state_tokens is None:
                raise RuntimeError("S12-A0 state token artifacts missing")
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
                    raise RuntimeError("S12-A0 schema token artifacts missing")

                logits, diag = binding(
                    projection=scorer.projection,
                    state_tokens=state_tokens,
                    state_mask=state_mask,
                    question_tokens=q.unsqueeze(0),
                    question_mask=qm.unsqueeze(0),
                    option_view_tokens=ov.unsqueeze(0),
                    option_view_token_mask=ovtm.unsqueeze(0),
                    option_view_mask=ovm.unsqueeze(0),
                )
                prefix = "canonical" if view == "canonical" else "paraphrase"
                if view == "canonical":
                    canonical_logits.append(logits[0].detach().cpu())
                    canonical_gold.append(gold)
                else:
                    paraphrase_logits.append(logits[0].detach().cpu())
                    paraphrase_gold.append(gold)

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
                diagnostics[f"{prefix}_pair"].append(
                    float(diag.mean_best_pair_score)
                )

    c = torch.stack(canonical_logits)
    p = torch.stack(paraphrase_logits)
    cg = torch.tensor(canonical_gold, dtype=torch.long)
    pg = torch.tensor(paraphrase_gold, dtype=torch.long)
    c_margin = relation_binding_gold_vs_max_wrong_margin(c, cg)
    p_margin = relation_binding_gold_vs_max_wrong_margin(p, pg)

    def avg(key: str) -> float:
        values = diagnostics[key]
        return sum(values) / len(values)

    return {
        "canonical_relation_binding_accuracy": float((c.argmax(-1) == cg).float().mean()),
        "paraphrase_relation_binding_accuracy": float((p.argmax(-1) == pg).float().mean()),
        "relation_binding_cross_view_selected_choice_agreement": float(
            (c.argmax(-1) == p.argmax(-1)).float().mean()
        ),
        "canonical_relation_binding_mean_gold_margin": float(c_margin.mean()),
        "paraphrase_relation_binding_mean_gold_margin": float(p_margin.mean()),
        "canonical_state_role_normalized_entropy": avg("canonical_state_entropy"),
        "paraphrase_state_role_normalized_entropy": avg("paraphrase_state_entropy"),
        "canonical_state_role_max_weight": avg("canonical_state_max"),
        "paraphrase_state_role_max_weight": avg("paraphrase_state_max"),
        "canonical_option_role_normalized_entropy": avg("canonical_option_entropy"),
        "paraphrase_option_role_normalized_entropy": avg("paraphrase_option_entropy"),
        "canonical_option_role_max_weight": avg("canonical_option_max"),
        "paraphrase_option_role_max_weight": avg("paraphrase_option_max"),
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
        raise RuntimeError("S12-A0 suite size changed")

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

    runtime = build_hira_v1_s12_relation_binding_core(
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
        raise RuntimeError("S12-A0 A13 identity changed")

    observed = _collect_decisions(runtime, suite, "projection_triadic")
    relation_binding = _collect_binding(runtime, suite)

    exact_logits = 0
    exact_choices = 0
    for key, (base_logits, base_choice) in base["records"].items():
        logits, choice = observed["records"][key]
        same_logits = torch.equal(base_logits, logits)
        same_choice = base_choice == choice
        exact_logits += int(same_logits)
        exact_choices += int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S12-A0 primary decision identity changed: {key}")

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S12-A0 projection scorer missing")
    physical = lora_params + scorer.projection_parameter_count
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )

    probe = RelationStructuredRoleValueBinding(
        role_temperature=ROLE_TEMPERATURE,
        contrastive_temperature=CONTRASTIVE_TEMPERATURE,
    )

    if lora_params != 16_384:
        raise RuntimeError("S12-A0 LoRA capacity changed")
    if scorer.projection_parameter_count != HIRA_V1_S12_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S12-A0 projection capacity changed")
    if physical != HIRA_V1_S12_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S12-A0 total capacity changed")
    if probe.parameter_count != 0:
        raise RuntimeError("S12-A0 binding unexpectedly adds parameters")
    if trainable != 0 or original_a13 != 0:
        raise RuntimeError("S12-A0 must be fully frozen")
    if observed["state_encode_calls"] != len(suite) * 2:
        raise RuntimeError("S12-A0 decision state-once changed")
    if observed["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S12-A0 primary option permutation changed")
    if observed["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S12-A0 probability mass changed")

    total_decisions = len(suite) * 4
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S12_A0_ROLE_BINDING_IDENTITY_LOCALIZATION_ONLY",
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
        **relation_binding,
        "option_order_flip_rate": observed["option_order_flip_rate"],
        "state_encode_calls": observed["state_encode_calls"],
        "max_probability_mass_error": observed["max_probability_mass_error"],
        "lora_parameter_count": lora_params,
        "projection_parameter_count": scorer.projection_parameter_count,
        "candidate_parameter_count": physical,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "relation_binding_added_parameter_count": probe.parameter_count,
        "role_temperature": ROLE_TEMPERATURE,
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
    print("HIRA_V1_S12_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
