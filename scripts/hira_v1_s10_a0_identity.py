from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_grounding import (
    QueryConditionedStateOptionGrounding,
    grounding_gold_vs_max_wrong_margin,
)
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core
from nmd.v1_s10_semantic_core import (
    HIRA_V1_S10_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S10_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s10_grounding_core,
)


SCHEMA_VERSION = "hira-v1-s10-a0-identity-v1"
OUTCOME = "HIRA_V1_S10_A0_IDENTITY_READY"
ATTENTION_TEMPERATURE = 0.10
GROUNDING_TEMPERATURE = 0.10


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

    @property
    def state_a(self) -> str:
        return (
            f"S10-A0 {self.noun} card {self.case_id} records "
            f"{self.field_a} {self.first} and {self.field_b} {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"In card {self.case_id}, {self.second} fills the {self.field_b} "
            f"field. The same S10-A0 {self.noun} card lists "
            f"{self.field_a} as {self.first}."
        )

    @property
    def qa1(self) -> str:
        return f"What {self.field_a} is recorded for {self.case_id}?"

    @property
    def qa2(self) -> str:
        return f"Which value occupies the {self.field_a} field on {self.case_id}?"

    @property
    def qb1(self) -> str:
        return f"What {self.field_b} is recorded for {self.case_id}?"

    @property
    def qb2(self) -> str:
        return f"Which value occupies the {self.field_b} field on {self.case_id}?"

    def options(self) -> tuple[LogicalOption, ...]:
        values = (self.first, self.second, self.wrong_a, self.wrong_b)
        fields = (self.field_a, self.field_b, self.field_a, self.field_b)
        return tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=(
                    f"for this S10-A0 {self.noun}, {fields[i]} equals {values[i]}"
                ),
                aliases=(
                    f"{values[i]} is the matching {fields[i]} entry "
                    f"for S10-A0 {self.case_id}",
                ),
            )
            for i in range(4)
        )


def cases() -> tuple[Case, ...]:
    return (
        Case("GA11", "brewery chiller", "coolant", "glycol", "setpoint", "3 C", "water", "8 C"),
        Case("GA22", "weather radar", "scan mode", "volume", "rotation rate", "6 rpm", "sector", "3 rpm"),
        Case("GA33", "aquarium pump", "impeller", "ceramic", "flow", "85 L/min", "polymer", "55 L/min"),
        Case("GA44", "theater winch", "cable type", "Dyneema", "load", "900 kg", "steel", "620 kg"),
        Case("GA55", "grain dryer", "fuel", "biogas", "airflow", "14 m3/s", "propane", "9 m3/s"),
        Case("GA66", "optical bench", "mirror coating", "silver", "beam height", "120 mm", "aluminum", "80 mm"),
        Case("GA77", "fish hatchery", "feed class", "starter", "ration", "4 percent", "grower", "7 percent"),
        Case("GA88", "stone saw", "blade", "diamond", "spindle speed", "1450 rpm", "carbide", "900 rpm"),
        Case("GA99", "seed cleaner", "screen", "2.4 mm", "fan speed", "680 rpm", "3.1 mm", "420 rpm"),
        Case("GB10", "tram depot", "charger", "pantograph", "power", "180 kW", "plug", "90 kW"),
        Case("GB21", "spectrometer", "detector", "InGaAs", "integration", "45 ms", "silicon", "20 ms"),
        Case("GB32", "food freezer", "refrigerant", "R744", "evaporator", "-31 C", "R134a", "-18 C"),
        Case("GB43", "paint booth", "filter", "bag", "air velocity", "0.45 m/s", "panel", "0.25 m/s"),
        Case("GB54", "power inverter", "switch", "SiC MOSFET", "carrier", "24 kHz", "IGBT", "12 kHz"),
        Case("GB65", "water jet", "orifice", "sapphire", "pressure", "310 MPa", "ruby", "220 MPa"),
        Case("GB76", "archive scanner", "illumination", "cross-polarized", "resolution", "720 dpi", "diffuse", "480 dpi"),
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
        for option in case.options():
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
        options = case.options()
        canonical_ok = []
        for view, state, qa, qb in (
            ("canonical", case.state_a, case.qa1, case.qb1),
            ("paraphrase", case.state_b, case.qa2, case.qb2),
        ):
            memory = runtime.compile_state(state)
            per_view = []
            for label, question, gold in (
                ("a", qa, 0),
                ("b", qb, 1),
            ):
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
                    raise RuntimeError("S10-A0 full-K changed")
                if not torch.equal(
                    out.hira.relation_delta,
                    torch.zeros_like(out.hira.relation_delta),
                ):
                    raise RuntimeError("S10-A0 relation delta changed")

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
def _collect_grounding(runtime, suite: tuple[Case, ...]) -> dict:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S10-A0 projection scorer missing")

    grounding = QueryConditionedStateOptionGrounding(
        attention_temperature=ATTENTION_TEMPERATURE,
        contrastive_temperature=GROUNDING_TEMPERATURE,
    )
    canonical_logits = []
    paraphrase_logits = []
    canonical_gold = []
    paraphrase_gold = []
    entropy = {"canonical": [], "paraphrase": []}
    concentration = {"canonical": [], "paraphrase": []}

    for case in suite:
        options = case.options()
        for view, state, qa, qb in (
            ("canonical", case.state_a, case.qa1, case.qb1),
            ("paraphrase", case.state_b, case.qa2, case.qb2),
        ):
            memory = runtime.compile_state(state)
            state_tokens = memory.content_token_embeddings
            if state_tokens is None:
                raise RuntimeError("S10-A0 state token artifacts missing")
            state_tokens = state_tokens.unsqueeze(0)
            state_mask = torch.ones(
                1,
                state_tokens.shape[1],
                dtype=torch.bool,
                device=state_tokens.device,
            )

            for question, gold in ((qa, 0), (qb, 1)):
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
                    raise RuntimeError("S10-A0 schema token artifacts missing")

                logits, diagnostics = grounding(
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
                    canonical_gold.append(gold)
                else:
                    paraphrase_logits.append(logits[0].detach().cpu())
                    paraphrase_gold.append(gold)
                entropy[view].append(float(diagnostics.normalized_attention_entropy))
                concentration[view].append(float(diagnostics.max_attention_weight))

    c = torch.stack(canonical_logits)
    p = torch.stack(paraphrase_logits)
    cg = torch.tensor(canonical_gold, dtype=torch.long)
    pg = torch.tensor(paraphrase_gold, dtype=torch.long)
    c_margin = grounding_gold_vs_max_wrong_margin(c, cg)
    p_margin = grounding_gold_vs_max_wrong_margin(p, pg)

    return {
        "canonical_grounding_accuracy": float((c.argmax(-1) == cg).float().mean()),
        "paraphrase_grounding_accuracy": float((p.argmax(-1) == pg).float().mean()),
        "grounding_cross_view_selected_choice_agreement": float(
            (c.argmax(-1) == p.argmax(-1)).float().mean()
        ),
        "canonical_grounding_mean_gold_margin": float(c_margin.mean()),
        "paraphrase_grounding_mean_gold_margin": float(p_margin.mean()),
        "canonical_attention_normalized_entropy": sum(entropy["canonical"]) / len(entropy["canonical"]),
        "paraphrase_attention_normalized_entropy": sum(entropy["paraphrase"]) / len(entropy["paraphrase"]),
        "canonical_attention_max_weight": sum(concentration["canonical"]) / len(concentration["canonical"]),
        "paraphrase_attention_max_weight": sum(concentration["paraphrase"]) / len(concentration["paraphrase"]),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S10-A0 suite size changed")

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

    runtime = build_hira_v1_s10_grounding_core(
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
        raise RuntimeError("S10-A0 A13 identity changed")

    observed = _collect_decisions(runtime, suite, "projection_triadic")
    grounding = _collect_grounding(runtime, suite)

    exact_logits = 0
    exact_choices = 0
    for key, (base_logits, base_choice) in base["records"].items():
        logits, choice = observed["records"][key]
        same_logits = torch.equal(base_logits, logits)
        same_choice = base_choice == choice
        exact_logits += int(same_logits)
        exact_choices += int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S10-A0 decision identity changed: {key}")

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S10-A0 projection scorer missing")
    physical = lora_params + scorer.projection_parameter_count
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )

    if lora_params != 16_384:
        raise RuntimeError("S10-A0 LoRA capacity changed")
    if scorer.projection_parameter_count != HIRA_V1_S10_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S10-A0 projection capacity changed")
    if physical != HIRA_V1_S10_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S10-A0 total capacity changed")
    if trainable != 0 or original_a13 != 0:
        raise RuntimeError("S10-A0 must be fully frozen")
    if observed["state_encode_calls"] != len(suite) * 2:
        raise RuntimeError("S10-A0 decision state-once changed")
    if observed["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S10-A0 option permutation changed")
    if observed["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S10-A0 probability mass changed")

    total_decisions = len(suite) * 4
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S10_A0_IDENTITY_LOCALIZATION_ONLY",
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
        **grounding,
        "option_order_flip_rate": observed["option_order_flip_rate"],
        "state_encode_calls": observed["state_encode_calls"],
        "max_probability_mass_error": observed["max_probability_mass_error"],
        "lora_parameter_count": lora_params,
        "projection_parameter_count": scorer.projection_parameter_count,
        "candidate_parameter_count": physical,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "grounding_added_parameter_count": 0,
        "attention_temperature": ATTENTION_TEMPERATURE,
        "grounding_temperature": GROUNDING_TEMPERATURE,
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
    print("HIRA_V1_S10_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
