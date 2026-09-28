from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_invariance import selected_choice_agreement, symmetric_js_divergence
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core
from nmd.v1_s8_semantic_core import (
    HIRA_V1_S8_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S8_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s8_invariant_core,
)


SCHEMA_VERSION = "hira-v1-s8-a0-identity-v1"
OUTCOME = "HIRA_V1_S8_A0_IDENTITY_READY"


@dataclass(frozen=True)
class Case:
    case_id: str
    state_a: str
    state_b: str
    qa1: str
    qa2: str
    qb1: str
    qb2: str
    values: tuple[str, str, str, str]
    gold_a: int = 0
    gold_b: int = 1

    def options(self) -> tuple[LogicalOption, ...]:
        return tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=f"the answer value is {value}",
                aliases=(f"{value} is the equivalent semantic option",),
            )
            for i, value in enumerate(self.values)
        )


def _case(
    cid: str,
    noun: str,
    first_label: str,
    first: str,
    second_label: str,
    second: str,
    x: str,
    y: str,
) -> Case:
    state_a = (
        f"S8-A0 {noun} card {cid} records {first_label} {first} "
        f"and {second_label} {second}."
    )
    state_b = (
        f"For the {noun} item {cid}, {second} is entered under {second_label}; "
        f"the {first_label} field contains {first}."
    )
    qa1 = f"What {first_label} is recorded on S8-A0 {noun} {cid}?"
    qa2 = f"Which value fills the {first_label} field for {cid}?"
    qb1 = f"What {second_label} is recorded on S8-A0 {noun} {cid}?"
    qb2 = f"Which value fills the {second_label} field for {cid}?"
    return Case(
        cid,
        state_a,
        state_b,
        qa1,
        qa2,
        qb1,
        qb2,
        (first, second, x, y),
    )


def cases() -> tuple[Case, ...]:
    return (
        _case("AX11", "aquarium controller", "pump mode", "pulse", "salinity target", "33 ppt", "steady", "28 ppt"),
        _case("AX22", "book bindery", "thread type", "linen", "spine width", "24 mm", "nylon", "16 mm"),
        _case("AX33", "cider batch", "apple variety", "Dabinett", "fermentation time", "11 days", "Yarlington", "7 days"),
        _case("AX44", "diamond saw", "blade matrix", "bronze", "rim speed", "38 m/s", "resin", "22 m/s"),
        _case("AX55", "evacuation drill", "assembly zone", "Zone K", "alarm delay", "42 seconds", "Zone M", "18 seconds"),
        _case("AX66", "fish sonar", "transducer type", "split-beam", "ping rate", "12 Hz", "single-beam", "6 Hz"),
        _case("AX77", "grain silo", "grain type", "spelt", "aeration setpoint", "14 C", "oat", "21 C"),
        _case("AX88", "heat exchanger", "plate alloy", "titanium", "flow setting", "48 L/min", "nickel", "31 L/min"),
        _case("AX99", "inkjet line", "ink family", "pigment", "drop size", "7 pL", "dye", "12 pL"),
        _case("BX10", "jelly process", "gelling agent", "pectin", "cook duration", "18 min", "agar", "11 min"),
        _case("BX21", "laser cutter", "assist gas", "nitrogen", "focus offset", "1.4 mm", "oxygen", "0.8 mm"),
        _case("BX32", "mushroom room", "substrate", "straw", "humidity setpoint", "88 percent", "sawdust", "74 percent"),
        _case("BX43", "navigation buoy", "light color", "amber", "flash cycle", "9 seconds", "green", "15 seconds"),
        _case("BX54", "optical bench", "mirror coating", "silver", "beam height", "125 mm", "aluminum", "85 mm"),
        _case("BX65", "paper mill", "pulp source", "eucalyptus", "basis weight", "82 gsm", "spruce", "64 gsm"),
        _case("BX76", "quartz furnace", "crucible type", "fused silica", "hold temperature", "1180 C", "alumina", "940 C"),
    )


def _text_bank(suite: tuple[Case, ...]) -> tuple[str, ...]:
    rows: list[str] = []
    for case in suite:
        rows.extend((
            case.state_a,
            case.state_b,
            case.qa1,
            case.qa2,
            case.qb1,
            case.qb2,
        ))
        rows.extend(f"the answer value is {value}" for value in case.values)
        rows.extend(f"{value} is the equivalent semantic option" for value in case.values)
    return tuple(rows)


@torch.inference_mode()
def _collect(runtime, suite: tuple[Case, ...], mode: str) -> dict:
    records: dict[tuple[str, str], tuple[torch.Tensor, str, bool]] = {}
    before = runtime.state_encode_calls
    correct = 0
    pair_both = 0
    flips = 0
    max_mass = 0.0
    cross_logits_a = []
    cross_logits_b = []

    for case in suite:
        options = case.options()
        view_correct = {"canonical": [], "paraphrase": []}
        view_logits: dict[str, list[torch.Tensor]] = {"canonical": [], "paraphrase": []}

        for view, state, qa, qb in (
            ("canonical", case.state_a, case.qa1, case.qb1),
            ("paraphrase", case.state_b, case.qa2, case.qb2),
        ):
            memory = runtime.compile_state(state)
            for label, question, gold in (
                ("a", qa, case.gold_a),
                ("b", qb, case.gold_b),
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
                view_correct[view].append(ok)
                view_logits[view].append(out.logits.detach().cpu().clone())
                records[(case.case_id, f"{view}-{label}")] = (
                    out.logits.detach().cpu().clone(),
                    out.selected_option_id,
                    ok,
                )
                max_mass = max(
                    max_mass,
                    abs(float(out.probabilities.sum()) - 1.0),
                )
                if int(out.hira.candidate_budget.item()) != 4:
                    raise RuntimeError("S8-A0 full-K changed")
                if not torch.equal(
                    out.hira.relation_delta,
                    torch.zeros_like(out.hira.relation_delta),
                ):
                    raise RuntimeError("S8-A0 relation delta changed")

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

        pair_both += int(all(view_correct["canonical"]))
        cross_logits_a.extend(view_logits["canonical"])
        cross_logits_b.extend(view_logits["paraphrase"])

    a = torch.stack(cross_logits_a)
    b = torch.stack(cross_logits_b)
    return {
        "records": records,
        "accuracy_all_views": correct / (len(suite) * 4),
        "canonical_paired_both_correct_rate": pair_both / len(suite),
        "cross_view_selected_choice_agreement": float(selected_choice_agreement(a, b)),
        "cross_view_mean_js": float(symmetric_js_divergence(a, b)),
        "option_order_flip_rate": flips / (len(suite) * 4),
        "state_encode_calls": runtime.state_encode_calls - before,
        "max_probability_mass_error": max_mass,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S8-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder
    encoder.eval()

    bank = _text_bank(suite)
    base_batch = encoder.encode_texts(bank)
    token_before = base_batch.token_embeddings.detach().cpu().clone()
    pooled_before = base_batch.pooled_embeddings.detach().cpu().clone()

    baseline = build_hira_v1_s3_parameter_free_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
    )
    base = _collect(baseline, suite, "parameter_free_triadic")

    runtime = build_hira_v1_s8_invariant_core(
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
        raise RuntimeError("S8-A0 A13 identity changed")

    observed = _collect(runtime, suite, "projection_triadic")

    exact_logits = 0
    exact_choices = 0
    for key, (base_logits, base_choice, _ok) in base["records"].items():
        logits, choice, _ = observed["records"][key]
        same_logits = torch.equal(base_logits, logits)
        same_choice = base_choice == choice
        exact_logits += int(same_logits)
        exact_choices += int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S8-A0 decision identity changed: {key}")

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S8-A0 projection scorer missing")

    physical = lora_params + scorer.projection_parameter_count
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )

    if lora_params != 16_384:
        raise RuntimeError("S8-A0 LoRA capacity changed")
    if scorer.projection_parameter_count != HIRA_V1_S8_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S8-A0 projection capacity changed")
    if physical != HIRA_V1_S8_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S8-A0 candidate capacity changed")
    if trainable != 0 or original_a13 != 0:
        raise RuntimeError("S8-A0 must be fully frozen")
    if observed["state_encode_calls"] != len(suite) * 2:
        raise RuntimeError("S8-A0 state-once per wording view changed")
    if observed["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S8-A0 option permutation changed")
    if observed["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S8-A0 probability mass changed")

    total_decisions = len(suite) * 4
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S8_A0_IDENTITY_LOCALIZATION_ONLY",
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
        "option_order_flip_rate": observed["option_order_flip_rate"],
        "state_encode_calls": observed["state_encode_calls"],
        "max_probability_mass_error": observed["max_probability_mass_error"],
        "lora_parameter_count": lora_params,
        "projection_parameter_count": scorer.projection_parameter_count,
        "candidate_parameter_count": physical,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
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
    print("HIRA_V1_S8_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
