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
from nmd.v1_margin import (
    gold_vs_max_wrong_margin,
    margin_satisfaction_rate,
    top1_top2_margin,
)
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core
from nmd.v1_s9_semantic_core import (
    HIRA_V1_S9_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S9_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s9_margin_core,
)


SCHEMA_VERSION = "hira-v1-s9-a0-identity-v1"
OUTCOME = "HIRA_V1_S9_A0_IDENTITY_READY"
MARGIN = 0.20


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
                criterion_text=f"for this S9-A0 item the answer value is {value}",
                aliases=(f"{value} is the matching S9-A0 semantic value",),
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
        f"S9-A0 {noun} note {cid} records {first_label} {first} "
        f"and {second_label} {second}."
    )
    state_b = (
        f"In {cid}, {second} appears in the {second_label} field; "
        f"the same {noun} note gives {first_label} as {first}."
    )
    qa1 = f"What {first_label} is recorded on S9-A0 {noun} {cid}?"
    qa2 = f"Which value occupies the {first_label} field for {cid}?"
    qb1 = f"What {second_label} is recorded on S9-A0 {noun} {cid}?"
    qb2 = f"Which value occupies the {second_label} field for {cid}?"
    return Case(
        cid, state_a, state_b, qa1, qa2, qb1, qb2,
        (first, second, x, y),
    )


def cases() -> tuple[Case, ...]:
    return (
        _case("MX11", "harbor crane", "hook type", "ramshorn", "lift limit", "42 tonnes", "single hook", "28 tonnes"),
        _case("MX22", "coffee roaster", "roast profile", "city plus", "drum speed", "54 rpm", "full city", "38 rpm"),
        _case("MX33", "planetarium projector", "lens set", "fisheye", "lamp current", "18 A", "rectilinear", "11 A"),
        _case("MX44", "cheese cave", "culture", "Penicillium roqueforti", "aging humidity", "92 percent", "Geotrichum", "78 percent"),
        _case("MX55", "freight elevator", "drive type", "traction", "rated load", "3200 kg", "hydraulic", "1800 kg"),
        _case("MX66", "river gauge", "sensor mode", "pressure", "sample interval", "45 seconds", "ultrasonic", "90 seconds"),
        _case("MX77", "solar kiln", "vent mode", "passive", "drying target", "14 percent", "forced", "9 percent"),
        _case("MX88", "wave tank", "generator type", "piston", "wave period", "2.8 s", "flap", "1.6 s"),
        _case("MX99", "orchard sprayer", "nozzle type", "hollow cone", "tank volume", "650 L", "flat fan", "420 L"),
        _case("NX10", "air sampler", "filter type", "PTFE", "flow rate", "16 L/min", "quartz", "9 L/min"),
        _case("NX21", "brick press", "binder", "lignosulfonate", "compaction time", "22 s", "starch", "14 s"),
        _case("NX32", "coral nursery", "frame type", "tree frame", "depth", "9 m", "table frame", "5 m"),
        _case("NX43", "flour mill", "sieve grade", "180 micron", "feed rate", "32 kg/min", "250 micron", "21 kg/min"),
        _case("NX54", "geothermal loop", "fluid", "propylene glycol", "return temperature", "41 C", "ethylene glycol", "29 C"),
        _case("NX65", "honey extractor", "basket type", "radial", "spin rate", "310 rpm", "tangential", "190 rpm"),
        _case("NX76", "ice rink", "refrigerant", "ammonia", "brine temperature", "-9 C", "CO2", "-5 C"),
    )


def _text_bank(suite: tuple[Case, ...]) -> tuple[str, ...]:
    rows: list[str] = []
    for case in suite:
        rows.extend((case.state_a, case.state_b, case.qa1, case.qa2, case.qb1, case.qb2))
        rows.extend(f"for this S9-A0 item the answer value is {v}" for v in case.values)
        rows.extend(f"{v} is the matching S9-A0 semantic value" for v in case.values)
    return tuple(rows)


@torch.inference_mode()
def _collect(runtime, suite: tuple[Case, ...], mode: str) -> dict:
    records = {}
    before = runtime.state_encode_calls
    correct = 0
    canonical_pair_both = 0
    flips = 0
    max_mass = 0.0
    canonical_logits = []
    canonical_gold = []
    paraphrase_logits = []
    paraphrase_gold = []

    for case in suite:
        options = case.options()
        canonical_ok = []
        per_view = {"canonical": [], "paraphrase": []}

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
                if view == "canonical":
                    canonical_ok.append(ok)
                per_view[view].append((out.logits.detach().cpu().clone(), gold))
                records[(case.case_id, f"{view}-{label}")] = (
                    out.logits.detach().cpu().clone(),
                    out.selected_option_id,
                )
                max_mass = max(max_mass, abs(float(out.probabilities.sum()) - 1.0))
                if int(out.hira.candidate_budget.item()) != 4:
                    raise RuntimeError("S9-A0 full-K changed")
                if not torch.equal(
                    out.hira.relation_delta,
                    torch.zeros_like(out.hira.relation_delta),
                ):
                    raise RuntimeError("S9-A0 relation delta changed")

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

        canonical_pair_both += int(all(canonical_ok))
        for logits, gold in per_view["canonical"]:
            canonical_logits.append(logits)
            canonical_gold.append(gold)
        for logits, gold in per_view["paraphrase"]:
            paraphrase_logits.append(logits)
            paraphrase_gold.append(gold)

    c = torch.stack(canonical_logits)
    p = torch.stack(paraphrase_logits)
    cg = torch.tensor(canonical_gold, dtype=torch.long)
    pg = torch.tensor(paraphrase_gold, dtype=torch.long)

    c_signed = gold_vs_max_wrong_margin(c, cg)
    p_signed = gold_vs_max_wrong_margin(p, pg)
    c_top = top1_top2_margin(c)
    p_top = top1_top2_margin(p)

    return {
        "records": records,
        "accuracy_all_views": correct / (len(suite) * 4),
        "canonical_paired_both_correct_rate": canonical_pair_both / len(suite),
        "cross_view_selected_choice_agreement": float(selected_choice_agreement(c, p)),
        "cross_view_mean_js": float(symmetric_js_divergence(c, p)),
        "canonical_mean_gold_vs_max_wrong_margin": float(c_signed.mean()),
        "paraphrase_mean_gold_vs_max_wrong_margin": float(p_signed.mean()),
        "canonical_margin_satisfaction_rate": float(
            margin_satisfaction_rate(c, cg, margin=MARGIN)
        ),
        "paraphrase_margin_satisfaction_rate": float(
            margin_satisfaction_rate(p, pg, margin=MARGIN)
        ),
        "canonical_mean_top1_top2_margin": float(c_top.mean()),
        "paraphrase_mean_top1_top2_margin": float(p_top.mean()),
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
        raise RuntimeError("S9-A0 suite size changed")

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

    runtime = build_hira_v1_s9_margin_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    runtime.encoder.eval()

    after_batch = runtime.encoder.encode_texts(bank)
    token_identity = torch.equal(after_batch.token_embeddings.detach().cpu(), token_before)
    pooled_identity = torch.equal(after_batch.pooled_embeddings.detach().cpu(), pooled_before)
    if not token_identity or not pooled_identity:
        raise RuntimeError("S9-A0 A13 identity changed")

    observed = _collect(runtime, suite, "projection_triadic")

    exact_logits = 0
    exact_choices = 0
    for key, (base_logits, base_choice) in base["records"].items():
        logits, choice = observed["records"][key]
        same_logits = torch.equal(base_logits, logits)
        same_choice = base_choice == choice
        exact_logits += int(same_logits)
        exact_choices += int(same_choice)
        if not same_logits or not same_choice:
            raise RuntimeError(f"S9-A0 decision identity changed: {key}")

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S9-A0 projection scorer missing")

    physical = lora_params + scorer.projection_parameter_count
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )

    if lora_params != 16_384:
        raise RuntimeError("S9-A0 LoRA capacity changed")
    if scorer.projection_parameter_count != HIRA_V1_S9_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S9-A0 projection capacity changed")
    if physical != HIRA_V1_S9_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S9-A0 candidate capacity changed")
    if trainable != 0 or original_a13 != 0:
        raise RuntimeError("S9-A0 must be fully frozen")
    if observed["state_encode_calls"] != len(suite) * 2:
        raise RuntimeError("S9-A0 state-once per wording view changed")
    if observed["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S9-A0 option permutation changed")
    if observed["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S9-A0 probability mass changed")

    total_decisions = len(suite) * 4
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S9_A0_IDENTITY_LOCALIZATION_ONLY",
        "semantic_case_count": len(suite),
        "decision_count": total_decisions,
        "state_view_count": len(suite) * 2,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "margin": MARGIN,
        "a13_token_output_identity": token_identity,
        "a13_pooled_output_identity": pooled_identity,
        "exact_logit_identity_rate": exact_logits / total_decisions,
        "exact_choice_identity_rate": exact_choices / total_decisions,
        "accuracy_all_views": observed["accuracy_all_views"],
        "canonical_paired_both_correct_rate": observed["canonical_paired_both_correct_rate"],
        "cross_view_selected_choice_agreement": observed["cross_view_selected_choice_agreement"],
        "cross_view_mean_js": observed["cross_view_mean_js"],
        "canonical_mean_gold_vs_max_wrong_margin": observed["canonical_mean_gold_vs_max_wrong_margin"],
        "paraphrase_mean_gold_vs_max_wrong_margin": observed["paraphrase_mean_gold_vs_max_wrong_margin"],
        "canonical_margin_satisfaction_rate": observed["canonical_margin_satisfaction_rate"],
        "paraphrase_margin_satisfaction_rate": observed["paraphrase_margin_satisfaction_rate"],
        "canonical_mean_top1_top2_margin": observed["canonical_mean_top1_top2_margin"],
        "paraphrase_mean_top1_top2_margin": observed["paraphrase_mean_top1_top2_margin"],
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
    print("HIRA_V1_S9_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
