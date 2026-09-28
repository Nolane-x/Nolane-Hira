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
from nmd.v1_role_value_binding import (
    RoleValueFactorizedBinding,
    binding_gold_vs_max_wrong_margin,
)
from nmd.v1_s11_semantic_core import (
    HIRA_V1_S11_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S11_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s11_role_value_core,
)


SCHEMA_VERSION = "hira-v1-s11-a0-role-value-v1"
OUTCOME = "HIRA_V1_S11_A0_ROLE_VALUE_READY"
ROLE_TEMPERATURE = 0.10
STATE_TEMPERATURE = 0.10


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
            f"S11-A0 {self.noun} file {self.case_id} reports "
            f"{self.field_a} {self.first}; {self.field_b} {self.second}."
        )

    @property
    def state_b(self) -> str:
        return (
            f"For {self.case_id}, {self.second} appears under {self.field_b}. "
            f"The same {self.noun} file records {self.first} for {self.field_a}."
        )

    @property
    def qa1(self) -> str:
        return f"Which {self.field_a} is specified in {self.case_id}?"

    @property
    def qa2(self) -> str:
        return f"Identify the {self.field_a} entry for file {self.case_id}."

    @property
    def qb1(self) -> str:
        return f"Which {self.field_b} is specified in {self.case_id}?"

    @property
    def qb2(self) -> str:
        return f"Identify the {self.field_b} entry for file {self.case_id}."

    def options(self) -> tuple[LogicalOption, ...]:
        values = (self.first, self.second, self.wrong_a, self.wrong_b)
        fields = (self.field_a, self.field_b, self.field_a, self.field_b)
        return tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=(
                    f"for this {self.noun}, {fields[i]} is {values[i]}"
                ),
                aliases=(
                    f"{values[i]} is the recorded {fields[i]} value "
                    f"for S11-A0 file {self.case_id}",
                ),
            )
            for i in range(4)
        )


def cases() -> tuple[Case, ...]:
    return (
        Case("RA11", "wind turbine", "blade pitch mode", "collective", "gearbox oil", "ISO VG 320", "individual", "ISO VG 220"),
        Case("RA22", "desalination skid", "membrane type", "SW30", "feed pressure", "61 bar", "BW30", "48 bar"),
        Case("RA33", "battery pack", "cell chemistry", "LFP", "coolant flow", "12 L/min", "NMC", "7 L/min"),
        Case("RA44", "observatory mount", "drive mode", "direct drive", "tracking rate", "sidereal", "worm gear", "lunar"),
        Case("RA55", "greenhouse bay", "cultivar", "Sakura", "irrigation pulse", "42 s", "Roma", "27 s"),
        Case("RA66", "textile loom", "weft yarn", "40 tex linen", "pick rate", "720 ppm", "30 tex cotton", "510 ppm"),
        Case("RA77", "drone mission", "camera payload", "multispectral", "cruise altitude", "115 m", "thermal", "75 m"),
        Case("RA88", "cryogenic tank", "insulation", "perlite vacuum", "boiloff rate", "0.18 percent/day", "foam glass", "0.42 percent/day"),
        Case("RA99", "data center rack", "accelerator", "MI325X", "inlet temperature", "21 C", "L40S", "27 C"),
        Case("RB10", "aquaculture tank", "stock species", "barramundi", "dissolved oxygen", "7.2 mg/L", "tilapia", "5.4 mg/L"),
        Case("RB21", "rail signal", "aspect plan", "four-aspect", "block length", "1350 m", "three-aspect", "900 m"),
        Case("RB32", "chemical reactor", "catalyst", "Pd/C", "jacket temperature", "68 C", "Raney nickel", "52 C"),
        Case("RB43", "medical sterilizer", "cycle gas", "ethylene oxide", "hold time", "165 min", "hydrogen peroxide", "95 min"),
        Case("RB54", "printing press", "ink system", "UV flexo", "web speed", "310 m/min", "water-based gravure", "190 m/min"),
        Case("RB65", "mining conveyor", "belt compound", "steel-cord FR", "motor power", "630 kW", "fabric EP", "420 kW"),
        Case("RB76", "spectrograph", "grating", "1200 lines/mm", "slit width", "80 um", "600 lines/mm", "140 um"),
    )


def _text_bank(suite: tuple[Case, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for case in suite:
        values.extend((case.state_a, case.state_b, case.qa1, case.qa2, case.qb1, case.qb2))
        for option in case.options():
            values.append(option.criterion_text)
            values.extend(option.aliases)
    return tuple(values)


def _binding_call(binding, scorer, memory, schema):
    state_tokens = memory.content_token_embeddings
    q = schema.question_token_embeddings
    qm = schema.question_content_token_mask
    ov = schema.option_view_token_embeddings
    ovtm = schema.option_view_token_mask
    ovm = schema.option_view_mask
    if state_tokens is None or any(x is None for x in (q, qm, ov, ovtm, ovm)):
        raise RuntimeError("S11-A0 token artifacts missing")
    st = state_tokens.unsqueeze(0)
    sm = torch.ones(
        1,
        st.shape[1],
        dtype=torch.bool,
        device=st.device,
    )
    return binding(
        projection=scorer.projection,
        state_tokens=st,
        state_mask=sm,
        question_tokens=q.unsqueeze(0),
        question_mask=qm.unsqueeze(0),
        option_view_tokens=ov.unsqueeze(0),
        option_view_token_mask=ovtm.unsqueeze(0),
        option_view_mask=ovm.unsqueeze(0),
    )


@torch.inference_mode()
def _collect(runtime, suite: tuple[Case, ...]) -> dict:
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S11-A0 projection scorer missing")
    binding = RoleValueFactorizedBinding(
        role_temperature=ROLE_TEMPERATURE,
        state_temperature=STATE_TEMPERATURE,
    )
    if binding.parameter_count != 0:
        raise RuntimeError("S11-A0 binding unexpectedly has parameters")

    canonical_logits = []
    paraphrase_logits = []
    canonical_gold = []
    paraphrase_gold = []
    canonical_pair_both = 0
    canonical_pair_changed = 0
    flips = 0
    max_mass = 0.0
    state_top2_hits = 0
    state_top2_total = 0
    role_entropy = {"canonical": [], "paraphrase": []}
    state_entropy = {"canonical": [], "paraphrase": []}
    expert_agreement = {"canonical": [], "paraphrase": []}
    state_encodes_before = runtime.state_encode_calls

    for case in suite:
        options = case.options()
        canonical_preds = []
        for view, state, qa, qb in (
            ("canonical", case.state_a, case.qa1, case.qb1),
            ("paraphrase", case.state_b, case.qa2, case.qb2),
        ):
            memory = runtime.compile_state(state)
            per_view_state_checked = False
            for label, question, gold in (("a", qa, 0), ("b", qb, 1)):
                schema, _ = runtime.compile_schema(
                    primitive="choice",
                    question_text=question,
                    options=options,
                    include_token_artifacts=True,
                    use_cache=False,
                )
                logits, role, state_scores, diag = _binding_call(
                    binding,
                    scorer,
                    memory,
                    schema,
                )
                pred = int(logits.argmax(-1).item())
                if view == "canonical":
                    canonical_logits.append(logits[0].detach().cpu())
                    canonical_gold.append(gold)
                    canonical_preds.append(pred)
                else:
                    paraphrase_logits.append(logits[0].detach().cpu())
                    paraphrase_gold.append(gold)

                role_entropy[view].append(float(diag.role_normalized_entropy))
                state_entropy[view].append(float(diag.state_normalized_entropy))
                expert_agreement[view].append(float(diag.expert_choice_agreement))

                if not per_view_state_checked:
                    top2 = set(state_scores[0].topk(2).indices.tolist())
                    state_top2_hits += int({0, 1}.issubset(top2))
                    state_top2_total += 1
                    per_view_state_checked = True

                probs = torch.softmax(logits, dim=-1)
                max_mass = max(
                    max_mass,
                    abs(float(probs.sum().item()) - 1.0),
                )
                if logits.shape[-1] != 4:
                    raise RuntimeError("S11-A0 full-K changed")

                reversed_options = tuple(reversed(options))
                reverse_schema, _ = runtime.compile_schema(
                    primitive="choice",
                    question_text=question,
                    options=reversed_options,
                    include_token_artifacts=True,
                    use_cache=False,
                )
                reverse_logits, _, _, _ = _binding_call(
                    binding,
                    scorer,
                    memory,
                    reverse_schema,
                )
                reverse_idx = int(reverse_logits.argmax(-1).item())
                original_id = options[pred].option_id
                reverse_id = reversed_options[reverse_idx].option_id
                flips += int(original_id != reverse_id)

            if view == "canonical":
                canonical_pair_both += int(
                    canonical_preds[-2] == 0 and canonical_preds[-1] == 1
                )
                canonical_pair_changed += int(
                    canonical_preds[-2] != canonical_preds[-1]
                )

    c = torch.stack(canonical_logits)
    p = torch.stack(paraphrase_logits)
    cg = torch.tensor(canonical_gold, dtype=torch.long)
    pg = torch.tensor(paraphrase_gold, dtype=torch.long)
    c_margin = binding_gold_vs_max_wrong_margin(c, cg)
    p_margin = binding_gold_vs_max_wrong_margin(p, pg)
    queries = len(suite) * 2

    return {
        "canonical_binding_accuracy": float((c.argmax(-1) == cg).float().mean()),
        "paraphrase_binding_accuracy": float((p.argmax(-1) == pg).float().mean()),
        "canonical_paired_both_correct_rate": canonical_pair_both / len(suite),
        "canonical_question_swap_choice_change_rate": canonical_pair_changed / len(suite),
        "cross_view_selected_choice_agreement": float(selected_choice_agreement(c, p)),
        "cross_view_mean_js": float(symmetric_js_divergence(c, p)),
        "canonical_binding_mean_gold_margin": float(c_margin.mean()),
        "paraphrase_binding_mean_gold_margin": float(p_margin.mean()),
        "canonical_role_normalized_entropy": sum(role_entropy["canonical"]) / len(role_entropy["canonical"]),
        "paraphrase_role_normalized_entropy": sum(role_entropy["paraphrase"]) / len(role_entropy["paraphrase"]),
        "canonical_state_normalized_entropy": sum(state_entropy["canonical"]) / len(state_entropy["canonical"]),
        "paraphrase_state_normalized_entropy": sum(state_entropy["paraphrase"]) / len(state_entropy["paraphrase"]),
        "canonical_expert_choice_agreement": sum(expert_agreement["canonical"]) / len(expert_agreement["canonical"]),
        "paraphrase_expert_choice_agreement": sum(expert_agreement["paraphrase"]) / len(expert_agreement["paraphrase"]),
        "paired_gold_state_support_top2_containment": state_top2_hits / state_top2_total,
        "option_order_flip_rate": flips / (len(suite) * 4),
        "max_probability_mass_error": max_mass,
        "state_encode_calls": runtime.state_encode_calls - state_encodes_before,
        "full_k": True,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S11-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder
    encoder.eval()

    bank = _text_bank(suite)
    before = encoder.encode_texts(bank)
    token_before = before.token_embeddings.detach().cpu().clone()
    pooled_before = before.pooled_embeddings.detach().cpu().clone()

    runtime = build_hira_v1_s11_role_value_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
        train_projection=False,
    )
    runtime.encoder.eval()

    after = runtime.encoder.encode_texts(bank)
    token_identity = torch.equal(after.token_embeddings.detach().cpu(), token_before)
    pooled_identity = torch.equal(after.pooled_embeddings.detach().cpu(), pooled_before)
    if not token_identity or not pooled_identity:
        raise RuntimeError("S11-A0 A13 identity changed")

    observed = _collect(runtime, suite)

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
    )
    scorer = runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S11-A0 projection scorer missing")
    physical = lora_params + scorer.projection_parameter_count
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    original_a13 = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )

    if lora_params != 16_384:
        raise RuntimeError("S11-A0 LoRA capacity changed")
    if scorer.projection_parameter_count != HIRA_V1_S11_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("S11-A0 projection capacity changed")
    if physical != HIRA_V1_S11_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S11-A0 total capacity changed")
    if trainable != 0 or original_a13 != 0:
        raise RuntimeError("S11-A0 must be fully frozen")
    if observed["state_encode_calls"] != len(suite) * 2:
        raise RuntimeError("S11-A0 state-once changed")
    if observed["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S11-A0 option permutation changed")
    if observed["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S11-A0 probability mass changed")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S11_A0_ROLE_VALUE_MECHANICS_ONLY",
        "semantic_case_count": len(suite),
        "decision_count": len(suite) * 4,
        "state_view_count": len(suite) * 2,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "a13_token_output_identity": token_identity,
        "a13_pooled_output_identity": pooled_identity,
        **observed,
        "lora_parameter_count": lora_params,
        "projection_parameter_count": scorer.projection_parameter_count,
        "candidate_parameter_count": physical,
        "runtime_trainable_parameter_count": trainable,
        "original_a13_trainable_parameter_count": original_a13,
        "binding_added_parameter_count": 0,
        "role_temperature": ROLE_TEMPERATURE,
        "state_temperature": STATE_TEMPERATURE,
        "relation_refinement": False,
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HIRA_V1_S11_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
