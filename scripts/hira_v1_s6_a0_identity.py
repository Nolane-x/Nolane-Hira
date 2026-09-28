from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_a13_lora import iter_a13_lora_modules
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core
from nmd.v1_s6_semantic_core import (
    HIRA_V1_S6_LORA_PARAMETER_COUNT,
    build_hira_v1_s6_a13_lora_core,
)


SCHEMA_VERSION = "hira-v1-s6-a0-identity-v1"
OUTCOME = "HIRA_V1_S6_A0_IDENTITY_READY"


@dataclass(frozen=True)
class Case:
    case_id: str
    state: str
    question_a: str
    question_b: str
    answers: tuple[str, str, str, str]
    aliases: tuple[str, str, str, str]
    gold_a: int
    gold_b: int

    def options(self) -> tuple[LogicalOption, ...]:
        return tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=self.answers[i],
                aliases=(self.aliases[i],),
            )
            for i in range(4)
        )


def cases() -> tuple[Case, ...]:
    return (
        Case("volcano", "Field note VN-14 records mineral olivine and station elevation 1840 meters.", "Which mineral is recorded in VN-14?", "What station elevation is recorded in VN-14?", ("the requested answer is olivine", "the requested answer is 1840 meters", "the requested answer is feldspar", "the requested answer is 920 meters"), ("olivine is the field-note value", "1840 meters is the field-note value", "feldspar is the field-note value", "920 meters is the field-note value"), 0, 1),
        Case("pharmacy", "Dispensing log DL-26 lists formulation capsule and storage band 15 C.", "Which formulation is listed in DL-26?", "What storage band is listed in DL-26?", ("the requested answer is capsule", "the requested answer is 15 C", "the requested answer is syrup", "the requested answer is 28 C"), ("capsule is the dispensing value", "15 C is the dispensing value", "syrup is the dispensing value", "28 C is the dispensing value"), 0, 1),
        Case("satellite", "Orbital card OR-39 names antenna Helix and downlink channel D12.", "Which antenna is named on OR-39?", "Which downlink channel is named on OR-39?", ("the requested answer is Helix", "the requested answer is channel D12", "the requested answer is Patch", "the requested answer is channel B4"), ("Helix is the orbital-card value", "channel D12 is the orbital-card value", "Patch is the orbital-card value", "channel B4 is the orbital-card value"), 0, 1),
        Case("brewery", "Fermentation sheet FS-48 specifies yeast Kveik and vessel temperature 31 C.", "Which yeast is specified in FS-48?", "What vessel temperature is specified in FS-48?", ("the requested answer is Kveik", "the requested answer is 31 C", "the requested answer is Lager", "the requested answer is 19 C"), ("Kveik is the fermentation value", "31 C is the fermentation value", "Lager is the fermentation value", "19 C is the fermentation value"), 0, 1),
        Case("court", "Docket DK-53 assigns chamber North and hearing slot 14:25.", "Which chamber is assigned on DK-53?", "What hearing slot is assigned on DK-53?", ("the requested answer is chamber North", "the requested answer is 14:25", "the requested answer is chamber West", "the requested answer is 09:40"), ("chamber North is the docket value", "14:25 is the docket value", "chamber West is the docket value", "09:40 is the docket value"), 0, 1),
        Case("greenhouse", "Climate card GC-62 records crop basil and humidity target 67 percent.", "Which crop is recorded on GC-62?", "What humidity target is recorded on GC-62?", ("the requested answer is basil", "the requested answer is 67 percent", "the requested answer is thyme", "the requested answer is 44 percent"), ("basil is the climate-card value", "67 percent is the climate-card value", "thyme is the climate-card value", "44 percent is the climate-card value"), 0, 1),
        Case("seismology", "Sensor bulletin SB-71 identifies axis vertical and sample rate 200 Hz.", "Which axis is identified in SB-71?", "What sample rate is identified in SB-71?", ("the requested answer is vertical", "the requested answer is 200 Hz", "the requested answer is lateral", "the requested answer is 80 Hz"), ("vertical is the bulletin value", "200 Hz is the bulletin value", "lateral is the bulletin value", "80 Hz is the bulletin value"), 0, 1),
        Case("shipping", "Port record PR-84 assigns berth Delta and tug call sign Mako.", "Which berth is assigned in PR-84?", "What tug call sign is assigned in PR-84?", ("the requested answer is berth Delta", "the requested answer is Mako", "the requested answer is berth Echo", "the requested answer is Lynx"), ("berth Delta is the port-record value", "Mako is the port-record value", "berth Echo is the port-record value", "Lynx is the port-record value"), 0, 1),
        Case("archive", "Preservation ticket PT-95 lists medium vellum and vault zone H3.", "Which medium is listed on PT-95?", "Which vault zone is listed on PT-95?", ("the requested answer is vellum", "the requested answer is zone H3", "the requested answer is linen", "the requested answer is zone C8"), ("vellum is the preservation value", "zone H3 is the preservation value", "linen is the preservation value", "zone C8 is the preservation value"), 0, 1),
        Case("mining", "Drill report DR-17 records core depth 126 meters and ore class sulfide.", "What core depth is recorded in DR-17?", "Which ore class is recorded in DR-17?", ("the requested answer is 126 meters", "the requested answer is sulfide", "the requested answer is 72 meters", "the requested answer is oxide"), ("126 meters is the drill-report value", "sulfide is the drill-report value", "72 meters is the drill-report value", "oxide is the drill-report value"), 0, 1),
        Case("broadcast", "Studio slate SS-32 names program Meridian and transmitter power 42 kW.", "Which program is named on SS-32?", "What transmitter power is named on SS-32?", ("the requested answer is Meridian", "the requested answer is 42 kW", "the requested answer is Horizon", "the requested answer is 18 kW"), ("Meridian is the studio-slate value", "42 kW is the studio-slate value", "Horizon is the studio-slate value", "18 kW is the studio-slate value"), 0, 1),
        Case("robotics", "Assembly note AN-43 assigns gripper soft and torque limit 18 Nm.", "Which gripper is assigned in AN-43?", "What torque limit is assigned in AN-43?", ("the requested answer is soft", "the requested answer is 18 Nm", "the requested answer is magnetic", "the requested answer is 9 Nm"), ("soft is the assembly-note value", "18 Nm is the assembly-note value", "magnetic is the assembly-note value", "9 Nm is the assembly-note value"), 0, 1),
        Case("ferry", "Voyage card VC-58 lists pier Juniper and boarding time 06:35.", "Which pier is listed on VC-58?", "What boarding time is listed on VC-58?", ("the requested answer is pier Juniper", "the requested answer is 06:35", "the requested answer is pier Maple", "the requested answer is 08:10"), ("pier Juniper is the voyage-card value", "06:35 is the voyage-card value", "pier Maple is the voyage-card value", "08:10 is the voyage-card value"), 0, 1),
        Case("optics", "Lens test LT-69 records coating fluorite and focal length 135 mm.", "Which coating is recorded in LT-69?", "What focal length is recorded in LT-69?", ("the requested answer is fluorite", "the requested answer is 135 mm", "the requested answer is quartz", "the requested answer is 85 mm"), ("fluorite is the lens-test value", "135 mm is the lens-test value", "quartz is the lens-test value", "85 mm is the lens-test value"), 0, 1),
        Case("reservoir", "Water log WL-77 names intake Cypress and turbidity 3.2 NTU.", "Which intake is named in WL-77?", "What turbidity is named in WL-77?", ("the requested answer is intake Cypress", "the requested answer is 3.2 NTU", "the requested answer is intake Alder", "the requested answer is 7.5 NTU"), ("intake Cypress is the water-log value", "3.2 NTU is the water-log value", "intake Alder is the water-log value", "7.5 NTU is the water-log value"), 0, 1),
        Case("printing", "Press sheet PS-88 specifies ink cyan and plate count 12.", "Which ink is specified on PS-88?", "What plate count is specified on PS-88?", ("the requested answer is cyan", "the requested answer is 12 plates", "the requested answer is magenta", "the requested answer is 6 plates"), ("cyan is the press-sheet value", "12 plates is the press-sheet value", "magenta is the press-sheet value", "6 plates is the press-sheet value"), 0, 1),
    )


def text_bank(suite: tuple[Case, ...]) -> tuple[str, ...]:
    values = []
    for case in suite:
        values.extend((case.state, case.question_a, case.question_b))
        values.extend(case.answers)
        values.extend(case.aliases)
    return tuple(values)


@torch.inference_mode()
def _collect_decisions(runtime, suite: tuple[Case, ...]):
    rows = {}
    before = runtime.state_encode_calls
    correct = 0
    both = 0
    order_flips = 0
    max_mass_error = 0.0

    for case in suite:
        options = case.options()
        memory = runtime.compile_state(case.state)
        pair = []
        for label, question, gold in (
            ("a", case.question_a, case.gold_a),
            ("b", case.question_b, case.gold_b),
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
                coarse_mode="parameter_free_triadic",
                relation_refinement=False,
            )
            mass_error = abs(float(out.probabilities.sum()) - 1.0)
            max_mass_error = max(max_mass_error, mass_error)
            index = int(out.probabilities.argmax())
            ok = index == gold
            correct += int(ok)
            pair.append(ok)
            rows[(case.case_id, label)] = {
                "logits": out.logits.detach().cpu().clone(),
                "selected": out.selected_option_id,
                "correct": ok,
            }

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
                coarse_mode="parameter_free_triadic",
                relation_refinement=False,
            )
            order_flips += int(reverse.selected_option_id != out.selected_option_id)
            if int(out.hira.candidate_budget.item()) != 4:
                raise RuntimeError("S6-A0 full-K changed")
            if not bool(out.hira.selected_mask.all()):
                raise RuntimeError("S6-A0 selected mask changed")
            if not torch.equal(
                out.hira.relation_delta,
                torch.zeros_like(out.hira.relation_delta),
            ):
                raise RuntimeError("S6-A0 relation delta changed")

        both += int(pair[0] and pair[1])

    state_calls = runtime.state_encode_calls - before
    queries = len(suite) * 2
    return {
        "rows": rows,
        "accuracy": correct / queries,
        "paired_both_correct_rate": both / len(suite),
        "option_order_flip_rate": order_flips / queries,
        "state_encode_calls": state_calls,
        "max_probability_mass_error": max_mass_error,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S6-A0 suite size changed")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder
    encoder.eval()

    bank = text_bank(suite)
    base_batch = encoder.encode_texts(bank)
    base_tokens = base_batch.token_embeddings.detach().cpu().clone()
    base_pooled = base_batch.pooled_embeddings.detach().cpu().clone()
    base_mask = base_batch.attention_mask.detach().cpu().clone()

    base_runtime = build_hira_v1_s3_parameter_free_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
    )
    base = _collect_decisions(base_runtime, suite)

    runtime = build_hira_v1_s6_a13_lora_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_lora=False,
    )
    runtime.encoder.eval()

    adapted_batch = runtime.encoder.encode_texts(bank)
    token_identity = torch.equal(
        adapted_batch.token_embeddings.detach().cpu(),
        base_tokens,
    )
    pooled_identity = torch.equal(
        adapted_batch.pooled_embeddings.detach().cpu(),
        base_pooled,
    )
    mask_identity = torch.equal(
        adapted_batch.attention_mask.detach().cpu(),
        base_mask,
    )
    if not token_identity or not pooled_identity or not mask_identity:
        raise RuntimeError("S6-A0 zero-init A13 identity changed")

    adapted = _collect_decisions(runtime, suite)

    exact_logits = 0
    exact_choices = 0
    per_case = []
    for case in suite:
        row = {"case_id": case.case_id}
        for label in ("a", "b"):
            before = base["rows"][(case.case_id, label)]
            after = adapted["rows"][(case.case_id, label)]
            same_logits = torch.equal(before["logits"], after["logits"])
            same_choice = before["selected"] == after["selected"]
            exact_logits += int(same_logits)
            exact_choices += int(same_choice)
            if not same_logits or not same_choice:
                raise RuntimeError(f"{case.case_id}/{label}: S6 identity changed")
            row[f"{label}_correct"] = bool(after["correct"])
        per_case.append(row)

    modules = iter_a13_lora_modules(runtime.encoder)
    lora_params = sum(
        p.numel()
        for module in modules
        for p in (module.lora_a, module.lora_b)
    )
    original_trainable = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    total_trainable = sum(
        p.numel() for p in runtime.parameters() if p.requires_grad
    )

    if lora_params != HIRA_V1_S6_LORA_PARAMETER_COUNT:
        raise RuntimeError("S6-A0 LoRA parameter count changed")
    if original_trainable != 0:
        raise RuntimeError("S6-A0 original A13 parameter became trainable")
    if total_trainable != 0:
        raise RuntimeError("S6-A0 candidate must be fully frozen")
    if adapted["state_encode_calls"] != len(suite):
        raise RuntimeError("S6-A0 state-once changed")
    if adapted["option_order_flip_rate"] != 0.0:
        raise RuntimeError("S6-A0 option-order invariance changed")
    if adapted["max_probability_mass_error"] > 1e-6:
        raise RuntimeError("S6-A0 probability mass failed")

    queries = len(suite) * 2
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S6_A0_IDENTITY_LOCALIZATION_ONLY",
        "case_count": len(suite),
        "query_count": queries,
        "language": "en",
        "k": 4,
        "views_per_option": 2,
        "a13_token_output_identity": token_identity,
        "a13_pooled_output_identity": pooled_identity,
        "attention_mask_identity": mask_identity,
        "exact_logit_identity_rate": exact_logits / queries,
        "exact_choice_identity_rate": exact_choices / queries,
        "accuracy": adapted["accuracy"],
        "paired_both_correct_rate": adapted["paired_both_correct_rate"],
        "option_order_flip_rate": adapted["option_order_flip_rate"],
        "state_encode_calls": adapted["state_encode_calls"],
        "max_probability_mass_error": adapted["max_probability_mass_error"],
        "lora_parameter_count": lora_params,
        "lora_trainable_parameter_count": 0,
        "original_a13_trainable_parameter_count": original_trainable,
        "runtime_trainable_parameter_count": total_trainable,
        "full_k": True,
        "relation_refinement": False,
        "m4_manifest": frozen.manifest.to_dict(),
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-case.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_case:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("HIRA_V1_S6_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
