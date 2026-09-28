from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_s4_semantic_core import build_hira_v1_s4_adapter_core
from nmd.v1_triadic_semantic import ParameterFreeTriadicScorer


SCHEMA_VERSION = "hira-v1-s4-a0-identity-v1"
OUTCOME = "HIRA_V1_S4_A0_IDENTITY_READY"


@dataclass(frozen=True)
class Case:
    case_id: str
    state: str
    question_a: str
    question_b: str
    answers: tuple[str, str, str, str]
    gold_a: int
    gold_b: int

    def options(self) -> tuple[LogicalOption, ...]:
        return tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{i:02d}",
                criterion_text=text,
            )
            for i, text in enumerate(self.answers)
        )


def cases() -> tuple[Case, ...]:
    return (
        Case("harbor", "Cargo lot HX-41 is stored in bay Cedar and weighs 74 kilograms.", "Which bay stores cargo lot HX-41?", "How much does cargo lot HX-41 weigh?", ("bay Cedar", "74 kilograms", "bay Aspen", "42 kilograms"), 0, 1),
        Case("clinic", "Patient tray CT-18 contains saline and is assigned to room 406.", "What fluid is on patient tray CT-18?", "Which room is assigned to patient tray CT-18?", ("saline", "room 406", "glucose", "room 219"), 0, 1),
        Case("observatory", "Observation card OC-52 uses filter violet and exposure 48 seconds.", "Which filter is used on observation card OC-52?", "What exposure is listed on observation card OC-52?", ("filter violet", "48 seconds", "filter amber", "22 seconds"), 0, 1),
        Case("warehouse", "Inventory pallet WP-63 is in aisle 17 and carries label QN-82.", "Which aisle contains pallet WP-63?", "What label does pallet WP-63 carry?", ("aisle 17", "QN-82", "aisle 9", "LM-31"), 0, 1),
        Case("garden", "Irrigation sector IG-27 uses valve copper and runs at 06:45.", "Which valve is used in irrigation sector IG-27?", "When does irrigation sector IG-27 run?", ("valve copper", "06:45", "valve silver", "08:20"), 0, 1),
        Case("studio", "Recording job RJ-35 uses microphone ribbon and sample rate 96 kHz.", "Which microphone type is used for recording job RJ-35?", "What sample rate is used for recording job RJ-35?", ("ribbon", "96 kHz", "condenser", "44 kHz"), 0, 1),
        Case("library", "Archive file AF-72 is in cabinet Birch and has retention code RC-14.", "Which cabinet contains archive file AF-72?", "What retention code belongs to archive file AF-72?", ("cabinet Birch", "RC-14", "cabinet Oak", "RC-39"), 0, 1),
        Case("factory", "Machine ticket MT-44 specifies lubricant ceramic and service interval 320 hours.", "Which lubricant is specified by machine ticket MT-44?", "What service interval is specified by machine ticket MT-44?", ("ceramic", "320 hours", "graphite", "180 hours"), 0, 1),
        Case("theater", "Stage cue SC-21 assigns spotlight Indigo and trigger time 19:12.", "Which spotlight is assigned by stage cue SC-21?", "What trigger time is assigned by stage cue SC-21?", ("Indigo", "19:12", "Crimson", "20:30"), 0, 1),
        Case("lab", "Culture batch CB-86 uses medium agarose and rotates at 140 rpm.", "Which medium is used for culture batch CB-86?", "At what speed does culture batch CB-86 rotate?", ("agarose", "140 rpm", "gelatin", "90 rpm"), 0, 1),
        Case("terminal", "Transit record TR-58 lists platform Luna and departure 17:28.", "Which platform is listed in transit record TR-58?", "What departure time is listed in transit record TR-58?", ("platform Luna", "17:28", "platform Nova", "15:05"), 0, 1),
        Case("aquifer", "Well report WR-33 records depth 84 meters and mineral class basaltic.", "What depth is recorded in well report WR-33?", "What mineral class is recorded in well report WR-33?", ("84 meters", "basaltic", "51 meters", "granitic"), 0, 1),
        Case("bakery", "Batch sheet BS-47 names flour rye and proofing duration 52 minutes.", "Which flour is named on batch sheet BS-47?", "What proofing duration is listed on batch sheet BS-47?", ("rye", "52 minutes", "barley", "35 minutes"), 0, 1),
        Case("network", "Router note RN-64 assigns subnet 10.4.8.0/24 and rack K7.", "Which subnet is assigned in router note RN-64?", "Which rack is assigned in router note RN-64?", ("10.4.8.0/24", "rack K7", "10.8.2.0/24", "rack M3"), 0, 1),
        Case("museum", "Loan tag LT-29 records lender Helios House and policy class Z4.", "Which lender is recorded on loan tag LT-29?", "What policy class is recorded on loan tag LT-29?", ("Helios House", "class Z4", "Orchid Hall", "class B6"), 0, 1),
        Case("farm", "Field card FC-91 lists crop lentil and harvest window week 38.", "Which crop is listed on field card FC-91?", "What harvest window is listed on field card FC-91?", ("lentil", "week 38", "sorghum", "week 24"), 0, 1),
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S4-A0 suite size changed")
    if any(case.gold_a == case.gold_b for case in suite):
        raise RuntimeError("S4-A0 paired gold must differ")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)
    encoder = frozen.runtime.encoder

    runtime = build_hira_v1_s4_adapter_core(
        encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
        train_adapter=False,
    )
    scorer = runtime.adapted_triadic_scorer
    if scorer is None:
        raise RuntimeError("S4 adapted scorer missing")

    base = ParameterFreeTriadicScorer(d_model=256, d_rel=128)
    base.load_projection_weight(
        scorer.projection.weight.detach().cpu(),
        freeze=True,
    )
    base.eval()

    if scorer.adapter_parameter_count != 16384:
        raise RuntimeError("S4 adapter parameter count changed")
    if scorer.adapter_trainable_parameter_count != 0:
        raise RuntimeError("S4 A0 adapter must be frozen")

    rows = []
    before_calls = runtime.state_encode_calls
    exact_logit = 0
    exact_choice = 0
    correct = 0
    pair_both = 0
    order_flips = 0
    max_mass_error = 0.0

    print("HIRA_V1_S4_A0_EXPOSURE_BEGIN", flush=True)

    for case in suite:
        options = case.options()
        memory = runtime.compile_state(case.state)
        state_tokens = memory.content_token_embeddings
        if state_tokens is None:
            raise RuntimeError(f"{case.case_id}: missing state content tokens")
        state_tokens = state_tokens.unsqueeze(0)
        state_mask = torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
            device=state_tokens.device,
        )

        pair = []
        for question, gold_index in (
            (case.question_a, case.gold_a),
            (case.question_b, case.gold_b),
        ):
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
                raise RuntimeError(f"{case.case_id}: schema token artifacts missing")

            kwargs = dict(
                state_tokens=state_tokens,
                state_mask=state_mask,
                question_tokens=q.unsqueeze(0),
                question_mask=qm.unsqueeze(0),
                option_view_tokens=ov.unsqueeze(0),
                option_view_token_mask=ovtm.unsqueeze(0),
                option_view_mask=ovm.unsqueeze(0),
            )
            expected = base(**kwargs)[0]
            observed = scorer(**kwargs)[0]

            logits_equal = torch.equal(observed, expected)
            exact_logit += int(logits_equal)
            if not logits_equal:
                raise RuntimeError(f"{case.case_id}: zero-init identity changed")

            probs = torch.softmax(observed, dim=-1)
            mass_error = abs(float(probs.sum()) - 1.0)
            max_mass_error = max(max_mass_error, mass_error)
            if mass_error > 1e-6:
                raise RuntimeError(f"{case.case_id}: probability mass failed")

            index = int(probs.argmax())
            base_index = int(torch.softmax(expected, dim=-1).argmax())
            same_choice = index == base_index
            exact_choice += int(same_choice)
            if not same_choice:
                raise RuntimeError(f"{case.case_id}: zero-init choice identity changed")

            ok = index == gold_index
            correct += int(ok)
            pair.append(ok)

            reversed_options = tuple(reversed(options))
            reversed_schema, _ = runtime.compile_schema(
                primitive="choice",
                question_text=question,
                options=reversed_options,
                include_token_artifacts=True,
                use_cache=False,
            )
            reverse_out = runtime.forward_compiled(
                memory,
                reversed_schema,
                coarse_mode="adapted_triadic",
                relation_refinement=False,
            )
            selected = options[index].option_id
            order_flips += int(reverse_out.selected_option_id != selected)

        pair_both += int(pair[0] and pair[1])
        rows.append(
            {
                "case_id": case.case_id,
                "a_correct": pair[0],
                "b_correct": pair[1],
            }
        )

    state_calls = runtime.state_encode_calls - before_calls
    if state_calls != len(suite):
        raise RuntimeError(
            f"S4-A0 state-once changed: {state_calls} != {len(suite)}"
        )

    queries = len(suite) * 2
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S4_A0_IDENTITY_LOCALIZATION_ONLY",
        "case_count": len(suite),
        "query_count": queries,
        "language": "en",
        "k": 4,
        "adapter_parameter_count": scorer.adapter_parameter_count,
        "adapter_trainable_parameter_count": scorer.adapter_trainable_parameter_count,
        "exact_logit_identity_rate": exact_logit / queries,
        "exact_choice_identity_rate": exact_choice / queries,
        "accuracy": correct / queries,
        "paired_both_correct_rate": pair_both / len(suite),
        "option_order_flip_rate": order_flips / queries,
        "state_encode_calls": state_calls,
        "max_probability_mass_error": max_mass_error,
        "full_k": True,
        "relation_refinement": False,
        "m4_manifest": frozen.manifest.to_dict(),
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-case.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("HIRA_V1_S4_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
