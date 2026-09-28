from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle, read_runtime_bundle_manifest
from nmd.v1_s3_semantic_core import build_hira_v1_s3_parameter_free_core


SCHEMA_VERSION = "hira-v1-s3-a0-parameter-free-triadic-v1"
OUTCOME = "HIRA_V1_S3_A0_PARAMETER_FREE_TRIADIC_READY"


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
        Case("buoy", "Ocean buoy B17 reports salinity 34 PSU and transmits beacon code BX7.", "What salinity does buoy B17 report?", "Which beacon code does buoy B17 transmit?", ("34 PSU", "BX7", "29 PSU", "CZ4"), 0, 1),
        Case("textile", "Textile roll T42 uses a twill weave and has width 145 centimeters.", "Which weave is used for textile roll T42?", "What width is recorded for textile roll T42?", ("twill", "145 centimeters", "satin", "120 centimeters"), 0, 1),
        Case("freezer", "Specimen F8 is stored on rack R12 at minus 70 degrees Celsius.", "Which rack stores specimen F8?", "At what temperature is specimen F8 stored?", ("rack R12", "minus 70 degrees Celsius", "rack R4", "minus 40 degrees Celsius"), 0, 1),
        Case("crane", "Harbor crane C9 has a 42 meter boom and a rated capacity of 18 tonnes.", "How long is the boom on crane C9?", "What is the rated capacity of crane C9?", ("42 meters", "18 tonnes", "30 meters", "12 tonnes"), 0, 1),
        Case("solar", "Solar array S31 routes through inverter Delta-3 and peaks at 480 kilowatts.", "Which inverter serves solar array S31?", "What peak output is recorded for solar array S31?", ("inverter Delta-3", "480 kilowatts", "inverter Omega-2", "320 kilowatts"), 0, 1),
        Case("library", "Preservation room P6 is held at 48 percent humidity and uses blue archival boxes.", "What humidity is maintained in preservation room P6?", "What kind of boxes are used in preservation room P6?", ("48 percent", "blue archival boxes", "61 percent", "gray plastic boxes"), 0, 1),
        Case("orchard", "Irrigation job I24 covers sector Lime for 22 minutes.", "Which sector is covered by irrigation job I24?", "How long does irrigation job I24 run?", ("sector Lime", "22 minutes", "sector Plum", "35 minutes"), 0, 1),
        Case("telescope", "Observation O11 uses the calcium K filter with an exposure of 75 seconds.", "Which filter is used for observation O11?", "What exposure time is used for observation O11?", ("calcium K filter", "75 seconds", "hydrogen beta filter", "40 seconds"), 0, 1),
        Case("ferry", "Ferry maintenance record M5 assigns dock 6 and inspection date May 14.", "Which dock is assigned in ferry maintenance record M5?", "What inspection date is listed in record M5?", ("dock 6", "May 14", "dock 3", "June 9"), 0, 1),
        Case("datacenter", "Server rack G19 draws power from circuit C-42 in hall North.", "Which server rack is described?", "Which power circuit feeds the rack?", ("rack G19", "circuit C-42", "rack D7", "circuit A-18"), 0, 1),
        Case("coffee", "Roast batch R33 uses Sidamo beans and a roast time of 11 minutes.", "What bean origin is used in roast batch R33?", "How long is roast batch R33 roasted?", ("Sidamo", "11 minutes", "Tarrazú", "17 minutes"), 0, 1),
        Case("battery", "Battery pack B52 uses LFP chemistry and has nominal voltage 51.2 volts.", "Which chemistry is used in battery pack B52?", "What nominal voltage does battery pack B52 have?", ("LFP", "51.2 volts", "NMC", "72 volts"), 0, 1),
        Case("permit", "Trail permit P18 assigns Cedar Spur and allows a group size of eight.", "Which trail is assigned by permit P18?", "What group size is allowed by permit P18?", ("Cedar Spur", "eight people", "Maple Ridge", "five people"), 0, 1),
        Case("studio", "Studio setup L4 places softbox West at a color temperature of 5600 kelvin.", "Which softbox is used in studio setup L4?", "What color temperature is set in studio setup L4?", ("softbox West", "5600 kelvin", "softbox East", "4200 kelvin"), 0, 1),
        Case("aquarium", "Aquarium tank A7 contains discus fish and is maintained at pH 6.8.", "Which fish are kept in aquarium tank A7?", "What pH is maintained in aquarium tank A7?", ("discus fish", "pH 6.8", "molly fish", "pH 7.6"), 0, 1),
        Case("rail", "Rail cargo manifest W21 lists wagon Q17 and destination Port Helix.", "Which wagon is listed in cargo manifest W21?", "What destination is listed in cargo manifest W21?", ("wagon Q17", "Port Helix", "wagon J8", "Port Orion"), 0, 1),
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16 or len({row.case_id for row in suite}) != 16:
        raise RuntimeError("S3-A0 suite identity changed")
    if any(row.gold_a == row.gold_b for row in suite):
        raise RuntimeError("S3-A0 paired gold must change with question")

    bundle = args.bundle.resolve()
    manifest = read_runtime_bundle_manifest(bundle)
    frozen = load_hira_v0_m4_bundle(bundle)

    runtime = build_hira_v1_s3_parameter_free_core(
        frozen.runtime.encoder,
        bundle / str(manifest["t0_checkpoint"]),
        expected_t0_sha256=str(manifest["t0_checkpoint_sha256"]),
    )
    runtime.eval()

    before = runtime.state_encode_calls
    triadic_correct = 0
    v0_correct = 0
    triadic_both = 0
    v0_both = 0
    triadic_changed = 0
    v0_changed = 0
    logits_changed = 0
    max_mass_error = 0.0
    rows = []

    print("HIRA_V1_S3_A0_EXPOSURE_BEGIN", flush=True)

    for case in suite:
        options = case.options()
        memory = runtime.compile_state(case.state)
        pair = []

        for label, question, gold_index in (
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
            triadic = runtime.forward_compiled(
                memory,
                schema,
                coarse_mode="parameter_free_triadic",
                relation_refinement=False,
            )
            v0 = frozen.runtime.forward_compiled(
                memory,
                schema,
                coarse_mode="coevidence_symmetric_semantic",
                relation_refinement=False,
            )

            if int(triadic.hira.candidate_budget.item()) != 4:
                raise RuntimeError(f"{case.case_id}: triadic full-K failed")
            if not bool(triadic.hira.selected_mask.all()):
                raise RuntimeError(f"{case.case_id}: triadic selected mask failed")
            if float(triadic.hira.relation_delta.abs().max()) != 0.0:
                raise RuntimeError(f"{case.case_id}: triadic relation delta changed")

            mass_error = abs(float(triadic.probabilities.sum()) - 1.0)
            max_mass_error = max(max_mass_error, mass_error)
            if mass_error > 1e-6:
                raise RuntimeError(f"{case.case_id}: probability mass failed")

            triadic_index = int(triadic.probabilities.argmax())
            v0_index = int(v0.probabilities.argmax())
            triadic_ok = triadic_index == gold_index
            v0_ok = v0_index == gold_index
            triadic_correct += int(triadic_ok)
            v0_correct += int(v0_ok)

            pair.append(
                {
                    "label": label,
                    "gold_index": gold_index,
                    "triadic_index": triadic_index,
                    "v0_index": v0_index,
                    "triadic_correct": triadic_ok,
                    "v0_correct": v0_ok,
                    "triadic_logits": triadic.logits.detach().cpu(),
                    "v0_logits": v0.logits.detach().cpu(),
                }
            )

        t_changed = pair[0]["triadic_index"] != pair[1]["triadic_index"]
        v_changed = pair[0]["v0_index"] != pair[1]["v0_index"]
        l_changed = not torch.equal(pair[0]["triadic_logits"], pair[1]["triadic_logits"])
        triadic_changed += int(t_changed)
        v0_changed += int(v_changed)
        logits_changed += int(l_changed)
        triadic_both += int(pair[0]["triadic_correct"] and pair[1]["triadic_correct"])
        v0_both += int(pair[0]["v0_correct"] and pair[1]["v0_correct"])

        rows.append(
            {
                "case_id": case.case_id,
                "triadic_question_changes_logits": l_changed,
                "triadic_question_changes_choice": t_changed,
                "v0_question_changes_choice": v_changed,
                "a": {
                    k: ([float(x) for x in v.tolist()] if isinstance(v, torch.Tensor) else v)
                    for k, v in pair[0].items()
                },
                "b": {
                    k: ([float(x) for x in v.tolist()] if isinstance(v, torch.Tensor) else v)
                    for k, v in pair[1].items()
                },
            }
        )

    state_calls = runtime.state_encode_calls - before
    if state_calls != 16:
        raise RuntimeError(f"S3-A0 state-once changed: {state_calls} != 16")

    queries = 32
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S3_A0_LOCALIZATION_ONLY",
        "case_count": 16,
        "query_count": queries,
        "language": "en",
        "k": 4,
        "added_parameter_count": 0,
        "trainable_parameter_count": 0,
        "a0": {
            "accuracy": triadic_correct / queries,
            "paired_both_correct_rate": triadic_both / 16,
            "question_changes_logits_rate": logits_changed / 16,
            "question_changes_choice_rate": triadic_changed / 16,
        },
        "fresh_v0_control": {
            "accuracy": v0_correct / queries,
            "paired_both_correct_rate": v0_both / 16,
            "question_changes_choice_rate": v0_changed / 16,
        },
        "runtime": {
            "state_encode_calls": state_calls,
            "full_k": True,
            "relation_refinement": False,
            "max_probability_mass_error": max_mass_error,
            "t0_checkpoint_sha256": str(manifest["t0_checkpoint_sha256"]),
        },
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print("HIRA_V1_S3_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
