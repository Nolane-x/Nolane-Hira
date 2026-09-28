from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle
from nmd.v1_query_token_fusion import QuestionAsEvidenceScorer
from nmd.w34_transfer_core import W34_CANDIDATE_KEYS


SCHEMA_VERSION = "hira-v1-s2-a0-question-as-evidence-v1"
OUTCOME = "HIRA_V1_S2_A0_QUESTION_AS_EVIDENCE_READY"


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
        Case("observatory", "Observation card OC17 lists mirror coating silver and tracking mode sidereal.", "What coating is used on the mirror?", "Which tracking mode is listed?", ("silver", "sidereal", "gold", "lunar"), 0, 1),
        Case("harvest", "Harvest lot HL28 was picked from block Pecan and packed in crate violet.", "Which block produced the harvest lot?", "What color is its packing crate?", ("block Pecan", "violet", "block Rowan", "orange"), 0, 1),
        Case("clinic", "Clinic tray CT39 contains syringe type Luer and carries batch code MX-72.", "Which syringe type is in the tray?", "What batch code is on the tray?", ("Luer", "MX-72", "Catheter", "PN-31"), 0, 1),
        Case("satellite", "Satellite pass SP46 uses antenna Kilo and begins at 05:42 UTC.", "Which antenna is assigned to the pass?", "When does the pass begin?", ("antenna Kilo", "05:42 UTC", "antenna Romeo", "07:18 UTC"), 0, 1),
        Case("kiln", "Ceramic firing CF55 uses glaze celadon and reaches 1180 degrees Celsius.", "Which glaze is used for the firing?", "What peak temperature is reached?", ("celadon", "1180 degrees Celsius", "tenmoku", "1040 degrees Celsius"), 0, 1),
        Case("library", "Archive request AR63 names collection Meridian and retrieval desk four.", "Which collection is requested?", "Which desk handles retrieval?", ("collection Meridian", "desk four", "collection Solstice", "desk nine"), 0, 1),
        Case("factory", "Assembly card AC74 specifies fastener M6 and torque setting 14 newton meters.", "Which fastener is specified?", "What torque setting is specified?", ("M6", "14 newton meters", "M8", "22 newton meters"), 0, 1),
        Case("radio", "Radio log RL82 records frequency 146.52 megahertz and operator call sign KQ7M.", "What frequency is recorded?", "What operator call sign is recorded?", ("146.52 megahertz", "KQ7M", "433.92 megahertz", "NZ4P"), 0, 1),
        Case("warehouse", "Pallet PL91 is stored in aisle Nectar and has handling class fragile.", "Which aisle stores the pallet?", "What handling class does it have?", ("aisle Nectar", "fragile", "aisle Quartz", "standard"), 0, 1),
        Case("theater", "Stage cue SQ14 assigns spotlight Indigo and trigger word horizon.", "Which spotlight is assigned?", "What trigger word activates the cue?", ("spotlight Indigo", "horizon", "spotlight Scarlet", "summit"), 0, 1),
        Case("marine", "Buoy record BR25 reports salinity 34 PSU and station label Pelican.", "What salinity is reported?", "What station label is attached?", ("34 PSU", "Pelican", "29 PSU", "Heron"), 0, 1),
        Case("printing", "Press job PJ36 uses paper stock linen and ink code C-18.", "Which paper stock is used?", "What ink code is used?", ("linen", "C-18", "vellum", "K-44"), 0, 1),
        Case("rail", "Freight consist FC47 assigns locomotive Atlas and brake test time 06:15.", "Which locomotive is assigned?", "When is the brake test scheduled?", ("locomotive Atlas", "06:15", "locomotive Boreal", "08:05"), 0, 1),
        Case("brewery", "Fermentation batch FB58 uses yeast strain Omega and vessel number 12.", "Which yeast strain is used?", "Which vessel contains the batch?", ("strain Omega", "vessel 12", "strain Sigma", "vessel 7"), 0, 1),
        Case("geology", "Core sample CS69 was taken at depth 430 meters and marked lithology basalt.", "At what depth was the core sample taken?", "What lithology is marked?", ("430 meters", "basalt", "275 meters", "granite"), 0, 1),
        Case("studio", "Recording session RS77 uses microphone Ribbon-3 and room designation Echo.", "Which microphone is used?", "What room designation is listed?", ("Ribbon-3", "Echo", "Condenser-8", "Pulse"), 0, 1),
    )


def _build_a0(base) -> QuestionAsEvidenceScorer:
    scorer = QuestionAsEvidenceScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(base.projection.weight.detach().cpu(), freeze=True)
    own = base.state_dict()
    scorer.load_candidate_state_dict(
        {key: own[key].detach().cpu() for key in W34_CANDIDATE_KEYS},
        freeze=True,
    )
    scorer.eval()
    if scorer.added_parameter_count != 0:
        raise RuntimeError("S2-A0 unexpectedly added parameters")
    if scorer.trainable_parameter_count != 0:
        raise RuntimeError("S2-A0 must be fully frozen")
    return scorer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S2-A0 suite size changed")
    if len({case.case_id for case in suite}) != len(suite):
        raise RuntimeError("S2-A0 duplicate case ID")
    if any(case.gold_a == case.gold_b for case in suite):
        raise RuntimeError("S2-A0 paired gold must change with question")

    model = load_hira_v0_m4_bundle(args.bundle)
    runtime = model.runtime
    base = runtime.coevidence_symmetric_semantic_scorer
    if base is None:
        raise RuntimeError("M4 runtime missing W34 co-evidence scorer")
    scorer = _build_a0(base)

    before_state_calls = runtime.state_encode_calls
    rows = []
    a0_correct = 0
    v0_correct = 0
    a0_both = 0
    v0_both = 0
    a0_choice_changes = 0
    v0_choice_changes = 0
    a0_logit_changes = 0
    max_mass_error = 0.0

    print("HIRA_V1_S2_A0_EXPOSURE_BEGIN", flush=True)

    for case in suite:
        options = case.options()
        memory = runtime.compile_state(case.state)
        state_tokens = memory.content_token_embeddings
        if state_tokens is None:
            raise RuntimeError(f"{case.case_id}: state content tokens missing")
        state_tokens_b = state_tokens.unsqueeze(0)
        state_mask = torch.ones(
            1,
            state_tokens.shape[0],
            dtype=torch.bool,
            device=state_tokens.device,
        )

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
            required = (
                schema.question_token_embeddings,
                schema.question_content_token_mask,
                schema.option_view_token_embeddings,
                schema.option_view_token_mask,
                schema.option_view_mask,
            )
            if any(value is None for value in required):
                raise RuntimeError(f"{case.case_id}: token artifacts missing")

            v0 = runtime.forward_compiled(
                memory,
                schema,
                coarse_mode="coevidence_symmetric_semantic",
                relation_refinement=False,
            )
            logits = scorer(
                state_tokens=state_tokens_b,
                state_mask=state_mask,
                question_tokens=schema.question_token_embeddings.unsqueeze(0),
                question_mask=schema.question_content_token_mask.unsqueeze(0),
                option_view_tokens=schema.option_view_token_embeddings.unsqueeze(0),
                option_view_token_mask=schema.option_view_token_mask.unsqueeze(0),
                option_view_mask=schema.option_view_mask.unsqueeze(0),
            )[0]
            probs = torch.softmax(logits, dim=-1)
            mass_error = abs(float(probs.sum()) - 1.0)
            max_mass_error = max(max_mass_error, mass_error)
            if mass_error > 1e-6:
                raise RuntimeError(f"{case.case_id}: probability mass failed")
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"{case.case_id}: non-finite A0 logits")
            if int(v0.hira.candidate_budget.item()) != 4:
                raise RuntimeError(f"{case.case_id}: v0 full-K failed")
            if not bool(v0.hira.selected_mask.all()):
                raise RuntimeError(f"{case.case_id}: v0 selected mask failed")
            if float(v0.hira.relation_delta.abs().max()) != 0.0:
                raise RuntimeError(f"{case.case_id}: v0 relation refinement changed")

            a0_index = int(probs.argmax())
            v0_index = int(v0.probabilities.argmax())
            a0_ok = a0_index == gold_index
            v0_ok = v0_index == gold_index
            a0_correct += int(a0_ok)
            v0_correct += int(v0_ok)

            pair.append(
                {
                    "label": label,
                    "gold_index": gold_index,
                    "a0_index": a0_index,
                    "v0_index": v0_index,
                    "a0_correct": a0_ok,
                    "v0_correct": v0_ok,
                    "a0_logits": logits.detach().cpu(),
                    "v0_logits": v0.logits.detach().cpu(),
                }
            )

        a0_changed = pair[0]["a0_index"] != pair[1]["a0_index"]
        v0_changed = pair[0]["v0_index"] != pair[1]["v0_index"]
        logits_changed = not torch.equal(pair[0]["a0_logits"], pair[1]["a0_logits"])
        a0_choice_changes += int(a0_changed)
        v0_choice_changes += int(v0_changed)
        a0_logit_changes += int(logits_changed)
        a0_both += int(pair[0]["a0_correct"] and pair[1]["a0_correct"])
        v0_both += int(pair[0]["v0_correct"] and pair[1]["v0_correct"])

        rows.append(
            {
                "case_id": case.case_id,
                "a0_logits_changed_with_question": logits_changed,
                "a0_choice_changed_with_question": a0_changed,
                "v0_choice_changed_with_question": v0_changed,
                "a": {
                    **{k: v for k, v in pair[0].items() if k not in {"a0_logits", "v0_logits"}},
                    "a0_logits": [float(x) for x in pair[0]["a0_logits"].tolist()],
                    "v0_logits": [float(x) for x in pair[0]["v0_logits"].tolist()],
                },
                "b": {
                    **{k: v for k, v in pair[1].items() if k not in {"a0_logits", "v0_logits"}},
                    "a0_logits": [float(x) for x in pair[1]["a0_logits"].tolist()],
                    "v0_logits": [float(x) for x in pair[1]["v0_logits"].tolist()],
                },
            }
        )

    state_calls = runtime.state_encode_calls - before_state_calls
    if state_calls != len(suite):
        raise RuntimeError(
            f"S2-A0 state-once changed: {state_calls} != {len(suite)}"
        )

    queries = len(suite) * 2
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S2_A0_LOCALIZATION_ONLY",
        "case_count": len(suite),
        "query_count": queries,
        "language": "en",
        "k": 4,
        "added_parameter_count": 0,
        "trainable_parameter_count": int(scorer.trainable_parameter_count),
        "a0": {
            "accuracy": a0_correct / queries,
            "paired_both_correct_rate": a0_both / len(suite),
            "question_changes_logits_rate": a0_logit_changes / len(suite),
            "question_changes_choice_rate": a0_choice_changes / len(suite),
        },
        "fresh_v0_control": {
            "accuracy": v0_correct / queries,
            "paired_both_correct_rate": v0_both / len(suite),
            "question_changes_choice_rate": v0_choice_changes / len(suite),
        },
        "runtime": {
            "state_encode_calls": state_calls,
            "full_k": True,
            "relation_refinement": False,
            "max_probability_mass_error": max_mass_error,
            "m4_manifest": model.manifest.to_dict(),
        },
        "used_for_model_selection": False,
        "production_ready_claimed": False,
        "claim_scope": (
            "Fresh English localization-only evidence for appending question "
            "content tokens to frozen W34 state evidence. This cannot qualify S2-A."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-case.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print("HIRA_V1_S2_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
