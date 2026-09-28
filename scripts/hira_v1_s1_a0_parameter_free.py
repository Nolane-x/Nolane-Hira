from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle
from nmd.v1_query_keyed_evidence import ParameterFreeQueryEvidenceScorer
from nmd.w34_transfer_core import W34_CANDIDATE_KEYS


SCHEMA_VERSION = "hira-v1-s1-a0-parameter-free-evidence-v1"
OUTCOME = "HIRA_V1_S1_A0_PARAMETER_FREE_EVIDENCE_READY"


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
        Case("museum", "The bronze sculpture is displayed in gallery seven.", "What material is the sculpture made from?", "Where is the sculpture displayed?", ("bronze", "gallery seven", "marble", "gallery two"), 0, 1),
        Case("rover", "The rover battery is at 68 percent and it is operating in sector Delta.", "What is the rover battery level?", "Which sector is the rover operating in?", ("68 percent", "sector Delta", "42 percent", "sector Kappa"), 0, 1),
        Case("tea", "The tea should steep at 85 degrees Celsius for four minutes.", "What temperature should be used?", "How long should the tea steep?", ("85 degrees Celsius", "four minutes", "95 degrees Celsius", "eight minutes"), 0, 1),
        Case("ferry", "The ferry departs from pier C at 16:40.", "Which pier does the ferry depart from?", "When does the ferry depart?", ("pier C", "16:40", "pier A", "18:10"), 0, 1),
        Case("archive", "The archive box is on shelf twelve and its code is RQ-47.", "Which shelf holds the archive box?", "What is the archive box code?", ("shelf twelve", "RQ-47", "shelf six", "TZ-18"), 0, 1),
        Case("telescope", "The telescope uses the H-alpha filter with a 90 second exposure.", "Which filter does the telescope use?", "What exposure time is used?", ("H-alpha", "90 seconds", "O-III", "30 seconds"), 0, 1),
        Case("bakery", "The bakery oven is set to 190 degrees and the batch uses six trays.", "What temperature is the oven set to?", "How many trays are in the batch?", ("190 degrees", "six trays", "160 degrees", "three trays"), 0, 1),
        Case("orchard", "The pear trees are in row fourteen and the harvest crate is yellow.", "Which row contains the pear trees?", "What color is the harvest crate?", ("row fourteen", "yellow", "row nine", "blue"), 0, 1),
        Case("drone", "The survey drone flies at 120 meters along route Echo.", "What altitude does the survey drone fly at?", "Which route does the drone follow?", ("120 meters", "route Echo", "80 meters", "route Foxtrot"), 0, 1),
        Case("aquarium", "Tank B contains angelfish and is maintained at 26 degrees Celsius.", "Which fish are in tank B?", "What temperature is tank B maintained at?", ("angelfish", "26 degrees Celsius", "clownfish", "22 degrees Celsius"), 0, 1),
        Case("workshop", "The torque wrench is stored at bench five and is calibrated to 60 newton meters.", "Where is the torque wrench stored?", "What torque is it calibrated to?", ("bench five", "60 newton meters", "bench two", "35 newton meters"), 0, 1),
        Case("stadium", "Spectators for section 214 enter through gate North.", "Which section is assigned?", "Which gate should spectators use?", ("section 214", "gate North", "section 118", "gate South"), 0, 1),
        Case("greenhouse", "Greenhouse zone F is maintained at 72 percent humidity.", "Which greenhouse zone is described?", "What humidity is maintained?", ("zone F", "72 percent", "zone B", "55 percent"), 0, 1),
        Case("cinema", "The documentary plays on screen nine at 20:25.", "Which screen shows the documentary?", "What time does the documentary play?", ("screen nine", "20:25", "screen four", "18:45"), 0, 1),
        Case("sensor", "The pressure sensor reports 3.3 volts on channel eight.", "What voltage does the pressure sensor report?", "Which channel carries the reading?", ("3.3 volts", "channel eight", "5 volts", "channel three"), 0, 1),
        Case("bookstore", "The history section is on the third floor beside the east staircase.", "Which floor contains the history section?", "Which staircase is beside the history section?", ("third floor", "east staircase", "first floor", "west staircase"), 0, 1),
    )


def _build_a0(base) -> ParameterFreeQueryEvidenceScorer:
    scorer = ParameterFreeQueryEvidenceScorer(d_model=256, d_rel=128, rank=8)
    scorer.load_projection_weight(base.projection.weight.detach().cpu(), freeze=True)
    own = base.state_dict()
    scorer.load_candidate_state_dict(
        {key: own[key].detach().cpu() for key in W34_CANDIDATE_KEYS},
        freeze=True,
    )
    scorer.eval()
    if scorer.added_parameter_count != 0:
        raise RuntimeError("S1-A0 unexpectedly added parameters")
    if scorer.trainable_parameter_count != 0:
        raise RuntimeError("S1-A0 must be fully frozen")
    return scorer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 16:
        raise RuntimeError("S1-A0 suite size changed")
    if len({c.case_id for c in suite}) != len(suite):
        raise RuntimeError("S1-A0 duplicate case id")
    if any(c.gold_a == c.gold_b for c in suite):
        raise RuntimeError("S1-A0 paired gold must change with question")

    model = load_hira_v0_m4_bundle(args.bundle)
    runtime = model.runtime
    base = runtime.coevidence_symmetric_semantic_scorer
    if base is None:
        raise RuntimeError("M4 runtime missing frozen W34 scorer")
    scorer = _build_a0(base)

    before_state_calls = runtime.state_encode_calls
    rows = []
    a0_correct = 0
    v0_correct = 0
    a0_both = 0
    v0_both = 0
    a0_choice_changes = 0
    v0_choice_changes = 0
    evidence_changes = 0
    max_mass_error = 0.0

    print("HIRA_V1_S1_A0_EXPOSURE_BEGIN", flush=True)

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
        evidence_signatures = []

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

            # Frozen v0 control on the same fresh row.
            v0 = runtime.forward_compiled(
                memory,
                schema,
                coarse_mode="coevidence_symmetric_semantic",
                relation_refinement=False,
            )

            evidence, _ = scorer.extract_evidence(
                state_tokens=state_tokens_b,
                state_mask=state_mask,
                question_tokens=schema.question_token_embeddings.unsqueeze(0),
                question_mask=schema.question_content_token_mask.unsqueeze(0),
            )
            evidence_signatures.append(evidence.detach().cpu())

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
                raise RuntimeError(f"{case.case_id}: A0 probability mass failed")
            if int(v0.hira.candidate_budget.item()) != len(options):
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
                    "a0_logits": [float(x) for x in logits.cpu().tolist()],
                    "v0_logits": [float(x) for x in v0.logits.cpu().tolist()],
                }
            )

        e_changed = not torch.equal(evidence_signatures[0], evidence_signatures[1])
        evidence_changes += int(e_changed)
        a0_changed = pair[0]["a0_index"] != pair[1]["a0_index"]
        v0_changed = pair[0]["v0_index"] != pair[1]["v0_index"]
        a0_choice_changes += int(a0_changed)
        v0_choice_changes += int(v0_changed)
        a0_both += int(pair[0]["a0_correct"] and pair[1]["a0_correct"])
        v0_both += int(pair[0]["v0_correct"] and pair[1]["v0_correct"])

        rows.append(
            {
                "case_id": case.case_id,
                "evidence_changed_with_question": e_changed,
                "a0_choice_changed_with_question": a0_changed,
                "v0_choice_changed_with_question": v0_changed,
                "a": pair[0],
                "b": pair[1],
            }
        )

    state_calls = runtime.state_encode_calls - before_state_calls
    if state_calls != len(suite):
        raise RuntimeError(
            f"S1-A0 state-once changed: {state_calls} != {len(suite)}"
        )

    queries = len(suite) * 2
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "S1_A0_LOCALIZATION_ONLY",
        "case_count": len(suite),
        "query_count": queries,
        "language": "en",
        "k": 4,
        "added_parameter_count": 0,
        "trainable_parameter_count": int(scorer.trainable_parameter_count),
        "a0": {
            "accuracy": a0_correct / queries,
            "paired_both_correct_rate": a0_both / len(suite),
            "question_changes_choice_rate": a0_choice_changes / len(suite),
            "question_changes_evidence_rate": evidence_changes / len(suite),
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
            "Fresh English localization-only evidence for zero-parameter "
            "query-keyed state evidence extraction. It cannot qualify S1-A."
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

    print("HIRA_V1_S1_A0_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
