from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hira_v1_s0_question_blindness import cases
from nmd.local_runtime import load_hira_v0_m4_bundle
from nmd.v1_parameter_free_query import ParameterFreeQueryCoEvidenceScorer
from nmd.w34_transfer_core import W34_CANDIDATE_KEYS


SCHEMA_VERSION = "hira-v1-s0-parameter-free-query-v1"
OUTCOME = "HIRA_V1_S0_PARAMETER_FREE_QUERY_BASELINE_READY"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--v0-result", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = load_hira_v0_m4_bundle(args.bundle)
    runtime = model.runtime
    base = runtime.coevidence_symmetric_semantic_scorer
    if base is None:
        raise RuntimeError("frozen M4 runtime has no W34 co-evidence scorer")

    scorer = ParameterFreeQueryCoEvidenceScorer(
        d_model=256,
        d_rel=128,
        rank=8,
    )
    scorer.load_projection_weight(
        base.projection.weight.detach().cpu(),
        freeze=True,
    )
    base_state = base.state_dict()
    scorer.load_candidate_state_dict(
        {
            key: base_state[key].detach().cpu()
            for key in W34_CANDIDATE_KEYS
        },
        freeze=True,
    )
    scorer.eval()

    if scorer.added_parameter_count != 0:
        raise RuntimeError("parameter-free baseline added parameters")
    if scorer.trainable_parameter_count != 0:
        raise RuntimeError("parameter-free baseline has trainable parameters")

    v0 = None
    if args.v0_result is not None:
        v0 = json.loads(args.v0_result.read_text(encoding="utf-8"))
        if v0.get("outcome") != "HIRA_V1_S0_QUESTION_BLINDNESS_CONFIRMED":
            raise RuntimeError("S0 v0 question-blindness authority is not qualified")

    suite = cases()
    before_state_calls = runtime.state_encode_calls
    rows = []
    changed_logits = 0
    changed_choice = 0
    correct = 0
    both_correct = 0
    max_mass_error = 0.0

    print("HIRA_V1_S0_PARAMETER_FREE_QUERY_EXPOSURE_BEGIN", flush=True)

    for case in suite:
        options = case.options()
        memory = runtime.compile_state(case.state)
        state_tokens = memory.content_token_embeddings.unsqueeze(0)
        state_mask = torch.ones(
            1,
            state_tokens.shape[1],
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

            logits = scorer(
                state_tokens=state_tokens,
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
                raise RuntimeError(f"{case.case_id}: non-finite logits")

            predicted_index = int(probs.argmax().item())
            is_correct = predicted_index == gold_index
            correct += int(is_correct)
            pair.append(
                {
                    "label": label,
                    "gold_index": gold_index,
                    "predicted_index": predicted_index,
                    "selected_option_id": options[predicted_index].option_id,
                    "correct": is_correct,
                    "logits": logits.detach().cpu(),
                    "probabilities": probs.detach().cpu(),
                }
            )

        logits_changed = not torch.equal(pair[0]["logits"], pair[1]["logits"])
        choice_changed = pair[0]["predicted_index"] != pair[1]["predicted_index"]
        changed_logits += int(logits_changed)
        changed_choice += int(choice_changed)
        both_correct += int(pair[0]["correct"] and pair[1]["correct"])

        rows.append(
            {
                "case_id": case.case_id,
                "language": case.language,
                "logits_changed_with_question": logits_changed,
                "choice_changed_with_question": choice_changed,
                "a": {
                    key: (
                        [float(x) for x in value.tolist()]
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in pair[0].items()
                },
                "b": {
                    key: (
                        [float(x) for x in value.tolist()]
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in pair[1].items()
                },
            }
        )

    state_calls = runtime.state_encode_calls - before_state_calls
    if state_calls != len(suite):
        raise RuntimeError(
            f"parameter-free state-once changed: {state_calls} != {len(suite)}"
        )

    query_count = len(suite) * 2
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "V1_S0_PARAMETER_FREE_LOCALIZATION_BASELINE",
        "case_count": len(suite),
        "query_count": query_count,
        "languages": ["en", "vi"],
        "added_parameter_count": 0,
        "candidate_parameter_count": int(scorer.candidate_parameter_count),
        "trainable_parameter_count": int(scorer.trainable_parameter_count),
        "question_changes_logits_rate": changed_logits / len(suite),
        "question_changes_choice_rate": changed_choice / len(suite),
        "accuracy": correct / query_count,
        "paired_both_correct_rate": both_correct / len(suite),
        "state_encode_calls": state_calls,
        "max_probability_mass_error": max_mass_error,
        "v0_localization": (
            None
            if v0 is None
            else {
                "artifact_outcome": v0["outcome"],
                "identical_logit_rate": v0["identical_logit_rate"],
                "identical_choice_rate": v0["identical_choice_rate"],
            }
        ),
        "interpretation_boundary": (
            "This zero-new-parameter baseline measures whether explicit "
            "question conditioning changes geometry on the S0 localization "
            "suite. It is not a promotion set and cannot establish external "
            "benchmark quality or production readiness."
        ),
        "used_for_final_model_selection": False,
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

    print("HIRA_V1_S0_PARAMETER_FREE_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
