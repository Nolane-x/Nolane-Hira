from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle


SCHEMA_VERSION = "hira-v1-s0-question-blindness-v1"
OUTCOME_CONFIRMED = "HIRA_V1_S0_QUESTION_BLINDNESS_CONFIRMED"
OUTCOME_NOT_CONFIRMED = "HIRA_V1_S0_QUESTION_BLINDNESS_NOT_CONFIRMED"


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    language: str
    state: str
    question_a: str
    question_b: str
    criteria: tuple[str, ...]
    gold_a: int
    gold_b: int

    def options(self) -> tuple[LogicalOption, ...]:
        return tuple(
            LogicalOption(
                option_id=f"{self.case_id}__opaque_{index:02d}",
                criterion_text=criterion,
            )
            for index, criterion in enumerate(self.criteria)
        )

    def to_dict(self) -> dict:
        row = asdict(self)
        row["criteria"] = list(self.criteria)
        return row


def cases() -> tuple[DiagnosticCase, ...]:
    return (
        DiagnosticCase(
            "en_bicycle",
            "en",
            "Mira bought a red bicycle on Tuesday.",
            "What color was the bicycle?",
            "On which day was the bicycle bought?",
            ("red", "Tuesday", "the requested fact is unknown"),
            0,
            1,
        ),
        DiagnosticCase(
            "en_package",
            "en",
            "The package weighs 12 kilograms and is stored in Berlin.",
            "How much does the package weigh?",
            "Where is the package stored?",
            ("12 kilograms", "Berlin", "the requested fact is unknown"),
            0,
            1,
        ),
        DiagnosticCase(
            "en_server",
            "en",
            "The server latency is 80 milliseconds and its status is healthy.",
            "What is the server latency?",
            "What is the server status?",
            ("80 milliseconds", "healthy", "the requested fact is unknown"),
            0,
            1,
        ),
        DiagnosticCase(
            "en_train",
            "en",
            "The train leaves from platform 4 at 09:30.",
            "Which platform does the train leave from?",
            "At what time does the train leave?",
            ("platform 4", "09:30", "the requested fact is unknown"),
            0,
            1,
        ),
        DiagnosticCase(
            "vi_ao",
            "vi",
            "Lan mua một chiếc áo màu xanh vào Chủ nhật.",
            "Chiếc áo có màu gì?",
            "Lan mua chiếc áo vào ngày nào?",
            ("màu xanh", "Chủ nhật", "không xác định được thông tin được hỏi"),
            0,
            1,
        ),
        DiagnosticCase(
            "vi_phong",
            "vi",
            "Phòng có nhiệt độ 21 độ C và độ ẩm 45 phần trăm.",
            "Nhiệt độ của phòng là bao nhiêu?",
            "Độ ẩm của phòng là bao nhiêu?",
            ("21 độ C", "45 phần trăm", "không xác định được thông tin được hỏi"),
            0,
            1,
        ),
        DiagnosticCase(
            "vi_hoa_don",
            "vi",
            "Hóa đơn có tổng tiền 240 đô la và hạn thanh toán là ngày 3 tháng 10.",
            "Tổng tiền của hóa đơn là bao nhiêu?",
            "Hạn thanh toán của hóa đơn là ngày nào?",
            ("240 đô la", "ngày 3 tháng 10", "không xác định được thông tin được hỏi"),
            0,
            1,
        ),
        DiagnosticCase(
            "vi_tau",
            "vi",
            "Tàu khởi hành ở sân ga số 6 lúc 14 giờ 15.",
            "Tàu khởi hành ở sân ga số mấy?",
            "Tàu khởi hành lúc mấy giờ?",
            ("sân ga số 6", "14 giờ 15", "không xác định được thông tin được hỏi"),
            0,
            1,
        ),
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suite = cases()
    if len(suite) != 8:
        raise RuntimeError("S0 question-blindness suite size changed")
    if sum(case.language == "en" for case in suite) != 4:
        raise RuntimeError("S0 English balance changed")
    if sum(case.language == "vi" for case in suite) != 4:
        raise RuntimeError("S0 Vietnamese balance changed")
    if any(case.gold_a == case.gold_b for case in suite):
        raise RuntimeError("every S0 case must change gold answer with question")

    model = load_hira_v0_m4_bundle(args.bundle)
    runtime = model.runtime
    before_state_calls = runtime.state_encode_calls

    rows = []
    identical_logits = 0
    identical_probabilities = 0
    identical_choices = 0
    question_tokens_changed = 0

    print("HIRA_V1_S0_QUESTION_BLINDNESS_EXPOSURE_BEGIN", flush=True)

    for case in suite:
        options = case.options()
        memory = runtime.compile_state(case.state)

        schema_a, _ = runtime.compile_schema(
            primitive="choice",
            question_text=case.question_a,
            options=options,
            include_token_artifacts=True,
            use_cache=False,
        )
        schema_b, _ = runtime.compile_schema(
            primitive="choice",
            question_text=case.question_b,
            options=options,
            include_token_artifacts=True,
            use_cache=False,
        )
        if schema_a.schema_hash == schema_b.schema_hash:
            raise RuntimeError(f"{case.case_id}: question swap did not change schema hash")
        if schema_a.question_token_embeddings is None or schema_b.question_token_embeddings is None:
            raise RuntimeError(f"{case.case_id}: missing question token artifacts")

        question_changed = not torch.equal(
            schema_a.question_token_embeddings,
            schema_b.question_token_embeddings,
        )
        question_tokens_changed += int(question_changed)

        out_a = runtime.forward_compiled(
            memory,
            schema_a,
            coarse_mode="coevidence_symmetric_semantic",
            relation_refinement=False,
        )
        out_b = runtime.forward_compiled(
            memory,
            schema_b,
            coarse_mode="coevidence_symmetric_semantic",
            relation_refinement=False,
        )

        logits_same = torch.equal(out_a.logits, out_b.logits)
        probs_same = torch.equal(out_a.probabilities, out_b.probabilities)
        choice_same = out_a.selected_option_id == out_b.selected_option_id
        identical_logits += int(logits_same)
        identical_probabilities += int(probs_same)
        identical_choices += int(choice_same)

        if int(out_a.hira.candidate_budget.item()) != len(options):
            raise RuntimeError(f"{case.case_id}: A full-K failed")
        if int(out_b.hira.candidate_budget.item()) != len(options):
            raise RuntimeError(f"{case.case_id}: B full-K failed")
        if not bool(out_a.hira.selected_mask.all()) or not bool(out_b.hira.selected_mask.all()):
            raise RuntimeError(f"{case.case_id}: selected mask is not full-K")
        if float(out_a.hira.relation_delta.abs().max()) != 0.0:
            raise RuntimeError(f"{case.case_id}: A relation refinement changed")
        if float(out_b.hira.relation_delta.abs().max()) != 0.0:
            raise RuntimeError(f"{case.case_id}: B relation refinement changed")

        rows.append(
            {
                "case_id": case.case_id,
                "language": case.language,
                "gold_a": options[case.gold_a].option_id,
                "gold_b": options[case.gold_b].option_id,
                "selected_a": out_a.selected_option_id,
                "selected_b": out_b.selected_option_id,
                "question_tokens_changed": question_changed,
                "identical_logits": logits_same,
                "identical_probabilities": probs_same,
                "identical_choice": choice_same,
                "logits_a": [float(x) for x in out_a.logits.cpu().tolist()],
                "logits_b": [float(x) for x in out_b.logits.cpu().tolist()],
            }
        )

    state_calls = runtime.state_encode_calls - before_state_calls
    if state_calls != len(suite):
        raise RuntimeError(
            f"S0 state-once changed: {state_calls} != {len(suite)}"
        )

    n = len(suite)
    logit_rate = identical_logits / n
    probability_rate = identical_probabilities / n
    choice_rate = identical_choices / n
    question_change_rate = question_tokens_changed / n

    confirmed = (
        logit_rate == 1.0
        and probability_rate == 1.0
        and choice_rate == 1.0
        and question_change_rate == 1.0
    )
    outcome = OUTCOME_CONFIRMED if confirmed else OUTCOME_NOT_CONFIRMED

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": outcome,
        "scientific_authority": "V1_S0_LOCALIZATION_ONLY",
        "case_count": n,
        "languages": ["en", "vi"],
        "gold_answer_changes_with_question_rate": 1.0,
        "question_token_embeddings_change_rate": question_change_rate,
        "identical_logit_rate": logit_rate,
        "identical_probability_rate": probability_rate,
        "identical_choice_rate": choice_rate,
        "state_encode_calls": state_calls,
        "runtime": {
            "manifest": model.manifest.to_dict(),
            "parameter_report": model.parameter_report().to_dict(),
            "coarse_mode": "coevidence_symmetric_semantic",
            "relation_refinement": False,
            "full_k": True,
        },
        "interpretation": (
            "With state and option schema fixed, the frozen Hira v0 W34 "
            "co-evidence coarse path produced identical decision logits while "
            "the compiled question token embeddings changed. This confirms "
            "question blindness for this coarse path; it does not prove that "
            "question omission is the only semantic-quality bottleneck."
        ),
        "used_for_model_selection": False,
        "production_ready_claimed": False,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "suite.json").write_text(
        json.dumps(
            {
                "schema_version": "hira-v1-s0-question-blindness-suite-v1",
                "cases": [case.to_dict() for case in suite],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (args.out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-case.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print("HIRA_V1_S0_QUESTION_BLINDNESS_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
