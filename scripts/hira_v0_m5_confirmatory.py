from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from nmd.contracts import LogicalOption
from nmd.local_runtime import load_hira_v0_m4_bundle


SCHEMA_VERSION = "hira-v0-mainline-m5-confirmatory-v1"
OUTCOME = "HIRA_V0_M5_CONFIRMATORY_READY"
QUESTION_COUNT_PER_STATE = 2


@dataclass(frozen=True)
class Case:
    case_id: str
    language: str
    category: str
    state: str
    question: str
    criteria: tuple[str, ...]
    gold_index: int

    def to_dict(self) -> dict:
        row = asdict(self)
        row["criteria"] = list(self.criteria)
        return row


def build_suite() -> tuple[Case, ...]:
    # Fresh confirmatory cases. None of these rows are sourced from the public
    # Laya/Jev benchmark datasets used in M5-B. The suite is intentionally
    # compact and balanced across English/Vietnamese and robustness phenomena.
    return (
        Case(
            "en_due_not_today", "en", "negation",
            "The report is due on Friday. Today is Monday.",
            "Is the report due today?",
            ("Yes, it is due today.", "No, it is not due today.",
             "The text does not give enough information.", "The question is not applicable."),
            1,
        ),
        Case(
            "en_server_healthy", "en", "negation",
            "The server is healthy and no errors are present.",
            "Is the server currently failing?",
            ("The server is failing.", "The server is not failing.",
             "The server state is unknown.", "Only the network state is known."),
            1,
        ),
        Case(
            "en_refund_unknown", "en", "insufficient_evidence",
            "Mira did not approve the refund.",
            "Who approved the refund?",
            ("Mira approved it.", "Another named person approved it.",
             "No approval occurred.", "The approver cannot be determined from the text."),
            3,
        ),
        Case(
            "en_before_noon", "en", "near_duplicate",
            "The shipment arrived at 11:50, ten minutes before noon.",
            "When did the shipment arrive?",
            ("Before noon.", "Exactly at noon.", "After noon.", "The arrival time is unknown."),
            0,
        ),
        Case(
            "en_badge_allowed", "en", "unseen_schema",
            "The policy allows blue or green badges. The worker has a green badge.",
            "Is the worker's badge allowed by the policy?",
            ("Allowed by the policy.", "Forbidden by the policy.",
             "The badge color is unknown.", "The policy does not mention badges."),
            0,
        ),
        Case(
            "en_weather_unknown", "en", "insufficient_evidence",
            "Alex owns a red bicycle. Nothing in the note describes the weather.",
            "Is it raining?",
            ("It is raining.", "It is not raining.",
             "The weather cannot be determined.", "The bicycle is not red."),
            2,
        ),
        Case(
            "en_budget_over", "en", "near_duplicate",
            "The spending cap is 100 dollars and the quoted price is 120 dollars.",
            "Is the quote within the spending cap?",
            ("Yes, it is within the cap.", "No, it exceeds the cap.",
             "It is exactly equal to the cap.", "The cap is unknown."),
            1,
        ),
        Case(
            "en_transitive_age", "en", "composition",
            "Sam is older than Lee. Lee is older than Kim.",
            "Is Sam older than Kim?",
            ("Yes.", "No.", "They are the same age.", "The relation cannot be determined."),
            0,
        ),
        Case(
            "en_exactly_one", "en", "composition",
            "Exactly one of service A and service B is active. Service A is inactive.",
            "Is service B active?",
            ("Yes, B is active.", "No, B is inactive.",
             "Both are active.", "The text is insufficient."),
            0,
        ),
        Case(
            "en_boundary_valid", "en", "near_duplicate",
            "The validator accepts integers from 1 through 5 inclusive. The input is 5.",
            "Is the input valid?",
            ("Valid.", "Invalid because 5 is above the range.",
             "Invalid because endpoints are excluded.", "The input is unknown."),
            0,
        ),
        Case(
            "en_refrigeration_unknown", "en", "insufficient_evidence",
            "The package is marked fragile. No storage temperature requirement is stated.",
            "Does the package require refrigeration?",
            ("Yes, refrigeration is required.", "No, refrigeration is forbidden.",
             "The requirement cannot be determined.", "The package is not fragile."),
            2,
        ),
        Case(
            "en_not_all", "en", "negation",
            "Not every request was approved.",
            "Were all requests approved?",
            ("Yes, all were approved.", "No, at least one was not approved.",
             "No requests were submitted.", "The text says nothing about approval."),
            1,
        ),
        Case(
            "vi_due_not_today", "vi", "negation",
            "Hạn nộp báo cáo là thứ Sáu. Hôm nay là thứ Hai.",
            "Báo cáo có đến hạn hôm nay không?",
            ("Có, đến hạn hôm nay.", "Không, chưa đến hạn hôm nay.",
             "Không đủ thông tin để biết.", "Câu hỏi không áp dụng."),
            1,
        ),
        Case(
            "vi_server_healthy", "vi", "negation",
            "Máy chủ đang hoạt động bình thường và không có lỗi.",
            "Máy chủ hiện có bị lỗi không?",
            ("Có, máy chủ đang lỗi.", "Không, máy chủ không bị lỗi.",
             "Không biết trạng thái máy chủ.", "Chỉ biết trạng thái mạng."),
            1,
        ),
        Case(
            "vi_refund_unknown", "vi", "insufficient_evidence",
            "Mai không duyệt khoản hoàn tiền.",
            "Ai đã duyệt khoản hoàn tiền?",
            ("Mai đã duyệt.", "Một người khác được nêu tên đã duyệt.",
             "Không hề có phê duyệt.", "Không thể xác định người duyệt từ thông tin đã cho."),
            3,
        ),
        Case(
            "vi_before_noon", "vi", "near_duplicate",
            "Chuyến hàng đến lúc 11 giờ 50, tức là trước buổi trưa mười phút.",
            "Chuyến hàng đến khi nào?",
            ("Trước buổi trưa.", "Đúng 12 giờ trưa.", "Sau buổi trưa.", "Không biết giờ đến."),
            0,
        ),
        Case(
            "vi_badge_allowed", "vi", "unseen_schema",
            "Quy định cho phép thẻ màu xanh dương hoặc xanh lá. Nhân viên có thẻ xanh lá.",
            "Thẻ của nhân viên có được phép không?",
            ("Được phép.", "Bị cấm.", "Không biết màu thẻ.", "Quy định không nói về thẻ."),
            0,
        ),
        Case(
            "vi_weather_unknown", "vi", "insufficient_evidence",
            "An có một chiếc xe đạp màu đỏ. Ghi chú không nói gì về thời tiết.",
            "Trời có đang mưa không?",
            ("Đang mưa.", "Không mưa.", "Không thể xác định thời tiết.", "Xe đạp không màu đỏ."),
            2,
        ),
        Case(
            "vi_budget_over", "vi", "near_duplicate",
            "Mức trần chi tiêu là 100 và báo giá là 120.",
            "Báo giá có nằm trong mức trần không?",
            ("Có, nằm trong mức trần.", "Không, vượt mức trần.",
             "Đúng bằng mức trần.", "Không biết mức trần."),
            1,
        ),
        Case(
            "vi_transitive_age", "vi", "composition",
            "Nam lớn tuổi hơn Bình. Bình lớn tuổi hơn Lan.",
            "Nam có lớn tuổi hơn Lan không?",
            ("Có.", "Không.", "Hai người bằng tuổi.", "Không thể xác định."),
            0,
        ),
        Case(
            "vi_exactly_one", "vi", "composition",
            "Chính xác một trong hai dịch vụ A và B đang hoạt động. Dịch vụ A không hoạt động.",
            "Dịch vụ B có đang hoạt động không?",
            ("Có, B đang hoạt động.", "Không, B không hoạt động.",
             "Cả hai đều hoạt động.", "Không đủ thông tin."),
            0,
        ),
        Case(
            "vi_boundary_invalid", "vi", "near_duplicate",
            "Bộ kiểm tra chỉ nhận số nguyên từ 1 đến 5, kể cả hai đầu. Đầu vào là 6.",
            "Đầu vào có hợp lệ không?",
            ("Hợp lệ.", "Không hợp lệ vì vượt quá 5.",
             "Không hợp lệ vì các đầu mút bị loại.", "Không biết đầu vào."),
            1,
        ),
        Case(
            "vi_temperature_over", "vi", "near_duplicate",
            "Khoảng nhiệt độ mục tiêu là từ 20 đến 22 độ C. Nhiệt độ hiện tại là 22,1 độ C.",
            "Nhiệt độ hiện tại có nằm trong khoảng mục tiêu không?",
            ("Có, nằm trong khoảng.", "Không, cao hơn khoảng mục tiêu.",
             "Đúng bằng giới hạn trên.", "Không biết nhiệt độ hiện tại."),
            1,
        ),
        Case(
            "vi_not_all", "vi", "negation",
            "Không phải tất cả yêu cầu đều được chấp thuận.",
            "Tất cả yêu cầu có được chấp thuận không?",
            ("Có, tất cả đều được chấp thuận.", "Không, có ít nhất một yêu cầu không được chấp thuận.",
             "Không có yêu cầu nào được gửi.", "Văn bản không nói về việc chấp thuận."),
            1,
        ),
    )


def suite_payload(cases: tuple[Case, ...]) -> dict:
    return {
        "schema_version": "hira-v0-mainline-m5-confirmatory-suite-v1",
        "created_after_public_m5_exposure": True,
        "used_for_model_selection": False,
        "languages": ["en", "vi"],
        "cases": [case.to_dict() for case in cases],
    }


def suite_sha256(cases: tuple[Case, ...]) -> str:
    payload = json.dumps(
        suite_payload(cases),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _options(case: Case) -> tuple[LogicalOption, ...]:
    return tuple(
        LogicalOption(
            option_id=f"{case.case_id}__opaque_{index:02d}",
            criterion_text=criterion,
        )
        for index, criterion in enumerate(case.criteria)
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = build_suite()
    if len(cases) != 24:
        raise RuntimeError("confirmatory suite size changed")
    if sum(case.language == "en" for case in cases) != 12:
        raise RuntimeError("confirmatory English balance changed")
    if sum(case.language == "vi" for case in cases) != 12:
        raise RuntimeError("confirmatory Vietnamese balance changed")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = suite_payload(cases)
    payload["suite_sha256"] = suite_sha256(cases)
    (args.out / "suite.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model = load_hira_v0_m4_bundle(args.bundle)
    if model.parameter_report().trainable_total != 0:
        raise RuntimeError("confirmatory runtime has trainable parameters")
    if model.manifest.production_ready:
        raise RuntimeError("confirmatory runtime unexpectedly claims production readiness")

    before_state_calls = model.runtime.state_encode_calls
    rows: list[dict] = []
    correct_original = 0
    correct_reversed = 0
    flips = 0
    max_mass_error = 0.0

    print("HIRA_V0_M5_CONFIRMATORY_FINAL_EXPOSURE_BEGIN", flush=True)

    for case in cases:
        options = _options(case)
        gold_id = options[case.gold_index].option_id
        session = model.open_session(case.state)

        outputs = []
        for order_name, ordered in (
            ("original", options),
            ("reversed", tuple(reversed(options))),
        ):
            out = session.decide(
                primitive="choice",
                question_text=case.question,
                options=ordered,
                use_schema_cache=False,
            )
            probs = out.probabilities.detach().cpu().to(torch.float64)
            if probs.numel() != len(options):
                raise RuntimeError(f"{case.case_id}: probability width changed")
            if not bool(torch.isfinite(probs).all()):
                raise RuntimeError(f"{case.case_id}: non-finite probabilities")
            mass_error = abs(float(probs.sum()) - 1.0)
            max_mass_error = max(max_mass_error, mass_error)
            if mass_error > 1e-6:
                raise RuntimeError(f"{case.case_id}: probability mass failed")
            if int(out.hira.candidate_budget.item()) != len(options):
                raise RuntimeError(f"{case.case_id}: full-K failed")
            if float(out.hira.relation_delta.detach().abs().max().cpu()) != 0.0:
                raise RuntimeError(f"{case.case_id}: relation refinement changed")
            outputs.append((order_name, out.selected_option_id, probs.tolist()))

        if session.query_count != QUESTION_COUNT_PER_STATE:
            raise RuntimeError(f"{case.case_id}: state-once query count changed")

        original_id = outputs[0][1]
        reversed_id = outputs[1][1]
        correct_original += int(original_id == gold_id)
        correct_reversed += int(reversed_id == gold_id)
        flips += int(original_id != reversed_id)

        rows.append(
            {
                "case_id": case.case_id,
                "language": case.language,
                "category": case.category,
                "state_sha256": sha256(case.state.encode("utf-8")).hexdigest(),
                "question_sha256": sha256(case.question.encode("utf-8")).hexdigest(),
                "gold_option_id": gold_id,
                "original_selected_option_id": original_id,
                "reversed_selected_option_id": reversed_id,
                "original_correct": original_id == gold_id,
                "reversed_correct": reversed_id == gold_id,
                "order_flip": original_id != reversed_id,
                "original_probabilities": outputs[0][2],
                "reversed_probabilities": outputs[1][2],
            }
        )

    state_calls = model.runtime.state_encode_calls - before_state_calls
    if state_calls != len(cases):
        raise RuntimeError(
            f"confirmatory state-once count changed: {state_calls} != {len(cases)}"
        )

    def accuracy(filtered: list[dict], key: str) -> float:
        return sum(bool(row[key]) for row in filtered) / len(filtered)

    by_language = {}
    for language in ("en", "vi"):
        subset = [row for row in rows if row["language"] == language]
        by_language[language] = {
            "cases": len(subset),
            "original_accuracy": accuracy(subset, "original_correct"),
            "reversed_accuracy": accuracy(subset, "reversed_correct"),
            "order_flip_rate": sum(row["order_flip"] for row in subset) / len(subset),
        }

    categories = sorted({row["category"] for row in rows})
    by_category = {}
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "cases": len(subset),
            "original_accuracy": accuracy(subset, "original_correct"),
            "reversed_accuracy": accuracy(subset, "reversed_correct"),
            "order_flip_rate": sum(row["order_flip"] for row in subset) / len(subset),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "outcome": OUTCOME,
        "scientific_authority": "M5_FRESH_CONFIRMATORY_ROBUSTNESS",
        "suite_sha256": suite_sha256(cases),
        "suite": {
            "base_cases": len(cases),
            "queries": len(cases) * QUESTION_COUNT_PER_STATE,
            "languages": ["en", "vi"],
            "opaque_option_ids": True,
            "order_permutations_per_case": 2,
            "fresh_rows_not_from_laya_jev_public_sets": True,
            "created_after_public_m5_exposure": True,
        },
        "scores": {
            "original_accuracy": correct_original / len(cases),
            "reversed_accuracy": correct_reversed / len(cases),
            "paired_both_correct_rate": sum(
                row["original_correct"] and row["reversed_correct"] for row in rows
            ) / len(rows),
            "order_flip_rate": flips / len(cases),
            "max_probability_mass_error": max_mass_error,
            "by_language": by_language,
            "by_category": by_category,
        },
        "runtime": {
            "manifest": model.manifest.to_dict(),
            "parameter_report": model.parameter_report().to_dict(),
            "state_encode_calls": state_calls,
            "queries_per_state": QUESTION_COUNT_PER_STATE,
        },
        "claim_scope": (
            "Fresh confirmatory robustness evidence only; no Laya/Jev parity or "
            "general superiority claim is authorized by this suite."
        ),
        "used_for_model_selection": False,
        "model_weights_changed_after_public_exposure": False,
        "production_ready_claimed": False,
        "authority_head": os.environ.get("GITHUB_SHA"),
    }

    (args.out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out / "per-case.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print("HIRA_V0_M5_CONFIRMATORY_RECEIPT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
