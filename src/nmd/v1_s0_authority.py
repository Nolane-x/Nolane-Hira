from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Literal


Split = Literal["train", "dev"]


@dataclass(frozen=True)
class S0PairCase:
    case_id: str
    split: Split
    domain: str
    language: str
    state: str
    question_a: str
    question_b: str
    option_texts: tuple[str, ...]
    option_ids: tuple[str, ...]
    gold_a: int
    gold_b: int

    def to_dict(self) -> dict:
        row = asdict(self)
        row["option_texts"] = list(self.option_texts)
        row["option_ids"] = list(self.option_ids)
        return row


_EN_ROOMS = ("Cedar", "Maple", "Quartz", "Willow", "Harbor", "Linden", "Opal", "Birch")
_EN_TIMES = ("07:20", "08:35", "09:50", "11:05", "13:25", "14:40", "16:15", "18:30")
_EN_PLANS = ("Aurora", "Beacon", "Cascade", "Delta", "Ember", "Fjord", "Grove", "Helix")
_EN_MONTHS = ("January", "February", "April", "May", "July", "August", "October", "December")

_VI_ROOMS = ("A12", "B07", "C14", "D03", "E18", "F09", "G21", "H05")
_VI_PERIODS = ("tiết 1", "tiết 2", "tiết 3", "tiết 4", "tiết 6", "tiết 7", "tiết 8", "tiết 9")
_VI_OWNERS = ("Huy", "Thảo", "Quân", "Ngọc", "Duy", "Vy", "Khang", "Mai")
_VI_PRIORITIES = ("rất thấp", "thấp", "trung bình", "cao", "rất cao", "khẩn", "thường", "ưu tiên")


def _shuffle_options(
    *,
    case_id: str,
    first: str,
    second: str,
    distractor_first: str,
    distractor_second: str,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    values = [
        ("a", first),
        ("b", second),
        ("x", distractor_first),
        ("y", distractor_second),
    ]
    rng = random.Random(seed)
    rng.shuffle(values)

    option_texts = tuple(text for _, text in values)
    option_ids = tuple(
        f"{case_id}__opaque_{index:02d}"
        for index in range(len(values))
    )
    gold_a = next(index for index, (kind, _) in enumerate(values) if kind == "a")
    gold_b = next(index for index, (kind, _) in enumerate(values) if kind == "b")
    return option_texts, option_ids, gold_a, gold_b


def _en_event(index: int, split: Split) -> S0PairCase:
    offset = index + (1000 if split == "dev" else 0)
    room = _EN_ROOMS[index % len(_EN_ROOMS)]
    time = _EN_TIMES[(index * 3 + 1) % len(_EN_TIMES)]
    other_room = _EN_ROOMS[(index + 3) % len(_EN_ROOMS)]
    other_time = _EN_TIMES[(index * 5 + 2) % len(_EN_TIMES)]
    case_id = f"{split}-en-event-{index:03d}"
    if split == "train":
        state = (
            f"Event record EV{offset:04d} lists venue Hall {room} and start time {time}."
        )
        question_a = "Which hall is listed as the event venue?"
        question_b = "What start time is listed for the event?"
    else:
        state = (
            f"For event code DV{offset:04d}, the scheduled location is Hall {room}; "
            f"the session begins at {time}."
        )
        question_a = "Where is this event scheduled to take place?"
        question_b = "When is this event scheduled to begin?"
    option_texts, option_ids, gold_a, gold_b = _shuffle_options(
        case_id=case_id,
        first=f"the requested answer is Hall {room}",
        second=f"the requested answer is {time}",
        distractor_first=f"the requested answer is Hall {other_room}",
        distractor_second=f"the requested answer is {other_time}",
        seed=5101 + offset,
    )
    return S0PairCase(
        case_id, split, "event_record", "en", state,
        question_a, question_b, option_texts, option_ids, gold_a, gold_b,
    )


def _en_account(index: int, split: Split) -> S0PairCase:
    offset = index + (2000 if split == "dev" else 0)
    plan = _EN_PLANS[index % len(_EN_PLANS)]
    month = _EN_MONTHS[(index * 3 + 2) % len(_EN_MONTHS)]
    other_plan = _EN_PLANS[(index + 5) % len(_EN_PLANS)]
    other_month = _EN_MONTHS[(index * 5 + 1) % len(_EN_MONTHS)]
    case_id = f"{split}-en-account-{index:03d}"
    if split == "train":
        state = (
            f"Account AC{offset:04d} is assigned plan {plan} and renews in {month}."
        )
        question_a = "Which plan is assigned to the account?"
        question_b = "In which month does the account renew?"
    else:
        state = (
            f"Customer file DC{offset:04d} shows subscription tier {plan}; "
            f"its next renewal month is {month}."
        )
        question_a = "What subscription tier does this customer file show?"
        question_b = "What is the next renewal month for this customer?"
    option_texts, option_ids, gold_a, gold_b = _shuffle_options(
        case_id=case_id,
        first=f"the requested answer is {plan}",
        second=f"the requested answer is {month}",
        distractor_first=f"the requested answer is {other_plan}",
        distractor_second=f"the requested answer is {other_month}",
        seed=5201 + offset,
    )
    return S0PairCase(
        case_id, split, "account_record", "en", state,
        question_a, question_b, option_texts, option_ids, gold_a, gold_b,
    )


def _vi_class(index: int, split: Split) -> S0PairCase:
    offset = index + (3000 if split == "dev" else 0)
    room = _VI_ROOMS[index % len(_VI_ROOMS)]
    period = _VI_PERIODS[(index * 3 + 1) % len(_VI_PERIODS)]
    other_room = _VI_ROOMS[(index + 3) % len(_VI_ROOMS)]
    other_period = _VI_PERIODS[(index * 5 + 2) % len(_VI_PERIODS)]
    case_id = f"{split}-vi-class-{index:03d}"
    if split == "train":
        state = (
            f"Hồ sơ lớp LH{offset:04d} ghi phòng học {room} và thời gian bắt đầu là {period}."
        )
        question_a = "Lớp học được xếp ở phòng nào?"
        question_b = "Lớp học bắt đầu vào tiết mấy?"
    else:
        state = (
            f"Phiếu học vụ DH{offset:04d} cho biết địa điểm là phòng {room}; "
            f"buổi học khởi đầu vào {period}."
        )
        question_a = "Địa điểm của buổi học này là phòng nào?"
        question_b = "Buổi học này khởi đầu vào tiết nào?"
    option_texts, option_ids, gold_a, gold_b = _shuffle_options(
        case_id=case_id,
        first=f"câu trả lời được hỏi là phòng {room}",
        second=f"câu trả lời được hỏi là {period}",
        distractor_first=f"câu trả lời được hỏi là phòng {other_room}",
        distractor_second=f"câu trả lời được hỏi là {other_period}",
        seed=5301 + offset,
    )
    return S0PairCase(
        case_id, split, "class_record", "vi", state,
        question_a, question_b, option_texts, option_ids, gold_a, gold_b,
    )


def _vi_ticket(index: int, split: Split) -> S0PairCase:
    offset = index + (4000 if split == "dev" else 0)
    owner = _VI_OWNERS[index % len(_VI_OWNERS)]
    priority = _VI_PRIORITIES[(index * 3 + 1) % len(_VI_PRIORITIES)]
    other_owner = _VI_OWNERS[(index + 5) % len(_VI_OWNERS)]
    other_priority = _VI_PRIORITIES[(index * 5 + 2) % len(_VI_PRIORITIES)]
    case_id = f"{split}-vi-ticket-{index:03d}"
    if split == "train":
        state = (
            f"Phiếu yêu cầu YC{offset:04d} do {owner} phụ trách và có mức ưu tiên {priority}."
        )
        question_a = "Ai là người phụ trách phiếu yêu cầu?"
        question_b = "Phiếu yêu cầu có mức ưu tiên nào?"
    else:
        state = (
            f"Biên bản hỗ trợ DH{offset:04d} nêu người xử lý là {owner}; "
            f"mức độ ưu tiên được ghi là {priority}."
        )
        question_a = "Người xử lý được ghi trong biên bản là ai?"
        question_b = "Mức độ ưu tiên ghi trong biên bản là gì?"
    option_texts, option_ids, gold_a, gold_b = _shuffle_options(
        case_id=case_id,
        first=f"câu trả lời được hỏi là {owner}",
        second=f"câu trả lời được hỏi là {priority}",
        distractor_first=f"câu trả lời được hỏi là {other_owner}",
        distractor_second=f"câu trả lời được hỏi là {other_priority}",
        seed=5401 + offset,
    )
    return S0PairCase(
        case_id, split, "support_record", "vi", state,
        question_a, question_b, option_texts, option_ids, gold_a, gold_b,
    )


def generate_s0_pairs(split: Split) -> tuple[S0PairCase, ...]:
    per_domain = 64 if split == "train" else 16
    builders = (_en_event, _en_account, _vi_class, _vi_ticket)
    output = []
    for builder in builders:
        output.extend(builder(index, split) for index in range(per_domain))
    return tuple(output)


def validate_s0_partitions(
    train: tuple[S0PairCase, ...],
    dev: tuple[S0PairCase, ...],
) -> None:
    if len(train) != 256 or len(dev) != 64:
        raise RuntimeError("Hira v1 S0 partition size changed")
    if sum(row.language == "en" for row in train) != 128:
        raise RuntimeError("Hira v1 S0 TRAIN English balance changed")
    if sum(row.language == "vi" for row in train) != 128:
        raise RuntimeError("Hira v1 S0 TRAIN Vietnamese balance changed")
    if sum(row.language == "en" for row in dev) != 32:
        raise RuntimeError("Hira v1 S0 DEV English balance changed")
    if sum(row.language == "vi" for row in dev) != 32:
        raise RuntimeError("Hira v1 S0 DEV Vietnamese balance changed")

    for rows in (train, dev):
        if len({row.case_id for row in rows}) != len(rows):
            raise RuntimeError("Hira v1 S0 duplicate case IDs")
        for row in rows:
            if len(row.option_texts) != 4 or len(row.option_ids) != 4:
                raise RuntimeError("Hira v1 S0 requires exact K=4")
            if len(set(row.option_ids)) != 4:
                raise RuntimeError("Hira v1 S0 option IDs must be unique")
            if row.gold_a == row.gold_b:
                raise RuntimeError("Hira v1 S0 paired questions need different golds")
            if not (0 <= row.gold_a < 4 and 0 <= row.gold_b < 4):
                raise RuntimeError("Hira v1 S0 gold index out of range")

    train_states = {row.state for row in train}
    dev_states = {row.state for row in dev}
    if train_states & dev_states:
        raise RuntimeError("Hira v1 S0 TRAIN/DEV state overlap")

    train_questions = {
        text
        for row in train
        for text in (row.question_a, row.question_b)
    }
    dev_questions = {
        text
        for row in dev
        for text in (row.question_a, row.question_b)
    }
    if train_questions & dev_questions:
        raise RuntimeError("Hira v1 S0 TRAIN/DEV exact question overlap")


__all__ = [
    "S0PairCase",
    "generate_s0_pairs",
    "validate_s0_partitions",
]
