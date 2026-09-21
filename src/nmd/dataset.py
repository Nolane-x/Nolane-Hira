from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .contracts import LogicalOption
from .training import DecisionExample


class DatasetContractError(ValueError):
    pass


def _option(raw: dict) -> LogicalOption:
    if not isinstance(raw, dict):
        raise DatasetContractError("each option must be an object")
    oid = str(raw.get("id", "")).strip()
    text = str(raw.get("text", "")).strip()
    if not oid or not text:
        raise DatasetContractError("each option requires non-empty id and text")
    return LogicalOption(
        option_id=oid,
        criterion_text=text,
        aliases=tuple(map(str, raw.get("aliases", ()))),
        exemplars=tuple(map(str, raw.get("exemplars", ()))),
        counterexamples=tuple(map(str, raw.get("counterexamples", ()))),
    )


def parse_example(raw: dict) -> DecisionExample:
    if not isinstance(raw, dict):
        raise DatasetContractError("example must be an object")
    primitive = raw.get("primitive")
    if primitive not in {"choice", "score", "noul"}:
        raise DatasetContractError("primitive must be choice, score or noul")
    state = str(raw.get("state", "")).strip()
    question = str(raw.get("question", "")).strip()
    if not state or not question:
        raise DatasetContractError("state and question must be non-empty")
    options = tuple(_option(x) for x in raw.get("options", ()))
    if len(options) < 2:
        raise DatasetContractError("at least two logical options are required")
    if len({o.option_id for o in options}) != len(options):
        raise DatasetContractError("option ids must be unique")
    if primitive == "noul" and len(options) != 2:
        raise DatasetContractError("noul requires exactly two logical options")
    try:
        gold_index = int(raw["gold_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetContractError("gold_index is required and must be an integer") from exc
    if not 0 <= gold_index < len(options):
        raise DatasetContractError("gold_index is out of range")

    teacher = raw.get("teacher_probs")
    teacher_probs = None
    if teacher is not None:
        if len(teacher) != len(options):
            raise DatasetContractError("teacher_probs length must equal logical option count")
        teacher_probs = tuple(float(x) for x in teacher)
        if any(x < 0 for x in teacher_probs) or sum(teacher_probs) <= 0:
            raise DatasetContractError("teacher_probs must be non-negative with positive mass")

    gold_score = raw.get("gold_score")
    if gold_score is not None:
        gold_score = float(gold_score)

    return DecisionExample(
        state_text=state,
        primitive=primitive,
        question_text=question,
        options=options,
        gold_index=gold_index,
        teacher_probs=teacher_probs,
        gold_score=gold_score,
    )


def load_jsonl(path: str | Path) -> list[DecisionExample]:
    path = Path(path)
    out = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            out.append(parse_example(raw))
        except (json.JSONDecodeError, DatasetContractError) as exc:
            raise DatasetContractError(f"{path}:{line_no}: {exc}") from exc
    if not out:
        raise DatasetContractError("dataset is empty")
    return out
