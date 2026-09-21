import json

import pytest

from nmd.dataset import DatasetContractError, load_jsonl, parse_example


def valid():
    return {
        "state": "the transfer is pending",
        "primitive": "choice",
        "question": "what happened?",
        "options": [
            {"id": "a", "text": "cash withdrawal issue"},
            {"id": "b", "text": "pending transfer"},
        ],
        "gold_index": 1,
        "teacher_probs": [0.1, 0.9],
    }


def test_parse_example():
    ex = parse_example(valid())
    assert ex.gold_index == 1
    assert ex.teacher_probs == (0.1, 0.9)


def test_noul_requires_two_options():
    raw = valid()
    raw["primitive"] = "noul"
    raw["options"].append({"id": "c", "text": "unknown"})
    with pytest.raises(DatasetContractError):
        parse_example(raw)


def test_jsonl_line_number(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(valid()) + "\n{broken\n", encoding="utf-8")
    with pytest.raises(DatasetContractError) as exc:
        load_jsonl(p)
    assert ":2:" in str(exc.value)
