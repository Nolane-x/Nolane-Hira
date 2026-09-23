import json
from types import SimpleNamespace

import pytest
import torch

from nmd.banking77_direct import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    EXPECTED_EXAMPLES,
    EXPECTED_LABELS,
    EXPECTED_MARKER,
    FORCED_BUDGET,
    LAYA_TARGET,
    QUESTION,
    evaluate_banking77_direct,
    load_and_validate_marker,
    load_banking77_authority,
)


LABELS = [f"intent_{i:02d}" for i in range(EXPECTED_LABELS)]


class FakeRows(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


def make_rows(n=500):
    return FakeRows(
        {
            "text": f"message {i}",
            "label_text": LABELS[i % EXPECTED_LABELS],
        }
        for i in range(n)
    )


def test_loader_requests_exact_pinned_test_source_and_first_400():
    calls = []

    def loader(dataset_id, *, split, revision):
        calls.append((dataset_id, split, revision))
        return make_rows(500)

    authority = load_banking77_authority(dataset_loader=loader)

    assert calls == [(DATASET_ID, DATASET_SPLIT, DATASET_REVISION)]
    assert len(authority.examples) == EXPECTED_EXAMPLES
    assert len(authority.options) == EXPECTED_LABELS
    assert len(authority.labels) == EXPECTED_LABELS
    assert authority.examples[0].state_text == '{"message":"message 0"}'
    assert authority.examples[-1].state_text == '{"message":"message 399"}'


def test_label_space_is_sorted_semantic_and_option_ids_are_opaque():
    rows = make_rows(500)
    rows.reverse()

    authority = load_banking77_authority(
        dataset_loader=lambda *args, **kwargs: rows
    )

    expected = tuple(sorted(label.replace("_", " ") for label in LABELS))
    assert authority.labels == expected
    assert [o.criterion_text for o in authority.options] == list(expected)
    assert [o.option_id for o in authority.options] == [
        f"intent-{i:03d}" for i in range(EXPECTED_LABELS)
    ]
    assert all(
        option.option_id not in option.criterion_text
        for option in authority.options
    )


def test_loader_fails_closed_on_short_or_wrong_label_space():
    with pytest.raises(ValueError, match="at least 400"):
        load_banking77_authority(
            dataset_loader=lambda *args, **kwargs: make_rows(399)
        )

    rows = make_rows(500)
    for row in rows:
        row["label_text"] = "only_one_label"
    with pytest.raises(ValueError, match="exactly 77"):
        load_banking77_authority(
            dataset_loader=lambda *args, **kwargs: rows
        )


def test_marker_is_exact_and_rejects_any_drift(tmp_path):
    p = tmp_path / "marker.json"
    p.write_text(json.dumps(EXPECTED_MARKER), encoding="utf-8")
    assert load_and_validate_marker(p) == EXPECTED_MARKER

    bad = dict(EXPECTED_MARKER)
    bad["selected_head_sha256"] = "0" * 64
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="marker mismatch"):
        load_and_validate_marker(p)

    extra = dict(EXPECTED_MARKER)
    extra["allow_training"] = True
    p.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthorized fields"):
        load_and_validate_marker(p)


class FakeModel:
    def __init__(self):
        self.state_encode_calls = 0
        self.compile_schema_calls = 0
        self.forward_calls = 0
        self.budgets = []
        self.adaptive = []

    def eval(self):
        return self

    def compile_schema(self, *, primitive, question_text, options, use_cache):
        self.compile_schema_calls += 1
        assert primitive == "choice"
        assert question_text == QUESTION
        assert use_cache is True
        assert len(options) == EXPECTED_LABELS
        schema = object()
        receipt = SimpleNamespace(option_count=EXPECTED_LABELS)
        return schema, receipt

    def compile_state(self, state_text, *, segment_tokens):
        self.state_encode_calls += 1
        idx = int(json.loads(state_text)["message"].split()[-1])
        return idx

    def forward_compiled(
        self,
        memory,
        schema,
        *,
        forced_budget,
        adaptive_budget,
    ):
        self.forward_calls += 1
        self.budgets.append(forced_budget)
        self.adaptive.append(adaptive_budget)
        gold = memory % EXPECTED_LABELS
        p = torch.full((EXPECTED_LABELS,), 0.1 / (EXPECTED_LABELS - 1))
        p[gold] = 0.9
        hira = SimpleNamespace(
            candidate_budget=torch.tensor([EXPECTED_LABELS]),
            tail_mass=torch.tensor([0.0]),
        )
        return SimpleNamespace(probabilities=p, hira=hira)


def test_evaluator_is_full_k_state_once_and_schema_once():
    authority = load_banking77_authority(
        dataset_loader=lambda *args, **kwargs: make_rows(500)
    )
    model = FakeModel()

    metrics = evaluate_banking77_direct(model, authority)

    assert metrics["case_count"] == 400
    assert metrics["label_count"] == 77
    assert metrics["accuracy"] == 1.0
    assert metrics["state_encode_calls"] == 400
    assert metrics["state_encode_calls_per_case"] == 1.0
    assert metrics["candidate_budget_min"] == 77
    assert metrics["candidate_budget_max"] == 77
    assert metrics["tail_mass_max"] == 0.0
    assert metrics["probability_mass_max_error"] < 1e-6
    assert metrics["laya_target"] == LAYA_TARGET
    assert metrics["laya_status"] == "WIN"

    assert model.compile_schema_calls == 1
    assert model.forward_calls == 400
    assert set(model.budgets) == {FORCED_BUDGET}
    assert set(model.adaptive) == {False}
