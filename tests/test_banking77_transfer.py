from __future__ import annotations

from nmd.banking77_transfer import (
    CANDIDATES,
    END_INDEX,
    EXAMPLE_COUNT,
    EXPECTED_LABELS,
    START_INDEX,
    classify_transfer,
    load_banking77_transfer_authority,
)


class FakeRows(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


def _fake_rows():
    rows = FakeRows()
    labels = [f"label_{i:02d}" for i in range(EXPECTED_LABELS)]
    for i in range(900):
        rows.append({"text": f"message {i}", "label_text": labels[i % EXPECTED_LABELS]})
    return rows


def test_transfer_loader_uses_rows_400_799_only():
    seen = {}
    def loader(dataset_id, *, split, revision):
        seen.update(dataset_id=dataset_id, split=split, revision=revision)
        return _fake_rows()

    authority = load_banking77_transfer_authority(dataset_loader=loader)
    assert len(authority.examples) == EXAMPLE_COUNT
    assert len(authority.options) == EXPECTED_LABELS
    assert START_INDEX == 400 and END_INDEX == 800
    assert "message 400" in authority.examples[0].state_text
    assert "message 799" in authority.examples[-1].state_text


def _metrics(frozen, primary, replica, typed=None, pair=None):
    typed = frozen if typed is None else typed
    pair = frozen if pair is None else pair
    def row(acc):
        return {
            "accuracy": acc,
            "state_encode_calls_per_case": 1.0,
            "probability_mass_max_error": 1e-7,
            "candidate_budget_min": 77,
            "candidate_budget_max": 77,
        }
    return {
        "frozen-w6e-control": row(frozen),
        "typed-only-retune": row(typed),
        "pair-only-retune": row(pair),
        "typed-plus-pair-primary": row(primary),
        "typed-plus-pair-replica": row(replica),
    }


def test_transfer_signal_gate():
    verdict, details = classify_transfer(_metrics(.10, .30, .28, typed=.20, pair=.11))
    assert verdict == "PUBLIC_HIGH_K_TRANSFER_SIGNAL"
    assert details["primary_gain_vs_frozen"] >= .10


def test_transfer_weak_gate():
    verdict, _ = classify_transfer(_metrics(.20, .26, .21))
    assert verdict == "PUBLIC_HIGH_K_TRANSFER_WEAK"


def test_transfer_absent_gate():
    verdict, _ = classify_transfer(_metrics(.30, .24, .25))
    assert verdict == "PUBLIC_HIGH_K_TRANSFER_ABSENT"


def test_candidate_set_is_frozen():
    assert CANDIDATES == (
        "frozen-w6e-control",
        "typed-only-retune",
        "pair-only-retune",
        "typed-plus-pair-primary",
        "typed-plus-pair-replica",
    )
