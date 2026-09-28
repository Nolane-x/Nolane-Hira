from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "hira_v0_m5_systems_cpu.py"
    spec = importlib.util.spec_from_file_location("hira_v0_m5_systems_cpu", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_systems_call_shape_matches_laya_source():
    m = load_module()
    assert m.LAYA_COMMIT == "42626c348753fbb17572a813127df2278a1ec527"
    assert m.LAYA_BENCH_LATENCY_BLOB == "e97ddbb499fcfd86ee626ceab238608cddbfb426"
    assert m.Q_COUNTS == (1, 5, 10, 50)
    assert m.WARMUP == 2
    assert m.REPS == 10
    assert m.Q_NOUL["instructions"] == (
        "Does `ticket.messages[0].text` express urgency?"
    )
    assert m.Q_CHOICE["instructions"] == "Which team should handle this?"


def test_systems_options_are_frozen():
    m = load_module()
    choice = m.choice_options()
    assert [x.option_id for x in choice] == ["billing", "technical", "sales"]
    assert [x.criterion_text for x in choice] == [
        "payments", "bugs and integrations", "pricing"
    ]
    noul = m.noul_options()
    assert [x.option_id for x in noul] == ["false", "true"]


def test_systems_t4_targets_are_not_cpu_targets():
    m = load_module()
    assert m.LAYA_T4_TARGETS == {
        1: 32.8,
        5: 40.1,
        10: 72.3,
        50: 337.4,
    }


def test_systems_summary_shape():
    m = load_module()
    out = m.summarize_ms([1.0, 2.0, 3.0, 4.0])
    assert out["min_ms"] == 1.0
    assert out["mean_ms"] == 2.5
    assert out["p50_ms"] == 2.5
    assert out["p95_ms"] >= out["p50_ms"]
