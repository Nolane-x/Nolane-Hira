from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "hira_v0_m5_xnli.py"
    spec = importlib.util.spec_from_file_location("hira_v0_m5_xnli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_xnli_protocol_is_frozen():
    m = load_module()
    assert m.DATASET_ID == "facebook/xnli"
    assert m.DATASET_REVISION == "b8dd5d7af51114dbda02c0e3f6133f332186418e"
    assert m.PER_LANGUAGE == 300
    assert len(m.LANGUAGES) == 15
    assert "vi" in m.LANGUAGES
    assert m.QUESTION == "What is the relationship between `premise` and `hypothesis`?"
    assert [o.option_id for o in m.OPTIONS] == [
        "entailment", "neutral", "contradiction"
    ]


def test_xnli_targets_are_exact():
    m = load_module()
    assert m.TARGETS == {
        "laya.xnli.en": 0.86,
        "laya.xnli.vi": 0.7233333333333334,
        "laya.xnli.other14_macro": 0.731,
    }


def test_xnli_state_is_canonical():
    m = load_module()
    state = m.canonical_state("p", "h")
    assert state == '{"hypothesis":"h","premise":"p"}'
    assert len(m.text_sha256(state)) == 64


def test_xnli_other14_definition_excludes_only_english():
    m = load_module()
    other = [x for x in m.LANGUAGES if x != "en"]
    assert len(other) == 14
    assert "vi" in other
