from __future__ import annotations

import importlib.util
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "hira_v0_m5_massive_intent.py"
    spec = importlib.util.spec_from_file_location("hira_v0_m5_massive_intent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_massive_protocol_constants_match_laya():
    m = load_module()
    assert m.LAYA_COMMIT == "42626c348753fbb17572a813127df2278a1ec527"
    assert m.LAYA_BUILD_BENCHMARK_BLOB == "8b131f2d0caa0c0b096c1654214464b02aef1461"
    assert m.DATASET_ID == "mteb/amazon_massive_intent"
    assert m.DATASET_REVISION == "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
    assert m.SEED == 13
    assert m.N_OPTS == 20
    assert m.PER_LANG == 300
    assert len(m.LANGUAGES) == 14
    assert m.QUESTION == "What is the user asking for in `utterance`?"


def test_massive_option_sampling_matches_laya_algorithm():
    m = load_module()
    labels = [f"label_{i}" for i in range(30)]
    a, ai, ak = m.build_options(random.Random(13), "label_7", labels)
    b, bi, bk = m.build_options(random.Random(13), "label_7", labels)
    assert ak == bk
    assert ai == bi
    assert len(a) == 20
    assert len(b) == 20
    assert ak[ai] == "label_7"
    assert len(set(ak)) == 20


def test_massive_criterion_transform_is_exact():
    m = load_module()
    assert m.criterion_text("foo_bar.baz") == "foo bar: baz"


def test_massive_targets_are_frozen():
    m = load_module()
    assert m.TARGETS == {
        "laya.massive.intent.en": 0.7833333333333333,
        "laya.massive.intent.other13_macro": 0.451,
    }
