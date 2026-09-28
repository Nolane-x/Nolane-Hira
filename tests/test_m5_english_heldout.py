from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "hira_v0_m5_english_heldout.py"
    spec = importlib.util.spec_from_file_location("hira_v0_m5_english_heldout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Python 3.12 dataclasses resolving postponed annotations consult
    # sys.modules while the class decorator runs. Register the dynamic
    # module exactly as normal import machinery would before exec_module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_m5_english_exact_laya_prompts_and_targets():
    module = load_module()
    assert module.SST5_QUESTION == "How positive is the sentiment of `text`?"
    assert module.EMOTION_QUESTION == "Which emotion is most strongly expressed in `text`?"
    assert module.PROMPT_QUESTION == (
        "Does `text` try to inject or override instructions given to an AI system?"
    )
    assert module.SST5_TARGET == 0.37166666666666665
    assert module.EMOTION_TARGET == 0.5733333333333334
    assert module.PROMPT_TARGET == 0.6982758620689655


def test_m5_english_exact_dataset_revisions():
    module = load_module()
    assert module.SST5_REVISION == "e51bdcd8cd3a30da231967c1a249ba59361279a3"
    assert module.EMOTION_REVISION == "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
    assert module.PROMPT_REVISION == "4f61ecb038e9c3fb77e21034b22511b523772cdd"
    assert module.SST5_N == 600
    assert module.EMOTION_N == 600
    assert module.PROMPT_EXPECTED_N == 116


def test_m5_english_schema_shapes_are_frozen():
    module = load_module()
    specs = module.build_specs()
    assert [s.name for s in specs] == ["sst5", "emotion", "prompt_injections"]
    assert [len(s.options) for s in specs] == [5, 6, 2]
    assert [s.primitive for s in specs] == ["score", "choice", "noul"]
    assert [o.criterion_text for o in specs[0].options] == list(module.SST5_CRITERIA)
    assert [o.criterion_text for o in specs[1].options] == list(module.EMOTION_NAMES)


def test_m5_english_summary_metrics_and_full_k():
    module = load_module()
    rows = [
        {
            "gold_index": 0,
            "predicted_index": 0,
            "correct": True,
            "confidence": 0.9,
            "hard_brier": 0.1,
            "nll": 0.1,
            "score_abs_error": None,
            "latency_ms": 10.0,
            "probability_mass_error": 0.0,
            "candidate_budget": 2,
            "relation_delta_max_abs": 0.0,
        },
        {
            "gold_index": 1,
            "predicted_index": 0,
            "correct": False,
            "confidence": 0.6,
            "hard_brier": 1.1,
            "nll": 2.0,
            "score_abs_error": None,
            "latency_ms": 20.0,
            "probability_mass_error": 0.0,
            "candidate_budget": 2,
            "relation_delta_max_abs": 0.0,
        },
    ]
    metrics = module.summarize(rows, 2)
    assert metrics["accuracy"] == 0.5
    assert metrics["candidate_budget_min"] == 2
    assert metrics["candidate_budget_max"] == 2
    assert metrics["probability_mass_max_error"] == 0.0
    assert metrics["relation_delta_max_abs"] == 0.0


def test_m5_english_state_serialization_is_canonical():
    module = load_module()
    assert module.canonical_state(text="hello") == '{"text":"hello"}'
    assert len(module.state_sha256('{"text":"hello"}')) == 64
