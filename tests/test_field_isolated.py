from __future__ import annotations

import pytest

from nmd.competitive import CompetitiveCoarseScorer
from nmd.field_isolated_authority import (
    generate_w17_confirm,
    generate_w17_dev,
    generate_w17_train,
)
from nmd.field_isolated_cache import compile_w17_cache
from nmd.field_isolated_eval import PATHS, PRIMITIVE_FIELDS, evaluate_w17, w17_verdict
from nmd.hira import HIRACore
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder


def test_w17_authority_shape_and_confirm_seal():
    train = generate_w17_train()
    dev = generate_w17_dev()
    assert len(train) == 384
    assert len(dev) == 96
    assert {row.domain_id for row in train} == {"CK", "CL", "CM", "CN"}
    assert {row.domain_id for row in dev} == {"CO"}
    for rows in (train, dev):
        for row in rows:
            assert len(row.typed.decisions) == 5
            assert row.diagnosis_k in {4, 8, 16}
            assert len(row.option_definitions) == 5

    with pytest.raises(RuntimeError):
        generate_w17_confirm("CP")
    cp = generate_w17_confirm("CP", allow_confirm=True)
    cq = generate_w17_confirm("CQ", allow_confirm=True)
    assert len(cp) == len(cq) == 96
    assert {row.domain_id for row in cp} == {"CP"}
    assert {row.domain_id for row in cq} == {"CQ"}


def test_w17_primitive_field_mapping_is_frozen():
    assert PRIMITIVE_FIELDS == {
        "diagnosis": ("intent",),
        "response": ("severity",),
        "needs_review": ("severity", "confidence"),
        "risk": ("severity",),
        "urgency": ("severity", "confidence"),
    }


def test_w17_download_free_one_case_execution():
    encoder = TrainableSemanticEncoder(
        vocab_size=1024,
        d_model=256,
        n_layers=1,
        n_heads=4,
        max_length=128,
    )
    hira = HIRACore(d_model=256, dropout=0.0)
    model = NolaneHira(encoder, hira)
    model.eval()

    case = generate_w17_train()[0]
    cache = compile_w17_cache(
        model,
        [case],
        expected_split_prefix="train",
    )
    scorer = CompetitiveCoarseScorer(d_model=256, d_rel=128)
    scorer.eval()
    result = evaluate_w17(hira, scorer, cache)

    assert set(result["paths"]) == set(PATHS)
    assert result["control"]["triplicate_prediction_identity_rate"] == 1.0
    assert result["control"]["triplicate_max_semantic_logit_diff"] <= 1e-5
    accounting = result["representation_accounting"]
    assert accounting["logical_state_compiles_per_candidate_case"] == {
        "FULL_SINGLE": 1,
        "FULL_TRIPLICATE": 1,
        "FIELD_ISOLATED": 1,
    }
    assert accounting["encoded_sequences_per_candidate_case"] == {
        "FULL_SINGLE": 1,
        "FULL_TRIPLICATE": 3,
        "FIELD_ISOLATED": 3,
    }
    for path in PATHS:
        assert result["paths"][path]["case_count"] == 1
        assert result["paths"][path]["decision_count"] == 5
        assert result["paths"][path]["probability_mass_max_error"] <= 1e-6


def _metrics_fixture(*, identity: float = 1.0, iso_k16: float = 0.72):
    full_diag = {
        "4": {"top1": 0.55},
        "8": {"top1": 0.45},
        "16": {"top1": 0.30},
    }
    iso_diag = {
        "4": {"top1": 0.90},
        "8": {"top1": 0.82},
        "16": {"top1": iso_k16},
    }
    full = {
        "accuracy": 0.60,
        "non_diagnosis_accuracy": 0.65,
        "probability_mass_max_error": 0.0,
        "diagnosis_per_k": full_diag,
    }
    iso = {
        "accuracy": 0.86,
        "non_diagnosis_accuracy": 0.84,
        "probability_mass_max_error": 0.0,
        "diagnosis_per_k": iso_diag,
        "primitive_accuracy": {"choice": 0.86, "noul": 0.84, "score": 0.83},
        "question_accuracy": {
            "diagnosis": 0.84,
            "response": 0.84,
            "needs_review": 0.84,
            "risk": 0.83,
            "urgency": 0.80,
        },
    }
    control = {
        "triplicate_prediction_identity_rate": identity,
        "triplicate_max_semantic_logit_diff": 0.0,
    }
    accounting = {
        "logical_state_compiles_per_candidate_case": {
            "FULL_SINGLE": 1,
            "FULL_TRIPLICATE": 1,
            "FIELD_ISOLATED": 1,
        },
        "a13_invocations_per_candidate_case": {
            "FULL_SINGLE": 1,
            "FULL_TRIPLICATE": 1,
            "FIELD_ISOLATED": 1,
        },
        "encoded_sequences_per_candidate_case": {
            "FULL_SINGLE": 1,
            "FULL_TRIPLICATE": 3,
            "FIELD_ISOLATED": 3,
        },
    }
    neutral = {"probability_mass_max_error": 0.0}
    return {
        "paths": {
            "FULL_SINGLE": full,
            "FULL_TRIPLICATE": neutral,
            "FIELD_ISOLATED": iso,
            "PRODUCTION_D0_FULL": neutral,
        },
        "control": control,
        "representation_accounting": accounting,
    }


def _reference_fixture():
    return {
        "diagnosis_per_k": {
            "4": {"top1": 0.90},
            "8": {"top1": 0.85},
            "16": {"top1": 0.80},
        }
    }


def test_w17_frozen_verdict_full_rescue_and_control_invalid():
    metrics = _metrics_fixture()
    reference = _reference_fixture()
    verdict, details = w17_verdict(metrics, metrics, reference, reference)
    assert verdict == "FIELD_ISOLATED_TYPED_RESCUE"
    assert details["CP"]["integrity"] is True
    assert details["CQ"]["competence"] is True
    assert details["CP"]["causal_gain"] is True

    invalid = _metrics_fixture(identity=0.99)
    verdict, _ = w17_verdict(invalid, metrics, reference, reference)
    assert verdict == "FIELD_ISOLATION_CONTROL_INVALID"
