from __future__ import annotations

import torch

from nmd.atomic_geometry_authority import (
    FACTOR_IDS,
    PARTITION_DOMAINS,
    all_w25_query_texts,
    all_w25_schema_texts,
    compose_severity,
    generate_w25_partition,
    severity_factors,
)
from nmd.atomic_geometry_eval import (
    AtomicFactorProbe,
    TRAIN_BATCH_SIZE,
    classify_w25,
    probe_parameter_count,
)


def test_w25_authority_partition_balance_and_decoder():
    train = generate_w25_partition("train")
    dev = generate_w25_partition("dev")
    confirm = generate_w25_partition("confirm")
    assert len(train) == 384
    assert len(dev) == 96
    assert len(confirm) == 192
    assert not (all_w25_query_texts() & all_w25_schema_texts())
    assert TRAIN_BATCH_SIZE == 32
    for partition, rows in (("train", train), ("dev", dev), ("confirm", confirm)):
        assert {row.domain_id for row in rows} == set(PARTITION_DOMAINS[partition])
        for domain in PARTITION_DOMAINS[partition]:
            subset = [row for row in rows if row.domain_id == domain]
            assert len(subset) == 96
            assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
    assert severity_factors(0) == (0, 0, 0)
    assert severity_factors(1) == (1, 0, 0)
    assert severity_factors(2) == (1, 1, 0)
    assert severity_factors(3) == (1, 1, 1)
    assert compose_severity((0, 0, 0)) == 0
    assert compose_severity((1, 0, 0)) == 1
    assert compose_severity((1, 1, 0)) == 2
    assert compose_severity((1, 1, 1)) == 3
    assert compose_severity((0, 1, 1)) is None


def test_w25_probe_parameter_counts_are_frozen():
    assert probe_parameter_count(AtomicFactorProbe(256)) == 771
    assert probe_parameter_count(AtomicFactorProbe(128)) == 387


def _factor(top1: float, ba: float | None = None):
    return {
        "n": 96,
        "top1": top1,
        "balanced_accuracy": top1 if ba is None else ba,
        "mrr": top1,
        "mean_margin": 0.2,
    }


def _row(
    *,
    factor: float,
    ba: float | None = None,
    vector: float,
    composed: float,
    invalid: float,
):
    return {
        "case_count": 96,
        "factors": {f: _factor(factor, ba) for f in FACTOR_IDS},
        "factor_vector_top1": vector,
        "invalid_factor_vector_rate": invalid,
        "composed_severity_top1": composed,
        "composed_severity_mae": 0.1,
        "factor_probability_mass_max_error": 0.0,
    }


def _evaluation(dy, dz):
    return {"per_domain": {"DY": dy, "DZ": dz}, "pooled": dy}


def _reference(pass_value: bool = True):
    row = (
        _row(factor=.96, ba=.95, vector=.90, composed=.91, invalid=.01)
        if pass_value
        else _row(factor=.80, ba=.80, vector=.60, composed=.65, invalid=.10)
    )
    return _evaluation(row, row)


def test_w25_reference_failure_has_precedence():
    weak = _row(factor=.50, vector=.20, composed=.20, invalid=.50)
    evals = {name: _evaluation(weak, weak) for name in ("A0", "P0", "Q0", "Q1", "T0", "T1")}
    out = classify_w25(evals, _reference(False))
    assert out["outcome"] == "W25_REFERENCE_INADEQUATE"


def test_w25_trainable_projection_rescue_requires_primary_and_replica_both_domains():
    p0 = _row(factor=.55, vector=.20, composed=.30, invalid=.30)
    q = _row(factor=.91, ba=.90, vector=.82, composed=.86, invalid=.05)
    t0 = _row(factor=.88, ba=.85, vector=.76, composed=.80, invalid=.08)
    t1 = _row(factor=.84, ba=.83, vector=.68, composed=.73, invalid=.10)
    evals = {
        "A0": _evaluation(p0, p0),
        "P0": _evaluation(p0, p0),
        "Q0": _evaluation(q, q),
        "Q1": _evaluation(q, q),
        "T0": _evaluation(t0, t0),
        "T1": _evaluation(t1, t1),
    }
    out = classify_w25(evals, _reference(True))
    assert out["outcome"] == "TRAINABLE_PROJECTION_RESCUE"


def test_w25_projection_information_loss_classification():
    p0 = _row(factor=.55, vector=.20, composed=.30, invalid=.30)
    q0 = _row(factor=.93, ba=.91, vector=.84, composed=.90, invalid=.04)
    q1 = _row(factor=.75, ba=.75, vector=.50, composed=.65, invalid=.20)
    t0 = _row(factor=.70, ba=.70, vector=.40, composed=.55, invalid=.20)
    t1 = _row(factor=.68, ba=.68, vector=.35, composed=.50, invalid=.25)
    evals = {
        "A0": _evaluation(p0, p0),
        "P0": _evaluation(p0, p0),
        "Q0": _evaluation(q0, q0),
        "Q1": _evaluation(q1, q1),
        "T0": _evaluation(t0, t0),
        "T1": _evaluation(t1, t1),
    }
    out = classify_w25(evals, _reference(True))
    assert out["outcome"] == "W9_PROJECTION_INFORMATION_LOSS"


def test_w25_semantic_matching_interface_limit_classification():
    p0 = _row(factor=.55, vector=.20, composed=.50, invalid=.25)
    q0 = _row(factor=.93, ba=.91, vector=.84, composed=.90, invalid=.04)
    q1 = _row(factor=.92, ba=.90, vector=.82, composed=.88, invalid=.04)
    t0 = _row(factor=.78, ba=.78, vector=.55, composed=.60, invalid=.18)
    t1 = _row(factor=.75, ba=.75, vector=.50, composed=.58, invalid=.20)
    evals = {
        "A0": _evaluation(p0, p0),
        "P0": _evaluation(p0, p0),
        "Q0": _evaluation(q0, q0),
        "Q1": _evaluation(q1, q1),
        "T0": _evaluation(t0, t0),
        "T1": _evaluation(t1, t1),
    }
    out = classify_w25(evals, _reference(True))
    assert out["outcome"] == "SEMANTIC_MATCHING_INTERFACE_LIMIT"


def test_w25_a13_linear_probe_limit_is_narrow():
    weak = _row(factor=.60, ba=.60, vector=.30, composed=.40, invalid=.30)
    p0 = _row(factor=.55, vector=.20, composed=.30, invalid=.30)
    t0 = _row(factor=.70, ba=.70, vector=.40, composed=.60, invalid=.20)
    t1 = _row(factor=.68, ba=.68, vector=.35, composed=.55, invalid=.25)
    evals = {
        "A0": _evaluation(weak, weak),
        "P0": _evaluation(p0, p0),
        "Q0": _evaluation(weak, weak),
        "Q1": _evaluation(weak, weak),
        "T0": _evaluation(t0, t0),
        "T1": _evaluation(t1, t1),
    }
    out = classify_w25(evals, _reference(True))
    assert out["outcome"] == "A13_LINEAR_PROBE_LIMIT"
