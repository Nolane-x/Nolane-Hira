from __future__ import annotations

from nmd.atomic_geometry_authority import all_w25_text_atoms
from nmd.projection_replication_authority import (
    FACTOR_IDS,
    PARTITION_DOMAINS,
    all_w26_query_texts,
    all_w26_schema_texts,
    all_w26_text_atoms,
    compose_severity,
    generate_w26_partition,
    severity_factors,
)
from nmd.projection_replication_reference import (
    confirm_domain_pass,
    evaluate_reference_panel,
    qualification_result,
)


def test_w26_partition_balance_freshness_and_decoder():
    expected = {
        "qualification": 192,
        "train": 384,
        "dev": 96,
        "confirm": 192,
    }
    for partition, count in expected.items():
        rows = generate_w26_partition(partition)
        assert len(rows) == count
        assert {row.domain_id for row in rows} == set(PARTITION_DOMAINS[partition])
        for domain in PARTITION_DOMAINS[partition]:
            subset = [row for row in rows if row.domain_id == domain]
            assert len(subset) == 96
            assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4

    assert not (all_w26_query_texts() & all_w26_schema_texts())
    assert not (all_w26_text_atoms() & all_w25_text_atoms())

    assert severity_factors(0) == (0, 0, 0)
    assert severity_factors(1) == (1, 0, 0)
    assert severity_factors(2) == (1, 1, 0)
    assert severity_factors(3) == (1, 1, 1)
    assert compose_severity((0, 0, 0)) == 0
    assert compose_severity((1, 0, 0)) == 1
    assert compose_severity((1, 1, 0)) == 2
    assert compose_severity((1, 1, 1)) == 3
    assert compose_severity((0, 1, 0)) is None


def _scores(rows, *, wrong_factor: str | None = None, wrong_every: int = 9999):
    out = {}
    for index, case in enumerate(rows):
        row = {}
        for f_index, factor_id in enumerate(FACTOR_IDS):
            gold = int(case.factor_vector[f_index])
            pred = gold
            if factor_id == wrong_factor and index % wrong_every == 0:
                pred = 1 - gold
            row[factor_id] = [0.95, 0.05] if pred == 0 else [0.05, 0.95]
        out[case.case_id] = row
    return out


def test_w26_reference_qualification_can_pass_clean_panel():
    rows = generate_w26_partition("qualification")
    model_scores = {
        "deberta_nli": _scores(rows),
        "roberta_nli": _scores(rows),
    }
    panel = evaluate_reference_panel([row.__dict__ for row in rows], model_scores)
    result = qualification_result(panel)
    assert result["status"] == "PASS"
    assert result["outcome"] == "REFERENCE_QUALIFIED"
    assert result["per_domain"] == {"EA": True, "EB": True}


def test_w26_reference_qualification_fails_before_hira_when_consensus_is_weak():
    rows = generate_w26_partition("qualification")
    # Corrupt F0 often enough in both references that the panel top1 misses .94.
    model_scores = {
        "deberta_nli": _scores(rows, wrong_factor="F0", wrong_every=5),
        "roberta_nli": _scores(rows, wrong_factor="F0", wrong_every=5),
    }
    panel = evaluate_reference_panel([row.__dict__ for row in rows], model_scores)
    result = qualification_result(panel)
    assert result["status"] == "FAIL"
    assert result["outcome"] == "W26_REFERENCE_QUALIFICATION_FAIL"
    assert not all(result["per_domain"].values())


def test_w26_reference_qualification_fails_on_cross_model_disagreement():
    rows = generate_w26_partition("qualification")
    model_scores = {
        "deberta_nli": _scores(rows),
        "roberta_nli": _scores(rows, wrong_factor="F1", wrong_every=4),
    }
    panel = evaluate_reference_panel([row.__dict__ for row in rows], model_scores)
    result = qualification_result(panel)
    assert result["status"] == "FAIL"


def test_w26_confirm_gate_uses_frozen_lower_but_still_strict_thresholds():
    rows = generate_w26_partition("confirm")
    model_scores = {
        "deberta_nli": _scores(rows),
        "roberta_nli": _scores(rows),
    }
    panel = evaluate_reference_panel([row.__dict__ for row in rows], model_scores)
    assert confirm_domain_pass(panel, "EH")
    assert confirm_domain_pass(panel, "EI")
