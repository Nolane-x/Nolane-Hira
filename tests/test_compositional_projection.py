from __future__ import annotations

from nmd.compositional_projection_authority import (
    PARTITION_DOMAINS,
    REFERENCE_ATOMS,
    all_w28_query_texts,
    all_w28_schema_texts,
    compose_f2,
    generate_w28_partition,
)
from nmd.compositional_projection_reference import (
    evaluate_reference_panel,
    qualification_result,
)
from nmd.f2_compositional_authority import all_w27_text_atoms


def test_w28_partitions_balance_and_freshness():
    expected = {
        "qualification": 192,
        "train": 384,
        "dev": 96,
        "confirm": 192,
    }
    for partition, count in expected.items():
        rows = generate_w28_partition(partition)
        assert len(rows) == count
        assert {row.domain_id for row in rows} == set(PARTITION_DOMAINS[partition])
        for domain in PARTITION_DOMAINS[partition]:
            subset = [row for row in rows if row.domain_id == domain]
            assert len(subset) == 96
            assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4

    assert not (all_w28_query_texts() & all_w28_schema_texts())
    assert not ({row.severity_field for row in generate_w28_partition("qualification")} & all_w27_text_atoms())
    assert compose_f2(0, 0) == 0
    assert compose_f2(1, 0) == 0
    assert compose_f2(0, 1) == 0
    assert compose_f2(1, 1) == 1


def _scores(rows, *, corrupt: str | None = None, direct_wrong: bool = False):
    out = {}
    for index, row in enumerate(rows):
        f0, f1, u, c = row.evidence_vector
        golds = {"F0": f0, "F1": f1, "U": u, "C": c, "F2": row.factor_vector[2]}
        out[row.case_id] = {}
        for task, gold in golds.items():
            pred = int(gold)
            if corrupt == task and index % 4 == 0:
                pred = 1 - pred
            if direct_wrong and task == "F2":
                pred = 1 - pred
            out[row.case_id][task] = [0.95, 0.05] if pred == 0 else [0.05, 0.95]
    return out


def test_w28_reference_qualification_can_pass_clean_panel():
    rows = generate_w28_partition("qualification")
    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        {
            "deberta_nli": _scores(rows),
            "roberta_nli": _scores(rows),
        },
    )
    result = qualification_result(panel)
    assert result["status"] == "PASS"
    assert result["outcome"] == "W28_REFERENCE_QUALIFIED"
    assert result["per_domain"] == {"EN": True, "EO": True}


def test_w28_direct_f2_is_diagnostic_only():
    rows = generate_w28_partition("qualification")
    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        {
            "deberta_nli": _scores(rows, direct_wrong=True),
            "roberta_nli": _scores(rows, direct_wrong=True),
        },
    )
    result = qualification_result(panel)
    assert result["status"] == "PASS"
    assert result["outcome"] == "W28_REFERENCE_QUALIFIED"


def test_w28_atomic_failure_blocks_hira():
    rows = generate_w28_partition("qualification")
    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        {
            "deberta_nli": _scores(rows, corrupt="C"),
            "roberta_nli": _scores(rows, corrupt="C"),
        },
    )
    result = qualification_result(panel)
    assert result["status"] == "FAIL"
    assert result["outcome"] == "W28_REFERENCE_QUALIFICATION_FAIL"
    assert not all(result["per_domain"].values())
