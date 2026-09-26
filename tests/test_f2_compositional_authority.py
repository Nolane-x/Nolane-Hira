from __future__ import annotations

from nmd.f2_compositional_authority import (
    DOMAINS,
    all_w27_query_texts,
    all_w27_schema_texts,
    compose_f2,
    generate_w27_domains,
)
from nmd.f2_compositional_reference import (
    evaluate_reference_panel,
    qualification_result,
)
from nmd.projection_replication_authority import all_w26_text_atoms


def test_w27_authority_balance_freshness_and_composition():
    rows = generate_w27_domains()
    assert len(rows) == 384
    assert {row.domain_id for row in rows} == set(DOMAINS)
    assert not (all_w27_query_texts() & all_w27_schema_texts())
    assert not ({row.text for row in rows} & all_w26_text_atoms())

    for domain in DOMAINS:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 96
        for u in (0, 1):
            for c in (0, 1):
                assert sum(
                    row.timing_atom == u and row.consequence_atom == c
                    for row in subset
                ) == 24
        assert sum(row.f2 == 1 for row in subset) == 24
        assert sum(row.f2 == 0 for row in subset) == 72

    assert compose_f2(0, 0) == 0
    assert compose_f2(1, 0) == 0
    assert compose_f2(0, 1) == 0
    assert compose_f2(1, 1) == 1


def _scores(rows, *, corrupt_task: str | None = None, direct_always_zero: bool = False):
    out = {}
    for index, row in enumerate(rows):
        golds = {"U": row.timing_atom, "C": row.consequence_atom, "F2": row.f2}
        out[row.case_id] = {}
        for task, gold in golds.items():
            pred = int(gold)
            if direct_always_zero and task == "F2":
                pred = 0
            if corrupt_task == task and index % 4 == 0:
                pred = 1 - pred
            out[row.case_id][task] = [0.95, 0.05] if pred == 0 else [0.05, 0.95]
    return out


def test_w27_fully_qualified_when_atoms_and_direct_are_stable():
    rows = generate_w27_domains()
    scores = {
        "deberta_nli": _scores(rows),
        "roberta_nli": _scores(rows),
    }
    panel = evaluate_reference_panel([row.__dict__ for row in rows], scores)
    result = qualification_result(panel)
    assert result["outcome"] == "F2_AUTHORITY_FULLY_QUALIFIED"
    assert result["atomic_qualified_domain_count"] == 4
    assert result["direct_qualified_domain_count"] == 4


def test_w27_atomic_failure_has_first_precedence():
    rows = generate_w27_domains()
    scores = {
        "deberta_nli": _scores(rows, corrupt_task="U"),
        "roberta_nli": _scores(rows, corrupt_task="U"),
    }
    panel = evaluate_reference_panel([row.__dict__ for row in rows], scores)
    result = qualification_result(panel)
    assert result["outcome"] == "W27_ATOMIC_REFERENCE_INADEQUATE"
    assert result["atomic_qualified_domain_count"] < 4


def test_w27_direct_packaging_limit_when_atoms_compose_but_direct_fails():
    rows = generate_w27_domains()
    scores = {
        "deberta_nli": _scores(rows, direct_always_zero=True),
        "roberta_nli": _scores(rows, direct_always_zero=True),
    }
    panel = evaluate_reference_panel([row.__dict__ for row in rows], scores)
    result = qualification_result(panel)
    assert result["atomic_qualified_domain_count"] == 4
    assert result["direct_qualified_domain_count"] == 0
    assert result["outcome"] == "F2_DIRECT_PACKAGING_LIMIT"
