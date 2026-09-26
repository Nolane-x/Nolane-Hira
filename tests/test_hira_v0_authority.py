from __future__ import annotations

from nmd.compositional_projection_reference import evaluate_reference_panel
from nmd.hira_v0_authority import (
    DOMAINS,
    FACTOR_IDS,
    PRIMITIVES,
    all_w29_query_texts,
    all_w29_schema_texts,
    compose_severity,
    generate_w29_domains,
)
from nmd.hira_v0_eval import classify_w29, summarize_hira_rows
from nmd.compositional_projection_authority import all_w28_text_atoms


def test_w29_authority_is_balanced_fresh_and_composable():
    rows = generate_w29_domains()
    assert len(rows) == 384
    assert {row.domain_id for row in rows} == set(DOMAINS)
    assert not (all_w29_query_texts() & all_w29_schema_texts())
    assert not ({row.state_text for row in rows} & all_w28_text_atoms())

    for domain in DOMAINS:
        subset = [row for row in rows if row.domain_id == domain]
        assert len(subset) == 96
        assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
        for row in subset:
            assert compose_severity(row.factor_vector) == row.severity


def _reference_scores(rows, *, corrupt: str | None = None):
    result = {}
    for index, row in enumerate(rows):
        f0, f1, u, c = row.evidence_vector
        golds = {
            "F0": int(f0),
            "F1": int(f1),
            "U": int(u),
            "C": int(c),
            "F2": int(row.factor_vector[2]),
        }
        result[row.case_id] = {}
        for task, gold in golds.items():
            pred = gold
            if corrupt == task and index % 3 == 0:
                pred = 1 - pred
            result[row.case_id][task] = (
                [0.98, 0.02] if pred == 0 else [0.02, 0.98]
            )
    return result


def _perfect_hira_rows(rows, *, runtime_bad: bool = False):
    out = []
    for row in rows:
        golds = {
            factor: int(row.factor_vector[index])
            for index, factor in enumerate(FACTOR_IDS)
        }
        predictions = {
            factor: {
                primitive: golds[factor]
                for primitive in PRIMITIVES
            }
            for factor in FACTOR_IDS
        }
        out.append(
            {
                "case_id": row.case_id,
                "domain_id": row.domain_id,
                "golds": golds,
                "predictions": predictions,
                "factor_vector_pred": tuple(row.factor_vector),
                "factor_vector_correct": True,
                "severity_pred": row.severity,
                "severity_correct": True,
                "invalid_factor_vector": False,
                "option_order_invariant": not runtime_bad,
                "state_encode_delta": 2 if runtime_bad else 1,
                "relation_delta_max_abs": 0.1 if runtime_bad else 0.0,
                "probability_mass_max_error": 0.0,
                "full_k": not runtime_bad,
            }
        )
    return out


def _runtime_integrity(*, bad: bool = False):
    return {
        "t0_checkpoint_exact": True,
        "projection_trainable_parameter_count": 0,
        "schema_cache_hit_after_first_compile": not bad,
        "relation_refinement_disabled": True,
        "candidate_truncation_used": False,
        "reference_outputs_used_as_model_inputs": False,
    }


def test_w29_ready_requires_all_three_gate_families():
    rows = generate_w29_domains()
    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        {
            "deberta_nli": _reference_scores(rows),
            "roberta_nli": _reference_scores(rows),
        },
    )
    hira = summarize_hira_rows(_perfect_hira_rows(rows))
    result = classify_w29(
        panel=panel,
        hira_summary=hira,
        runtime_integrity=_runtime_integrity(),
    )
    assert result["outcome"] == "HIRA_V0_SEMANTIC_CORE_READY"
    assert all(result["reference_per_domain"].values())
    assert all(result["hira_per_domain"].values())
    assert all(result["runtime_per_domain"].values())
    assert result["global_runtime_integrity"]


def test_w29_reference_failure_has_absolute_precedence():
    rows = generate_w29_domains()
    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        {
            "deberta_nli": _reference_scores(rows, corrupt="C"),
            "roberta_nli": _reference_scores(rows, corrupt="C"),
        },
    )
    hira = summarize_hira_rows(_perfect_hira_rows(rows))
    result = classify_w29(
        panel=panel,
        hira_summary=hira,
        runtime_integrity=_runtime_integrity(),
    )
    assert result["outcome"] == "W29_REFERENCE_INADEQUATE"


def test_w29_runtime_failure_cannot_promote_semantic_core():
    rows = generate_w29_domains()
    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        {
            "deberta_nli": _reference_scores(rows),
            "roberta_nli": _reference_scores(rows),
        },
    )
    hira = summarize_hira_rows(_perfect_hira_rows(rows, runtime_bad=True))
    result = classify_w29(
        panel=panel,
        hira_summary=hira,
        runtime_integrity=_runtime_integrity(bad=True),
    )
    assert result["outcome"] == "W29_RUNTIME_INTEGRATION_FAIL"
