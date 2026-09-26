from __future__ import annotations

from nmd.compositional_projection_authority import (
    PARTITION_DOMAINS,
    REFERENCE_ATOMS,
    all_w28_query_texts,
    all_w28_schema_texts,
    compose_f2,
    generate_w28_partition,
)
from nmd.compositional_projection_eval import (
    CANDIDATE_SEEDS,
    PROJECTION_EPOCHS,
    PROJECTION_LR,
    PROJECTION_TEMPERATURE,
    PROJECTION_WEIGHT_DECAY,
    TRAIN_BATCH_SIZE,
    primary_rescue,
    replica_rescue,
)
from nmd.compositional_projection_reference import (
    confirm_domain_pass,
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


def test_w28_projection_recipe_is_exact_w25_recipe():
    assert CANDIDATE_SEEDS == {"T0": 2519, "T1": 2539}
    assert PROJECTION_EPOCHS == 8
    assert TRAIN_BATCH_SIZE == 32
    assert PROJECTION_LR == 3e-4
    assert PROJECTION_WEIGHT_DECAY == 0.01
    assert PROJECTION_TEMPERATURE == 0.07


def test_w28_confirm_reference_can_pass_clean_compositional_panel():
    rows = generate_w28_partition("confirm")
    panel = evaluate_reference_panel(
        [row.__dict__ for row in rows],
        {
            "deberta_nli": _scores(rows, direct_wrong=True),
            "roberta_nli": _scores(rows, direct_wrong=True),
        },
    )
    assert confirm_domain_pass(panel, "EU")
    assert confirm_domain_pass(panel, "EV")


def _hira_summary(*, factor_top1, factor_ba, vector, composed, invalid, mass=0.0):
    return {
        "factors": {
            f: {"top1": factor_top1, "balanced_accuracy": factor_ba}
            for f in ("F0", "F1", "F2")
        },
        "factor_vector_top1": vector,
        "composed_severity_top1": composed,
        "invalid_factor_vector_rate": invalid,
        "factor_probability_mass_max_error": mass,
    }


def test_w28_primary_and_replica_rescue_gates_are_frozen():
    p0 = _hira_summary(
        factor_top1=.40, factor_ba=.50, vector=.05, composed=.10, invalid=.90
    )
    t0 = _hira_summary(
        factor_top1=.86, factor_ba=.83, vector=.72, composed=.76, invalid=.10
    )
    t1 = _hira_summary(
        factor_top1=.83, factor_ba=.81, vector=.66, composed=.71, invalid=.14
    )
    assert primary_rescue(t0, p0)
    assert replica_rescue(t1, p0)


def test_w28_primary_rescue_requires_all_factor_thresholds():
    p0 = _hira_summary(
        factor_top1=.40, factor_ba=.50, vector=.05, composed=.10, invalid=.90
    )
    almost = _hira_summary(
        factor_top1=.84, factor_ba=.90, vector=.90, composed=.90, invalid=.01
    )
    assert not primary_rescue(almost, p0)
