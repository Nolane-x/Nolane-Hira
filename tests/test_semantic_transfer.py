from __future__ import annotations

from collections import defaultdict

from nmd.semantic_transfer import (
    DOMAIN_MIXED,
    OUTCOME_MIXED,
    OUTCOME_STABLE,
    checkpoint_stability,
    transfer_classification,
    transfer_stability,
)
from nmd.semantic_transfer_authority import (
    BASES_PER_DOMAIN,
    DOMAINS,
    INTENTS,
    K_VALUES,
    VIEW_IDS,
    generate_w8_domain,
)


def test_w8_domains_have_exact_intent_and_base_shape():
    assert set(DOMAINS) == {"AU", "AV", "AW", "AX"}
    for domain_id in DOMAINS:
        intents = INTENTS[domain_id]
        assert len(intents) == 16
        assert len({row.intent_id for row in intents}) == 16
        assert len({row.terse_label for row in intents}) == 16
        rows = generate_w8_domain(domain_id)
        assert len(rows) == BASES_PER_DOMAIN * len(K_VALUES) * len(VIEW_IDS)
        assert len({row.base_id for row in rows}) == BASES_PER_DOMAIN


def test_w8_views_preserve_semantic_identity_and_nested_candidates():
    rows = generate_w8_domain("AU")
    by_base = defaultdict(list)
    for row in rows:
        by_base[row.base_id].append(row)

    for base_rows in by_base.values():
        assert len(base_rows) == 12
        assert len({row.state_text for row in base_rows}) == 1
        assert len({row.intent_id for row in base_rows}) == 1
        for k in K_VALUES:
            same_k = [row for row in base_rows if row.diagnosis_k == k]
            assert {row.view_id for row in same_k} == set(VIEW_IDS)
            assert len({row.option_ids for row in same_k}) == 1
            assert len({row.gold_index for row in same_k}) == 1
            assert all(len(row.option_ids) == k for row in same_k)

        ids4 = next(
            row.option_ids for row in base_rows
            if row.diagnosis_k == 4 and row.view_id == "V0"
        )
        ids8 = next(
            row.option_ids for row in base_rows
            if row.diagnosis_k == 8 and row.view_id == "V0"
        )
        ids16 = next(
            row.option_ids for row in base_rows
            if row.diagnosis_k == 16 and row.view_id == "V0"
        )
        assert set(ids4) < set(ids8) < set(ids16)
        assert [x for x in ids8 if x in ids4] == list(ids4)
        assert [x for x in ids16 if x in ids8] == list(ids8)


def _metrics(**overrides):
    row = {
        "v0_k4_top1": .60,
        "v1_k4_top1": .61,
        "v2_k4_top1": .62,
        "v3_k4_top1": .63,
        "v1_k16_top1": .60,
        "v3_k16_top1": .60,
        "k16_coarse_wrong_given_final_error_rate": .50,
    }
    row.update(overrides)
    return row


def test_schema_label_interface_gate():
    out = transfer_classification(_metrics(
        v0_k4_top1=.40,
        v1_k4_top1=.70,
        v2_k4_top1=.75,
        v3_k4_top1=.75,
        v1_k16_top1=.68,
        v3_k16_top1=.68,
    ))
    assert out["classification"] == "SCHEMA_LABEL_INTERFACE_LIMIT"


def test_synthetic_format_dependence_gate():
    out = transfer_classification(_metrics(
        v0_k4_top1=.40,
        v1_k4_top1=.50,
        v2_k4_top1=.75,
        v3_k4_top1=.55,
        v1_k16_top1=.50,
        v3_k16_top1=.55,
    ))
    assert out["classification"] == "SYNTHETIC_FORMAT_DEPENDENCE"


def test_state_schema_alignment_gate():
    out = transfer_classification(_metrics(
        v0_k4_top1=.50,
        v1_k4_top1=.50,
        v2_k4_top1=.55,
        v3_k4_top1=.75,
        v1_k16_top1=.50,
        v3_k16_top1=.72,
    ))
    assert out["classification"] == "STATE_SCHEMA_ALIGNMENT_LIMIT"


def test_general_semantic_transfer_gate():
    out = transfer_classification(_metrics(
        v0_k4_top1=.50,
        v1_k4_top1=.55,
        v2_k4_top1=.58,
        v3_k4_top1=.57,
        v1_k16_top1=.50,
        v3_k16_top1=.50,
    ))
    assert out["classification"] == "GENERAL_SEMANTIC_TRANSFER_LIMIT"


def test_cardinality_amplification_gate():
    out = transfer_classification(_metrics(
        v0_k4_top1=.60,
        v1_k4_top1=.80,
        v2_k4_top1=.65,
        v3_k4_top1=.70,
        v1_k16_top1=.55,
        v3_k16_top1=.60,
        k16_coarse_wrong_given_final_error_rate=.80,
    ))
    assert out["classification"] == "CARDINALITY_AMPLIFICATION_AFTER_TRANSFER"


def test_multiple_gates_are_mixed_not_precedence():
    out = transfer_classification(_metrics(
        v0_k4_top1=.40,
        v1_k4_top1=.80,
        v2_k4_top1=.82,
        v3_k4_top1=.82,
        v1_k16_top1=.55,
        v3_k16_top1=.60,
        k16_coarse_wrong_given_final_error_rate=.80,
    ))
    assert out["classification"] == DOMAIN_MIXED
    assert set(out["active_rules"]) == {
        "SCHEMA_LABEL_INTERFACE_LIMIT",
        "CARDINALITY_AMPLIFICATION_AFTER_TRANSFER",
    }


def _domains(label):
    return {
        "AU": {"classification": label},
        "AV": {"classification": label},
        "AW": {"classification": label},
        "AX": {"classification": "SEMANTIC_TRANSFER_UNRESOLVED"},
    }


def test_checkpoint_requires_three_fresh_domains():
    row = checkpoint_stability(_domains("SCHEMA_LABEL_INTERFACE_LIMIT"))
    assert row["stable"] is True
    assert row["classification"] == "SCHEMA_LABEL_INTERFACE_LIMIT"
    assert row["domains"] == ["AU", "AV", "AW"]


def test_cross_checkpoint_stability_requires_base_and_retuned_agreement():
    rows = {
        "frozen-w6e-control": _domains("SCHEMA_LABEL_INTERFACE_LIMIT"),
        "typed-only-retune": _domains("SCHEMA_LABEL_INTERFACE_LIMIT"),
        "typed-plus-pair-primary": _domains("SEMANTIC_TRANSFER_UNRESOLVED"),
        "typed-plus-pair-replica": _domains("SEMANTIC_TRANSFER_UNRESOLVED"),
    }
    out = transfer_stability(rows)
    assert out["outcome"] == OUTCOME_STABLE
    assert out["stable_classification"] == "SCHEMA_LABEL_INTERFACE_LIMIT"


def test_opposing_stable_retuned_checkpoint_makes_mixed():
    rows = {
        "frozen-w6e-control": _domains("SCHEMA_LABEL_INTERFACE_LIMIT"),
        "typed-only-retune": _domains("SCHEMA_LABEL_INTERFACE_LIMIT"),
        "typed-plus-pair-primary": _domains("SYNTHETIC_FORMAT_DEPENDENCE"),
        "typed-plus-pair-replica": _domains("SEMANTIC_TRANSFER_UNRESOLVED"),
    }
    out = transfer_stability(rows)
    assert out["outcome"] == OUTCOME_MIXED
    assert out["stable_classification"] is None
