from collections import Counter

import pytest

from nmd.typed_reliability_authority import (
    CONFIDENCE_LEVELS,
    CONFIRM_K_COUNTS,
    CONFIRM_SEED,
    CONFIRM_TEMPLATES,
    DEV_ANOMALIES,
    DEV_CHANNELS,
    DEV_COMPONENTS,
    DEV_K_COUNTS,
    DEV_SEED,
    DEV_ZONES,
    TRAIN_K_COUNTS,
    TRAIN_SEED,
    TRAIN_TEMPLATES,
    _render_state,
    all_w6c_values,
    generate_w6c_authority,
)


def _gold_signature(case):
    decision = case.typed.decisions[0]
    return decision.options[decision.gold_index].criterion_text


def test_counts_seeds_and_confirm_seal_are_frozen():
    assert sum(TRAIN_K_COUNTS.values()) == 384
    assert sum(DEV_K_COUNTS.values()) == 192
    assert sum(CONFIRM_K_COUNTS.values()) == 192
    assert (TRAIN_SEED, DEV_SEED, CONFIRM_SEED) == (
        171137,
        172239,
        173341,
    )
    train = generate_w6c_authority("train")
    dev = generate_w6c_authority("dev")
    assert len(train) == 384
    assert len(dev) == 192
    assert sum(len(case.typed.decisions) for case in train) == 1920
    assert sum(len(case.typed.decisions) for case in dev) == 960
    with pytest.raises(RuntimeError, match="sealed until post-selection"):
        generate_w6c_authority("confirm")


def test_train_and_dev_are_exactly_balanced_across_joint_strata():
    for split, expected_per_stratum in (
        ("train", 8),
        ("dev", 4),
    ):
        cases = generate_w6c_authority(split)
        counts = Counter(
            (case.diagnosis_k, case.severity, case.confidence)
            for case in cases
        )
        assert len(counts) == 4 * 4 * 3
        assert set(counts.values()) == {expected_per_stratum}
        for k in (8, 16, 32, 64):
            for severity in range(4):
                for confidence in CONFIDENCE_LEVELS:
                    assert counts[(k, severity, confidence)] == (
                        expected_per_stratum
                    )


def test_every_case_has_five_frozen_typed_decisions():
    for case in (
        generate_w6c_authority("train")[:48]
        + generate_w6c_authority("dev")[:48]
    ):
        decisions = case.typed.decisions
        assert [decision.question_id for decision in decisions] == [
            "diagnosis",
            "response",
            "needs_review",
            "risk",
            "urgency",
        ]
        assert [decision.primitive for decision in decisions] == [
            "choice",
            "choice",
            "noul",
            "score",
            "score",
        ]
        assert len(decisions[0].options) == case.diagnosis_k
        for decision in decisions:
            assert abs(sum(decision.gold_probabilities) - 1.0) < 1e-12
            assert max(
                range(len(decision.gold_probabilities)),
                key=decision.gold_probabilities.__getitem__,
            ) == decision.gold_index


def test_noul_rule_is_joint_strata_deterministic():
    for case in generate_w6c_authority("dev"):
        review = next(
            decision
            for decision in case.typed.decisions
            if decision.question_id == "needs_review"
        )
        expected = int(
            case.confidence == "uncertain"
            or case.severity == 3
        )
        assert review.gold_index == expected


def test_train_dev_gold_signatures_are_unique_and_disjoint():
    train = generate_w6c_authority("train")
    dev = generate_w6c_authority("dev")
    train_signatures = [_gold_signature(case) for case in train]
    dev_signatures = [_gold_signature(case) for case in dev]
    assert len(set(train_signatures)) == len(train_signatures)
    assert len(set(dev_signatures)) == len(dev_signatures)
    assert set(train_signatures).isdisjoint(dev_signatures)


def test_w6c_values_are_fresh_against_w6b_and_w5():
    from nmd.typed_competitive_authority import all_w6b_values
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd import semantic_alignment_probes as w5c
    from nmd import semantic_capacity_control as w5e
    from nmd import semantic_encoder_adaptation as w5d
    from nmd import semantic_routing_curriculum as w5a
    from nmd import semantic_token_curriculum as w5b

    prior = (
        set(all_w6b_values())
        | set(all_w5f_vocab())
        | set(all_w5g_vocab())
        | set(all_w5h_vocab())
        | set(all_w5i_vocab())
    )
    for module in (w5a, w5b, w5c, w5d, w5e):
        for name, value in vars(module).items():
            if not (
                name.startswith("TRAIN_")
                or name.startswith("CONFIRM_")
            ):
                continue
            if isinstance(value, tuple) and all(
                isinstance(item, str) for item in value
            ):
                prior.update(value)
    overlap = all_w6c_values() & prior
    assert not overlap, overlap


def test_w6c_templates_are_split_disjoint_and_render_fresh_scaffolds():
    assert set(TRAIN_TEMPLATES).isdisjoint(CONFIRM_TEMPLATES)
    placeholder = (
        "component placeholder",
        "zone placeholder",
        "fault placeholder",
        "channel placeholder",
    )
    rendered = []
    for template in (
        *TRAIN_TEMPLATES,
        "w6c-dev-checkout-record",
        "w6c-dev-status-ledger",
        *CONFIRM_TEMPLATES,
    ):
        rendered.append(
            _render_state(
                placeholder,
                severity=2,
                confidence="provisional",
                template_id=template,
            )
        )
    text = "\n".join(rendered)
    forbidden = (
        "Maintenance capsule reports",
        "Operations folio:",
        "Diagnostic slate places",
        "Control summary identifies",
        "Inspection brief records",
        "Service tableau associates",
        "Verification sheet lists",
        "Assurance card:",
        "Reliability panel links",
        "Survey lattice states",
        "Catalog prism maps",
        "Signal folio notes",
        "Coordinate slate gives",
        "Inspection grid carries",
        "Verifier ribbon maps",
        "Crosscheck tableau states",
    )
    for phrase in forbidden:
        assert phrase not in text


def test_noul_true_false_is_exactly_balanced_per_k_on_train_and_dev():
    for split in ("train", "dev"):
        cases = generate_w6c_authority(split)
        for k in (8, 16, 32, 64):
            rows = [case for case in cases if case.diagnosis_k == k]
            labels = []
            for case in rows:
                review = next(
                    decision
                    for decision in case.typed.decisions
                    if decision.question_id == "needs_review"
                )
                labels.append(review.gold_index)
            assert labels.count(0) == labels.count(1)


def test_train_dev_confirm_field_lexicons_are_pairwise_disjoint():
    from nmd.typed_reliability_authority import (
        CONFIRM_ANOMALIES,
        CONFIRM_CHANNELS,
        CONFIRM_COMPONENTS,
        CONFIRM_ZONES,
    )
    train_values = set().union(
        TRAIN_COMPONENTS,
        TRAIN_ZONES,
        TRAIN_ANOMALIES,
        TRAIN_CHANNELS,
    )
    dev_values = set().union(
        DEV_COMPONENTS,
        DEV_ZONES,
        DEV_ANOMALIES,
        DEV_CHANNELS,
    )
    confirm_values = set().union(
        CONFIRM_COMPONENTS,
        CONFIRM_ZONES,
        CONFIRM_ANOMALIES,
        CONFIRM_CHANNELS,
    )
    assert train_values.isdisjoint(dev_values)
    assert train_values.isdisjoint(confirm_values)
    assert dev_values.isdisjoint(confirm_values)
