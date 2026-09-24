import math

import pytest

from nmd.typed_competitive_authority import (
    CONFIDENCE_MASS,
    CONFIRM_ANOMALIES,
    CONFIRM_CHANNELS,
    CONFIRM_EQUIPMENT,
    CONFIRM_K_COUNTS,
    CONFIRM_SEED,
    CONFIRM_TEMPLATES,
    CONFIRM_ZONES,
    DEV_K_COUNTS,
    DEV_SEED,
    TRAIN_ANOMALIES,
    TRAIN_CHANNELS,
    TRAIN_EQUIPMENT,
    TRAIN_K_COUNTS,
    TRAIN_SEED,
    TRAIN_TEMPLATES,
    TRAIN_ZONES,
    _render_state,
    all_w6b_values,
    generate_w6b_authority,
)


def test_w6b_counts_seeds_and_confirm_seal_are_frozen():
    assert sum(TRAIN_K_COUNTS.values()) == 320
    assert sum(DEV_K_COUNTS.values()) == 128
    assert sum(CONFIRM_K_COUNTS.values()) == 160
    assert (TRAIN_SEED, DEV_SEED, CONFIRM_SEED) == (
        161127,
        162229,
        163331,
    )

    train = generate_w6b_authority("train")
    dev = generate_w6b_authority("dev")
    assert len(train) == 320
    assert len(dev) == 128
    assert sum(len(case.typed.decisions) for case in train) == 1600
    assert sum(len(case.typed.decisions) for case in dev) == 640
    assert {case.typed.case_id for case in train}.isdisjoint(
        {case.typed.case_id for case in dev}
    )
    with pytest.raises(RuntimeError, match="sealed until post-selection"):
        generate_w6b_authority("confirm")


def test_every_case_has_five_frozen_typed_decisions_and_valid_soft_targets():
    for case in (
        generate_w6b_authority("train")[:24]
        + generate_w6b_authority("dev")[:24]
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
        assert len(decisions[1].options) == 4
        assert len(decisions[2].options) == 2
        assert len(decisions[3].options) == 4
        assert len(decisions[4].options) == 4

        expected_mass = CONFIDENCE_MASS[case.confidence]
        for decision in decisions:
            probs = decision.gold_probabilities
            assert math.isclose(sum(probs), 1.0, abs_tol=1e-12)
            assert probs[decision.gold_index] == pytest.approx(
                expected_mass
            )
            assert max(range(len(probs)), key=probs.__getitem__) == (
                decision.gold_index
            )

        for decision in decisions[3:]:
            assert decision.gold_score is not None
            expected = sum(
                float(option.value) * probability
                for option, probability in zip(
                    decision.options,
                    decision.gold_probabilities,
                )
            )
            assert decision.gold_score == pytest.approx(expected)


def test_review_and_urgency_rules_match_frozen_contract():
    for case in (
        generate_w6b_authority("train")[:96]
        + generate_w6b_authority("dev")[:64]
    ):
        decisions = {
            decision.question_id: decision
            for decision in case.typed.decisions
        }
        expected_review = int(
            case.confidence == "uncertain" or case.severity == 3
        )
        expected_urgency = min(
            3,
            case.severity + int(case.confidence == "uncertain"),
        )
        assert decisions["needs_review"].gold_index == expected_review
        assert decisions["risk"].gold_index == case.severity
        assert decisions["urgency"].gold_index == expected_urgency
        assert decisions["response"].gold_index == case.severity


def test_confirm_value_lexicons_are_reserved_and_disjoint():
    train_dev = set().union(
        TRAIN_EQUIPMENT,
        TRAIN_ZONES,
        TRAIN_ANOMALIES,
        TRAIN_CHANNELS,
    )
    confirm = set().union(
        CONFIRM_EQUIPMENT,
        CONFIRM_ZONES,
        CONFIRM_ANOMALIES,
        CONFIRM_CHANNELS,
    )
    assert not (train_dev & confirm)
    assert len(train_dev) == 96
    assert len(confirm) == 64
    assert len(all_w6b_values()) == 160


def test_templates_are_split_disjoint_and_avoid_prior_w5_rendered_scaffolds():
    assert set(TRAIN_TEMPLATES).isdisjoint(CONFIRM_TEMPLATES)
    placeholder = (
        "equipment placeholder",
        "zone placeholder",
        "anomaly placeholder",
        "channel placeholder",
    )
    rendered = []
    for template in (
        *TRAIN_TEMPLATES,
        "w6b-dev-inspection-brief",
        "w6b-dev-service-tableau",
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
        "The active route uses",
        "Route attributes:",
        "Dispatch record says",
        "Record: color=",
        "Envelope:",
        "Card fields:",
        "Manifest fields:",
        "Roster fields:",
        "Ledger fields:",
        "Observation matrix has",
        "Courier packet lists",
        "Reference strip reads",
        "Keymap:",
        "Beacon record encodes",
        "Notebook trace:",
        "Specimen matrix contains",
        "Dispatch packet records",
        "Quadrant note assigns",
        "Survey lattice states",
        "Catalog prism maps",
        "Signal folio notes",
        "Coordinate slate gives",
        "Transit mosaic lists",
        "Archive compass marks",
        "Inspection grid carries",
        "Verifier ribbon maps",
        "Crosscheck tableau states",
    )
    for phrase in forbidden:
        assert phrase not in text


def test_w6b_value_lexicon_has_no_exact_overlap_with_w5_authority_values():
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
        set(all_w5f_vocab())
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

    overlap = all_w6b_values() & prior
    assert not overlap, overlap


def test_train_and_dev_have_no_semantic_diagnosis_signature_collision():
    train = generate_w6b_authority("train")
    dev = generate_w6b_authority("dev")

    def gold_signature(case):
        diagnosis = case.typed.decisions[0]
        return diagnosis.options[diagnosis.gold_index].criterion_text

    train_signatures = [gold_signature(case) for case in train]
    dev_signatures = [gold_signature(case) for case in dev]
    assert len(set(train_signatures)) == len(train_signatures)
    assert len(set(dev_signatures)) == len(dev_signatures)
    assert set(train_signatures).isdisjoint(dev_signatures)
