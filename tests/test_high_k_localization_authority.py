from collections import Counter, defaultdict

from nmd.high_k_localization_authority import (
    BASES_PER_DOMAIN,
    DISTANCE_TARGETS,
    DOMAIN_N_SEED,
    DOMAIN_O_SEED,
    DOMAIN_P_SEED,
    DOMAINS,
    K_VALUES,
    all_w6f_values,
    distance_histogram,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w6f_diagnostics,
    generate_w6f_domain,
)


def _gold_text(view):
    decision = view.typed.decisions[0]
    return decision.options[decision.gold_index].criterion_text


def _option_texts(view):
    return {
        option.criterion_text
        for option in view.typed.decisions[0].options
    }


def _option_ids(view):
    return {
        option.option_id
        for option in view.typed.decisions[0].options
    }


def test_w6f_seeds_counts_and_domain_identity_are_frozen():
    assert (DOMAIN_N_SEED, DOMAIN_O_SEED, DOMAIN_P_SEED) == (
        201173,
        201181,
        201187,
    )
    assert K_VALUES == (8, 16, 32, 64)
    assert BASES_PER_DOMAIN == 96
    assert set(DOMAINS) == {"N", "O", "P"}

    all_rows = generate_w6f_diagnostics()
    assert len(all_rows) == 3 * 96 * 4

    by_domain = Counter(row.domain_id for row in all_rows)
    assert by_domain == {"N": 384, "O": 384, "P": 384}

    by_domain_k = Counter(
        (row.domain_id, row.diagnosis_k)
        for row in all_rows
    )
    assert set(by_domain_k.values()) == {96}


def test_w6f_base_severity_confidence_strata_are_exactly_balanced():
    for domain_id in ("N", "O", "P"):
        rows = generate_w6f_domain(domain_id)
        bases = {}
        for row in rows:
            bases.setdefault(
                row.base_id,
                (row.severity, row.confidence),
            )
            assert bases[row.base_id] == (
                row.severity,
                row.confidence,
            )
        assert len(bases) == 96
        strata = Counter(bases.values())
        assert len(strata) == 12
        assert set(strata.values()) == {8}


def test_every_base_uses_same_state_gold_and_nested_candidate_membership():
    for domain_id in ("N", "O", "P"):
        rows = generate_w6f_domain(domain_id)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.base_id].append(row)
        assert len(grouped) == 96

        golds = []
        for base_id, views in grouped.items():
            ordered = sorted(views, key=lambda row: row.diagnosis_k)
            assert [row.diagnosis_k for row in ordered] == [8, 16, 32, 64]
            assert len({row.typed.state_text for row in ordered}) == 1
            assert len({row.gold_signature for row in ordered}) == 1
            assert len({_gold_text(row) for row in ordered}) == 1
            golds.append(ordered[0].gold_signature)

            for left, right in zip(ordered, ordered[1:]):
                assert _option_texts(left) < _option_texts(right)
                assert _option_ids(left) < _option_ids(right)
                assert _gold_text(left) in _option_texts(right)

            for view in ordered:
                decision = view.typed.decisions[0]
                assert decision.question_id == "diagnosis"
                assert decision.primitive == "choice"
                assert len(decision.options) == view.diagnosis_k
                assert len(view.option_signatures) == view.diagnosis_k
                assert len(view.option_distances) == view.diagnosis_k
                assert view.option_distances[decision.gold_index] == 0
                assert abs(sum(decision.gold_probabilities) - 1.0) < 1e-12

        assert len(golds) == len(set(golds)) == 96


def test_nested_views_match_frozen_hard_negative_distance_targets():
    for row in generate_w6f_diagnostics():
        assert distance_histogram(row) == DISTANCE_TARGETS[row.diagnosis_k]


def test_w6f_domains_are_pairwise_disjoint_in_values_templates_and_roles():
    value_sets = domain_value_sets()
    template_sets = domain_template_sets()
    role_sets = domain_role_sets()
    ids = sorted(DOMAINS)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            assert value_sets[left].isdisjoint(value_sets[right])
            assert template_sets[left].isdisjoint(template_sets[right])
            assert role_sets[left].isdisjoint(role_sets[right])
    assert len(all_w6f_values()) == 3 * 4 * 24


def test_w6f_values_are_exactly_fresh_against_prior_authorities():
    from nmd.typed_competitive_authority import all_w6b_values
    from nmd.typed_reliability_authority import all_w6c_values
    from nmd.typed_domain_generalization_authority import all_w6d_values
    from nmd.typed_joint_replication_authority import all_w6e_values
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab

    prior = (
        set(all_w6b_values())
        | set(all_w6c_values())
        | set(all_w6d_values())
        | set(all_w6e_values())
        | set(all_w5f_vocab())
        | set(all_w5g_vocab())
        | set(all_w5h_vocab())
        | set(all_w5i_vocab())
    )
    overlap = all_w6f_values() & prior
    assert not overlap, overlap


def test_w6f_templates_and_role_wording_are_fresh_against_prior_authorities():
    from nmd import semantic_balanced_binding as w5h
    from nmd import semantic_contrastive_salience as w5g
    from nmd import semantic_cross_candidate_binding as w5i
    from nmd import semantic_late_interaction as w5f
    from nmd import typed_competitive_authority as w6b
    from nmd import typed_reliability_authority as w6c
    from nmd import typed_domain_generalization_authority as w6d
    from nmd import typed_joint_replication_authority as w6e

    def module_literal_strings(module):
        rows = set()
        for value in vars(module).values():
            if isinstance(value, str):
                rows.add(value)
            elif isinstance(value, (tuple, list, set)) and all(
                isinstance(item, str) for item in value
            ):
                rows.update(value)
        return rows

    prior_literals = set()
    for module in (w5f, w5g, w5h, w5i, w6b, w6c):
        prior_literals.update(module_literal_strings(module))

    prior_templates = set(w6b.TRAIN_TEMPLATES)
    prior_templates.update(w6b.DEV_TEMPLATES)
    prior_templates.update(w6b.CONFIRM_TEMPLATES)
    prior_templates.update(w6c.TRAIN_TEMPLATES)
    prior_templates.update(w6c.DEV_TEMPLATES)
    prior_templates.update(w6c.CONFIRM_TEMPLATES)
    prior_roles = {
        "equipment", "zone", "anomaly", "channel",
        "component", "compartment", "fault", "telemetry",
    }

    for spec in w6d.DOMAINS.values():
        prior_templates.update(spec.templates)
        prior_roles.update(spec.roles)
    for spec in w6e.DOMAINS.values():
        prior_templates.update(spec.templates)
        prior_roles.update(spec.roles)

    prior_templates.update(
        value for value in prior_literals if "w5" in value
    )

    w6f_templates = {
        template
        for spec in DOMAINS.values()
        for template in spec.templates
    }
    w6f_roles = {
        role
        for spec in DOMAINS.values()
        for role in spec.roles
    }
    assert w6f_templates.isdisjoint(prior_templates)
    assert w6f_templates.isdisjoint(prior_literals)
    assert w6f_roles.isdisjoint(prior_roles)
    assert w6f_roles.isdisjoint(prior_literals)


def test_w6f_is_diagnostic_only_and_contains_no_prior_case_ids():
    rows = generate_w6f_diagnostics()
    ids = [row.typed.case_id for row in rows]
    assert len(ids) == len(set(ids))
    assert all(case_id.startswith("w6f-diag-") for case_id in ids)
    assert all(len(row.typed.decisions) == 1 for row in rows)
    assert all(row.typed.decisions[0].question_id == "diagnosis" for row in rows)
