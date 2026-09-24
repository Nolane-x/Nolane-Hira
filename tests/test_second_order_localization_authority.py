from __future__ import annotations

from collections import Counter, defaultdict

from nmd.second_order_localization_authority import (
    BASES_PER_DOMAIN,
    DENSE_VIEW_IDS,
    DOMAIN_Q_SEED,
    DOMAIN_R_SEED,
    DOMAIN_S_SEED,
    DOMAINS,
    PAIR_VIEW_IDS,
    ROLE_KEYS,
    VIEW_IDS,
    all_w6g_values,
    changed_roles,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w6g_diagnostics,
    generate_w6g_domain,
    signature_distance,
    view_histogram,
)


def _decision(view):
    return view.typed.decisions[0]


def _option_map(view):
    return {
        option.option_id: option.criterion_text
        for option in _decision(view).options
    }


def _signature_map(view):
    return {
        option.option_id: signature
        for option, signature in zip(
            _decision(view).options,
            view.option_signatures,
        )
    }


def test_w6g_seeds_counts_and_view_identity_are_frozen():
    assert (DOMAIN_Q_SEED, DOMAIN_R_SEED, DOMAIN_S_SEED) == (
        211193,
        211199,
        211213,
    )
    assert BASES_PER_DOMAIN == 64
    assert ROLE_KEYS == ("entity", "location", "anomaly", "channel")
    assert len(VIEW_IDS) == 10
    assert set(PAIR_VIEW_IDS) == {
        "pair-entity",
        "pair-location",
        "pair-anomaly",
        "pair-channel",
    }
    assert set(DENSE_VIEW_IDS) == {
        "dense-entity64",
        "dense-location64",
        "dense-anomaly64",
        "dense-channel64",
    }

    rows = generate_w6g_diagnostics()
    assert len(rows) == 3 * 64 * 10
    assert Counter(row.domain_id for row in rows) == {
        "Q": 640,
        "R": 640,
        "S": 640,
    }
    histogram = view_histogram(rows)
    assert set(histogram) == set(VIEW_IDS)
    assert set(histogram.values()) == {3 * 64}


def test_each_base_has_exact_ten_views_and_stable_state_gold():
    for domain_id in ("Q", "R", "S"):
        grouped = defaultdict(list)
        for row in generate_w6g_domain(domain_id):
            grouped[row.base_id].append(row)
        assert len(grouped) == 64

        for base_id, views in grouped.items():
            assert len(views) == 10
            assert {row.view_id for row in views} == set(VIEW_IDS)
            assert len({row.typed.state_text for row in views}) == 1
            assert len({row.gold_signature for row in views}) == 1
            assert len({row.gold_option_id for row in views}) == 1
            assert all(row.typed.case_id.startswith(base_id) for row in views)
            assert all(len(row.typed.decisions) == 1 for row in views)
            assert all(_decision(row).question_id == "diagnosis" for row in views)
            assert all(_decision(row).primitive == "choice" for row in views)


def test_pair_views_are_exact_one_field_counterfactuals():
    for row in generate_w6g_diagnostics():
        if not row.view_id.startswith("pair-"):
            continue
        assert row.diagnosis_k == 2
        assert row.target_role in ROLE_KEYS
        assert row.target_role_index == ROLE_KEYS.index(row.target_role)
        assert row.target_negative_signature is not None
        assert row.target_negative_option_id is not None
        assert signature_distance(
            row.gold_signature,
            row.target_negative_signature,
        ) == 1
        assert changed_roles(
            row.gold_signature,
            row.target_negative_signature,
        ) == (row.target_role,)
        assert set(row.option_distances) == {0, 1}
        assert row.gold_option_id in _option_map(row)
        assert row.target_negative_option_id in _option_map(row)


def test_core_is_preserved_byte_for_byte_in_far_and_dense_views():
    for domain_id in ("Q", "R", "S"):
        grouped = defaultdict(dict)
        for row in generate_w6g_domain(domain_id):
            grouped[row.base_id][row.view_id] = row

        for views in grouped.values():
            core = views["core-k8"]
            core_options = _option_map(core)
            core_signatures = _signature_map(core)
            assert core.diagnosis_k == 8
            assert Counter(core.option_distances) == {0: 1, 1: 4, 2: 3}

            for view_id in ("far64", *DENSE_VIEW_IDS):
                view = views[view_id]
                assert view.diagnosis_k == 64
                options = _option_map(view)
                signatures = _signature_map(view)
                assert set(core_options) < set(options)
                for option_id, text in core_options.items():
                    assert options[option_id] == text
                    assert signatures[option_id] == core_signatures[option_id]


def test_far64_adds_only_three_or_four_field_spectators():
    for domain_id in ("Q", "R", "S"):
        grouped = defaultdict(dict)
        for row in generate_w6g_domain(domain_id):
            grouped[row.base_id][row.view_id] = row

        for views in grouped.values():
            core_ids = set(_option_map(views["core-k8"]))
            far = views["far64"]
            far_map = _signature_map(far)
            added = [
                signature
                for option_id, signature in far_map.items()
                if option_id not in core_ids
            ]
            assert len(added) == 56
            assert all(
                signature_distance(far.gold_signature, signature) >= 3
                for signature in added
            )


def test_dense64_adds_only_same_role_one_field_spectators():
    for domain_id in ("Q", "R", "S"):
        grouped = defaultdict(dict)
        for row in generate_w6g_domain(domain_id):
            grouped[row.base_id][row.view_id] = row

        for views in grouped.values():
            core_ids = set(_option_map(views["core-k8"]))
            for role in ROLE_KEYS:
                dense = views[f"dense-{role}64"]
                dense_map = _signature_map(dense)
                added = [
                    signature
                    for option_id, signature in dense_map.items()
                    if option_id not in core_ids
                ]
                assert len(added) == 56
                assert all(
                    signature_distance(
                        dense.gold_signature,
                        signature,
                    ) == 1
                    for signature in added
                )
                assert all(
                    changed_roles(
                        dense.gold_signature,
                        signature,
                    ) == (role,)
                    for signature in added
                )


def test_target_pair_is_stable_across_pair_core_far_and_matching_dense():
    for domain_id in ("Q", "R", "S"):
        grouped = defaultdict(dict)
        for row in generate_w6g_domain(domain_id):
            grouped[row.base_id][row.view_id] = row

        for views in grouped.values():
            core_map = _option_map(views["core-k8"])
            far_map = _option_map(views["far64"])
            for role in ROLE_KEYS:
                pair = views[f"pair-{role}"]
                dense = views[f"dense-{role}64"]
                dense_map = _option_map(dense)
                pair_map = _option_map(pair)
                target_id = pair.target_negative_option_id
                assert target_id is not None
                for option_id in (pair.gold_option_id, target_id):
                    text = pair_map[option_id]
                    assert core_map[option_id] == text
                    assert far_map[option_id] == text
                    assert dense_map[option_id] == text


def test_base_severity_confidence_strata_are_nearly_balanced():
    for domain_id in ("Q", "R", "S"):
        bases = {}
        for row in generate_w6g_domain(domain_id):
            bases.setdefault(
                row.base_id,
                (row.severity, row.confidence),
            )
            assert bases[row.base_id] == (
                row.severity,
                row.confidence,
            )
        assert len(bases) == 64
        counts = Counter(bases.values())
        assert len(counts) == 12
        assert set(counts.values()) <= {5, 6}
        assert sum(counts.values()) == 64


def test_domains_are_pairwise_disjoint_in_values_templates_and_roles():
    value_sets = domain_value_sets()
    template_sets = domain_template_sets()
    role_sets = domain_role_sets()
    ids = sorted(DOMAINS)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            assert value_sets[left].isdisjoint(value_sets[right])
            assert template_sets[left].isdisjoint(template_sets[right])
            assert role_sets[left].isdisjoint(role_sets[right])
    assert len(all_w6g_values()) == 3 * 4 * 64


def test_w6g_values_are_fresh_against_all_prior_authority_and_diagnostic_values():
    from nmd.high_k_localization_authority import all_w6f_values
    from nmd.semantic_balanced_binding import all_w5h_vocab
    from nmd.semantic_contrastive_salience import all_w5g_vocab
    from nmd.semantic_cross_candidate_binding import all_w5i_vocab
    from nmd.semantic_late_interaction import all_w5f_vocab
    from nmd.typed_competitive_authority import all_w6b_values
    from nmd.typed_domain_generalization_authority import all_w6d_values
    from nmd.typed_joint_replication_authority import all_w6e_values
    from nmd.typed_reliability_authority import all_w6c_values

    prior = (
        set(all_w5f_vocab())
        | set(all_w5g_vocab())
        | set(all_w5h_vocab())
        | set(all_w5i_vocab())
        | set(all_w6b_values())
        | set(all_w6c_values())
        | set(all_w6d_values())
        | set(all_w6e_values())
        | set(all_w6f_values())
    )
    overlap = all_w6g_values() & prior
    assert not overlap, overlap


def test_w6g_templates_and_roles_are_fresh_against_prior_w6_domains():
    from nmd.high_k_localization_authority import DOMAINS as W6F_DOMAINS
    from nmd.typed_competitive_authority import (
        CONFIRM_TEMPLATES as W6B_CONFIRM,
        DEV_TEMPLATES as W6B_DEV,
        TRAIN_TEMPLATES as W6B_TRAIN,
    )
    from nmd.typed_domain_generalization_authority import DOMAINS as W6D_DOMAINS
    from nmd.typed_joint_replication_authority import DOMAINS as W6E_DOMAINS
    from nmd.typed_reliability_authority import (
        CONFIRM_TEMPLATES as W6C_CONFIRM,
        DEV_TEMPLATES as W6C_DEV,
        TRAIN_TEMPLATES as W6C_TRAIN,
    )

    prior_templates = set(
        (*W6B_TRAIN, *W6B_DEV, *W6B_CONFIRM, *W6C_TRAIN, *W6C_DEV, *W6C_CONFIRM)
    )
    prior_roles = {
        "equipment",
        "zone",
        "anomaly",
        "channel",
        "component",
        "compartment",
        "fault",
        "telemetry",
    }
    for collection in (W6D_DOMAINS, W6E_DOMAINS, W6F_DOMAINS):
        for spec in collection.values():
            prior_templates.update(spec.templates)
            prior_roles.update(spec.roles)

    current_templates = {
        template
        for spec in DOMAINS.values()
        for template in spec.templates
    }
    current_roles = {
        role
        for spec in DOMAINS.values()
        for role in spec.roles
    }
    assert current_templates.isdisjoint(prior_templates)
    assert current_roles.isdisjoint(prior_roles)


def test_case_ids_and_option_ids_are_deterministic_and_unique_per_view():
    first = generate_w6g_diagnostics()
    second = generate_w6g_diagnostics()

    assert [row.typed.case_id for row in first] == [
        row.typed.case_id for row in second
    ]
    assert [
        tuple(option.option_id for option in _decision(row).options)
        for row in first
    ] == [
        tuple(option.option_id for option in _decision(row).options)
        for row in second
    ]

    case_ids = [row.typed.case_id for row in first]
    assert len(case_ids) == len(set(case_ids))
    assert all(case_id.startswith("w6g-diag-") for case_id in case_ids)
    for row in first:
        ids = [option.option_id for option in _decision(row).options]
        assert len(ids) == len(set(ids))
        assert len(ids) == row.diagnosis_k
        assert abs(sum(_decision(row).gold_probabilities) - 1.0) < 1e-12
