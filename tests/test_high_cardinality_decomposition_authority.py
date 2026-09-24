from __future__ import annotations

from collections import Counter, defaultdict

from nmd.high_cardinality_decomposition_authority import (
    BASES_PER_DOMAIN,
    DOMAINS,
    DOMAIN_AD_SEED,
    DOMAIN_AE_SEED,
    DOMAIN_AF_SEED,
    VIEW_IDS,
    all_w6j_values,
    changed_roles,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w6j_diagnostics,
    generate_w6j_domain,
    signature_distance,
    view_histogram,
)


def _decision(view):
    return view.typed.decisions[0]


def _signature_map(view):
    return {
        signature: option
        for signature, option in zip(
            view.option_signatures,
            _decision(view).options,
        )
    }


def test_w6j_seeds_counts_and_views_are_frozen():
    assert (DOMAIN_AD_SEED, DOMAIN_AE_SEED, DOMAIN_AF_SEED) == (
        241301,
        241307,
        241319,
    )
    assert BASES_PER_DOMAIN == 64
    assert VIEW_IDS == ("k8", "k16", "k32", "k64")

    rows = generate_w6j_diagnostics()
    assert len(rows) == 3 * 64 * 4
    assert Counter(row.domain_id for row in rows) == {
        "AD": 256,
        "AE": 256,
        "AF": 256,
    }
    assert view_histogram(rows) == {
        "k8": 192,
        "k16": 192,
        "k32": 192,
        "k64": 192,
    }


def test_each_base_has_nested_views_stable_state_and_gold():
    for domain_id in ("AD", "AE", "AF"):
        grouped = defaultdict(dict)
        for row in generate_w6j_domain(domain_id):
            grouped[row.base_id][row.view_id] = row

        assert len(grouped) == 64
        for views in grouped.values():
            assert set(views) == set(VIEW_IDS)
            assert len({row.typed.state_text for row in views.values()}) == 1
            assert len({row.gold_signature for row in views.values()}) == 1
            assert len({row.gold_option_id for row in views.values()}) == 1

            maps = {
                view_id: _signature_map(row)
                for view_id, row in views.items()
            }
            assert set(maps["k8"]) < set(maps["k16"])
            assert set(maps["k16"]) < set(maps["k32"])
            assert set(maps["k32"]) < set(maps["k64"])

            for smaller, larger in (
                ("k8", "k16"),
                ("k16", "k32"),
                ("k32", "k64"),
            ):
                for signature, option in maps[smaller].items():
                    other = maps[larger][signature]
                    assert other.option_id == option.option_id
                    assert other.criterion_text == option.criterion_text


def test_master_distance_histogram_is_exact():
    for domain_id in ("AD", "AE", "AF"):
        for row in generate_w6j_domain(domain_id):
            if row.view_id != "k64":
                continue
            assert Counter(row.option_distances) == {
                0: 1,
                1: 12,
                2: 20,
                3: 15,
                4: 16,
            }
            assert row.option_distances[_decision(row).gold_index] == 0


def test_nested_distance_composition_is_frozen():
    expected = {
        "k8": {0: 1, 1: 4, 2: 3},
        "k16": {0: 1, 1: 8, 2: 5, 3: 2},
        "k32": {0: 1, 1: 12, 2: 12, 3: 7},
        "k64": {0: 1, 1: 12, 2: 20, 3: 15, 4: 16},
    }
    for row in generate_w6j_diagnostics():
        assert Counter(row.option_distances) == expected[row.view_id]


def test_distance_and_changed_roles_agree():
    for row in generate_w6j_diagnostics()[:128]:
        for signature, distance in zip(
            row.option_signatures,
            row.option_distances,
        ):
            assert signature_distance(row.gold_signature, signature) == distance
            assert len(changed_roles(row.gold_signature, signature)) == distance


def test_domains_are_pairwise_disjoint():
    values = domain_value_sets()
    templates = domain_template_sets()
    roles = domain_role_sets()
    ids = sorted(DOMAINS)
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            assert values[left].isdisjoint(values[right])
            assert templates[left].isdisjoint(templates[right])
            assert roles[left].isdisjoint(roles[right])


def test_w6j_values_are_fresh_against_all_prior_w5_w6_lanes():
    from nmd.field_semantic_rescue_authority import all_w6h_values
    from nmd.high_k_localization_authority import all_w6f_values
    from nmd.representation_bridge_authority import all_w6i_values
    from nmd.second_order_localization_authority import all_w6g_values
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
        | set(all_w6g_values())
        | set(all_w6h_values())
        | set(all_w6i_values())
    )
    overlap = all_w6j_values() & prior
    assert not overlap, overlap


def test_w6j_templates_and_roles_are_fresh_against_prior_w6_lanes():
    from nmd.field_semantic_rescue_authority import DOMAINS as W6H_DOMAINS
    from nmd.high_k_localization_authority import DOMAINS as W6F_DOMAINS
    from nmd.representation_bridge_authority import DOMAINS as W6I_DOMAINS
    from nmd.second_order_localization_authority import DOMAINS as W6G_DOMAINS
    from nmd.typed_domain_generalization_authority import DOMAINS as W6D_DOMAINS
    from nmd.typed_joint_replication_authority import DOMAINS as W6E_DOMAINS

    prior_templates = set()
    prior_roles = set()
    for collection in (
        W6D_DOMAINS,
        W6E_DOMAINS,
        W6F_DOMAINS,
        W6G_DOMAINS,
        W6H_DOMAINS,
        W6I_DOMAINS,
    ):
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


def test_case_and_option_ids_are_deterministic():
    first = generate_w6j_diagnostics()
    second = generate_w6j_diagnostics()

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
    assert all(case_id.startswith("w6j-diag-") for case_id in case_ids)
    for row in first:
        ids = [option.option_id for option in _decision(row).options]
        assert len(ids) == len(set(ids))
        assert len(ids) == row.diagnosis_k
        assert abs(sum(_decision(row).gold_probabilities) - 1.0) < 1e-12
