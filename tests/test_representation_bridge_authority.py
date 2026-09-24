from __future__ import annotations

from collections import Counter, defaultdict

from nmd.representation_bridge_authority import (
    BASES_PER_DOMAIN,
    DOMAIN_AA_SEED,
    DOMAIN_AB_SEED,
    DOMAIN_AC_SEED,
    DOMAINS,
    PAIR_VIEW_IDS,
    ROLE_KEYS,
    VIEW_IDS,
    all_w6i_values,
    canonical_tagged_text,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    factorized_role_value_phrases,
    generate_w6i_diagnostics,
    generate_w6i_domain,
    signature_distance,
    view_histogram,
)


def _decision(view):
    return view.typed.decisions[0]


def _options(view):
    return {
        option.option_id: option.criterion_text
        for option in _decision(view).options
    }


def _signatures(view):
    return {
        option.option_id: signature
        for option, signature in zip(
            _decision(view).options,
            view.option_signatures,
        )
    }


def test_w6i_seeds_counts_and_view_contracts_are_frozen():
    assert (DOMAIN_AA_SEED, DOMAIN_AB_SEED, DOMAIN_AC_SEED) == (
        231269,
        231271,
        231277,
    )
    assert BASES_PER_DOMAIN == 64
    assert ROLE_KEYS == ("entity", "location", "anomaly", "channel")
    assert set(PAIR_VIEW_IDS) == {
        "pair-entity",
        "pair-location",
        "pair-anomaly",
        "pair-channel",
    }
    assert VIEW_IDS == (
        "pair-entity",
        "pair-location",
        "pair-anomaly",
        "pair-channel",
        "core-k8",
        "master-k64",
    )

    rows = generate_w6i_diagnostics()
    assert len(rows) == 3 * 64 * 6
    assert Counter(row.domain_id for row in rows) == {
        "AA": 384,
        "AB": 384,
        "AC": 384,
    }
    histogram = view_histogram(rows)
    assert set(histogram) == set(VIEW_IDS)
    assert set(histogram.values()) == {3 * 64}


def test_each_base_has_exact_same_state_gold_and_six_views():
    for domain_id in ("AA", "AB", "AC"):
        grouped = defaultdict(list)
        for row in generate_w6i_domain(domain_id):
            grouped[row.base_id].append(row)
        assert len(grouped) == 64

        for base_id, views in grouped.items():
            assert len(views) == 6
            assert {view.view_id for view in views} == set(VIEW_IDS)
            assert len({view.typed.state_text for view in views}) == 1
            assert len({view.gold_signature for view in views}) == 1
            assert len({view.gold_option_id for view in views}) == 1
            assert all(view.typed.case_id.startswith(base_id) for view in views)
            assert all(len(view.typed.decisions) == 1 for view in views)
            assert all(_decision(view).primitive == "choice" for view in views)
            assert all(_decision(view).question_id == "diagnosis" for view in views)


def test_pair_views_use_exact_same_one_field_counterfactuals():
    for row in generate_w6i_diagnostics():
        if row.view_id not in PAIR_VIEW_IDS:
            continue
        assert row.diagnosis_k == 2
        assert row.target_role_index is not None
        assert row.target_role == ROLE_KEYS[row.target_role_index]
        assert row.target_negative_signature is not None
        assert signature_distance(
            row.gold_signature,
            row.target_negative_signature,
        ) == 1
        for index, (gold, negative) in enumerate(
            zip(row.gold_signature, row.target_negative_signature)
        ):
            if index == row.target_role_index:
                assert gold != negative
            else:
                assert gold == negative


def test_k8_is_nested_byte_identically_in_k64():
    for domain_id in ("AA", "AB", "AC"):
        grouped = defaultdict(dict)
        for row in generate_w6i_domain(domain_id):
            grouped[row.base_id][row.view_id] = row

        for views in grouped.values():
            core = views["core-k8"]
            master = views["master-k64"]
            assert core.diagnosis_k == 8
            assert master.diagnosis_k == 64
            core_options = _options(core)
            master_options = _options(master)
            core_signatures = _signatures(core)
            master_signatures = _signatures(master)
            assert set(core_options) < set(master_options)
            for option_id, text in core_options.items():
                assert master_options[option_id] == text
                assert master_signatures[option_id] == core_signatures[option_id]


def test_every_one_field_pair_is_byte_identical_across_pair_core_master():
    for domain_id in ("AA", "AB", "AC"):
        grouped = defaultdict(dict)
        for row in generate_w6i_domain(domain_id):
            grouped[row.base_id][row.view_id] = row

        for views in grouped.values():
            core = _options(views["core-k8"])
            master = _options(views["master-k64"])
            for role in ROLE_KEYS:
                pair = views[f"pair-{role}"]
                pair_options = _options(pair)
                assert pair.target_negative_option_id is not None
                for option_id in (
                    pair.gold_option_id,
                    pair.target_negative_option_id,
                ):
                    assert core[option_id] == pair_options[option_id]
                    assert master[option_id] == pair_options[option_id]


def test_master_has_expected_controlled_distance_structure():
    for domain_id in ("AA", "AB", "AC"):
        grouped = defaultdict(dict)
        for row in generate_w6i_domain(domain_id):
            grouped[row.base_id][row.view_id] = row

        for views in grouped.values():
            master = views["master-k64"]
            distances = Counter(
                signature_distance(master.gold_signature, signature)
                for signature in master.option_signatures
            )
            assert distances[0] == 1
            assert distances[1] == 4
            assert distances[2] >= 12
            assert distances[3] + distances[4] >= 36
            assert sum(distances.values()) == 64


def test_canonical_and_factorized_renderings_preserve_only_same_signature():
    for domain_id, spec in DOMAINS.items():
        rows = generate_w6i_domain(domain_id)
        sample = rows[0].gold_signature
        canonical = canonical_tagged_text(sample)
        assert canonical == " ".join(
            f"[F{index}] {value}"
            for index, value in enumerate(sample)
        )
        phrases = factorized_role_value_phrases(spec, sample)
        assert len(phrases) == 4
        for index, phrase in enumerate(phrases):
            assert phrase == f"{spec.roles[index]} {sample[index]}"


def test_domains_are_pairwise_disjoint_in_values_templates_and_roles():
    values = domain_value_sets()
    templates = domain_template_sets()
    roles = domain_role_sets()
    ids = sorted(DOMAINS)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            assert values[left].isdisjoint(values[right])
            assert templates[left].isdisjoint(templates[right])
            assert roles[left].isdisjoint(roles[right])
    assert len(all_w6i_values()) == 3 * 4 * 64


def test_w6i_values_are_fresh_against_all_prior_w5_w6_lanes():
    from nmd.field_semantic_rescue_authority import all_w6h_values
    from nmd.high_k_localization_authority import all_w6f_values
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
    )
    overlap = all_w6i_values() & prior
    assert not overlap, overlap


def test_w6i_templates_and_roles_are_fresh_against_prior_w6_lanes():
    from nmd.field_semantic_rescue_authority import DOMAINS as W6H_DOMAINS
    from nmd.high_k_localization_authority import DOMAINS as W6F_DOMAINS
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
    first = generate_w6i_diagnostics()
    second = generate_w6i_diagnostics()

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
    assert all(case_id.startswith("w6i-diag-") for case_id in case_ids)
    for row in first:
        ids = [option.option_id for option in _decision(row).options]
        assert len(ids) == len(set(ids))
        assert len(ids) == row.diagnosis_k
        assert abs(sum(_decision(row).gold_probabilities) - 1.0) < 1e-12
