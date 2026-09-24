from collections import Counter
from pathlib import Path

import pytest

from nmd.typed_competitive_authority import all_w6b_values
from nmd.typed_reliability_authority import all_w6c_values
from nmd.typed_domain_generalization_authority import all_w6d_values
from nmd.typed_joint_replication_authority import (
    CONFIRM_L_SEED,
    CONFIRM_M_SEED,
    DEV_K_SEED,
    DOMAINS,
    PRIMARY_TRAIN_SEED,
    REPLICA_TRAIN_SEED,
    SOURCE_G_SEED,
    SOURCE_H_SEED,
    SOURCE_I_SEED,
    SOURCE_J_SEED,
    all_w6e_values,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w6e_confirm,
    generate_w6e_dev,
    generate_w6e_multi_train,
)


def _prior_w5_values() -> set[str]:
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
            if not (name.startswith("TRAIN_") or name.startswith("CONFIRM_")):
                continue
            if isinstance(value, tuple) and all(
                isinstance(item, str) for item in value
            ):
                prior.update(value)
    return prior


def _gold_signature(case):
    diagnosis = case.typed.decisions[0]
    return diagnosis.options[diagnosis.gold_index].criterion_text


def test_w6e_seeds_and_confirm_seal_are_frozen():
    assert (
        SOURCE_G_SEED,
        SOURCE_H_SEED,
        SOURCE_I_SEED,
        SOURCE_J_SEED,
        DEV_K_SEED,
        CONFIRM_L_SEED,
        CONFIRM_M_SEED,
        PRIMARY_TRAIN_SEED,
        REPLICA_TRAIN_SEED,
    ) == (
        191161,
        191167,
        191173,
        191179,
        192283,
        193387,
        194489,
        1409,
        2411,
    )

    with pytest.raises(RuntimeError, match="sealed until post-freeze"):
        generate_w6e_confirm("L")
    with pytest.raises(RuntimeError, match="sealed until post-freeze"):
        generate_w6e_confirm("M")
    with pytest.raises(RuntimeError, match="sealed until post-freeze"):
        generate_w6e_confirm("L", allow_confirm=False)


def test_w6e_train_is_exactly_balanced_across_four_source_domains():
    rows = generate_w6e_multi_train()
    assert len(rows) == 384
    assert sum(len(row.typed.decisions) for row in rows) == 1920
    assert Counter(row.domain_id for row in rows) == {
        "G": 96,
        "H": 96,
        "I": 96,
        "J": 96,
    }

    strata = Counter(
        (row.domain_id, row.diagnosis_k, row.severity, row.confidence)
        for row in rows
    )
    assert len(strata) == 4 * 4 * 4 * 3
    assert set(strata.values()) == {2}


def test_w6e_dev_k_is_exactly_balanced_and_disjoint_from_train():
    train = generate_w6e_multi_train()
    dev = generate_w6e_dev()

    assert len(dev) == 192
    assert sum(len(row.typed.decisions) for row in dev) == 960
    assert {row.domain_id for row in dev} == {"K"}

    strata = Counter(
        (row.diagnosis_k, row.severity, row.confidence)
        for row in dev
    )
    assert len(strata) == 4 * 4 * 3
    assert set(strata.values()) == {4}

    train_ids = {row.typed.case_id for row in train}
    dev_ids = {row.typed.case_id for row in dev}
    assert train_ids.isdisjoint(dev_ids)

    train_gold = {_gold_signature(row) for row in train}
    dev_gold = {_gold_signature(row) for row in dev}
    assert train_gold.isdisjoint(dev_gold)


def test_w6e_case_identity_is_fresh_and_workflows_match_domain():
    rows = generate_w6e_multi_train() + generate_w6e_dev()
    ids = [row.typed.case_id for row in rows]
    assert len(ids) == len(set(ids))
    assert all(case_id.startswith("w6e-") for case_id in ids)
    assert all(not case_id.startswith("w6d-") for case_id in ids)
    for row in rows:
        assert row.typed.workflow == DOMAINS[row.domain_id].workflow


def test_w6e_domain_values_templates_and_roles_are_pairwise_disjoint():
    values = domain_value_sets()
    templates = domain_template_sets()
    roles = domain_role_sets()

    assert set(values) == set("GHIJKLM")
    for left in values:
        assert len(values[left]) == 96
        assert len(templates[left]) >= 3
        assert len(roles[left]) == 4
        for right in values:
            if left >= right:
                continue
            assert values[left].isdisjoint(values[right])
            assert templates[left].isdisjoint(templates[right])
            assert roles[left].isdisjoint(roles[right])

    assert len(all_w6e_values()) == 7 * 96


def test_w6e_values_do_not_overlap_prior_w5_w6_authorities():
    current = all_w6e_values()
    prior = (
        _prior_w5_values()
        | set(all_w6b_values())
        | set(all_w6c_values())
        | set(all_w6d_values())
    )
    overlap = current & prior
    assert not overlap, overlap


def test_reserved_confirm_domains_are_static_and_never_materialized_in_unit_suite():
    # Static reservation checks only. Never call allow_confirm=True here.
    assert DOMAINS["L"].workflow == "w6e-mining-heldout-confirm-l"
    assert DOMAINS["M"].workflow == "w6e-datacenter-heldout-confirm-m"
    assert set(DOMAINS["L"].templates).isdisjoint(DOMAINS["M"].templates)
    assert domain_value_sets()["L"].isdisjoint(domain_value_sets()["M"])


def test_only_post_freeze_confirm_script_may_eventually_request_confirm_capability():
    tests_root = Path(__file__).resolve().parent
    source = (tests_root / "test_typed_joint_replication_authority.py").read_text()
    sentinel = "allow_" + "confirm=True"
    assert sentinel not in source


def test_confirm_capability_exists_only_in_post_freeze_evaluator():
    root = Path(__file__).resolve().parents[1]
    sentinel = "allow_" + "confirm=True"
    hits = []
    patterns = (
        "src/nmd/*joint_replication*.py",
        "scripts/r8_w6e_*.py",
        "tests/test_typed_joint_replication*.py",
        "research/R8-W6E-HANDOFF.md",
        ".github/workflows/r8-w6e-*.yml",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            text = path.read_text(encoding="utf-8")
            count = text.count(sentinel)
            if count:
                hits.append((path.relative_to(root).as_posix(), count))
    assert hits == [("scripts/r8_w6e_confirm.py", 2)]
