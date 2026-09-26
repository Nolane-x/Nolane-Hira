from nmd.compositional_projection_authority import all_w28_text_atoms
from nmd.hira_v0_authority import all_w29_text_atoms
from nmd.semantic_transfer_authority import (
    DOMAIN_STYLE,
    FACTOR_IDS,
    PARTITION_DOMAINS,
    SCHEMA_VIEWS,
    all_w30_query_texts,
    all_w30_schema_texts,
    all_w30_text_atoms,
    compose_severity,
    factor_options,
    generate_w30_partition,
)


def test_w30_partitions_are_balanced_and_style_separated():
    all_rows = []
    for partition, domains in PARTITION_DOMAINS.items():
        rows = generate_w30_partition(partition)
        assert len(rows) == 96 * len(domains)
        assert {row.domain_id for row in rows} == set(domains)
        assert all(row.partition == partition for row in rows)
        all_rows.extend(rows)
        for domain in domains:
            subset = [row for row in rows if row.domain_id == domain]
            assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
            assert all(compose_severity(row.factor_vector) == row.severity for row in subset)

    train_styles = {DOMAIN_STYLE[d] for d in PARTITION_DOMAINS["train"]}
    dev_styles = {DOMAIN_STYLE[d] for d in PARTITION_DOMAINS["dev"]}
    confirm_styles = {DOMAIN_STYLE[d] for d in PARTITION_DOMAINS["confirm"]}
    assert train_styles.isdisjoint(dev_styles)
    assert train_styles.isdisjoint(confirm_styles)
    assert dev_styles.isdisjoint(confirm_styles)

    assert len({row.case_id for row in all_rows}) == len(all_rows)


def test_w30_exact_text_is_fresh_against_w28_and_w29():
    assert not (all_w30_query_texts() & all_w30_schema_texts())
    assert not (all_w30_text_atoms() & all_w29_text_atoms())
    assert not (all_w30_text_atoms() & all_w28_text_atoms())


def test_w30_schema_views_are_partition_held_out():
    train = set()
    dev = set()
    confirm = set()
    for domain in PARTITION_DOMAINS["train"]:
        style = DOMAIN_STYLE[domain]
        for factor in FACTOR_IDS:
            for value in (0, 1):
                train.update(SCHEMA_VIEWS[style][factor][value])
    for domain in PARTITION_DOMAINS["dev"]:
        style = DOMAIN_STYLE[domain]
        for factor in FACTOR_IDS:
            for value in (0, 1):
                dev.update(SCHEMA_VIEWS[style][factor][value])
    for domain in PARTITION_DOMAINS["confirm"]:
        style = DOMAIN_STYLE[domain]
        for factor in FACTOR_IDS:
            for value in (0, 1):
                confirm.update(SCHEMA_VIEWS[style][factor][value])

    assert train.isdisjoint(dev)
    assert train.isdisjoint(confirm)
    assert dev.isdisjoint(confirm)


def test_w30_factor_options_keep_semantic_identity_and_values():
    for partition, domains in PARTITION_DOMAINS.items():
        for domain in domains:
            for factor in FACTOR_IDS:
                options = factor_options(domain, factor)
                assert len(options) == 2
                assert tuple(float(o.value) for o in options) == (0.0, 1.0)
                assert all(len(o.aliases) == 1 for o in options)
                assert all(len(o.exemplars) == 1 for o in options)
                assert options[0].option_id.endswith("-0")
                assert options[1].option_id.endswith("-1")
