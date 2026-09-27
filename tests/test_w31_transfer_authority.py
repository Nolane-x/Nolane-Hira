from nmd.compositional_projection_authority import all_w28_text_atoms
from nmd.hira_v0_authority import all_w29_text_atoms
from nmd.w30_transfer_authority import all_w30_text_atoms
from nmd.w31_transfer_authority import (
    DOMAIN_STYLE,
    FACTOR_IDS,
    PARTITION_DOMAINS,
    all_w31_query_texts,
    all_w31_schema_texts,
    all_w31_text_atoms,
    compose_severity,
    factor_options,
    generate_w31_partition,
)


def test_w31_partitions_are_balanced_and_disjoint():
    seen = set()
    partition_styles = {}
    for partition, domains in PARTITION_DOMAINS.items():
        rows = generate_w31_partition(partition)
        assert len(rows) == 96 * len(domains)
        assert {row.domain_id for row in rows} == set(domains)
        assert all(row.partition == partition for row in rows)

        partition_styles[partition] = {DOMAIN_STYLE[d] for d in domains}
        for domain in domains:
            subset = [row for row in rows if row.domain_id == domain]
            assert len(subset) == 96
            assert [sum(row.severity == s for row in subset) for s in range(4)] == [24] * 4
            assert all(compose_severity(row.factor_vector) == row.severity for row in subset)

        ids = {row.case_id for row in rows}
        assert not (seen & ids)
        seen |= ids

    names = tuple(partition_styles)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert partition_styles[left].isdisjoint(partition_styles[right])


def test_w31_exact_text_is_fresh_against_prior_authorities():
    assert not (all_w31_query_texts() & all_w31_schema_texts())
    assert not (all_w31_text_atoms() & all_w30_text_atoms())
    assert not (all_w31_text_atoms() & all_w29_text_atoms())
    assert not (all_w31_text_atoms() & all_w28_text_atoms())


def test_w31_qualification_is_model_evidence_separate():
    qualification = generate_w31_partition("qualification")
    train = generate_w31_partition("train")
    dev = generate_w31_partition("dev")
    confirm = generate_w31_partition("confirm")

    qualification_text = {row.state_text for row in qualification}
    train_text = {row.state_text for row in train}
    dev_text = {row.state_text for row in dev}
    confirm_text = {row.state_text for row in confirm}

    assert qualification_text.isdisjoint(train_text)
    assert qualification_text.isdisjoint(dev_text)
    assert qualification_text.isdisjoint(confirm_text)
    assert train_text.isdisjoint(dev_text)
    assert train_text.isdisjoint(confirm_text)
    assert dev_text.isdisjoint(confirm_text)


def test_w31_factor_options_keep_dynamic_semantic_identity():
    for domains in PARTITION_DOMAINS.values():
        for domain in domains:
            for factor in FACTOR_IDS:
                options = factor_options(domain, factor)
                assert len(options) == 2
                assert tuple(float(option.value) for option in options) == (0.0, 1.0)
                assert all(len(option.aliases) == 1 for option in options)
                assert all(len(option.exemplars) == 1 for option in options)
                assert options[0].option_id.endswith("-0")
                assert options[1].option_id.endswith("-1")
