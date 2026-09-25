from nmd.freeform_attribution_authority import (
    DOMAINS,
    all_w7b_values,
    diagnosis_distance_histogram,
    domain_role_sets,
    domain_template_sets,
    domain_value_sets,
    generate_w7b_confirm,
    generate_w7b_dev,
    generate_w7b_train,
)


def test_w7b_train_dev_shape_and_balance():
    train = generate_w7b_train()
    dev = generate_w7b_dev()
    assert len(train) == 384
    assert len(dev) == 192
    assert {row.domain_id for row in train} == {"AN", "AO", "AP", "AQ"}
    assert {row.domain_id for row in dev} == {"AR"}
    for domain in ("AN", "AO", "AP", "AQ"):
        assert sum(row.domain_id == domain for row in train) == 96
    assert all(row.split == "train" for row in train)
    assert all(row.split == "dev-ar" for row in dev)


def test_w7b_k64_has_frozen_semantic_distance_pressure():
    cases = [row for row in generate_w7b_dev() if row.diagnosis_k == 64]
    assert len(cases) == 48
    for row in cases:
        hist = diagnosis_distance_histogram(row)
        assert hist == {0: 1, 1: 12, 2: 20, 3: 15, 4: 16}


def test_w7b_domains_are_pairwise_disjoint():
    values = domain_value_sets()
    templates = domain_template_sets()
    roles = domain_role_sets()
    keys = sorted(DOMAINS)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            assert values[left].isdisjoint(values[right])
            assert templates[left].isdisjoint(templates[right])
            assert roles[left].isdisjoint(roles[right])
    assert len(all_w7b_values()) == sum(len(v) for v in values.values())


def test_w7b_confirm_is_sealed():
    for domain in ("AS", "AT"):
        try:
            generate_w7b_confirm(domain)
        except RuntimeError:
            pass
        else:
            raise AssertionError("W7b CONFIRM must be sealed")
