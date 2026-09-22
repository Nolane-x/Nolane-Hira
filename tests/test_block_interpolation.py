import torch

from nmd.block_interpolation import (
    ATOMIC_GROUPS,
    atomic_group_for_key,
    balanced_ranked_structural_non_entailment_indices,
    interpolate_selected_groups,
    partition_state_keys,
    unselected_keys_are_bit_identical,
)
from nmd.hira import HIRACore, count_parameters


def test_every_hira_state_key_maps_to_exactly_one_atomic_group():
    model = HIRACore(dropout=0.0)
    parts = partition_state_keys(model.state_dict())
    assert set(parts) == set(ATOMIC_GROUPS)
    flattened = [key for keys in parts.values() for key in keys]
    assert sorted(flattened) == sorted(model.state_dict())
    assert len(flattened) == len(set(flattened))
    for key in model.state_dict():
        assert atomic_group_for_key(key) in ATOMIC_GROUPS


def test_relation_interpolation_changes_only_relation_keys():
    torch.manual_seed(11)
    base_model = HIRACore(dropout=0.0)
    torch.manual_seed(17)
    direction_model = HIRACore(dropout=0.0)
    base = base_model.state_dict()
    direction = direction_model.state_dict()

    fused = interpolate_selected_groups(
        base, direction, groups=("relation",), alpha=0.5
    )
    assert unselected_keys_are_bit_identical(
        base, fused, groups=("relation",)
    )

    relation_changed = 0
    for key in base:
        group = atomic_group_for_key(key)
        if group == "relation" and torch.is_floating_point(base[key]):
            assert torch.allclose(
                fused[key], (base[key] + direction[key]) / 2
            )
            if not torch.equal(base[key], direction[key]):
                relation_changed += 1
        else:
            assert torch.equal(fused[key], base[key])
    assert relation_changed > 0

    model = HIRACore(dropout=0.0)
    model.load_state_dict(fused, strict=True)
    assert count_parameters(model) == 422_159


def test_budget_control_is_output_negative_control_in_nonadaptive_mode():
    torch.manual_seed(23)
    base_model = HIRACore(dropout=0.0)
    torch.manual_seed(29)
    direction_model = HIRACore(dropout=0.0)

    fused_state = interpolate_selected_groups(
        base_model.state_dict(),
        direction_model.state_dict(),
        groups=("budget_control",),
        alpha=1.0,
    )
    fused_model = HIRACore(dropout=0.0)
    fused_model.load_state_dict(fused_state, strict=True)

    torch.manual_seed(31)
    batch, segments, k, d = 4, 5, 3, 256
    question = torch.randn(batch, d)
    state_segments = torch.randn(batch, segments, d)
    options = torch.randn(batch, k, d)
    qtype = torch.zeros(batch, dtype=torch.long)
    segment_mask = torch.ones(batch, segments, dtype=torch.bool)

    base_model.eval()
    fused_model.eval()
    with torch.no_grad():
        a = base_model(
            question,
            state_segments,
            options,
            qtype,
            segment_mask=segment_mask,
            adaptive_budget=False,
        )
        b = fused_model(
            question,
            state_segments,
            options,
            qtype,
            segment_mask=segment_mask,
            adaptive_budget=False,
        )
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.probabilities, b.probabilities)
    assert torch.equal(a.candidate_budget, b.candidate_budget)


def test_ranked_structural_selection_is_balanced_and_deterministic():
    premises = [
        "a b c d", "a x b c", "d c b a", "a q c",
        "u v w x", "u y v w", "x w v u", "u q w",
        "noise words here", "other tokens here",
    ]
    hypotheses = [
        "a b c", "a b c", "a b c", "a b c",
        "u v w", "u v w", "u v w", "u v w",
        "a b c", "u v w",
    ]
    labels = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2]

    a, stats_a = balanced_ranked_structural_non_entailment_indices(
        premises, hypotheses, labels, per_label=3, min_hypothesis_tokens=3
    )
    b, stats_b = balanced_ranked_structural_non_entailment_indices(
        premises, hypotheses, labels, per_label=3, min_hypothesis_tokens=3
    )
    assert a == b
    assert stats_a == stats_b
    assert stats_a["selected_total"] == 6
    assert stats_a["selected_per_label"] == 3
    assert stats_a["eligible_by_label"] == {"1": 5, "2": 5}


def test_unknown_group_is_rejected():
    model = HIRACore(dropout=0.0)
    state = model.state_dict()
    try:
        interpolate_selected_groups(
            state, state, groups=("not-a-group",), alpha=0.1
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown group must fail")
