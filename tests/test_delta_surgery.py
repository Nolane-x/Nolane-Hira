import torch

from nmd.delta_surgery import (
    apply_relation_mask,
    balanced_ranked_structural_window_indices,
    coordinate_scores,
    flatten_relation,
    nested_masks,
    relation_delta,
    relation_layout,
    sparse_identity_invariants,
    top_fraction_mask,
)
from nmd.hira import HIRACore, count_parameters


def test_relation_layout_is_deterministic_and_covers_only_relation():
    model = HIRACore(dropout=0.0)
    state = model.state_dict()
    a = relation_layout(state)
    b = relation_layout(state)
    assert a == b
    assert a[0].start == 0
    assert all(x.end > x.start for x in a)
    assert all(a[i].end == a[i + 1].start for i in range(len(a) - 1))
    flat = flatten_relation(state, layout=a)
    assert flat.numel() == a[-1].end
    assert flat.dtype == torch.float64


def test_sparse_relation_mask_changes_only_selected_coordinates():
    torch.manual_seed(101)
    base_model = HIRACore(dropout=0.0)
    torch.manual_seed(103)
    direction_model = HIRACore(dropout=0.0)
    base = base_model.state_dict()
    direction = direction_model.state_dict()
    layout = relation_layout(base)
    delta = relation_delta(base, direction, layout=layout)

    mask = torch.zeros(delta.numel(), dtype=torch.bool)
    mask[0] = True
    mask[len(mask) // 2] = True
    mask[-1] = True

    candidate = apply_relation_mask(
        base, direction, mask, alpha=0.5, layout=layout
    )
    invariants = sparse_identity_invariants(
        base, candidate, mask, layout=layout
    )
    assert invariants == {
        "non_relation_bit_identical": True,
        "masked_off_relation_bit_identical": True,
    }

    base_flat = flatten_relation(base, layout=layout)
    candidate_flat = flatten_relation(candidate, layout=layout)
    direction_flat = flatten_relation(direction, layout=layout)
    assert torch.equal(candidate_flat[~mask], base_flat[~mask])
    assert torch.allclose(
        candidate_flat[mask],
        base_flat[mask] + 0.5 * (direction_flat[mask] - base_flat[mask]),
    )

    loaded = HIRACore(dropout=0.0)
    loaded.load_state_dict(candidate, strict=True)
    assert count_parameters(loaded) == 422_159


def test_coordinate_scoring_and_masks_are_nested_and_stable():
    structural_gradient = torch.tensor(
        [-2.0, -1.0, 1.0, -4.0, 0.0, -2.0], dtype=torch.float64
    )
    retention_energy = torch.tensor(
        [1.0, 4.0, 1.0, 16.0, 1.0, 1.0], dtype=torch.float64
    )
    delta = torch.tensor(
        [1.0, 1.0, 1.0, 0.5, 1.0, 2.0], dtype=torch.float64
    )
    scores = coordinate_scores(
        structural_gradient, retention_energy, delta
    )
    positive = scores["positive_mask"]
    assert positive.tolist() == [True, True, False, True, False, True]

    fractions = (0.25, 0.5, 0.75, 1.0)
    for key in ("benefit_score", "benefit_over_cost_score"):
        masks = [
            top_fraction_mask(scores[key], positive, fraction)
            for fraction in fractions
        ]
        assert nested_masks(masks)
        assert int(masks[0].sum()) == 1
        assert int(masks[1].sum()) == 2
        assert int(masks[2].sum()) == 3
        assert int(masks[3].sum()) == 4

        again = [
            top_fraction_mask(scores[key], positive, fraction)
            for fraction in fractions
        ]
        assert all(torch.equal(a, b) for a, b in zip(masks, again))


def test_top_fraction_tie_break_prefers_lower_global_index():
    score = torch.tensor([3.0, 3.0, 3.0, 1.0], dtype=torch.float64)
    positive = torch.ones(4, dtype=torch.bool)
    mask = top_fraction_mask(score, positive, 0.5)
    assert mask.tolist() == [True, True, False, False]


def test_ranked_structural_window_is_disjoint_from_top_window():
    premises = [
        "a b c d",
        "a x b c",
        "a b y c",
        "c b a",
        "a z c",
        "u v w x",
        "u y v w",
        "u v y w",
        "w v u",
        "u z w",
    ]
    hypotheses = [
        "a b c",
        "a b c",
        "a b c",
        "a b c",
        "a b c",
        "u v w",
        "u v w",
        "u v w",
        "u v w",
        "u v w",
    ]
    labels = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    top, _ = balanced_ranked_structural_window_indices(
        premises,
        hypotheses,
        labels,
        start_per_label=0,
        count_per_label=2,
        min_hypothesis_tokens=3,
    )
    next_window, stats = balanced_ranked_structural_window_indices(
        premises,
        hypotheses,
        labels,
        start_per_label=2,
        count_per_label=2,
        min_hypothesis_tokens=3,
    )
    assert set(top).isdisjoint(next_window)
    assert len(next_window) == 4
    assert stats["selected_total"] == 4
    assert stats["start_per_label"] == 2
    assert stats["count_per_label"] == 2


def test_apply_relation_mask_rejects_wrong_mask_shape():
    model = HIRACore(dropout=0.0)
    state = model.state_dict()
    layout = relation_layout(state)
    try:
        apply_relation_mask(
            state,
            state,
            torch.zeros(layout[-1].end - 1, dtype=torch.bool),
            alpha=1.0,
            layout=layout,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("wrong mask shape must fail")
