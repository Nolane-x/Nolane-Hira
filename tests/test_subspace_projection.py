import torch

from nmd.delta_surgery import (\n    balanced_ranked_structural_window_indices,\n    relation_delta,\n    relation_layout,\n)
from nmd.hira import HIRACore, count_parameters
from nmd.subspace_projection import (
    apply_relation_update,
    build_retention_subspace,
    non_relation_bit_identical,
    projection_metrics,
    project_onto_retention_subspace,
    rank_for_coverage,
    rotate_delta_away_from_retention,
)


def test_retention_subspace_coverage_and_projection_geometry():
    rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    subspace = build_retention_subspace(rows)
    assert subspace.rank == 2
    assert subspace.orthonormal_error < 1e-10

    rank50, coverage50 = rank_for_coverage(subspace, 0.50)
    rank90, coverage90 = rank_for_coverage(subspace, 0.90)
    assert rank50 == 1
    assert coverage50 >= 0.50
    assert rank90 == 2
    assert coverage90 >= 0.90

    x = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    p = project_onto_retention_subspace(x, subspace, rank=rank50)
    assert torch.allclose(p, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))

    rotated, projected = rotate_delta_away_from_retention(
        x, subspace, rank=rank50, removal_strength=1.0
    )
    assert torch.allclose(projected, p)
    assert torch.allclose(rotated, torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64))


def test_partial_removal_is_exact_convex_at_projection_component():
    rows = torch.eye(2, 3, dtype=torch.float64)
    subspace = build_retention_subspace(rows)
    x = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
    rotated, projected = rotate_delta_away_from_retention(
        x, subspace, rank=2, removal_strength=0.5
    )
    assert torch.allclose(projected, torch.tensor([2.0, 4.0, 0.0], dtype=torch.float64))
    assert torch.allclose(rotated, torch.tensor([1.0, 2.0, 8.0], dtype=torch.float64))


def test_relation_update_preserves_non_relation_state_and_parameter_count():
    torch.manual_seed(201)
    base_model = HIRACore(dropout=0.0)
    torch.manual_seed(203)
    direction_model = HIRACore(dropout=0.0)
    base = base_model.state_dict()
    direction = direction_model.state_dict()
    layout = relation_layout(base)
    delta = relation_delta(base, direction, layout=layout)

    candidate = apply_relation_update(
        base, delta, alpha=0.25, layout=layout
    )
    assert non_relation_bit_identical(base, candidate, layout=layout)

    loaded = HIRACore(dropout=0.0)
    loaded.load_state_dict(candidate, strict=True)
    assert count_parameters(loaded) == 422_159


def test_projection_metrics_report_removed_energy_and_structural_benefit():
    d = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    r = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    g = torch.tensor([-2.0, 1.0, 0.0], dtype=torch.float64)
    metrics = projection_metrics(d, r, g)
    assert abs(metrics["removed_delta_energy_fraction"] - 0.5) < 1e-12
    assert abs(metrics["first_order_structural_predicted_benefit"] - 2.0) < 1e-12
    assert metrics["changed_relation_coordinates"] == 1


def test_invalid_coverage_and_rank_fail_closed():
    subspace = build_retention_subspace(torch.eye(2, 3, dtype=torch.float64))
    for target in (0.0, 1.01):
        try:
            rank_for_coverage(subspace, target)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid coverage must fail")

    try:
        project_onto_retention_subspace(
            torch.ones(3, dtype=torch.float64),
            subspace,
            rank=subspace.rank + 1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid rank must fail")


def test_structural_ranking_exclusions_are_respected_before_selection():
    premises = [
        "a b c d", "a x b c", "a b y c", "a z b c",
        "u v w x", "u y v w", "u v y w", "u z v w",
    ]
    hypotheses = ["a b c"] * 4 + ["u v w"] * 4
    labels = [1, 1, 1, 1, 2, 2, 2, 2]
    selected, stats = balanced_ranked_structural_window_indices(
        premises,
        hypotheses,
        labels,
        start_per_label=0,
        count_per_label=2,
        min_hypothesis_tokens=3,
        exclude_indices=[0, 4],
    )
    assert 0 not in selected
    assert 4 not in selected
    assert len(selected) == 4
    assert stats["excluded_source_indices"] == 2
