import torch

from nmd.hira import HIRACore
from nmd.interpolation import (
    balanced_near_structural_non_entailment_indices,
    interpolate_state_dicts,
    multiset_recall,
    ordered_lcs_recall,
)
from nmd.hira import count_parameters


def test_near_structural_recalls_are_distinct_and_bounded():
    p = ("a", "x", "b", "y", "c")
    h = ("a", "b", "c")
    assert multiset_recall(p, h) == 1.0
    assert ordered_lcs_recall(p, h) == 1.0

    p2 = ("c", "b", "a")
    assert multiset_recall(p2, h) == 1.0
    assert ordered_lcs_recall(p2, h) == 1 / 3


def test_balanced_near_structural_selection_is_deterministic():
    premises = [
        "a b c d",
        "a x b c",
        "d c b a",
        "a b c z",
        "q r s t",
        "q x r s",
        "t s r q",
        "q r s z",
        "noise only here",
        "different tokens entirely",
    ]
    hypotheses = [
        "a b c",
        "a b c",
        "a b c",
        "a b c",
        "q r s",
        "q r s",
        "q r s",
        "q r s",
        "a b c",
        "q r s",
    ]
    labels = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2]
    a, stats_a = balanced_near_structural_non_entailment_indices(
        premises,
        hypotheses,
        labels,
        threshold=0.80,
        min_hypothesis_tokens=3,
        max_per_label=3,
    )
    b, stats_b = balanced_near_structural_non_entailment_indices(
        premises,
        hypotheses,
        labels,
        threshold=0.80,
        min_hypothesis_tokens=3,
        max_per_label=3,
    )
    assert a == b
    assert stats_a == stats_b
    assert stats_a["available_by_label"] == {"1": 4, "2": 4}
    assert stats_a["selected_per_label"] == 3
    assert len(a) == 6


def test_interpolation_endpoints_and_midpoint():
    torch.manual_seed(3)
    a_model = HIRACore(dropout=0.0)
    torch.manual_seed(5)
    b_model = HIRACore(dropout=0.0)
    a = a_model.state_dict()
    b = b_model.state_dict()

    zero = interpolate_state_dicts(a, b, 0.0)
    one = interpolate_state_dicts(a, b, 1.0)
    half = interpolate_state_dicts(a, b, 0.5)

    for key in a:
        assert torch.equal(zero[key], a[key])
        assert torch.equal(one[key], b[key])
        if torch.is_floating_point(a[key]):
            assert torch.allclose(half[key], (a[key] + b[key]) / 2)

    fused = HIRACore(dropout=0.0)
    fused.load_state_dict(half, strict=True)
    assert count_parameters(fused) == 422_159


def test_interpolation_rejects_invalid_alpha():
    model = HIRACore(dropout=0.0)
    state = model.state_dict()
    for alpha in (-0.01, 1.01):
        try:
            interpolate_state_dicts(state, state, alpha)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid alpha must fail")
