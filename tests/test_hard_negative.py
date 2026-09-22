import torch

from nmd.hard_negative import (
    anti_entailment_margin_loss,
    balanced_top_indices,
    lexical_overlap_score,
    top_non_entailment_indices,
    train_hard_negative_repair,
)
from nmd.hira import HIRACore
from nmd.relation_cache import RelationCache


def make_cache(n=36, d=256, k=3):
    torch.manual_seed(21)
    return RelationCache(
        state_segments=torch.randn(n, 3, d),
        state_mask=torch.ones(n, 3, dtype=torch.bool),
        question_embeddings=torch.randn(n, d),
        option_embeddings=torch.randn(k, d),
        labels=torch.arange(n) % k,
        metadata={"fixture": 1},
    )


def test_lexical_overlap_is_hypothesis_multiset_recall():
    s = lexical_overlap_score(
        "the cat is on the mat",
        "the cat is not on the mat",
    )
    assert abs(s - (6 / 7)) < 1e-12


def test_balanced_curriculum_is_deterministic_and_class_balanced():
    premises = [
        "a b c", "a b", "x y", "x y z", "m n", "m n o",
        "a c", "x z", "m o",
    ]
    hypotheses = [
        "a b c", "a q", "x y", "x y z", "m n", "m n o",
        "a c", "x z", "m o",
    ]
    labels = [0, 0, 1, 1, 2, 2, 0, 1, 2]
    a, stats_a = balanced_top_indices(
        premises, hypotheses, labels, per_label=2
    )
    b, stats_b = balanced_top_indices(
        premises, hypotheses, labels, per_label=2
    )
    assert a == b
    assert stats_a == stats_b
    assert len(a) == 6
    selected_labels = [labels[i] for i in a]
    assert selected_labels.count(0) == 2
    assert selected_labels.count(1) == 2
    assert selected_labels.count(2) == 2


def test_hard_validation_contains_only_neutral_and_contradiction():
    premises = ["a b", "a", "x y", "x", "m n", "m"]
    hypotheses = ["a b", "a b", "x y", "x z", "m n", "m q"]
    labels = [0, 0, 1, 1, 2, 2]
    idx, stats = top_non_entailment_indices(
        premises, hypotheses, labels, per_label=1
    )
    assert len(idx) == 2
    assert {labels[i] for i in idx} == {1, 2}
    assert stats["total"] == 2


def test_anti_entailment_margin_ignores_entailment_gold():
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],  # entailment gold: ignored
            [2.0, 1.0, 0.0],  # neutral gold below entailment
            [2.0, 0.0, 1.5],  # contradiction gold slightly below desired margin
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2])
    loss = anti_entailment_margin_loss(logits, labels, margin=1.0)
    # Row 2: 1 - 1 + 2 = 2. Row 3: 1 - 1.5 + 2 = 1.5.
    assert abs(loss.item() - 1.75) < 1e-6
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad[0], torch.zeros(3))


def test_repair_training_can_select_an_eligible_epoch():
    train = make_cache()
    matched = make_cache()
    hard = make_cache()
    hira = HIRACore(dropout=0.0)
    history, state, result = train_hard_negative_repair(
        hira,
        train,
        matched,
        hard,
        epochs=1,
        batch_size=12,
        lr=1e-4,
        matched_accuracy_floor=0.0,
    )
    assert len(history) == 1
    assert state
    assert "baseline" in result and "selected" in result
