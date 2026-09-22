import torch

from nmd.hira import HIRACore
from nmd.relation_cache import RelationCache
from nmd.retention import cached_teacher_probabilities
from nmd.structural import (
    StructuralRetentionConfig,
    balanced_structural_non_entailment_indices,
    is_contiguous_subsequence,
    is_multiset_subset,
    is_ordered_subsequence,
    structural_flags,
    train_structural_candidate,
)


def test_structural_predicates():
    p = ("the", "doctor", "saw", "the", "lawyer")
    assert is_contiguous_subsequence(p, ("the", "lawyer"))
    assert is_ordered_subsequence(p, ("doctor", "the", "lawyer"))
    assert is_multiset_subset(p, ("the", "the", "lawyer"))
    f = structural_flags("The doctor saw the lawyer", "doctor lawyer")
    assert f.ordered and f.multiset_subset and not f.contiguous


def test_balanced_structural_mining_is_deterministic():
    premises = [
        "a b c", "a b c", "x y z", "x y z", "other words", "other words"
    ]
    hypotheses = ["a b", "a c", "x y", "x z", "nothing", "nothing"]
    labels = [1, 1, 2, 2, 1, 2]
    a, stats_a = balanced_structural_non_entailment_indices(
        premises, hypotheses, labels, max_per_label=2
    )
    b, stats_b = balanced_structural_non_entailment_indices(
        premises, hypotheses, labels, max_per_label=2
    )
    assert a == b
    assert stats_a == stats_b
    assert len(a) == 4


def make_cache(n=36, d=256, k=3, labels=None):
    torch.manual_seed(29)
    states = torch.randn(n, 3, d)
    mask = torch.ones(n, 3, dtype=torch.bool)
    q = torch.randn(n, d)
    options = torch.randn(k, d)
    if labels is None:
        labels = torch.arange(n) % k
    return RelationCache(states, mask, q, options, labels.clone(), {"fixture": "1"})


def test_structural_retention_trainer_runs_and_preserves_contract():
    replay = make_cache()
    structural_labels = torch.tensor(([1, 2] * 18))
    structural = make_cache(labels=structural_labels)
    structural.option_embeddings = replay.option_embeddings.clone()
    matched = make_cache()
    matched.option_embeddings = replay.option_embeddings.clone()
    validation = make_cache(labels=structural_labels)
    validation.option_embeddings = replay.option_embeddings.clone()

    teacher = HIRACore(dropout=0.0)
    teacher_probs = cached_teacher_probabilities(teacher, replay, batch_size=12)
    student = HIRACore(dropout=0.0)
    student.load_state_dict(teacher.state_dict())

    history, state, result = train_structural_candidate(
        student,
        structural,
        replay,
        teacher_probs,
        matched,
        validation,
        config=StructuralRetentionConfig(
            epochs=1,
            batch_size=12,
            lr=1e-4,
            hard_replay_ratio=2,
            teacher_kl_weight=0.5,
            matched_accuracy_floor=0.0,
        ),
    )
    assert len(history) == 1
    assert state
    assert result["selected_eligible"]
    assert 0 <= result["selected"]["structural"]["non_entailment_accuracy"] <= 1
