import torch

from nmd.hira import HIRACore
from nmd.relation_cache import RelationCache
from nmd.retention import (
    RetentionConfig,
    cached_teacher_probabilities,
    teacher_kl,
    train_retention_candidate,
)


def make_cache(n=48, d=256, k=3, seed=7):
    torch.manual_seed(seed)
    states = torch.randn(n, 3, d)
    mask = torch.ones(n, 3, dtype=torch.bool)
    q = torch.randn(n, d)
    options = torch.randn(k, d)
    labels = torch.arange(n) % k
    return RelationCache(states, mask, q, options, labels, {"fixture": seed})


def shared_caches():
    base = make_cache()
    return (
        base,
        RelationCache(
            base.state_segments.clone(),
            base.state_mask.clone(),
            base.question_embeddings.clone(),
            base.option_embeddings.clone(),
            base.labels.clone(),
            {"fixture": "replay"},
        ),
        RelationCache(
            base.state_segments.clone(),
            base.state_mask.clone(),
            base.question_embeddings.clone(),
            base.option_embeddings.clone(),
            base.labels.clone(),
            {"fixture": "matched"},
        ),
        RelationCache(
            base.state_segments.clone(),
            base.state_mask.clone(),
            base.question_embeddings.clone(),
            base.option_embeddings.clone(),
            base.labels.clone(),
            {"fixture": "hard"},
        ),
    )


def test_teacher_kl_zero_for_same_distribution():
    logits = torch.tensor([[2.0, 1.0, -1.0]])
    probs = torch.softmax(logits, -1)
    assert abs(float(teacher_kl(logits, probs))) < 1e-6


def test_retention_candidate_runs_and_preserves_parameter_ledger():
    hard, replay, matched, hard_val = shared_caches()
    torch.manual_seed(3)
    teacher = HIRACore(dropout=0.0)
    teacher_probs = cached_teacher_probabilities(teacher, replay, batch_size=16)

    student = HIRACore(dropout=0.0)
    student.load_state_dict(teacher.state_dict())
    before = student.q_proj.weight.detach().clone()

    history, state, result = train_retention_candidate(
        student,
        hard,
        replay,
        teacher_probs,
        matched,
        hard_val,
        config=RetentionConfig(
            epochs=1,
            batch_size=16,
            lr=1e-4,
            teacher_kl_weight=0.5,
            matched_accuracy_floor=0.0,
        ),
    )
    assert len(history) == 1
    assert result["selected_eligible"] is True
    assert not torch.equal(before, student.q_proj.weight)
    assert sum(p.numel() for p in student.parameters()) == 422_159
    assert set(["matched", "hard"]) <= set(result["selected"])
