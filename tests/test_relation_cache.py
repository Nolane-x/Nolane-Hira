from pathlib import Path

import torch

from nmd.hira import HIRACore
from nmd.relation_cache import RelationCache, evaluate_cached, train_cached


def make_cache(n=48, d=256, k=3):
    torch.manual_seed(5)
    states = torch.randn(n, 3, d)
    mask = torch.ones(n, 3, dtype=torch.bool)
    q = torch.randn(n, d)
    options = torch.randn(k, d)
    labels = torch.arange(n) % k
    return RelationCache(states, mask, q, options, labels, {"fixture": "1"})


def test_cache_roundtrip(tmp_path: Path):
    cache = make_cache()
    p = cache.save(tmp_path / "cache.pt")
    restored = RelationCache.load(p)
    assert torch.equal(cache.labels, restored.labels)
    assert torch.equal(cache.option_embeddings, restored.option_embeddings)


def test_cached_training_changes_hira_and_returns_metrics():
    train = make_cache()
    val = make_cache()
    hira = HIRACore(dropout=0.0)
    before = hira.q_proj.weight.detach().clone()
    history, best = train_cached(hira, train, val, epochs=1, batch_size=16, lr=1e-3)
    assert len(history) == 1
    assert 0 <= history[0]["accuracy"] <= 1
    assert history[0]["brier"] >= 0
    assert not torch.equal(before, hira.q_proj.weight)
    metrics = evaluate_cached(hira, val)
    assert set(["accuracy", "brier", "ece", "nll"]) <= set(metrics)
