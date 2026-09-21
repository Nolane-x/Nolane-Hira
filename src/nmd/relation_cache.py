from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
import torch.nn.functional as F

from .hira import HIRACore


@dataclass
class RelationCache:
    state_segments: Tensor
    state_mask: Tensor
    question_embeddings: Tensor
    option_embeddings: Tensor
    labels: Tensor
    metadata: dict[str, str | int | float]

    def validate(self) -> None:
        n = self.labels.shape[0]
        if self.state_segments.ndim != 3:
            raise ValueError("state_segments must be [N,S,D]")
        if self.state_mask.shape != self.state_segments.shape[:2]:
            raise ValueError("state_mask must be [N,S]")
        if self.question_embeddings.shape != (n, self.state_segments.shape[-1]):
            raise ValueError("question_embeddings must be [N,D]")
        if self.option_embeddings.ndim != 2:
            raise ValueError("option_embeddings must be [K,D]")
        if self.option_embeddings.shape[-1] != self.state_segments.shape[-1]:
            raise ValueError("option/state hidden sizes differ")
        if self.labels.ndim != 1 or self.labels.shape[0] != n:
            raise ValueError("labels must be [N]")
        if n == 0:
            raise ValueError("cache is empty")
        if self.labels.min().item() < 0 or self.labels.max().item() >= self.option_embeddings.shape[0]:
            raise ValueError("label outside option range")

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_segments": self.state_segments,
                "state_mask": self.state_mask,
                "question_embeddings": self.question_embeddings,
                "option_embeddings": self.option_embeddings,
                "labels": self.labels,
                "metadata": self.metadata,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RelationCache":
        raw = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(**raw)
        obj.validate()
        return obj

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def minibatches(n: int, batch_size: int, *, seed: int, epoch: int, shuffle: bool) -> Iterator[Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shuffle:
        g = torch.Generator().manual_seed(int(seed) + int(epoch))
        order = torch.randperm(n, generator=g)
    else:
        order = torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def forward_cached(hira: HIRACore, cache: RelationCache, indices: Tensor):
    states = cache.state_segments[indices].float()
    state_mask = cache.state_mask[indices].bool()
    questions = cache.question_embeddings[indices].float()
    options = cache.option_embeddings.float().unsqueeze(0).expand(indices.shape[0], -1, -1)
    qtype = torch.zeros(indices.shape[0], dtype=torch.long)
    return hira(
        questions,
        states,
        options,
        qtype,
        segment_mask=state_mask,
        forced_budget=cache.option_embeddings.shape[0],
    )


def multiclass_brier(probs: Tensor, labels: Tensor) -> Tensor:
    onehot = F.one_hot(labels, num_classes=probs.shape[-1]).to(probs.dtype)
    return ((probs - onehot) ** 2).sum(-1).mean()


def ece15(probs: Tensor, labels: Tensor) -> float:
    conf, pred = probs.max(-1)
    correct = pred.eq(labels).float()
    total = probs.new_tensor(0.0)
    edges = torch.linspace(0, 1, 16, device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total = total + sel.float().mean() * (conf[sel].mean() - correct[sel].mean()).abs()
    return float(total.cpu())


@torch.no_grad()
def predict_cached(hira: HIRACore, cache: RelationCache, *, batch_size: int = 128) -> Tensor:
    cache.validate()
    hira.eval()
    probs_all = []
    for idx in minibatches(len(cache), batch_size, seed=0, epoch=0, shuffle=False):
        probs_all.append(forward_cached(hira, cache, idx).probabilities.cpu())
    return torch.cat(probs_all)


@torch.no_grad()
def evaluate_cached(hira: HIRACore, cache: RelationCache, *, batch_size: int = 128) -> dict[str, float]:
    cache.validate()
    hira.eval()
    probs_all = []
    labels_all = []
    losses = []
    for idx in minibatches(len(cache), batch_size, seed=0, epoch=0, shuffle=False):
        out = forward_cached(hira, cache, idx)
        labels = cache.labels[idx].long()
        probs_all.append(out.probabilities.cpu())
        labels_all.append(labels.cpu())
        losses.append(float(F.cross_entropy(out.logits, labels).cpu()) * len(idx))
    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all)
    accuracy = float(probs.argmax(-1).eq(labels).float().mean())
    return {
        "accuracy": accuracy,
        "brier": float(multiclass_brier(probs, labels)),
        "ece": ece15(probs, labels),
        "nll": sum(losses) / len(cache),
    }


def train_cached(
    hira: HIRACore,
    train: RelationCache,
    validation: RelationCache,
    *,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 2e-3,
    seed: int = 13,
    brier_weight: float = 0.1,
) -> tuple[list[dict[str, float]], dict[str, Tensor]]:
    train.validate()
    validation.validate()
    if not torch.equal(train.option_embeddings, validation.option_embeddings):
        raise ValueError("train/validation option embeddings differ")
    optimizer = torch.optim.AdamW(hira.parameters(), lr=lr)
    history = []
    best = None
    best_acc = -1.0

    for epoch in range(int(epochs)):
        hira.train()
        loss_sum = 0.0
        seen = 0
        for idx in minibatches(len(train), batch_size, seed=seed, epoch=epoch, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            out = forward_cached(hira, train, idx)
            labels = train.labels[idx].long()
            ce = F.cross_entropy(out.logits, labels)
            brier = multiclass_brier(out.probabilities, labels)
            loss = ce + float(brier_weight) * brier
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
            seen += len(idx)

        metrics = evaluate_cached(hira, validation, batch_size=batch_size)
        metrics["epoch"] = float(epoch + 1)
        metrics["train_loss"] = loss_sum / max(1, seen)
        history.append(metrics)
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best = {k: v.detach().cpu().clone() for k, v in hira.state_dict().items()}

    if best is None:
        raise RuntimeError("training produced no checkpoint")
    hira.load_state_dict(best)
    return history, best


@torch.no_grad()
def evaluate_option_permutation(
    hira: HIRACore,
    cache: RelationCache,
    permutation: Tensor,
    *,
    batch_size: int = 128,
) -> dict[str, float]:
    """Verify that option order is a routing permutation, not semantic signal."""
    cache.validate()
    k = cache.option_embeddings.shape[0]
    permutation = permutation.long().cpu()
    if permutation.shape != (k,) or sorted(permutation.tolist()) != list(range(k)):
        raise ValueError("permutation must contain each option index exactly once")
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(k)

    base_probs = predict_cached(hira, cache, batch_size=batch_size)
    permuted = RelationCache(
        state_segments=cache.state_segments,
        state_mask=cache.state_mask,
        question_embeddings=cache.question_embeddings,
        option_embeddings=cache.option_embeddings[permutation],
        labels=inverse[cache.labels],
        metadata={**cache.metadata, "option_permutation": ",".join(map(str, permutation.tolist()))},
    )
    perm_probs = predict_cached(hira, permuted, batch_size=batch_size)
    restored_probs = perm_probs[:, inverse]
    base_pred = base_probs.argmax(-1)
    restored_pred = restored_probs.argmax(-1)
    perm_accuracy = float(
        perm_probs.argmax(-1).eq(permuted.labels).float().mean()
    )
    base_accuracy = float(base_pred.eq(cache.labels).float().mean())
    return {
        "base_accuracy": base_accuracy,
        "permuted_accuracy": perm_accuracy,
        "accuracy_delta": perm_accuracy - base_accuracy,
        "prediction_flip_rate": float(restored_pred.ne(base_pred).float().mean()),
        "max_probability_equivariance_error": float((restored_probs - base_probs).abs().max()),
        "mean_probability_equivariance_error": float((restored_probs - base_probs).abs().mean()),
    }
