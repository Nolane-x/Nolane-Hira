import math

import torch

from nmd.contracts import LogicalOption
from nmd.hira import HIRACore
from nmd.losses import LossWeights
from nmd.typed_decisions import TypedDecision, TypedDecisionCase
from nmd.typed_feature_cache import (
    cache_role_cases,
    dev_selection_key,
    evaluate_w3_cached_cases,
    load_w3_feature_cache,
    loss_w3_cached_case,
    save_w3_feature_cache,
    split_w3_train_dev,
    validate_w3_feature_cache,
)


def shell_case(case_id: str, workflow: str) -> TypedDecisionCase:
    return TypedDecisionCase(
        case_id=case_id,
        workflow=workflow,
        state_text="{}",
        decisions=(),
    )


def test_w3_split_is_deterministic_240_60_per_workflow():
    workflows = (
        "agent_trace_observability",
        "customer_service",
        "invoice_processing",
        "security_incidents",
    )
    cases = [
        shell_case(f"{workflow}-{i:03d}", workflow)
        for workflow in workflows
        for i in range(300)
    ]

    train_a, dev_a = split_w3_train_dev(cases)
    train_b, dev_b = split_w3_train_dev(list(reversed(cases)))

    assert len(train_a) == 960
    assert len(dev_a) == 240
    assert [c.case_id for c in train_a] == [c.case_id for c in train_b]
    assert [c.case_id for c in dev_a] == [c.case_id for c in dev_b]
    assert not ({c.case_id for c in train_a} & {c.case_id for c in dev_a})

    for workflow in workflows:
        assert sum(c.workflow == workflow for c in train_a) == 240
        assert sum(c.workflow == workflow for c in dev_a) == 60


def cached_decision(
    primitive: str,
    *,
    k: int,
    gold_index: int,
):
    g = torch.Generator().manual_seed(k * 17 + gold_index)
    probs = torch.full((k,), 0.05)
    probs[gold_index] = 1.0 - 0.05 * (k - 1)
    if primitive == "score":
        support = torch.arange(k, dtype=torch.float32) * 2.0 + 1.0
        gold_score = float((probs * support).sum())
    else:
        support = torch.empty(0)
        gold_score = None
    return {
        "question_id": f"{primitive}-{k}",
        "primitive": primitive,
        "question_embedding": torch.randn(256, generator=g).half(),
        "option_embeddings": torch.randn(k, 256, generator=g).half(),
        "schema_hash": f"schema-{primitive}-{k}",
        "gold_index": gold_index,
        "gold_probabilities": probs,
        "gold_score": gold_score,
        "score_support": support,
    }


def cached_case(case_id: str, role: str):
    g = torch.Generator().manual_seed(
        sum(ord(ch) for ch in case_id)
    )
    return {
        "case_id": case_id,
        "workflow": "fixture",
        "role": role,
        "state_segments": torch.randn(2, 256, generator=g).half(),
        "decisions": [
            cached_decision("choice", k=3, gold_index=1),
            cached_decision("choice", k=4, gold_index=2),
            cached_decision("noul", k=2, gold_index=1),
            cached_decision("score", k=4, gold_index=1),
            cached_decision("score", k=4, gold_index=2),
        ],
    }


def tiny_cache():
    return {
        "metadata": {
            "schema_version": "r8-w3-typed-feature-cache-v1",
            "case_count": 3,
            "train_case_count": 2,
            "dev_case_count": 1,
            "decision_count": 15,
            "state_encode_calls": 3,
            "state_encode_calls_per_case": 1.0,
            "train_case_id_sha256": "train",
            "dev_case_id_sha256": "dev",
            "segment_tokens": 32,
        },
        "cases": [
            cached_case("train-a", "train"),
            cached_case("train-b", "train"),
            cached_case("dev-a", "dev"),
        ],
    }


def test_cached_w3_loss_and_evaluator_are_full_k_and_finite():
    cache = tiny_cache()
    validate_w3_feature_cache(cache)
    train = cache_role_cases(cache, "train")
    dev = cache_role_cases(cache, "dev")

    torch.manual_seed(13)
    model = HIRACore(d_model=256, dropout=0.0)
    loss = loss_w3_cached_case(
        model,
        train[0],
        weights=LossWeights(
            hard_ce=0.5,
            teacher_kl=1.0,
            brier=0.05,
            soft_brier=1.0,
            ordinal_mae=0.2,
        ),
    )
    loss.backward()

    assert math.isfinite(float(loss))
    assert model.cross_score[0].weight.grad is not None

    metrics = evaluate_w3_cached_cases(model, dev)
    assert metrics["case_count"] == 1
    assert metrics["decision_count"] == 5
    assert metrics["primitive_counts"] == {
        "choice": 2,
        "score": 2,
        "noul": 1,
    }
    assert metrics["cached_feature_evaluation"] is True
    assert metrics["state_encode_calls_during_eval"] == 0
    assert metrics["source_state_encode_calls_per_case"] == 1.0
    assert metrics["decisions_per_source_state_encode"] == 5.0
    assert metrics["probability_mass_max_error"] < 1e-6
    for key in (
        "accuracy",
        "soft_accuracy",
        "hard_brier",
        "soft_brier",
        "nll",
        "kl_gold_to_prediction",
        "ece",
        "score_mae",
    ):
        assert math.isfinite(float(metrics[key]))


def test_dev_selection_key_matches_frozen_lexicographic_order():
    base = {
        "accuracy": 0.7,
        "hard_brier": 0.2,
        "soft_accuracy": 0.5,
        "score_mae": 0.4,
        "ece": 0.1,
    }
    better_accuracy = {**base, "accuracy": 0.71}
    better_brier = {**base, "hard_brier": 0.19}
    better_soft = {**base, "soft_accuracy": 0.51}

    assert dev_selection_key(
        better_accuracy,
        epoch=8,
    ) < dev_selection_key(base, epoch=1)
    assert dev_selection_key(
        better_brier,
        epoch=8,
    ) < dev_selection_key(base, epoch=1)
    assert dev_selection_key(
        better_soft,
        epoch=8,
    ) < dev_selection_key(base, epoch=1)
    assert dev_selection_key(
        base,
        epoch=1,
    ) < dev_selection_key(base, epoch=2)


def test_w3_cache_validator_rejects_bad_probability_mass():
    cache = tiny_cache()
    cache["cases"][0]["decisions"][0]["gold_probabilities"] = torch.tensor(
        [0.9, 0.9, 0.1]
    )
    try:
        validate_w3_feature_cache(cache)
    except ValueError as exc:
        assert "probability mass" in str(exc)
    else:
        raise AssertionError("invalid probability mass was accepted")



def test_w3_feature_cache_roundtrip_is_weights_only_safe(tmp_path):
    cache = tiny_cache()
    path = save_w3_feature_cache(cache, tmp_path / "cache.pt")
    restored = load_w3_feature_cache(path)

    assert restored["metadata"] == cache["metadata"]
    assert len(restored["cases"]) == len(cache["cases"])
    assert torch.equal(
        restored["cases"][0]["state_segments"],
        cache["cases"][0]["state_segments"],
    )
    assert torch.equal(
        restored["cases"][0]["decisions"][0]["gold_probabilities"],
        cache["cases"][0]["decisions"][0]["gold_probabilities"],
    )
