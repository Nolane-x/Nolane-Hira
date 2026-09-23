import math

import torch

from nmd.semantic_alignment_probes import (
    BilinearAlignmentProbe,
    PairMLPAlignmentProbe,
    all_w5c_vocab,
    classify_confirm,
    evaluate_probe,
    generate_alignment_authority,
    probe_parameter_count,
    validate_alignment_cache,
)
from nmd import semantic_routing_curriculum as w5a
from nmd import semantic_token_curriculum as w5b


def prior_vocab():
    groups = (
        w5a.TRAIN_MATERIALS,
        w5a.TRAIN_ENTITIES,
        w5a.TRAIN_ACTIONS,
        w5a.TRAIN_LOCATIONS,
        w5a.CONFIRM_SYMBOLS,
        w5a.CONFIRM_ITEMS,
        w5a.CONFIRM_ACTIONS,
        w5a.CONFIRM_LOCATIONS,
        w5b.TRAIN_TAGS,
        w5b.TRAIN_ROLES,
        w5b.TRAIN_ACTIONS,
        w5b.TRAIN_SITES,
        w5b.CONFIRM_TAGS,
        w5b.CONFIRM_ROLES,
        w5b.CONFIRM_ACTIONS,
        w5b.CONFIRM_SITES,
    )
    return {item for group in groups for item in group}


def test_w5c_vocab_is_exactly_disjoint_from_w5a_and_w5b():
    assert all_w5c_vocab().isdisjoint(prior_vocab())


def test_w5c_authority_counts_and_split_isolation():
    train = generate_alignment_authority("train")
    dev = generate_alignment_authority("dev")
    confirm = generate_alignment_authority("confirm")

    assert len(train) == 768
    assert len(dev) == 192
    assert len(confirm) == 256
    assert {row.split for row in train} == {"train"}
    assert {row.split for row in dev} == {"dev"}
    assert {row.split for row in confirm} == {"confirm"}

    assert sum(row.k == 128 for row in train) == 112
    assert sum(row.k == 255 for row in train) == 64
    assert sum(row.k == 128 for row in dev) == 32
    assert sum(row.k == 255 for row in dev) == 32
    assert sum(row.k == 128 for row in confirm) == 64
    assert sum(row.k == 255 for row in confirm) == 64

    train_vocab = {
        token
        for row in train
        for option in row.options
        for token in option.criterion_text.split()
    }
    confirm_vocab = {
        token
        for row in confirm
        for option in row.options
        for token in option.criterion_text.split()
    }
    # Field labels repeat by design; semantic values must be disjoint.
    for label in {"region", "sport", "trait", "object"}:
        train_vocab.discard(label)
        confirm_vocab.discard(label)
    assert not (all_w5c_vocab() & prior_vocab())


def fake_case(k: int, gold: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    state_global = torch.randn(256, generator=g).half()
    state_tokens = torch.randn(8, 256, generator=g).half()
    question = torch.randn(256, generator=g).half()
    options = torch.randn(k, 256, generator=g).half()
    option_tokens = torch.randn(k, 6, 256, generator=g).half()
    mask = torch.ones(k, 6, dtype=torch.bool)
    return {
        "case_id": f"fake-{k}-{seed}",
        "split": "dev",
        "template_id": "fake",
        "k": k,
        "gold_index": gold,
        "state_global": state_global,
        "state_tokens": state_tokens,
        "question_embedding": question,
        "option_embeddings": options,
        "option_tokens": option_tokens,
        "option_token_mask": mask,
    }


def fake_cache():
    cases = [
        fake_case(32, 3, 1),
        fake_case(64, 7, 2),
        fake_case(128, 11, 3),
        fake_case(255, 13, 4),
    ]
    return {
        "schema_version": "r8-w5c-alignment-cache-v1",
        "case_count": len(cases),
        "state_encode_calls": len(cases),
        "state_encode_calls_per_case": 1.0,
        "cases": cases,
    }


def test_probe_evaluators_are_full_k_and_probability_preserving():
    cache = fake_cache()
    validate_alignment_cache(cache, expected_split="dev")

    bilinear = BilinearAlignmentProbe()
    pair = PairMLPAlignmentProbe(dropout=0.0)
    assert probe_parameter_count(bilinear) > 0
    assert probe_parameter_count(pair) > probe_parameter_count(bilinear)

    for name, model in (
        ("pooled_cosine", None),
        ("token_max", None),
        ("bilinear", bilinear),
        ("pair_mlp", pair),
    ):
        metrics = evaluate_probe(name, cache, model=model)
        assert metrics["case_count"] == 4
        assert set(metrics["per_k"]) == {"32", "64", "128", "255"}
        assert metrics["probability_mass_max_error"] < 1e-6
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["top5_recall"] <= 1.0
        assert math.isfinite(metrics["mrr"])
        assert math.isfinite(metrics["hard_brier"])


def passing_metrics():
    return {
        "accuracy": 0.7,
        "top5_recall": 0.9,
        "mrr": 0.75,
        "hard_brier": 0.2,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "32": {"accuracy": 0.7, "top5_recall": 0.9, "mrr": 0.75},
            "64": {"accuracy": 0.7, "top5_recall": 0.9, "mrr": 0.75},
            "128": {"accuracy": 0.5, "top5_recall": 0.8, "mrr": 0.6},
            "255": {"accuracy": 0.4, "top5_recall": 0.8, "mrr": 0.5},
        },
    }


def failing_metrics():
    x = passing_metrics()
    x["accuracy"] = 0.1
    x["per_k"] = {
        key: dict(value, accuracy=0.0, top5_recall=0.1)
        for key, value in x["per_k"].items()
    }
    return x


def test_confirm_classification_has_frozen_precedence():
    fail = failing_metrics()
    passed = passing_metrics()

    verdict, _ = classify_confirm({
        "pooled_cosine": passed,
        "token_max": passed,
        "bilinear": passed,
        "pair_mlp": passed,
    })
    assert verdict == "DIRECT_POOLED_SIGNAL"

    verdict, _ = classify_confirm({
        "pooled_cosine": fail,
        "token_max": passed,
        "bilinear": passed,
        "pair_mlp": passed,
    })
    assert verdict == "DIRECT_TOKEN_SIGNAL"

    verdict, _ = classify_confirm({
        "pooled_cosine": fail,
        "token_max": fail,
        "bilinear": passed,
        "pair_mlp": passed,
    })
    assert verdict == "LINEAR_RECOVERABLE_SIGNAL"

    verdict, _ = classify_confirm({
        "pooled_cosine": fail,
        "token_max": fail,
        "bilinear": fail,
        "pair_mlp": passed,
    })
    assert verdict == "NONLINEAR_RECOVERABLE_SIGNAL"

    verdict, _ = classify_confirm({
        "pooled_cosine": fail,
        "token_max": fail,
        "bilinear": fail,
        "pair_mlp": fail,
    })
    assert verdict == "A13_PROBE_FAIL"
