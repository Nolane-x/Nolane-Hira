import math

import torch

from nmd.hira import HIRACore
from nmd.semantic_routing_curriculum import (
    CONFIRM_K_COUNTS,
    CONFIRM_TEMPLATES,
    CONFIRM_SYMBOLS,
    DEV_K_COUNTS,
    DEV_TEMPLATES,
    TRAIN_K_COUNTS,
    TRAIN_MATERIALS,
    TRAIN_TEMPLATES,
    dev_selection_key,
    evaluate_routing_cache,
    generate_authority,
    generate_routing_case,
    train_candidate,
    validate_routing_cache,
)


def fake_case(case_id: str, split: str, k: int, gold_index: int):
    g=torch.Generator().manual_seed(sum(ord(c) for c in case_id)+k)
    return {
        "case_id":case_id,
        "split":split,
        "template_id":"fixture",
        "k":k,
        "gold_index":gold_index,
        "state_segments":torch.randn(2,256,generator=g).half(),
        "question_embedding":torch.randn(256,generator=g).half(),
        "option_embeddings":torch.randn(k,256,generator=g).half(),
        "schema_hash":f"schema-{case_id}",
    }


def fake_cache(split: str, cases):
    return {
        "schema_version":"r8-w5a-routing-cache-v1",
        "case_count":len(cases),
        "state_encode_calls":len(cases),
        "state_encode_calls_per_case":1.0,
        "cases":cases,
    }


def test_authority_counts_and_k_distribution_are_frozen():
    for split, expected in (
        ("train",TRAIN_K_COUNTS),
        ("dev",DEV_K_COUNTS),
        ("confirm",CONFIRM_K_COUNTS),
    ):
        rows=generate_authority(split)
        assert len(rows)==sum(expected.values())
        observed={}
        for row in rows:
            observed[row.k]=observed.get(row.k,0)+1
        assert observed==expected


def test_train_dev_confirm_templates_are_disjoint():
    assert set(TRAIN_TEMPLATES).isdisjoint(DEV_TEMPLATES)
    assert set(TRAIN_TEMPLATES).isdisjoint(CONFIRM_TEMPLATES)
    assert set(DEV_TEMPLATES).isdisjoint(CONFIRM_TEMPLATES)


def test_confirm_primary_vocabulary_is_disjoint_from_train_primary_vocab():
    assert set(TRAIN_MATERIALS).isdisjoint(CONFIRM_SYMBOLS)


def test_generator_is_deterministic_opaque_and_has_unique_options():
    for split,k in (("train",128),("dev",255),("confirm",255)):
        a=generate_routing_case(split=split,case_index=7,k=k)
        b=generate_routing_case(split=split,case_index=7,k=k)
        assert a==b
        assert len(a.options)==k
        assert len({o.criterion_text for o in a.options})==k
        assert [o.option_id for o in a.options]==[
            f"route-{i:03d}" for i in range(k)
        ]
        assert 0<=a.gold_index<k


def test_train_and_confirm_case_texts_do_not_share_templates_or_vocab_tokens():
    train=generate_routing_case(split="train",case_index=1,k=32)
    confirm=generate_routing_case(split="confirm",case_index=1,k=32)
    assert train.template_id in TRAIN_TEMPLATES
    assert confirm.template_id in CONFIRM_TEMPLATES
    assert train.state_text != confirm.state_text
    assert train.question_text != confirm.question_text


def test_cached_metrics_are_finite_full_k_and_selection_key_prefers_accuracy():
    torch.manual_seed(9)
    hira=HIRACore(d_model=256,dropout=0.0)
    cases=[
        fake_case("dev-1","dev",128,3),
        fake_case("dev-2","dev",255,5),
        fake_case("dev-3","dev",128,7),
        fake_case("dev-4","dev",255,9),
    ]
    cache=fake_cache("dev",cases)
    validate_routing_cache(cache,expected_split="dev")
    metrics=evaluate_routing_cache(hira,cache)
    assert metrics["case_count"]==4
    assert metrics["candidate_budget_min"]==128
    assert metrics["candidate_budget_max"]==255
    assert metrics["probability_mass_max_error"]<1e-6
    assert math.isfinite(metrics["hard_brier"])
    assert set(metrics["per_k"])=={"128","255"}

    better=dict(metrics)
    better["accuracy"]=metrics["accuracy"]+0.01
    assert dev_selection_key(better,6)<dev_selection_key(metrics,1)


def test_small_candidate_training_runs_and_returns_selected_checkpoint():
    torch.manual_seed(11)
    hira=HIRACore(d_model=256,dropout=0.0)
    train=fake_cache("train",[
        fake_case("train-a","train",8,1),
        fake_case("train-b","train",16,2),
        fake_case("train-c","train",32,3),
    ])
    dev=fake_cache("dev",[
        fake_case("dev-a","dev",128,4),
        fake_case("dev-b","dev",255,5),
    ])
    history,state,best=train_candidate(
        hira,train,dev,lr=3e-4,epochs=2,seed=91
    )
    assert len(history)==2
    assert best["epoch"] in {1,2}
    assert state
    assert all(torch.isfinite(t).all() for t in state.values())
