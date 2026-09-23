import math

import pytest
import torch

from nmd.semantic_late_interaction import (
    CONFIRM_K_COUNTS,
    DEV_K_COUNTS,
    TRAIN_K_COUNTS,
    LateInteractionMatcher,
    all_w5f_vocab,
    competence_gates,
    confirm_verdict,
    dev_selection_key,
    evaluate_matcher,
    evaluate_pooled_baseline,
    generate_late_interaction_authority,
    matcher_case_loss,
    validate_late_interaction_cache,
)


def synthetic_case(*, gold_index: int = 0, k: int = 3):
    d = 256
    # Context has two orthogonal semantic tokens.
    state = torch.zeros(4, d)
    question = torch.zeros(3, d)
    state[1, 0] = 1.0
    state[2, 1] = 1.0
    question[1, 2] = 1.0
    state_mask = torch.tensor([False, True, True, False])
    question_mask = torch.tensor([False, True, False])

    options = torch.zeros(k, 4, d)
    option_mask = torch.tensor(
        [[False, True, True, False]] * k
    )
    # Gold covers both state semantic dimensions.
    options[gold_index, 1, 0] = 1.0
    options[gold_index, 2, 1] = 1.0

    cursor = 0
    for i in range(k):
        if i == gold_index:
            continue
        # Distractors cover one field but miss the other.
        options[i, 1, 0] = 1.0
        options[i, 2, 10 + cursor] = 1.0
        cursor += 1

    pooled = options[:, 1:3].mean(1)
    return {
        "case_id": "synthetic",
        "split": "dev",
        "template_id": "fixture",
        "k": k,
        "gold_index": gold_index,
        "state_tokens": state.half(),
        "state_mask": state_mask,
        "question_tokens": question.half(),
        "question_mask": question_mask,
        "option_tokens": options.half(),
        "option_mask": option_mask,
        "state_pooled": state[1:3].mean(0).half(),
        "question_pooled": question[1].half(),
        "option_pooled": pooled.half(),
    }


def synthetic_cache(case):
    return {
        "schema_version": "r8-w5f-late-interaction-cache-v1",
        "case_count": 1,
        "encoder_calls": 1,
        "state_text_encodes": 1,
        "state_text_encodes_per_case": 1.0,
        "cases": [case],
    }


def test_w5f_authority_counts_are_frozen():
    assert sum(TRAIN_K_COUNTS.values()) == 512
    assert sum(DEV_K_COUNTS.values()) == 176
    assert sum(CONFIRM_K_COUNTS.values()) == 192

    train = generate_late_interaction_authority("train")
    dev = generate_late_interaction_authority("dev")
    confirm = generate_late_interaction_authority("confirm")

    assert len(train) == 512
    assert len(dev) == 176
    assert len(confirm) == 192
    assert {case.case_id for case in train}.isdisjoint(
        {case.case_id for case in dev}
    )
    assert all(case.split == "confirm" for case in confirm)


def test_w5f_vocab_is_disjoint_from_prior_semantic_waves():
    from nmd import semantic_alignment_probes as w5c
    from nmd import semantic_capacity_control as w5e
    from nmd import semantic_encoder_adaptation as w5d
    from nmd import semantic_routing_curriculum as w5a
    from nmd import semantic_token_curriculum as w5b

    prior = set()
    for module in (w5a, w5b, w5c, w5d, w5e):
        for name, value in vars(module).items():
            if not (
                name.startswith("TRAIN_")
                or name.startswith("CONFIRM_")
            ):
                continue
            if isinstance(value, tuple) and all(
                isinstance(item, str) for item in value
            ):
                prior.update(value)

    overlap = all_w5f_vocab() & prior
    assert overlap == set()


def test_generated_case_has_opaque_ids_and_exact_target_once():
    case = generate_late_interaction_authority("train")[0]
    assert len(case.options) == case.k
    assert [o.option_id for o in case.options] == [
        f"route-{i:03d}" for i in range(case.k)
    ]
    target = case.options[case.gold_index].criterion_text
    assert sum(o.criterion_text == target for o in case.options) == 1


def test_raw_maxsim_prefers_full_token_coverage():
    case = synthetic_case(gold_index=1, k=3)
    cache = synthetic_cache(case)
    validate_late_interaction_cache(cache, expected_split="dev")

    matcher = LateInteractionMatcher(None)
    logits = matcher.forward_case(case)
    assert int(logits.argmax()) == 1

    metrics = evaluate_matcher(matcher, cache)
    assert metrics["accuracy"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["probability_mass_max_error"] < 1e-6


def test_projected_matcher_has_small_trainable_surface_and_backprop():
    torch.manual_seed(4)
    case = synthetic_case(gold_index=0, k=3)
    matcher = LateInteractionMatcher(64)
    params = sum(p.numel() for p in matcher.parameters())
    assert params == 256 * 64 + 1

    loss = matcher_case_loss(matcher, case)
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert matcher.projection.weight.grad is not None
    assert matcher.log_scale.grad is not None


def test_pooled_baseline_uses_same_cache_and_is_finite():
    cache = synthetic_cache(synthetic_case(gold_index=0, k=3))
    metrics = evaluate_pooled_baseline(cache)
    assert metrics["case_count"] == 1
    assert math.isfinite(float(metrics["hard_brier"]))
    assert metrics["probability_mass_max_error"] < 1e-6


def test_dev_selection_key_prioritizes_accuracy_then_high_k():
    base = {
        "accuracy": 0.5,
        "top5_recall": 0.6,
        "mrr": 0.2,
        "hard_brier": 0.4,
        "per_k": {
            "128": {"accuracy": 0.2},
            "255": {"accuracy": 0.1},
        },
    }
    higher = {
        **base,
        "accuracy": 0.51,
    }
    assert dev_selection_key(higher, 6) < dev_selection_key(base, 1)


def test_confirm_verdict_requires_all_rescue_gates():
    pooled = {
        "accuracy": 0.1,
        "mrr": 0.1,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.1},
            "255": {"accuracy": 0.05, "top5_recall": 0.2},
        },
    }
    rescued = {
        "accuracy": 0.7,
        "mrr": 0.7,
        "probability_mass_max_error": 1e-7,
        "per_k": {
            "128": {"accuracy": 0.5},
            "255": {"accuracy": 0.4, "top5_recall": 0.8},
        },
    }
    verdict, gates = confirm_verdict(rescued, pooled)
    assert verdict == "LATE_INTERACTION_RESCUE"
    assert all(gates.values())

    weak = {
        **rescued,
        "accuracy": 0.19,
        "mrr": 0.11,
        "per_k": {
            "128": {"accuracy": 0.1},
            "255": {"accuracy": 0.05, "top5_recall": 0.2},
        },
    }
    verdict, _ = confirm_verdict(weak, pooled)
    assert verdict == "LATE_INTERACTION_FAIL"


def test_cache_validator_rejects_empty_content_mask():
    case = synthetic_case()
    case["state_mask"] = torch.zeros_like(case["state_mask"])
    with pytest.raises(ValueError, match="no content tokens"):
        validate_late_interaction_cache(synthetic_cache(case))
