import copy
import json
import math

import pytest
import torch

from nmd.hira import HIRACore
from nmd.losses import LossWeights, ordinal_expected_mae
from nmd.runtime import NolaneHira
from nmd.semantic import TrainableSemanticEncoder
from nmd.training import loss_typed_case
from nmd.typed_decisions import (
    TYPED_DECISIONS_DATASET_ID,
    TYPED_DECISIONS_REVISION,
    TYPED_DECISIONS_FINAL_CONFIG,
    TYPED_DECISIONS_FINAL_SPLIT,
    EXPECTED_FINAL_CASES,
    TypedDecisionContractError,
    execute_typed_case,
    load_pinned_typed_decisions_final,
    load_pinned_typed_decisions_train,
    parse_typed_decisions_final_row,
    parse_typed_decisions_train_row,
)
from nmd.typed_eval import evaluate_typed_cases


def typed_row(
    *,
    case_id: str = "tr_agent_trace_observability_000000",
    workflow: str = "agent_trace_observability",
    split: str = "train",
):
    questions = {
        "action": {
            "type": "choice",
            "instructions": "What should the system do with this trace?",
            "criteria": {
                "continue": "Let the agent proceed.",
                "human_review": "Queue this trace for human review.",
                "stop": "Halt the agent now.",
            },
        },
        "needs_review": {
            "type": "noul",
            "instructions": "Does this trace need human review?",
            "criteria": {
                "false": "Human review is not needed.",
                "true": "Human review is needed.",
            },
        },
        "outcome": {
            "type": "choice",
            "instructions": "What was the outcome?",
            "criteria": {
                "failure": "The task failed.",
                "partial": "The task was partly completed.",
                "success": "The task completed successfully.",
            },
        },
        "risk": {
            "type": "score",
            "instructions": "How risky was this run?",
            "criteria": [
                "No meaningful risk.",
                "Low risk.",
                "Material risk.",
                "Severe risk.",
            ],
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is intervention?",
            "criteria": [
                "No urgency.",
                "Low urgency.",
                "High urgency.",
                "Immediate intervention.",
            ],
        },
    }
    gold = {
        "action": {
            "type": "choice",
            "label": "human_review",
            "confidence": 0.4,
            "probabilities": {
                "continue": 0.2,
                "human_review": 0.7,
                "stop": 0.1,
            },
        },
        "needs_review": {
            "type": "noul",
            "label": "true",
            "confidence": 0.6,
            "noul": 0.8,
            "probabilities": {
                "false": 0.2,
                "true": 0.8,
            },
        },
        "outcome": {
            "type": "choice",
            "label": "partial",
            "confidence": 0.3,
            "probabilities": {
                "failure": 0.1,
                "partial": 0.6,
                "success": 0.3,
            },
        },
        "risk": {
            "type": "score",
            "label": 1,
            "confidence": 0.5,
            "score": 1.3,
            "probabilities": {
                "0": 0.1,
                "1": 0.6,
                "2": 0.2,
                "3": 0.1,
            },
        },
        "urgency": {
            "type": "score",
            "label": 2,
            "confidence": 0.4,
            "score": 1.8,
            "probabilities": {
                "0": 0.1,
                "1": 0.2,
                "2": 0.5,
                "3": 0.2,
            },
        },
    }
    return {
        "id": case_id,
        "workflow": workflow,
        "split": split,
        "state": json.dumps(
            {
                "task": "Delete the queued records.",
                "trace": {
                    "steps": 7,
                    "tool_errors": 1,
                },
            }
        ),
        "questions": json.dumps(questions),
        "gold": json.dumps(gold),
        "factors": json.dumps(
            {"latent_only": "must never reach the model"}
        ),
        "label_agreement": json.dumps(
            {"action": {"argmax_agree": False}}
        ),
        "n_questions": 5,
    }


def make_model():
    torch.manual_seed(123)
    encoder = TrainableSemanticEncoder(
        vocab_size=512,
        d_model=256,
        n_layers=1,
        n_heads=4,
    )
    return NolaneHira(
        encoder,
        HIRACore(dropout=0.0),
    )


def test_adapter_preserves_typed_semantics_and_soft_gold():
    case = parse_typed_decisions_train_row(typed_row())

    assert case.case_id == "tr_agent_trace_observability_000000"
    assert len(case.decisions) == 5
    assert [d.question_id for d in case.decisions] == [
        "action",
        "needs_review",
        "outcome",
        "risk",
        "urgency",
    ]

    action = case.decisions[0]
    assert action.primitive == "choice"
    assert [o.option_id for o in action.options] == [
        "continue",
        "human_review",
        "stop",
    ]
    assert action.options[1].criterion_text == (
        "Queue this trace for human review."
    )
    assert action.gold_index == 1
    assert math.isclose(sum(action.gold_probabilities), 1.0)

    noul = case.decisions[1]
    assert [o.value for o in noul.options] == [0.0, 1.0]
    assert noul.gold_index == 1

    risk = case.decisions[3]
    assert [o.value for o in risk.options] == [0.0, 1.0, 2.0, 3.0]
    assert risk.gold_score == 1.3


def test_adapter_is_order_canonical_and_ignores_latent_fields():
    raw_a = typed_row()
    raw_b = copy.deepcopy(raw_a)

    questions = json.loads(raw_b["questions"])
    questions["action"]["criteria"] = {
        "stop": "Halt the agent now.",
        "continue": "Let the agent proceed.",
        "human_review": "Queue this trace for human review.",
    }
    raw_b["questions"] = json.dumps(questions)

    gold = json.loads(raw_b["gold"])
    gold["action"]["probabilities"] = {
        "stop": 0.1,
        "continue": 0.2,
        "human_review": 0.7,
    }
    raw_b["gold"] = json.dumps(gold)
    raw_b["factors"] = json.dumps({"leak": "different"})
    raw_b["label_agreement"] = json.dumps(
        {"action": {"argmax_agree": True}}
    )

    assert parse_typed_decisions_train_row(
        raw_a
    ) == parse_typed_decisions_train_row(raw_b)


def test_w2_adapter_fails_closed_on_test_split_and_bad_probability_mass():
    raw = typed_row(split="test")
    with pytest.raises(TypedDecisionContractError, match="train-only"):
        parse_typed_decisions_train_row(raw)

    raw = typed_row()
    gold = json.loads(raw["gold"])
    gold["action"]["probabilities"] = {
        "continue": 0.5,
        "human_review": 0.7,
        "stop": 0.1,
    }
    raw["gold"] = json.dumps(gold)
    with pytest.raises(
        TypedDecisionContractError,
        match="probability mass",
    ):
        parse_typed_decisions_train_row(raw)


def test_pinned_loader_never_requests_test_split_or_mutable_revision():
    calls = []

    def loader(dataset_id, config, *, split, revision):
        calls.append((dataset_id, config, split, revision))
        return [
            typed_row(
                case_id=f"tr_{config}_{i:06d}",
                workflow=config,
            )
            for i in range(300)
        ]

    cases = load_pinned_typed_decisions_train(
        ("customer_service",),
        dataset_loader=loader,
    )

    assert len(cases) == 300
    assert calls == [
        (
            TYPED_DECISIONS_DATASET_ID,
            "customer_service",
            "train",
            TYPED_DECISIONS_REVISION,
        )
    ]


def test_state_once_execution_answers_all_five_questions_with_one_encode():
    case = parse_typed_decisions_train_row(typed_row())
    model = make_model()
    model.eval()

    before_model = model.state_encode_calls
    before_encoder = model.encoder.state_encode_calls
    execution = execute_typed_case(model, case)

    assert execution.receipt.decision_count == 5
    assert execution.receipt.state_encode_calls == 1
    assert model.state_encode_calls - before_model == 1
    assert model.encoder.state_encode_calls - before_encoder == 1
    assert len(set(execution.receipt.schema_hashes)) == 5
    assert len(execution.outputs) == 5
    for output in execution.outputs:
        assert torch.isclose(
            output.probabilities.sum(),
            output.probabilities.new_tensor(1.0),
            atol=1e-6,
        )


def test_case_training_reuses_one_state_graph_and_reaches_encoder_and_hira():
    case = parse_typed_decisions_train_row(typed_row())
    model = make_model()
    model.train()

    before = model.state_encode_calls
    loss, parts, receipt = loss_typed_case(
        model,
        case,
        weights=LossWeights(
            hard_ce=1.0,
            teacher_kl=0.2,
            brier=0.1,
            soft_brier=0.1,
            ordinal_mae=0.2,
        ),
    )
    loss.backward()

    assert receipt.state_encode_calls == 1
    assert receipt.decision_count == 5
    assert model.state_encode_calls - before == 1
    assert "teacher_kl" in parts
    assert "soft_brier" in parts
    assert "ordinal_mae" in parts
    assert model.encoder.embedding.weight.grad is not None
    assert model.hira.cross_score[0].weight.grad is not None
    assert torch.isfinite(loss)


def test_explicit_score_support_is_not_assumed_to_be_option_index():
    logits = torch.tensor([[-10.0, 10.0, -10.0]])
    gold_score = torch.tensor([5.0])
    explicit = ordinal_expected_mae(
        logits,
        gold_score,
        score_support=torch.tensor([1.0, 5.0, 10.0]),
    )
    index_based = ordinal_expected_mae(logits, gold_score)

    assert explicit.item() < 1e-3
    assert index_based.item() > 3.9


def test_train_only_evaluator_reports_full_distribution_and_state_once_metrics():
    case = parse_typed_decisions_train_row(typed_row())
    model = make_model()

    metrics = evaluate_typed_cases(model, [case])

    assert metrics["case_count"] == 1
    assert metrics["decision_count"] == 5
    assert metrics["primitive_counts"] == {
        "choice": 2,
        "score": 2,
        "noul": 1,
    }
    assert metrics["state_encode_calls"] == 1
    assert metrics["state_encode_calls_per_case"] == 1.0
    assert metrics["decisions_per_state_encode"] == 5.0
    assert metrics["score_count"] == 2
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
        assert metrics[key] is not None
        assert math.isfinite(float(metrics[key]))


def test_criteria_text_is_semantic_input_not_a_bare_label_list():
    raw_a = typed_row()
    raw_b = typed_row(case_id="tr_agent_trace_observability_000001")
    questions = json.loads(raw_b["questions"])
    questions["action"]["criteria"]["human_review"] = (
        "Escalate immediately to a senior human operator."
    )
    raw_b["questions"] = json.dumps(questions)

    case_a = parse_typed_decisions_train_row(raw_a)
    case_b = parse_typed_decisions_train_row(raw_b)
    model = make_model()
    model.eval()

    action_a = case_a.decisions[0]
    action_b = case_b.decisions[0]
    schema_a, _ = model.compile_schema(
        primitive=action_a.primitive,
        question_text=action_a.question_text,
        options=action_a.options,
        use_cache=False,
    )
    schema_b, _ = model.compile_schema(
        primitive=action_b.primitive,
        question_text=action_b.question_text,
        options=action_b.options,
        use_cache=False,
    )

    assert not torch.allclose(
        schema_a.option_embeddings,
        schema_b.option_embeddings,
    )



def test_score_requires_ordered_list_not_numeric_key_map():
    raw = typed_row()
    questions = json.loads(raw["questions"])
    questions["risk"]["criteria"] = {
        "0": "No meaningful risk.",
        "1": "Low risk.",
        "2": "Material risk.",
        "3": "Severe risk.",
    }
    raw["questions"] = json.dumps(questions)

    with pytest.raises(
        TypedDecisionContractError,
        match="ordered list",
    ):
        parse_typed_decisions_train_row(raw)


def test_json_content_descriptions_are_canonical_semantic_text():
    raw = typed_row()
    questions = json.loads(raw["questions"])
    questions["action"]["criteria"]["human_review"] = {
        "meaning": "Queue for review",
        "severity": 2,
    }
    questions["needs_review"]["criteria"]["true"] = {
        "meaning": "Review is needed",
        "examples": ["unsafe", "ambiguous"],
    }
    raw["questions"] = json.dumps(questions)

    case = parse_typed_decisions_train_row(raw)
    action = next(
        decision
        for decision in case.decisions
        if decision.question_id == "action"
    )
    noul = next(
        decision
        for decision in case.decisions
        if decision.question_id == "needs_review"
    )

    assert action.options[1].criterion_text == (
        '{"meaning":"Queue for review","severity":2}'
    )
    assert noul.options[1].criterion_text == (
        '{"examples":["unsafe","ambiguous"],"meaning":"Review is needed"}'
    )



def test_noul_allows_missing_criteria_and_derives_semantics_from_question():
    raw = typed_row()
    questions = json.loads(raw["questions"])
    del questions["needs_review"]["criteria"]
    raw["questions"] = json.dumps(questions)

    case = parse_typed_decisions_train_row(raw)
    decision = next(
        d for d in case.decisions
        if d.question_id == "needs_review"
    )

    assert [option.option_id for option in decision.options] == [
        "false",
        "true",
    ]
    assert [option.value for option in decision.options] == [0.0, 1.0]
    assert "Does this trace need human review?" in (
        decision.options[0].criterion_text
    )
    assert "Does this trace need human review?" in (
        decision.options[1].criterion_text
    )


def test_noul_rejects_nonstandard_supplied_criteria_shape():
    raw = typed_row()
    questions = json.loads(raw["questions"])
    questions["needs_review"]["criteria"] = ["no", "yes"]
    raw["questions"] = json.dumps(questions)

    with pytest.raises(
        TypedDecisionContractError,
        match="absent or a false/true object",
    ):
        parse_typed_decisions_train_row(raw)



def test_final_parser_accepts_only_test_and_train_parser_stays_train_only():
    final_case = parse_typed_decisions_final_row(
        typed_row(
            case_id="te_agent_trace_observability_000000",
            split="test",
        )
    )
    assert final_case.case_id.startswith("te_")

    with pytest.raises(TypedDecisionContractError, match="split='test'"):
        parse_typed_decisions_final_row(typed_row(split="train"))

    with pytest.raises(TypedDecisionContractError, match="split='train'"):
        parse_typed_decisions_train_row(typed_row(split="test"))


def test_pinned_final_loader_requests_exact_all_test_revision_and_counts():
    calls = []
    workflows = (
        "agent_trace_observability",
        "customer_service",
        "invoice_processing",
        "security_incidents",
    )

    def loader(dataset_id, config, *, split, revision):
        calls.append((dataset_id, config, split, revision))
        return [
            typed_row(
                case_id=f"te_{workflows[i % 4]}_{i:06d}",
                workflow=workflows[i % 4],
                split="test",
            )
            for i in range(EXPECTED_FINAL_CASES)
        ]

    cases = load_pinned_typed_decisions_final(dataset_loader=loader)

    assert len(cases) == 400
    assert sum(len(case.decisions) for case in cases) == 2000
    assert calls == [
        (
            TYPED_DECISIONS_DATASET_ID,
            TYPED_DECISIONS_FINAL_CONFIG,
            TYPED_DECISIONS_FINAL_SPLIT,
            TYPED_DECISIONS_REVISION,
        )
    ]


def test_final_loader_fails_closed_on_wrong_count_and_duplicate_ids():
    def short_loader(*args, **kwargs):
        return [
            typed_row(
                case_id=f"te_customer_service_{i:06d}",
                workflow="customer_service",
                split="test",
            )
            for i in range(399)
        ]

    with pytest.raises(
        TypedDecisionContractError,
        match="exactly 400 cases",
    ):
        load_pinned_typed_decisions_final(
            dataset_loader=short_loader
        )

    def duplicate_loader(*args, **kwargs):
        rows = [
            typed_row(
                case_id=f"te_customer_service_{i:06d}",
                workflow="customer_service",
                split="test",
            )
            for i in range(400)
        ]
        rows[-1]["id"] = rows[0]["id"]
        return rows

    with pytest.raises(
        TypedDecisionContractError,
        match="duplicate typed-decisions final case id",
    ):
        load_pinned_typed_decisions_final(
            dataset_loader=duplicate_loader
        )
