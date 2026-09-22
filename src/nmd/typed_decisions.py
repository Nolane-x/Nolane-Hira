from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Callable, Iterable, Mapping, Sequence

from .contracts import LogicalOption, Primitive
from .runtime import DecisionOutput, NolaneHira


TYPED_DECISIONS_DATASET_ID = "LocalLLaMA/typed-decisions"
TYPED_DECISIONS_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
TYPED_DECISIONS_WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)
EXPECTED_TRAIN_CASES_PER_WORKFLOW = 300
EXPECTED_QUESTIONS_PER_CASE = 5
PROBABILITY_MASS_TOLERANCE = 5e-4


class TypedDecisionContractError(ValueError):
    pass


@dataclass(frozen=True)
class TypedDecision:
    question_id: str
    primitive: Primitive
    question_text: str
    options: tuple[LogicalOption, ...]
    gold_index: int
    gold_probabilities: tuple[float, ...]
    gold_score: float | None = None


@dataclass(frozen=True)
class TypedDecisionCase:
    case_id: str
    workflow: str
    state_text: str
    decisions: tuple[TypedDecision, ...]


@dataclass(frozen=True)
class TypedCaseExecutionReceipt:
    case_id: str
    workflow: str
    state_encode_calls: int
    decision_count: int
    schema_hashes: tuple[str, ...]


@dataclass
class TypedCaseExecution:
    outputs: tuple[DecisionOutput, ...]
    receipt: TypedCaseExecutionReceipt


def _parse_json_object(value: object, field: str) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TypedDecisionContractError(
                f"{field} must contain valid JSON"
            ) from exc
    if not isinstance(value, dict):
        raise TypedDecisionContractError(f"{field} must be a JSON object")
    return value


def _canonical_json_object(value: object, field: str) -> str:
    parsed = _parse_json_object(value, field)
    return json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _criteria_options(
    primitive: Primitive,
    criteria: object,
) -> tuple[LogicalOption, ...]:
    if not isinstance(criteria, Mapping) or len(criteria) < 2:
        raise TypedDecisionContractError(
            "question criteria must contain at least two options"
        )

    rows: list[tuple[str, str, float | None]] = []
    for raw_key, raw_text in criteria.items():
        option_id = str(raw_key).strip()
        text = str(raw_text).strip()
        if not option_id or not text:
            raise TypedDecisionContractError(
                "criteria option IDs and descriptions must be non-empty"
            )
        value: float | None = None
        if primitive == "score":
            try:
                value = float(option_id)
            except ValueError as exc:
                raise TypedDecisionContractError(
                    "score criteria IDs must be numeric rubric values"
                ) from exc
            if not math.isfinite(value):
                raise TypedDecisionContractError(
                    "score rubric values must be finite"
                )
        rows.append((option_id, text, value))

    if primitive == "score":
        rows.sort(key=lambda row: (float(row[2]), row[0]))
        numeric = [float(row[2]) for row in rows]
        if len(set(numeric)) != len(numeric):
            raise TypedDecisionContractError(
                "score criteria must have unique numeric rubric values"
            )
    elif primitive == "noul":
        by_lower = {row[0].lower(): row for row in rows}
        if set(by_lower) != {"false", "true"} or len(rows) != 2:
            raise TypedDecisionContractError(
                "noul criteria must be exactly false and true"
            )
        rows = [by_lower["false"], by_lower["true"]]
        rows[0] = (rows[0][0], rows[0][1], 0.0)
        rows[1] = (rows[1][0], rows[1][1], 1.0)
    else:
        rows.sort(key=lambda row: row[0])

    return tuple(
        LogicalOption(
            option_id=option_id,
            criterion_text=text,
            value=value,
        )
        for option_id, text, value in rows
    )


def _gold_distribution(
    raw_gold: object,
    *,
    primitive: Primitive,
    options: tuple[LogicalOption, ...],
) -> tuple[int, tuple[float, ...], float | None]:
    if not isinstance(raw_gold, Mapping):
        raise TypedDecisionContractError(
            "gold entry must be an object"
        )
    gold_type = raw_gold.get("type")
    if gold_type is not None and str(gold_type) != primitive:
        raise TypedDecisionContractError(
            "gold type does not match question type"
        )

    raw_probabilities = raw_gold.get("probabilities")
    if not isinstance(raw_probabilities, Mapping):
        raise TypedDecisionContractError(
            "gold probabilities must be an object"
        )
    probabilities = {
        str(key): float(value)
        for key, value in raw_probabilities.items()
    }
    option_ids = tuple(option.option_id for option in options)
    if set(probabilities) != set(option_ids):
        raise TypedDecisionContractError(
            "gold probability labels must exactly match criteria labels"
        )

    ordered = tuple(probabilities[option_id] for option_id in option_ids)
    if any((not math.isfinite(value)) or value < 0.0 for value in ordered):
        raise TypedDecisionContractError(
            "gold probabilities must be finite and non-negative"
        )
    mass = sum(ordered)
    if mass <= 0.0 or abs(mass - 1.0) > PROBABILITY_MASS_TOLERANCE:
        raise TypedDecisionContractError(
            f"gold probability mass must be one within tolerance; got {mass}"
        )
    normalized = tuple(value / mass for value in ordered)

    if "label" not in raw_gold:
        raise TypedDecisionContractError("gold label is required")
    raw_label = str(raw_gold["label"])
    id_to_index = {option_id: i for i, option_id in enumerate(option_ids)}
    if raw_label in id_to_index:
        gold_index = id_to_index[raw_label]
    elif primitive == "noul":
        lower_to_index = {
            option_id.lower(): i
            for i, option_id in enumerate(option_ids)
        }
        if raw_label.lower() not in lower_to_index:
            raise TypedDecisionContractError(
                "gold noul label is not in criteria"
            )
        gold_index = lower_to_index[raw_label.lower()]
    else:
        raise TypedDecisionContractError(
            "gold label is not in criteria"
        )

    gold_score = None
    if primitive == "score":
        if raw_gold.get("score") is None:
            gold_score = sum(
                probability * float(option.value)
                for probability, option in zip(normalized, options)
            )
        else:
            gold_score = float(raw_gold["score"])
        if not math.isfinite(gold_score):
            raise TypedDecisionContractError(
                "gold score must be finite"
            )

    return gold_index, normalized, gold_score


def parse_typed_decisions_train_row(
    row: Mapping[str, object],
) -> TypedDecisionCase:
    if not isinstance(row, Mapping):
        raise TypedDecisionContractError("typed-decisions row must be an object")

    split = str(row.get("split", "train")).strip().lower()
    if split != "train":
        raise TypedDecisionContractError(
            "W2 adapter is train-only; non-train rows are forbidden"
        )

    case_id = str(row.get("id", "")).strip()
    workflow = str(row.get("workflow", "")).strip()
    if not case_id or not workflow:
        raise TypedDecisionContractError(
            "typed-decisions row requires id and workflow"
        )

    state_text = _canonical_json_object(row.get("state"), "state")
    questions = _parse_json_object(row.get("questions"), "questions")
    gold = _parse_json_object(row.get("gold"), "gold")

    if set(questions) != set(gold):
        raise TypedDecisionContractError(
            "question IDs and gold IDs must match exactly"
        )
    if len(questions) != EXPECTED_QUESTIONS_PER_CASE:
        raise TypedDecisionContractError(
            f"typed-decisions case must contain exactly "
            f"{EXPECTED_QUESTIONS_PER_CASE} questions"
        )
    if row.get("n_questions") is not None:
        try:
            n_questions = int(row["n_questions"])
        except (TypeError, ValueError) as exc:
            raise TypedDecisionContractError(
                "n_questions must be an integer"
            ) from exc
        if n_questions != EXPECTED_QUESTIONS_PER_CASE:
            raise TypedDecisionContractError(
                "n_questions disagrees with frozen dataset contract"
            )

    decisions: list[TypedDecision] = []
    for question_id in sorted(map(str, questions.keys())):
        raw_question = questions[question_id]
        if not isinstance(raw_question, Mapping):
            raise TypedDecisionContractError(
                f"question {question_id!r} must be an object"
            )
        primitive = raw_question.get("type")
        if primitive not in {"choice", "score", "noul"}:
            raise TypedDecisionContractError(
                f"question {question_id!r} has unsupported type"
            )
        primitive = str(primitive)
        question_text = str(
            raw_question.get("instructions", "")
        ).strip()
        if not question_text:
            raise TypedDecisionContractError(
                f"question {question_id!r} requires instructions"
            )

        options = _criteria_options(
            primitive,
            raw_question.get("criteria"),
        )
        gold_index, probabilities, gold_score = _gold_distribution(
            gold[question_id],
            primitive=primitive,
            options=options,
        )
        decisions.append(
            TypedDecision(
                question_id=question_id,
                primitive=primitive,
                question_text=question_text,
                options=options,
                gold_index=gold_index,
                gold_probabilities=probabilities,
                gold_score=gold_score,
            )
        )

    # factors and label_agreement are deliberately never read here.
    return TypedDecisionCase(
        case_id=case_id,
        workflow=workflow,
        state_text=state_text,
        decisions=tuple(decisions),
    )


def load_pinned_typed_decisions_train(
    workflows: Sequence[str] = TYPED_DECISIONS_WORKFLOWS,
    *,
    dataset_loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> list[TypedDecisionCase]:
    requested = tuple(workflows)
    unknown = sorted(set(requested) - set(TYPED_DECISIONS_WORKFLOWS))
    if unknown:
        raise TypedDecisionContractError(
            f"unknown typed-decisions workflow(s): {unknown}"
        )
    if len(set(requested)) != len(requested):
        raise TypedDecisionContractError(
            "typed-decisions workflow list contains duplicates"
        )

    if dataset_loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional research stack to load typed-decisions"
            ) from exc
        dataset_loader = load_dataset

    cases: list[TypedDecisionCase] = []
    seen_ids: set[str] = set()
    for workflow in requested:
        rows = dataset_loader(
            TYPED_DECISIONS_DATASET_ID,
            workflow,
            split="train",
            revision=TYPED_DECISIONS_REVISION,
        )
        if len(rows) != EXPECTED_TRAIN_CASES_PER_WORKFLOW:
            raise TypedDecisionContractError(
                f"{workflow} train split must contain exactly "
                f"{EXPECTED_TRAIN_CASES_PER_WORKFLOW} rows"
            )
        for row in rows:
            case = parse_typed_decisions_train_row(row)
            if case.workflow != workflow:
                raise TypedDecisionContractError(
                    "row workflow does not match requested config"
                )
            if case.case_id in seen_ids:
                raise TypedDecisionContractError(
                    f"duplicate typed-decisions case id: {case.case_id}"
                )
            seen_ids.add(case.case_id)
            cases.append(case)
    return cases


def execute_typed_case(
    model: NolaneHira,
    case: TypedDecisionCase,
    *,
    forced_budget: int | None = None,
    adaptive_budget: bool = False,
    use_schema_cache: bool = True,
) -> TypedCaseExecution:
    before = model.state_encode_calls
    memory = model.compile_state(case.state_text)
    outputs: list[DecisionOutput] = []
    schema_hashes: list[str] = []
    for decision in case.decisions:
        schema, _ = model.compile_schema(
            primitive=decision.primitive,
            question_text=decision.question_text,
            options=decision.options,
            use_cache=use_schema_cache,
        )
        schema_hashes.append(schema.schema_hash)
        outputs.append(
            model.forward_compiled(
                memory,
                schema,
                forced_budget=forced_budget,
                adaptive_budget=adaptive_budget,
            )
        )
    state_encode_calls = model.state_encode_calls - before
    if state_encode_calls != 1:
        raise RuntimeError(
            "typed case execution violated state-once semantics"
        )
    return TypedCaseExecution(
        outputs=tuple(outputs),
        receipt=TypedCaseExecutionReceipt(
            case_id=case.case_id,
            workflow=case.workflow,
            state_encode_calls=state_encode_calls,
            decision_count=len(outputs),
            schema_hashes=tuple(schema_hashes),
        ),
    )
