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
TYPED_DECISIONS_FINAL_CONFIG = "all"
TYPED_DECISIONS_FINAL_SPLIT = "test"
EXPECTED_FINAL_CASES = 400
EXPECTED_FINAL_DECISIONS = 2000
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


def _semantic_json_text(
    value: object,
    *,
    field: str,
) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TypedDecisionContractError(
                f"{field} must not be empty"
            )
        return text
    if value is None:
        raise TypedDecisionContractError(
            f"{field} must contain a semantic description"
        )
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypedDecisionContractError(
            f"{field} must be JSON-compatible semantic content"
        ) from exc
    if not text:
        raise TypedDecisionContractError(
            f"{field} must not be empty"
        )
    return text


def _criteria_options(
    primitive: Primitive,
    criteria: object,
    *,
    question_text: str,
) -> tuple[LogicalOption, ...]:
    if primitive == "score":
        if (
            not isinstance(criteria, Sequence)
            or isinstance(criteria, (str, bytes, bytearray))
            or not 2 <= len(criteria) <= 10
        ):
            raise TypedDecisionContractError(
                "score criteria must be an ordered list of 2 to 10 levels"
            )
        return tuple(
            LogicalOption(
                option_id=str(index),
                criterion_text=_semantic_json_text(
                    description,
                    field=f"score criteria[{index}]",
                ),
                value=float(index),
            )
            for index, description in enumerate(criteria)
        )

    if primitive == "noul":
        if criteria is None or criteria == {}:
            # System One Noul does not require a criteria object. Keep the
            # two logical values explicit while grounding their semantics in
            # the question text rather than treating routing IDs as meaning.
            return (
                LogicalOption(
                    option_id="false",
                    criterion_text=f"False / no for: {question_text}",
                    value=0.0,
                ),
                LogicalOption(
                    option_id="true",
                    criterion_text=f"True / yes for: {question_text}",
                    value=1.0,
                ),
            )
        if not isinstance(criteria, Mapping):
            raise TypedDecisionContractError(
                "noul criteria must be absent or a false/true object"
            )
        by_lower = {
            str(key).strip().lower(): value
            for key, value in criteria.items()
        }
        if set(by_lower) != {"false", "true"} or len(criteria) != 2:
            raise TypedDecisionContractError(
                "noul criteria must be exactly false and true when supplied"
            )
        return (
            LogicalOption(
                option_id="false",
                criterion_text=_semantic_json_text(
                    by_lower["false"],
                    field="noul criteria.false",
                ),
                value=0.0,
            ),
            LogicalOption(
                option_id="true",
                criterion_text=_semantic_json_text(
                    by_lower["true"],
                    field="noul criteria.true",
                ),
                value=1.0,
            ),
        )

    if not isinstance(criteria, Mapping):
        raise TypedDecisionContractError(
            f"{primitive} criteria must be an object"
        )

    if primitive != "choice":
        raise TypedDecisionContractError(
            f"unsupported primitive: {primitive}"
        )
    if len(criteria) < 2 or len(criteria) > 255:
        raise TypedDecisionContractError(
            "choice criteria must contain 2 to 255 options"
        )

    rows: list[tuple[str, str]] = []
    for raw_key, raw_description in criteria.items():
        option_id = str(raw_key).strip()
        if not option_id:
            raise TypedDecisionContractError(
                "choice criteria option IDs must be non-empty"
            )
        rows.append(
            (
                option_id,
                _semantic_json_text(
                    raw_description,
                    field=f"choice criteria.{option_id}",
                ),
            )
        )
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise TypedDecisionContractError(
            "choice criteria option IDs must be unique"
        )
    return tuple(
        LogicalOption(
            option_id=option_id,
            criterion_text=description,
        )
        for option_id, description in rows
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


def _parse_typed_decisions_row(
    row: Mapping[str, object],
    *,
    expected_split: str,
    authority: str,
) -> TypedDecisionCase:
    if not isinstance(row, Mapping):
        raise TypedDecisionContractError(
            "typed-decisions row must be an object"
        )

    split = str(
        row.get("split", expected_split)
    ).strip().lower()
    if split != expected_split:
        raise TypedDecisionContractError(
            f"{authority} accepts only split={expected_split!r}; "
            f"got {split!r}"
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
        question_text = _semantic_json_text(
            raw_question.get("instructions"),
            field=f"question {question_id!r} instructions",
        )
        options = _criteria_options(
            primitive,
            raw_question.get("criteria"),
            question_text=question_text,
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


def parse_typed_decisions_train_row(
    row: Mapping[str, object],
) -> TypedDecisionCase:
    return _parse_typed_decisions_row(
        row,
        expected_split="train",
        authority="TRAIN adapter",
    )


def parse_typed_decisions_final_row(
    row: Mapping[str, object],
) -> TypedDecisionCase:
    return _parse_typed_decisions_row(
        row,
        expected_split=TYPED_DECISIONS_FINAL_SPLIT,
        authority="W3b final adapter",
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



def load_pinned_typed_decisions_final(
    *,
    dataset_loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> list[TypedDecisionCase]:
    if dataset_loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional research stack to load typed-decisions"
            ) from exc
        dataset_loader = load_dataset

    rows = dataset_loader(
        TYPED_DECISIONS_DATASET_ID,
        TYPED_DECISIONS_FINAL_CONFIG,
        split=TYPED_DECISIONS_FINAL_SPLIT,
        revision=TYPED_DECISIONS_REVISION,
    )
    if len(rows) != EXPECTED_FINAL_CASES:
        raise TypedDecisionContractError(
            "typed-decisions final authority must contain exactly "
            f"{EXPECTED_FINAL_CASES} cases"
        )

    cases: list[TypedDecisionCase] = []
    seen_ids: set[str] = set()
    for row in rows:
        case = parse_typed_decisions_final_row(row)
        if case.workflow not in TYPED_DECISIONS_WORKFLOWS:
            raise TypedDecisionContractError(
                f"unknown workflow in final authority: {case.workflow!r}"
            )
        if case.case_id in seen_ids:
            raise TypedDecisionContractError(
                f"duplicate typed-decisions final case id: {case.case_id}"
            )
        seen_ids.add(case.case_id)
        cases.append(case)

    decisions = sum(len(case.decisions) for case in cases)
    if decisions != EXPECTED_FINAL_DECISIONS:
        raise TypedDecisionContractError(
            "typed-decisions final authority must contain exactly "
            f"{EXPECTED_FINAL_DECISIONS} decisions"
        )
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
