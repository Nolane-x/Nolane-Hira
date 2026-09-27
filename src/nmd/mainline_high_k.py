from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable, Sequence

import torch

from .contracts import CompiledSchema, LogicalOption
from .mainline import HIRA_V0_MAX_K, HiraV0Mainline, HiraV0Session
from .runtime import DecisionOutput

M2_K_LADDER = (4, 8, 16, 32, 64, 128, 255)
M2_MASS_TOL = 1e-6
M2_PERMUTATION_TOL = 2e-6


@dataclass(frozen=True)
class HighKQueryReport:
    requested_k: int
    actual_candidate_budget: int
    state_encode_delta: int
    schema_compile_ms: float
    decision_ms: float
    total_query_ms: float
    schema_tensor_bytes: int
    probability_mass_error: float
    finite: bool
    full_k: bool
    relation_delta_max: float
    selected_option_id: str | None

    @property
    def mechanics_pass(self) -> bool:
        return bool(
            self.state_encode_delta == 0
            and self.actual_candidate_budget == self.requested_k
            and self.finite
            and self.full_k
            and self.probability_mass_error <= M2_MASS_TOL
            and self.relation_delta_max == 0.0
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mechanics_pass"] = self.mechanics_pass
        return payload


@dataclass(frozen=True)
class HighKPermutationReport:
    k: int
    probability_max_error: float
    selected_option_invariant: bool

    @property
    def permutation_pass(self) -> bool:
        return bool(
            self.probability_max_error <= M2_PERMUTATION_TOL
            and self.selected_option_invariant
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["permutation_pass"] = self.permutation_pass
        return payload


@dataclass(frozen=True)
class HighKMechanicsSuite:
    k_values: tuple[int, ...]
    state_encode_delta: int
    query_count: int
    query_reports: tuple[HighKQueryReport, ...]
    permutation_reports: tuple[HighKPermutationReport, ...]

    @property
    def mechanics_pass(self) -> bool:
        return bool(
            self.k_values == M2_K_LADDER
            and self.state_encode_delta == 1
            and self.query_count == 2 * len(self.k_values)
            and all(report.mechanics_pass for report in self.query_reports)
            and all(
                report.permutation_pass
                for report in self.permutation_reports
            )
        )

    @property
    def max_probability_mass_error(self) -> float:
        return max(
            (report.probability_mass_error for report in self.query_reports),
            default=0.0,
        )

    @property
    def max_permutation_error(self) -> float:
        return max(
            (
                report.probability_max_error
                for report in self.permutation_reports
            ),
            default=0.0,
        )

    @property
    def max_schema_tensor_bytes(self) -> int:
        return max(
            (report.schema_tensor_bytes for report in self.query_reports),
            default=0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "k_values": list(self.k_values),
            "state_encode_delta": self.state_encode_delta,
            "query_count": self.query_count,
            "query_reports": [
                report.to_dict() for report in self.query_reports
            ],
            "permutation_reports": [
                report.to_dict() for report in self.permutation_reports
            ],
            "max_probability_mass_error": self.max_probability_mass_error,
            "max_permutation_error": self.max_permutation_error,
            "max_schema_tensor_bytes": self.max_schema_tensor_bytes,
            "mechanics_pass": self.mechanics_pass,
        }


def compiled_schema_tensor_bytes(schema: CompiledSchema) -> int:
    total = 0
    for value in vars(schema).values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel() * value.element_size())
    return total


def _validate_high_k(k: int) -> None:
    if k < 4:
        raise ValueError("M2 high-K mechanics requires K>=4")
    if k > HIRA_V0_MAX_K:
        raise ValueError(
            f"M2 high-K mechanics supports at most K={HIRA_V0_MAX_K}"
        )


def _profile_query(
    session: HiraV0Session,
    *,
    question_text: str,
    options: Sequence[LogicalOption],
) -> tuple[HighKQueryReport, DecisionOutput]:
    logical_options = tuple(options)
    k = len(logical_options)
    _validate_high_k(k)

    runtime = session._mainline.runtime
    before_state = runtime.state_encode_calls

    compile_start = time.perf_counter()
    schema, _ = runtime.compile_schema(
        primitive="choice",
        question_text=question_text,
        options=logical_options,
        use_cache=False,
        include_token_artifacts=True,
    )
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    decision_start = time.perf_counter()
    out = session.decide_compiled(schema)
    decision_ms = (time.perf_counter() - decision_start) * 1000.0

    state_delta = runtime.state_encode_calls - before_state
    relation_delta_max = float(out.hira.relation_delta.abs().max())
    mass_error = abs(float(out.probabilities.sum()) - 1.0)
    finite = bool(
        torch.isfinite(out.logits).all()
        and torch.isfinite(out.probabilities).all()
    )
    full_k = bool(
        int(out.hira.candidate_budget.item()) == k
        and bool(out.hira.selected_mask.all())
        and out.probabilities.numel() == k
        and out.logits.numel() == k
    )

    report = HighKQueryReport(
        requested_k=k,
        actual_candidate_budget=int(out.hira.candidate_budget.item()),
        state_encode_delta=state_delta,
        schema_compile_ms=compile_ms,
        decision_ms=decision_ms,
        total_query_ms=compile_ms + decision_ms,
        schema_tensor_bytes=compiled_schema_tensor_bytes(schema),
        probability_mass_error=mass_error,
        finite=finite,
        full_k=full_k,
        relation_delta_max=relation_delta_max,
        selected_option_id=out.selected_option_id,
    )
    return report, out


def profile_high_k_query(
    session: HiraV0Session,
    *,
    question_text: str,
    options: Sequence[LogicalOption],
) -> HighKQueryReport:
    report, _ = _profile_query(
        session,
        question_text=question_text,
        options=options,
    )
    return report


def _deterministic_permutation(
    options: tuple[LogicalOption, ...],
) -> tuple[LogicalOption, ...]:
    k = len(options)
    shift = max(1, k // 3)
    return options[shift:] + options[:shift]


def _probability_by_id(
    out: DecisionOutput,
    options: tuple[LogicalOption, ...],
) -> dict[str, float]:
    return {
        option.option_id: float(out.probabilities[index])
        for index, option in enumerate(options)
    }


def profile_high_k_permutation(
    session: HiraV0Session,
    *,
    question_text: str,
    options: Sequence[LogicalOption],
) -> tuple[
    HighKQueryReport,
    HighKPermutationReport,
]:
    canonical_options = tuple(options)
    k = len(canonical_options)
    _validate_high_k(k)
    if len({option.option_id for option in canonical_options}) != k:
        raise ValueError("M2 permutation contract requires unique option IDs")

    canonical_report, canonical = _profile_query(
        session,
        question_text=question_text,
        options=canonical_options,
    )
    permuted_options = _deterministic_permutation(canonical_options)
    _, permuted = _profile_query(
        session,
        question_text=question_text,
        options=permuted_options,
    )

    canonical_by_id = _probability_by_id(canonical, canonical_options)
    permuted_by_id = _probability_by_id(permuted, permuted_options)
    max_error = max(
        abs(canonical_by_id[option_id] - permuted_by_id[option_id])
        for option_id in canonical_by_id
    )
    permutation = HighKPermutationReport(
        k=k,
        probability_max_error=max_error,
        selected_option_invariant=(
            canonical.selected_option_id == permuted.selected_option_id
        ),
    )
    return canonical_report, permutation


def run_high_k_mechanics_suite(
    model: HiraV0Mainline,
    *,
    state_text: str,
    question_text: str,
    option_factory: Callable[[int], Sequence[LogicalOption]],
) -> HighKMechanicsSuite:
    before_state = model.runtime.state_encode_calls
    session = model.open_session(state_text)

    query_reports: list[HighKQueryReport] = []
    permutation_reports: list[HighKPermutationReport] = []

    for k in M2_K_LADDER:
        options = tuple(option_factory(k))
        if len(options) != k:
            raise ValueError(
                f"M2 option factory returned {len(options)} options for K={k}"
            )
        query_report, permutation_report = profile_high_k_permutation(
            session,
            question_text=question_text,
            options=options,
        )
        query_reports.append(query_report)
        permutation_reports.append(permutation_report)

    state_delta = model.runtime.state_encode_calls - before_state
    return HighKMechanicsSuite(
        k_values=M2_K_LADDER,
        state_encode_delta=state_delta,
        query_count=session.query_count,
        query_reports=tuple(query_reports),
        permutation_reports=tuple(permutation_reports),
    )


__all__ = [
    "HighKMechanicsSuite",
    "HighKPermutationReport",
    "HighKQueryReport",
    "M2_K_LADDER",
    "M2_MASS_TOL",
    "M2_PERMUTATION_TOL",
    "compiled_schema_tensor_bytes",
    "profile_high_k_permutation",
    "profile_high_k_query",
    "run_high_k_mechanics_suite",
]
