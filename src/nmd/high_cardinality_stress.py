from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import statistics
import time
from typing import Sequence

import torch
from torch import Tensor

from .contracts import LogicalOption
from .hira import HIRACore
from .runtime import NolaneHira


MECHANICAL_CASES = 64
SEMANTIC_CASES = 128
K_VALUES = (128, 255)
MECHANICAL_SEEDS = {128: 404, 255: 808}
SEMANTIC_SEEDS = {128: 128128, 255: 255255}
FORCED_BUDGET = 255
D_MODEL = 256
STATE_SEGMENTS = 4

MASS_TOL = 1e-6
PERM_TOL = 2e-6
REPEAT_TOL = 1e-7

COLORS = (
    "amber","aqua","azure","beige","black","blue","bronze","brown",
    "coral","crimson","cyan","emerald","gold","gray","green","indigo",
    "ivory","jade","lavender","lime","magenta","maroon","navy","olive",
    "orange","pink","purple","red","silver","teal","violet","yellow",
)
OBJECTS = (
    "anchor","badge","beacon","bridge","cabin","candle","circle","column",
    "compass","crystal","cube","door","drum","feather","flag","gate",
    "gem","harbor","key","lantern","leaf","mirror","orb","pyramid",
    "ring","shield","sphere","star","stone","tower","wheel","window",
)
ACTIONS = (
    "align","carry","circle","close","collect","cross","enter","follow",
    "guard","hold","lift","mark","open","place","return","signal",
)
LOCATIONS = (
    "atrium","bay","courtyard","deck","field","garden","hall","island",
    "junction","meadow","plaza","ridge","station","terrace","valley","yard",
)


@dataclass(frozen=True)
class MechanicalResult:
    k: int
    case_count: int
    finite_pass: bool
    budget_min: int
    budget_max: int
    tail_mass_max: float
    probability_mass_max_error: float
    permutation_max_error: float
    argmax_invariant_count: int
    repeatability_max_error: float
    p50_ms_cpu: float
    p95_ms_cpu: float
    option_tensor_bytes_per_case: int
    mechanics_pass: bool


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@torch.inference_mode()
def evaluate_mechanical_stress(
    hira: HIRACore,
    *,
    k: int,
    cases: int = MECHANICAL_CASES,
) -> MechanicalResult:
    if k not in K_VALUES:
        raise ValueError(f"unsupported frozen K: {k}")
    hira.eval()
    gen = torch.Generator(device="cpu").manual_seed(MECHANICAL_SEEDS[k])

    finite_pass = True
    budget_values: list[int] = []
    tail_values: list[float] = []
    mass_errors: list[float] = []
    perm_errors: list[float] = []
    repeat_errors: list[float] = []
    argmax_invariant = 0
    latencies: list[float] = []

    for case_index in range(cases):
        question = torch.randn(1, D_MODEL, generator=gen)
        segments = torch.randn(1, STATE_SEGMENTS, D_MODEL, generator=gen)
        options = torch.randn(1, k, D_MODEL, generator=gen)
        qtype = torch.zeros(1, dtype=torch.long)

        start = time.perf_counter()
        out = hira(
            question,
            segments,
            options,
            qtype,
            forced_budget=FORCED_BUDGET,
            adaptive_budget=False,
        )
        latencies.append((time.perf_counter() - start) * 1000.0)

        repeated = hira(
            question,
            segments,
            options,
            qtype,
            forced_budget=FORCED_BUDGET,
            adaptive_budget=False,
        )
        repeat_error = float(
            (out.probabilities - repeated.probabilities).abs().max().item()
        )
        repeat_errors.append(repeat_error)

        perm_gen = torch.Generator(device="cpu").manual_seed(
            9000 + k * 100 + case_index
        )
        perm = torch.randperm(k, generator=perm_gen)
        permuted = hira(
            question,
            segments,
            options[:, perm, :],
            qtype,
            forced_budget=FORCED_BUDGET,
            adaptive_budget=False,
        )
        restored = torch.empty_like(permuted.probabilities)
        restored[:, perm] = permuted.probabilities
        perm_error = float(
            (out.probabilities - restored).abs().max().item()
        )
        perm_errors.append(perm_error)

        canonical_argmax = int(out.probabilities.argmax(-1).item())
        restored_argmax = int(restored.argmax(-1).item())
        argmax_invariant += int(canonical_argmax == restored_argmax)

        finite_pass = finite_pass and bool(
            torch.isfinite(out.logits).all()
            and torch.isfinite(out.probabilities).all()
            and torch.isfinite(permuted.logits).all()
            and torch.isfinite(permuted.probabilities).all()
        )
        budget_values.append(int(out.candidate_budget.item()))
        tail_values.append(float(out.tail_mass.item()))
        mass_errors.append(abs(float(out.probabilities.sum().item()) - 1.0))

    probability_mass_max_error = max(mass_errors)
    permutation_max_error = max(perm_errors)
    repeatability_max_error = max(repeat_errors)
    tail_mass_max = max(abs(v) for v in tail_values)
    budget_min = min(budget_values)
    budget_max = max(budget_values)

    mechanics_pass = bool(
        finite_pass
        and budget_min == k
        and budget_max == k
        and tail_mass_max == 0.0
        and probability_mass_max_error <= MASS_TOL
        and permutation_max_error <= PERM_TOL
        and argmax_invariant == cases
        and repeatability_max_error <= REPEAT_TOL
    )
    return MechanicalResult(
        k=k,
        case_count=cases,
        finite_pass=finite_pass,
        budget_min=budget_min,
        budget_max=budget_max,
        tail_mass_max=tail_mass_max,
        probability_mass_max_error=probability_mass_max_error,
        permutation_max_error=permutation_max_error,
        argmax_invariant_count=argmax_invariant,
        repeatability_max_error=repeatability_max_error,
        p50_ms_cpu=_percentile(latencies, 0.50),
        p95_ms_cpu=_percentile(latencies, 0.95),
        option_tensor_bytes_per_case=k * D_MODEL * 4,
        mechanics_pass=mechanics_pass,
    )


def _phrase_space() -> list[str]:
    return [
        f"{color} {obj} {action} {location}"
        for color in COLORS
        for obj in OBJECTS
        for action in ACTIONS
        for location in LOCATIONS
    ]


def generate_semantic_key_case(
    *,
    k: int,
    case_index: int,
) -> tuple[str, str, tuple[LogicalOption, ...], int]:
    if k not in K_VALUES:
        raise ValueError(f"unsupported frozen K: {k}")
    if not 0 <= case_index < SEMANTIC_CASES:
        raise ValueError("semantic case_index outside frozen range")

    rng = random.Random(SEMANTIC_SEEDS[k] + case_index)
    space = _phrase_space()
    indices = rng.sample(range(len(space)), k)
    phrases = [space[i] for i in indices]
    correct_index = rng.randrange(k)
    correct_phrase = phrases[correct_index]

    state_text = json.dumps(
        {"routing_key": correct_phrase},
        sort_keys=True,
        separators=(",", ":"),
    )
    question = "Which option exactly matches the routing_key?"
    options = tuple(
        LogicalOption(
            option_id=f"route-{i:03d}",
            criterion_text=phrase,
        )
        for i, phrase in enumerate(phrases)
    )
    return state_text, question, options, correct_index


@torch.inference_mode()
def evaluate_semantic_key_stress(
    model: NolaneHira,
    *,
    k: int,
    cases: int = SEMANTIC_CASES,
) -> dict[str, float | int]:
    if cases != SEMANTIC_CASES:
        raise ValueError("semantic case count is frozen at 128")
    model.eval()
    before = model.state_encode_calls
    correct = 0
    top5 = 0
    reciprocal_ranks: list[float] = []
    correct_probs: list[float] = []
    confidences: list[float] = []
    mass_errors: list[float] = []

    for case_index in range(cases):
        state_text, question, options, gold_index = generate_semantic_key_case(
            k=k,
            case_index=case_index,
        )
        memory = model.compile_state(state_text, segment_tokens=32)
        schema, _ = model.compile_schema(
            primitive="choice",
            question_text=question,
            options=options,
            use_cache=False,
        )
        output = model.forward_compiled(
            memory,
            schema,
            forced_budget=FORCED_BUDGET,
            adaptive_budget=False,
        )
        p = output.probabilities.detach().cpu()
        order = torch.argsort(p, descending=True)
        predicted = int(order[0].item())
        correct += int(predicted == gold_index)
        top5 += int(gold_index in set(order[:5].tolist()))
        rank = int((order == gold_index).nonzero(as_tuple=False)[0].item()) + 1
        reciprocal_ranks.append(1.0 / rank)
        correct_probs.append(float(p[gold_index].item()))
        confidences.append(float(p.max().item()))
        mass_errors.append(abs(float(p.sum().item()) - 1.0))

    state_calls = model.state_encode_calls - before
    if state_calls != cases:
        raise RuntimeError("semantic-key stress violated state-once authority")

    return {
        "k": k,
        "case_count": cases,
        "accuracy": correct / cases,
        "top5_recall": top5 / cases,
        "mrr": statistics.fmean(reciprocal_ranks),
        "mean_correct_option_probability": statistics.fmean(correct_probs),
        "mean_confidence": statistics.fmean(confidences),
        "probability_mass_max_error": max(mass_errors),
        "state_encode_calls": state_calls,
        "state_encode_calls_per_case": state_calls / cases,
        "candidate_budget": k,
    }
