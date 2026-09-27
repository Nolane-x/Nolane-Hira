from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
from torch import nn

from .contracts import LogicalOption, Primitive, StateMemory
from .hira import HIRACore
from .runtime import DecisionOutput, NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .w34_transfer_core import (
    W34_CANDIDATE_PARAMETER_COUNT,
    build_hira_v0_w34_core,
)

HIRA_V0_MAINLINE_VERSION = "0.0-m0"
W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256 = (
    "d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c"
)

ModuleMaturity = Literal[
    "available",
    "provisional",
    "frozen_research_base",
    "pending",
]


@dataclass(frozen=True)
class HiraV0Manifest:
    version: str
    semantic_frontend: ModuleMaturity
    projection: ModuleMaturity
    transfer_core: ModuleMaturity
    typed_runtime: ModuleMaturity
    reliability_ood_abstention: ModuleMaturity
    high_k: ModuleMaturity
    multilingual: ModuleMaturity
    production_ready: bool
    transfer_core_promoted: bool
    default_coarse_mode: str
    relation_refinement: bool
    adaptive_budget: bool
    t0_checkpoint_sha256: str
    transfer_checkpoint_sha256: str
    transfer_candidate_parameter_count: int
    transfer_authority: str

    @classmethod
    def m0_provisional(cls) -> "HiraV0Manifest":
        return cls(
            version=HIRA_V0_MAINLINE_VERSION,
            semantic_frontend="provisional",
            projection="frozen_research_base",
            transfer_core="provisional",
            typed_runtime="available",
            reliability_ood_abstention="pending",
            high_k="pending",
            multilingual="pending",
            production_ready=False,
            transfer_core_promoted=False,
            default_coarse_mode="coevidence_symmetric_semantic",
            relation_refinement=False,
            adaptive_budget=False,
            t0_checkpoint_sha256=W28_T0_CHECKPOINT_SHA256,
            transfer_checkpoint_sha256=W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
            transfer_candidate_parameter_count=W34_CANDIDATE_PARAMETER_COUNT,
            transfer_authority="R8-W34:W34_COEVIDENCE_COMPOSITION_FAIL:PROVISIONAL_TRANSFER_CORE",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HiraV0ParameterReport:
    resident_total: int
    trainable_total: int
    semantic_frontend_resident: int
    relation_core_resident: int
    transfer_scorer_resident: int
    projection_parameters: int
    transfer_candidate_parameters: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class HiraV0Session:
    """One compiled state serving many dynamic typed queries."""

    def __init__(self, mainline: "HiraV0Mainline", memory: StateMemory):
        self._mainline = mainline
        self.memory = memory
        self.query_count = 0

    @property
    def state_hash(self) -> str:
        return self.memory.state_hash

    def decide(
        self,
        *,
        primitive: Primitive,
        question_text: str,
        options: Iterable[LogicalOption],
        use_schema_cache: bool = True,
    ) -> DecisionOutput:
        logical_options = tuple(options)
        if len(logical_options) < 2:
            raise ValueError("Hira v0 mainline requires at least two logical options")

        runtime = self._mainline.runtime
        before = runtime.state_encode_calls
        schema, _ = runtime.compile_schema(
            primitive=primitive,
            question_text=question_text,
            options=logical_options,
            use_cache=use_schema_cache,
            include_token_artifacts=True,
        )
        out = runtime.forward_compiled(
            self.memory,
            schema,
            forced_budget=None,
            adaptive_budget=False,
            relation_mode="pooled",
            coarse_mode=self._mainline.manifest.default_coarse_mode,
            relation_refinement=False,
        )
        if runtime.state_encode_calls != before:
            raise RuntimeError("Hira v0 session re-encoded state during a query")
        if int(out.hira.candidate_budget.item()) != len(logical_options):
            raise RuntimeError("Hira v0 M0 must evaluate the full schema")
        if not bool(out.hira.selected_mask.all()):
            raise RuntimeError("Hira v0 M0 full-K selected mask changed")
        if not torch.equal(
            out.hira.relation_delta,
            torch.zeros_like(out.hira.relation_delta),
        ):
            raise RuntimeError("Hira v0 M0 relation refinement must remain disabled")
        if not bool(torch.isfinite(out.probabilities).all()):
            raise RuntimeError("Hira v0 M0 produced non-finite probabilities")
        if abs(float(out.probabilities.sum()) - 1.0) > 1e-6:
            raise RuntimeError("Hira v0 M0 probability mass changed")
        self.query_count += 1
        return out


class HiraV0Mainline(nn.Module):
    """Integrated Hira v0 M0 inference shell.

    M0 deliberately freezes the current runtime. Training/calibration pipelines
    are introduced in later mainline phases rather than silently mutating this
    provisional research package.
    """

    def __init__(
        self,
        runtime: NolaneHira,
        *,
        manifest: HiraV0Manifest | None = None,
        freeze_runtime: bool = True,
    ):
        super().__init__()
        self.runtime = runtime
        self.manifest = manifest or HiraV0Manifest.m0_provisional()

        if freeze_runtime:
            for parameter in self.runtime.parameters():
                parameter.requires_grad_(False)
        self.eval()
        self._assert_m0_invariants()

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError(
                "Hira v0 M0 is a frozen inference shell; "
                "use a qualified later-phase training pipeline"
            )
        return super().train(False)

    def _assert_m0_invariants(self) -> None:
        if int(self.runtime.encoder.d_model) != 256:
            raise RuntimeError("Hira v0 M0 requires d_model=256")
        scorer = self.runtime.coevidence_symmetric_semantic_scorer
        if scorer is None:
            raise RuntimeError("Hira v0 M0 requires the provisional co-evidence core")
        if scorer.candidate_parameter_count != W34_CANDIDATE_PARAMETER_COUNT:
            raise RuntimeError("Hira v0 M0 transfer candidate capacity changed")
        if scorer.trainable_parameter_count != 0:
            raise RuntimeError("Hira v0 M0 packaged transfer scorer must be frozen")
        if scorer.projection.weight.requires_grad:
            raise RuntimeError("Hira v0 M0 W28 projection must be frozen")
        if self.runtime.reliability_calibrator is not None:
            raise RuntimeError(
                "Hira v0 M0 reliability calibration is not mainline-qualified"
            )
        if self.manifest.production_ready:
            raise RuntimeError("Hira v0 M0 cannot claim production readiness")
        if self.manifest.transfer_core_promoted:
            raise RuntimeError("W34 transfer core was not production-promoted")
        if self.manifest.transfer_core != "provisional":
            raise RuntimeError("Hira v0 M0 transfer core must remain provisional")
        if self.manifest.default_coarse_mode != "coevidence_symmetric_semantic":
            raise RuntimeError("Hira v0 M0 default coarse mode changed")
        if self.manifest.relation_refinement:
            raise RuntimeError("Hira v0 M0 relation refinement must default off")
        if self.manifest.adaptive_budget:
            raise RuntimeError("Hira v0 M0 adaptive budget must default off")

    def open_session(
        self,
        state_text: str,
        *,
        segment_tokens: int = 32,
    ) -> HiraV0Session:
        before = self.runtime.state_encode_calls
        memory = self.runtime.compile_state(
            state_text,
            segment_tokens=segment_tokens,
        )
        if self.runtime.state_encode_calls != before + 1:
            raise RuntimeError("Hira v0 mainline state-once contract changed")
        return HiraV0Session(self, memory)

    def decide_text(
        self,
        state_text: str,
        *,
        primitive: Primitive,
        question_text: str,
        options: Iterable[LogicalOption],
        use_schema_cache: bool = True,
    ) -> DecisionOutput:
        return self.open_session(state_text).decide(
            primitive=primitive,
            question_text=question_text,
            options=options,
            use_schema_cache=use_schema_cache,
        )

    def parameter_report(self) -> HiraV0ParameterReport:
        scorer = self.runtime.coevidence_symmetric_semantic_scorer
        if scorer is None:
            raise RuntimeError("Hira v0 M0 transfer scorer missing")

        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        return HiraV0ParameterReport(
            resident_total=count(self.runtime),
            trainable_total=sum(
                parameter.numel()
                for parameter in self.runtime.parameters()
                if parameter.requires_grad
            ),
            semantic_frontend_resident=count(self.runtime.encoder),
            relation_core_resident=count(self.runtime.hira),
            transfer_scorer_resident=count(scorer),
            projection_parameters=int(scorer.projection.weight.numel()),
            transfer_candidate_parameters=int(scorer.candidate_parameter_count),
        )


def build_hira_v0_mainline(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    transfer_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_transfer_sha256: str = W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
) -> HiraV0Mainline:
    """Bind exact R8 provenance into the first integrated Hira v0 shell."""
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()

    runtime = build_hira_v0_w34_core(
        encoder,
        t0_checkpoint_path,
        transfer_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        expected_candidate_sha256=expected_transfer_sha256,
        hira=HIRACore(d_model=256, dropout=0.0),
        include_unbridged_baseline=False,
    )
    return HiraV0Mainline(runtime, manifest=HiraV0Manifest.m0_provisional())


__all__ = [
    "HIRA_V0_MAINLINE_VERSION",
    "W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256",
    "HiraV0Mainline",
    "HiraV0Manifest",
    "HiraV0ParameterReport",
    "HiraV0Session",
    "build_hira_v0_mainline",
]
