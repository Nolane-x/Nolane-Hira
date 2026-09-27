from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import torch
from torch import nn

from .contracts import CompiledSchema, LogicalOption, Primitive, StateMemory
from .hira import HIRACore
from .runtime import DecisionOutput, NolaneHira
from .semantic import TextSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
if TYPE_CHECKING:
    from .mainline_reliability import HiraV0ReliabilityPolicy, HiraV0ReliableDecision

from .w34_transfer_core import (
    W34_CANDIDATE_PARAMETER_COUNT,
    build_hira_v0_w34_core,
)

HIRA_V0_MAINLINE_VERSION = "0.0-m0"
HIRA_V0_MAINLINE_M1_VERSION = "0.0-m1a"
HIRA_V0_MAINLINE_M2_VERSION = "0.0-m2a"
HIRA_V0_MAINLINE_M3_VERSION = "0.0-m3a"
HIRA_V0_MAINLINE_M4_VERSION = "0.0-m4a"
HIRA_V0_MAX_K = 255
M2_MECHANICS_AUTHORITY = (
    "run:36310118240;artifact:10928771833;"
    "digest:sha256:99e8c02aeb48fa32910a591d71b92e3e6f96258c0fa8501f38682ea4a2b7f453"
)
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
    high_k_mechanics: ModuleMaturity
    high_k_mechanics_authority: str | None
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
            high_k_mechanics="pending",
            high_k_mechanics_authority=None,
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

    @classmethod
    def m1_mechanism_provisional(cls) -> "HiraV0Manifest":
        base = cls.m0_provisional().to_dict()
        base["version"] = HIRA_V0_MAINLINE_M1_VERSION
        base["reliability_ood_abstention"] = "provisional"
        return cls(**base)

    @classmethod
    def m2_mechanics_provisional(cls) -> "HiraV0Manifest":
        base = cls.m1_mechanism_provisional().to_dict()
        base["version"] = HIRA_V0_MAINLINE_M2_VERSION
        base["high_k"] = "provisional"
        base["high_k_mechanics"] = "provisional"
        return cls(**base)

    @classmethod
    def m2_mechanics_available(cls) -> "HiraV0Manifest":
        base = cls.m2_mechanics_provisional().to_dict()
        base["high_k_mechanics"] = "available"
        base["high_k_mechanics_authority"] = M2_MECHANICS_AUTHORITY
        return cls(**base)

    @classmethod
    def m3_multilingual_provisional(cls) -> "HiraV0Manifest":
        base = cls.m2_mechanics_available().to_dict()
        base["version"] = HIRA_V0_MAINLINE_M3_VERSION
        base["multilingual"] = "provisional"
        return cls(**base)

    @classmethod
    def m4_runtime_provisional(cls) -> "HiraV0Manifest":
        base = cls.m3_multilingual_provisional().to_dict()
        base["version"] = HIRA_V0_MAINLINE_M4_VERSION
        return cls(**base)

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

    @staticmethod
    def _validate_option_count(count: int) -> None:
        if count < 2:
            raise ValueError("Hira v0 mainline requires at least two logical options")
        if count > HIRA_V0_MAX_K:
            raise ValueError(
                f"Hira v0 supports at most {HIRA_V0_MAX_K} logical options"
            )

    def decide_compiled(self, schema: CompiledSchema) -> DecisionOutput:
        """Execute an already-compiled dynamic schema against this state."""
        self._validate_option_count(len(schema.options))
        runtime = self._mainline.runtime
        before = runtime.state_encode_calls
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
        if int(out.hira.candidate_budget.item()) != len(schema.options):
            raise RuntimeError("Hira v0 mainline must evaluate the full schema")
        if not bool(out.hira.selected_mask.all()):
            raise RuntimeError("Hira v0 full-K selected mask changed")
        if not torch.equal(
            out.hira.relation_delta,
            torch.zeros_like(out.hira.relation_delta),
        ):
            raise RuntimeError("Hira v0 relation refinement must remain disabled")
        if not bool(
            torch.isfinite(out.logits).all()
            and torch.isfinite(out.probabilities).all()
        ):
            raise RuntimeError("Hira v0 produced non-finite decision tensors")
        if abs(float(out.probabilities.sum()) - 1.0) > 1e-6:
            raise RuntimeError("Hira v0 probability mass changed")
        self.query_count += 1
        return out

    def decide(
        self,
        *,
        primitive: Primitive,
        question_text: str,
        options: Iterable[LogicalOption],
        use_schema_cache: bool = True,
    ) -> DecisionOutput:
        logical_options = tuple(options)
        self._validate_option_count(len(logical_options))

        schema, _ = self._mainline.runtime.compile_schema(
            primitive=primitive,
            question_text=question_text,
            options=logical_options,
            use_cache=use_schema_cache,
            include_token_artifacts=True,
        )
        return self.decide_compiled(schema)

    def decide_reliable(
        self,
        *,
        primitive: Primitive,
        question_text: str,
        options: Iterable[LogicalOption],
        use_schema_cache: bool = True,
        probabilities_calibrated: bool = False,
        ood_score: float | None = None,
        distribution_shift: bool = False,
        ood_calibrator_valid: bool = True,
    ) -> "HiraV0ReliableDecision":
        from .mainline_reliability import HiraV0ReliabilityPolicy

        decision = self.decide(
            primitive=primitive,
            question_text=question_text,
            options=options,
            use_schema_cache=use_schema_cache,
        )
        policy = self._mainline.reliability_policy
        if policy is None:
            policy = HiraV0ReliabilityPolicy.m1_mechanism_fail_closed()
        return policy.evaluate(
            decision,
            probabilities_calibrated=probabilities_calibrated,
            ood_score=ood_score,
            distribution_shift=distribution_shift,
            ood_calibrator_valid=ood_calibrator_valid,
        )


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
        reliability_policy: "HiraV0ReliabilityPolicy | None" = None,
        freeze_runtime: bool = True,
    ):
        super().__init__()
        self.runtime = runtime
        self.manifest = manifest or HiraV0Manifest.m0_provisional()
        self.reliability_policy = reliability_policy

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
        if (
            self.manifest.reliability_ood_abstention == "available"
            and self.reliability_policy is None
        ):
            raise RuntimeError(
                "available mainline reliability requires an explicit policy"
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

    def schema_cache_info(self) -> dict[str, int]:
        return self.runtime.schema_cache_info()

    def configure_schema_cache(
        self,
        *,
        max_entries: int,
        max_bytes: int,
        clear: bool = True,
    ) -> None:
        self.runtime.configure_schema_cache(
            max_entries=max_entries,
            max_bytes=max_bytes,
            clear=clear,
        )

    def clear_schema_cache(self) -> None:
        self.runtime.clear_schema_cache()

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


def build_hira_v0_m1_mechanism(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    transfer_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_transfer_sha256: str = W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    reliability_policy: "HiraV0ReliabilityPolicy | None" = None,
) -> HiraV0Mainline:
    """Build M1-A with explicit fail-closed reliability mechanism only."""
    from .mainline_reliability import HiraV0ReliabilityPolicy

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
    return HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m1_mechanism_provisional(),
        reliability_policy=(
            reliability_policy
            or HiraV0ReliabilityPolicy.m1_mechanism_fail_closed()
        ),
    )


def build_hira_v0_m2_mechanics(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    transfer_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_transfer_sha256: str = W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    reliability_policy: "HiraV0ReliabilityPolicy | None" = None,
) -> HiraV0Mainline:
    """Build the M2-A frozen mainline with explicit K<=255 mechanics."""
    from .mainline_reliability import HiraV0ReliabilityPolicy

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
    return HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m2_mechanics_available(),
        reliability_policy=(
            reliability_policy
            or HiraV0ReliabilityPolicy.m1_mechanism_fail_closed()
        ),
    )


def build_hira_v0_m3_baseline(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    transfer_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_transfer_sha256: str = W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    reliability_policy: "HiraV0ReliabilityPolicy | None" = None,
) -> HiraV0Mainline:
    """Build the M3-A zero-training multilingual baseline.

    M3-A intentionally keeps the current semantic front-end and every trained
    Hira weight frozen. It measures EN/VI transfer before any multilingual
    rescue is allowed.
    """
    from .mainline_reliability import HiraV0ReliabilityPolicy

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
    return HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m3_multilingual_provisional(),
        reliability_policy=(
            reliability_policy
            or HiraV0ReliabilityPolicy.m1_mechanism_fail_closed()
        ),
    )


def build_hira_v0_m4_runtime(
    encoder: TextSemanticEncoder,
    t0_checkpoint_path: str | Path,
    transfer_checkpoint_path: str | Path,
    *,
    expected_t0_sha256: str = W28_T0_CHECKPOINT_SHA256,
    expected_transfer_sha256: str = W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256,
    reliability_policy: "HiraV0ReliabilityPolicy | None" = None,
    schema_cache_max_entries: int | None = None,
    schema_cache_max_bytes: int | None = None,
) -> HiraV0Mainline:
    """Build the M4 runtime-optimized package without maturity promotion."""
    from .mainline_reliability import HiraV0ReliabilityPolicy

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
    model = HiraV0Mainline(
        runtime,
        manifest=HiraV0Manifest.m4_runtime_provisional(),
        reliability_policy=(
            reliability_policy
            or HiraV0ReliabilityPolicy.m1_mechanism_fail_closed()
        ),
    )
    if schema_cache_max_entries is not None or schema_cache_max_bytes is not None:
        current = model.schema_cache_info()
        model.configure_schema_cache(
            max_entries=(
                current["max_entries"]
                if schema_cache_max_entries is None
                else int(schema_cache_max_entries)
            ),
            max_bytes=(
                current["max_bytes"]
                if schema_cache_max_bytes is None
                else int(schema_cache_max_bytes)
            ),
            clear=True,
        )
    return model


__all__ = [
    "HIRA_V0_MAINLINE_VERSION",
    "HIRA_V0_MAINLINE_M1_VERSION",
    "HIRA_V0_MAINLINE_M2_VERSION",
    "HIRA_V0_MAINLINE_M3_VERSION",
    "HIRA_V0_MAINLINE_M4_VERSION",
    "HIRA_V0_MAX_K",
    "M2_MECHANICS_AUTHORITY",
    "W34_PROVISIONAL_TRANSFER_CHECKPOINT_SHA256",
    "HiraV0Mainline",
    "HiraV0Manifest",
    "HiraV0ParameterReport",
    "HiraV0Session",
    "build_hira_v0_mainline",
    "build_hira_v0_m1_mechanism",
    "build_hira_v0_m2_mechanics",
    "build_hira_v0_m3_baseline",
    "build_hira_v0_m4_runtime",
]
