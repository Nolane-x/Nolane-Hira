from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor, nn

from .calibration import TypedReliabilityCalibrator
from .competitive import CompetitiveCoarseScorer
from .contracts import CompiledSchema, LogicalOption, Primitive, StateMemory
from .hira import HIRACore, HIRAOutput
from .schema import SchemaCompiler, SchemaCompileReceipt
from .semantic import TextSemanticEncoder
from .semantic_transfer_bridge import BridgedSymmetricSemanticScorer
from .symmetric_semantic import SymmetricSemanticScorer

PRIMITIVE_TO_ID: dict[Primitive, int] = {"choice": 0, "score": 1, "noul": 2}
RelationMode = Literal["pooled", "option_tokens", "state_tokens", "dual_tokens"]
RELATION_MODES: tuple[RelationMode, ...] = (
    "pooled",
    "option_tokens",
    "state_tokens",
    "dual_tokens",
)
CoarseMode = Literal[
    "legacy",
    "competitive",
    "symmetric_semantic",
    "bridged_symmetric_semantic",
]
COARSE_MODES: tuple[CoarseMode, ...] = (
    "legacy",
    "competitive",
    "symmetric_semantic",
    "bridged_symmetric_semantic",
)
COARSE_ONLY_SEMANTIC_MODES: tuple[CoarseMode, ...] = (
    "symmetric_semantic",
    "bridged_symmetric_semantic",
)


@dataclass
class DecisionOutput:
    primitive: Primitive
    probabilities: Tensor
    logits: Tensor
    value: Tensor
    selected_option_id: str | None
    hira: HIRAOutput


class NolaneHira(nn.Module):
    """End-to-end trainable HIRA model with state-once semantics."""

    def __init__(
        self,
        encoder: TextSemanticEncoder,
        hira: HIRACore | None = None,
        coarse_scorer: CompetitiveCoarseScorer | None = None,
        symmetric_semantic_scorer: SymmetricSemanticScorer | None = None,
        bridged_symmetric_semantic_scorer: BridgedSymmetricSemanticScorer | None = None,
        reliability_calibrator: TypedReliabilityCalibrator | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.hira = hira or HIRACore(d_model=encoder.d_model)
        if (
            coarse_scorer is not None
            and coarse_scorer.d_model != encoder.d_model
        ):
            raise ValueError("competitive coarse scorer d_model mismatch")
        self.coarse_scorer = coarse_scorer
        if (
            symmetric_semantic_scorer is not None
            and symmetric_semantic_scorer.d_model != encoder.d_model
        ):
            raise ValueError("symmetric semantic scorer d_model mismatch")
        self.symmetric_semantic_scorer = symmetric_semantic_scorer
        if (
            bridged_symmetric_semantic_scorer is not None
            and bridged_symmetric_semantic_scorer.d_model != encoder.d_model
        ):
            raise ValueError("bridged symmetric semantic scorer d_model mismatch")
        self.bridged_symmetric_semantic_scorer = bridged_symmetric_semantic_scorer
        self.reliability_calibrator = reliability_calibrator
        self.schema_compiler = SchemaCompiler(encoder)
        self.state_encode_calls = 0

    def train(self, mode: bool = True):
        if mode:
            self.schema_compiler.clear()
        return super().train(mode)

    def compile_state(self, text: str, *, segment_tokens: int = 32) -> StateMemory:
        self.state_encode_calls += 1
        return self.encoder.encode_state(text, segment_tokens=segment_tokens)

    def compile_schema(
        self, *, primitive: Primitive, question_text: str,
        options: Iterable[LogicalOption], use_cache: bool = True,
        include_token_artifacts: bool = False,
    ) -> tuple[CompiledSchema, SchemaCompileReceipt]:
        return self.schema_compiler.compile(
            primitive=primitive,
            question_text=question_text,
            options=options,
            use_cache=use_cache,
            include_token_artifacts=include_token_artifacts,
        )

    @staticmethod
    def _batch_state(
        memory: StateMemory,
        *,
        use_tokens: bool = False,
    ) -> tuple[Tensor, Tensor]:
        segments = (
            memory.token_embeddings
            if use_tokens
            else memory.segment_embeddings
        )
        if segments is None:
            raise ValueError(
                "requested state token relation mode but "
                "StateMemory.token_embeddings is unavailable"
            )
        if segments.ndim != 2:
            raise ValueError("state relation memory must be [S,D]")
        return segments.unsqueeze(0), torch.ones(
            1,
            segments.shape[0],
            dtype=torch.bool,
            device=segments.device,
        )

    def _competitive_coarse(
        self,
        memory: StateMemory,
        schema: CompiledSchema,
    ) -> Tensor:
        if self.coarse_scorer is None:
            raise ValueError(
                "competitive coarse mode requires CompetitiveCoarseScorer"
            )
        if memory.content_token_embeddings is None:
            raise ValueError(
                "competitive coarse mode requires state content tokens"
            )
        required = (
            schema.question_token_embeddings,
            schema.question_content_token_mask,
            schema.option_token_embeddings,
            schema.option_token_ids,
            schema.option_content_token_mask,
        )
        if any(value is None for value in required):
            raise ValueError(
                "competitive coarse mode requires schema token artifacts"
            )

        state_tokens = memory.content_token_embeddings.unsqueeze(0)
        state_mask = torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
            device=state_tokens.device,
        )
        question_tokens = schema.question_token_embeddings.unsqueeze(0)
        question_mask = schema.question_content_token_mask.unsqueeze(0)
        option_tokens = schema.option_token_embeddings.unsqueeze(0)
        option_token_ids = schema.option_token_ids.unsqueeze(0)
        option_mask = schema.option_content_token_mask.unsqueeze(0)
        return self.coarse_scorer(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_tokens=option_tokens,
            option_token_ids=option_token_ids,
            option_mask=option_mask,
        )


    def _symmetric_semantic_coarse(
        self,
        memory: StateMemory,
        schema: CompiledSchema,
    ) -> Tensor:
        scorer = self.symmetric_semantic_scorer
        if scorer is None:
            raise ValueError(
                "symmetric_semantic coarse mode requires SymmetricSemanticScorer"
            )
        if memory.content_token_embeddings is None:
            raise ValueError(
                "symmetric_semantic coarse mode requires state content tokens"
            )
        required = (
            schema.option_view_token_embeddings,
            schema.option_view_token_mask,
            schema.option_view_mask,
        )
        if any(value is None for value in required):
            raise ValueError(
                "symmetric_semantic coarse mode requires multi-view schema artifacts"
            )

        state_tokens = memory.content_token_embeddings.unsqueeze(0)
        state_mask = torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
            device=state_tokens.device,
        )
        return scorer(
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=schema.option_view_token_embeddings.unsqueeze(0),
            option_view_token_mask=schema.option_view_token_mask.unsqueeze(0),
            option_view_mask=schema.option_view_mask.unsqueeze(0),
        )

    def _bridged_symmetric_semantic_coarse(
        self,
        memory: StateMemory,
        schema: CompiledSchema,
    ) -> Tensor:
        scorer = self.bridged_symmetric_semantic_scorer
        if scorer is None:
            raise ValueError(
                "bridged_symmetric_semantic coarse mode requires "
                "BridgedSymmetricSemanticScorer"
            )
        if memory.content_token_embeddings is None:
            raise ValueError(
                "bridged_symmetric_semantic coarse mode requires state content tokens"
            )
        required = (
            schema.option_view_token_embeddings,
            schema.option_view_token_mask,
            schema.option_view_mask,
        )
        if any(value is None for value in required):
            raise ValueError(
                "bridged_symmetric_semantic coarse mode requires "
                "multi-view schema artifacts"
            )

        state_tokens = memory.content_token_embeddings.unsqueeze(0)
        state_mask = torch.ones(
            1,
            state_tokens.shape[1],
            dtype=torch.bool,
            device=state_tokens.device,
        )
        return scorer(
            state_tokens=state_tokens,
            state_mask=state_mask,
            option_view_tokens=schema.option_view_token_embeddings.unsqueeze(0),
            option_view_token_mask=schema.option_view_token_mask.unsqueeze(0),
            option_view_mask=schema.option_view_mask.unsqueeze(0),
        )

    def _coarse_only_output(
        self,
        memory: StateMemory,
        coarse: Tensor,
    ) -> HIRAOutput:
        if coarse.ndim != 2 or coarse.shape[0] != 1 or coarse.shape[1] < 2:
            raise ValueError("coarse-only logits must be [1,K] with K>=2")
        if not torch.isfinite(coarse).all():
            raise ValueError("coarse-only logits must be finite")

        probabilities = torch.softmax(coarse, dim=-1)
        k = coarse.shape[1]
        indices = torch.arange(
            k,
            device=coarse.device,
            dtype=torch.long,
        ).unsqueeze(0)
        selected_mask = torch.ones_like(indices, dtype=torch.bool)
        relation_delta = torch.zeros_like(coarse)
        budget = torch.tensor([k], device=coarse.device, dtype=torch.long)
        budget_logits = torch.zeros(
            1,
            len(self.hira.budget_buckets),
            device=coarse.device,
            dtype=coarse.dtype,
        )
        tail_mass = torch.zeros(1, device=coarse.device, dtype=coarse.dtype)

        segment_count = int(memory.segment_embeddings.shape[0])
        if segment_count < 1:
            raise ValueError("state memory must expose at least one segment")
        state_attention = torch.full(
            (1, segment_count),
            1.0 / float(segment_count),
            device=coarse.device,
            dtype=coarse.dtype,
        )

        return HIRAOutput(
            logits=coarse,
            probabilities=probabilities,
            coarse_logits=coarse,
            selected_indices=indices,
            selected_mask=selected_mask,
            relation_delta=relation_delta,
            candidate_budget=budget,
            budget_logits=budget_logits,
            tail_mass=tail_mass,
            state_attention=state_attention,
        )

    def _apply_reliability_calibration(
        self,
        out: HIRAOutput,
        schema: CompiledSchema,
        qtype: Tensor,
    ) -> HIRAOutput:
        calibrator = self.reliability_calibrator
        if calibrator is None:
            return out

        noul_true_mask = None
        if schema.primitive == "noul":
            raw_values = [option.value for option in schema.options]
            if (
                any(value is None for value in raw_values)
                or sorted(float(value) for value in raw_values) != [0.0, 1.0]
            ):
                raise ValueError(
                    "noul calibration requires option values exactly 0 and 1"
                )
            noul_true_mask = torch.tensor(
                [[float(value) == 1.0 for value in raw_values]],
                dtype=torch.bool,
                device=out.logits.device,
            )

        logits = calibrator(
            out.logits,
            qtype,
            noul_true_mask=noul_true_mask,
        )
        probabilities = torch.softmax(logits, dim=-1)

        reranked = torch.zeros_like(probabilities, dtype=torch.bool)
        reranked.scatter_(1, out.selected_indices, out.selected_mask)
        tail_mass = probabilities.masked_fill(reranked, 0).sum(-1)

        return HIRAOutput(
            logits=logits,
            probabilities=probabilities,
            coarse_logits=out.coarse_logits,
            selected_indices=out.selected_indices,
            selected_mask=out.selected_mask,
            relation_delta=out.relation_delta,
            candidate_budget=out.candidate_budget,
            budget_logits=out.budget_logits,
            tail_mass=tail_mass,
            state_attention=out.state_attention,
        )

    def forward_compiled(
        self,
        memory: StateMemory,
        schema: CompiledSchema,
        *,
        forced_budget: int | None = None,
        adaptive_budget: bool = False,
        relation_mode: RelationMode = "pooled",
        coarse_mode: CoarseMode = "legacy",
        relation_refinement: bool | None = None,
    ) -> DecisionOutput:
        if memory.model_hash != schema.encoder_hash:
            raise ValueError("state/schema encoder hash mismatch")
        if relation_mode not in RELATION_MODES:
            raise ValueError(f"unsupported relation_mode: {relation_mode}")
        if coarse_mode not in COARSE_MODES:
            raise ValueError(f"unsupported coarse_mode: {coarse_mode}")
        if relation_refinement is None:
            relation_refinement = coarse_mode not in COARSE_ONLY_SEMANTIC_MODES
        primitive = schema.primitive
        qtype = torch.tensor(
            [PRIMITIVE_TO_ID[primitive]], dtype=torch.long,
            device=schema.question_embedding.device
        )
        use_state_tokens = relation_mode in {"state_tokens", "dual_tokens"}
        use_option_tokens = relation_mode in {
            "option_tokens",
            "dual_tokens",
        }
        segments, segment_mask = self._batch_state(
            memory,
            use_tokens=use_state_tokens,
        )
        option_tokens = None
        option_token_mask = None
        if use_option_tokens:
            if (
                schema.option_token_embeddings is None
                or schema.option_token_mask is None
            ):
                raise ValueError(
                    "option-token relation mode requires schema token artifacts"
                )
            option_tokens = schema.option_token_embeddings.unsqueeze(0)
            option_token_mask = schema.option_token_mask.unsqueeze(0)
        coarse_override = None
        if coarse_mode == "competitive":
            coarse_override = self._competitive_coarse(memory, schema)
        elif coarse_mode == "symmetric_semantic":
            coarse_override = self._symmetric_semantic_coarse(memory, schema)
        elif coarse_mode == "bridged_symmetric_semantic":
            coarse_override = self._bridged_symmetric_semantic_coarse(memory, schema)

        if coarse_mode in COARSE_ONLY_SEMANTIC_MODES and not relation_refinement:
            out = self._coarse_only_output(memory, coarse_override)
        else:
            out = self.hira(
            schema.question_embedding.unsqueeze(0),
            segments,
            schema.option_embeddings.unsqueeze(0),
            qtype,
            segment_mask=segment_mask,
            option_tokens=option_tokens,
            option_token_mask=option_token_mask,
            coarse_override=coarse_override,
            forced_budget=forced_budget,
                adaptive_budget=adaptive_budget,
            )

        # Calibration is a post-W29 gate. The semantic-core mode preserves the
        # frozen rescued logits exactly until a calibrated layer is separately
        # qualified.
        if coarse_mode not in COARSE_ONLY_SEMANTIC_MODES:
            out = self._apply_reliability_calibration(
                out,
                schema,
                qtype,
            )
        p = out.probabilities[0]
        selected_option_id = None
        if primitive == "choice":
            selected_index = int(p.argmax().item())
            value = p.new_tensor(float(selected_index))
            selected_option_id = schema.options[selected_index].option_id
        elif primitive == "score":
            raw_values = [o.value for o in schema.options]
            if any(v is None for v in raw_values):
                raise ValueError("score requires explicit numeric value for every logical option")
            support = torch.tensor(raw_values, device=p.device, dtype=p.dtype)
            value = (p * support).sum()
        elif primitive == "noul":
            if p.shape[-1] != 2:
                raise ValueError("noul requires exactly two logical options")
            raw_values = [o.value for o in schema.options]
            if any(v is None for v in raw_values) or sorted(float(v) for v in raw_values) != [0.0, 1.0]:
                raise ValueError("noul requires option values exactly 0 and 1")
            true_index = next(i for i, v in enumerate(raw_values) if float(v) == 1.0)
            value = p[true_index]
        else:
            raise ValueError(f"unsupported primitive: {primitive}")
        return DecisionOutput(primitive, p, out.logits[0], value, selected_option_id, out)

    def decide_text(
        self,
        state_text: str,
        *,
        primitive: Primitive,
        question_text: str,
        options: Iterable[LogicalOption],
        forced_budget: int | None = None,
        use_schema_cache: bool = True,
        relation_mode: RelationMode = "pooled",
        coarse_mode: CoarseMode = "legacy",
        relation_refinement: bool | None = None,
    ) -> DecisionOutput:
        memory = self.compile_state(state_text)
        needs_option_tokens = (
            relation_mode in {"option_tokens", "dual_tokens"}
            or coarse_mode in {
                "competitive",
                "symmetric_semantic",
                "bridged_symmetric_semantic",
            }
        )
        schema, _ = self.compile_schema(
            primitive=primitive,
            question_text=question_text,
            options=options,
            use_cache=use_schema_cache,
            include_token_artifacts=needs_option_tokens,
        )
        return self.forward_compiled(
            memory,
            schema,
            forced_budget=forced_budget,
            relation_mode=relation_mode,
            coarse_mode=coarse_mode,
            relation_refinement=relation_refinement,
        )
