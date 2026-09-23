from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor, nn

from .contracts import CompiledSchema, LogicalOption, Primitive, StateMemory
from .hira import HIRACore, HIRAOutput
from .schema import SchemaCompiler, SchemaCompileReceipt
from .semantic import TextSemanticEncoder

PRIMITIVE_TO_ID: dict[Primitive, int] = {"choice": 0, "score": 1, "noul": 2}
RelationMode = Literal["pooled", "option_tokens", "state_tokens", "dual_tokens"]
RELATION_MODES: tuple[RelationMode, ...] = (
    "pooled",
    "option_tokens",
    "state_tokens",
    "dual_tokens",
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

    def __init__(self, encoder: TextSemanticEncoder, hira: HIRACore | None = None):
        super().__init__()
        self.encoder = encoder
        self.hira = hira or HIRACore(d_model=encoder.d_model)
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

    def forward_compiled(
        self,
        memory: StateMemory,
        schema: CompiledSchema,
        *,
        forced_budget: int | None = None,
        adaptive_budget: bool = False,
        relation_mode: RelationMode = "pooled",
    ) -> DecisionOutput:
        if memory.model_hash != schema.encoder_hash:
            raise ValueError("state/schema encoder hash mismatch")
        if relation_mode not in RELATION_MODES:
            raise ValueError(f"unsupported relation_mode: {relation_mode}")
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
        out = self.hira(
            schema.question_embedding.unsqueeze(0),
            segments,
            schema.option_embeddings.unsqueeze(0),
            qtype,
            segment_mask=segment_mask,
            option_tokens=option_tokens,
            option_token_mask=option_token_mask,
            forced_budget=forced_budget,
            adaptive_budget=adaptive_budget,
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
    ) -> DecisionOutput:
        memory = self.compile_state(state_text)
        needs_option_tokens = relation_mode in {
            "option_tokens",
            "dual_tokens",
        }
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
        )
