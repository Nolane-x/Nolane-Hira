from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .contracts import CompiledSchema, LogicalOption, Primitive, StateMemory
from .hira import HIRACore, HIRAOutput
from .schema import SchemaCompiler, SchemaCompileReceipt
from .semantic import TextSemanticEncoder

PRIMITIVE_TO_ID: dict[Primitive, int] = {"choice": 0, "score": 1, "noul": 2}


@dataclass
class DecisionOutput:
    primitive: Primitive
    probabilities: Tensor
    logits: Tensor
    value: Tensor
    hira: HIRAOutput


class NolaneHira(nn.Module):
    """End-to-end trainable HIRA model with state-once semantics."""

    def __init__(self, encoder: TextSemanticEncoder, hira: HIRACore | None = None):
        super().__init__()
        self.encoder = encoder
        self.hira = hira or HIRACore(d_model=encoder.d_model)
        self.schema_compiler = SchemaCompiler(encoder)

    def compile_state(self, text: str, *, segment_tokens: int = 32) -> StateMemory:
        return self.encoder.encode_state(text, segment_tokens=segment_tokens)

    def compile_schema(
        self, *, primitive: Primitive, question_text: str,
        options: Iterable[LogicalOption], use_cache: bool = True
    ) -> tuple[CompiledSchema, SchemaCompileReceipt]:
        return self.schema_compiler.compile(
            primitive=primitive, question_text=question_text,
            options=options, use_cache=use_cache
        )

    @staticmethod
    def _batch_state(memory: StateMemory) -> tuple[Tensor, Tensor]:
        segments = memory.segment_embeddings
        if segments.ndim != 2:
            raise ValueError("StateMemory.segment_embeddings must be [S,D]")
        return segments.unsqueeze(0), torch.ones(
            1, segments.shape[0], dtype=torch.bool, device=segments.device
        )

    def forward_compiled(
        self, memory: StateMemory, schema: CompiledSchema, *,
        forced_budget: int | None = None, adaptive_budget: bool = False
    ) -> DecisionOutput:
        if memory.model_hash != schema.encoder_hash:
            raise ValueError("state/schema encoder hash mismatch")
        primitive = schema.primitive
        qtype = torch.tensor(
            [PRIMITIVE_TO_ID[primitive]], dtype=torch.long,
            device=schema.question_embedding.device
        )
        segments, segment_mask = self._batch_state(memory)
        out = self.hira(
            schema.question_embedding.unsqueeze(0),
            segments,
            schema.option_embeddings.unsqueeze(0),
            qtype,
            segment_mask=segment_mask,
            forced_budget=forced_budget,
            adaptive_budget=adaptive_budget,
        )
        p = out.probabilities[0]
        if primitive == "choice":
            value = p.argmax().to(p.dtype)
        elif primitive == "score":
            support = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
            value = (p * support).sum()
        elif primitive == "noul":
            if p.shape[-1] != 2:
                raise ValueError("noul requires exactly two logical options: false/true")
            value = p[1]
        else:
            raise ValueError(f"unsupported primitive: {primitive}")
        return DecisionOutput(primitive, p, out.logits[0], value, out)

    def decide_text(
        self, state_text: str, *, primitive: Primitive,
        question_text: str, options: Iterable[LogicalOption],
        forced_budget: int | None = None, use_schema_cache: bool = True
    ) -> DecisionOutput:
        memory = self.compile_state(state_text)
        schema, _ = self.compile_schema(
            primitive=primitive, question_text=question_text,
            options=options, use_cache=use_schema_cache
        )
        return self.forward_compiled(memory, schema, forced_budget=forced_budget)
