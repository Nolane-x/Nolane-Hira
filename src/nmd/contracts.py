from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import torch

Primitive = Literal["choice", "score", "noul"]
Mode = Literal["D", "R", "RF"]


@dataclass(frozen=True)
class StateMemory:
    model_hash: str
    tokenizer_hash: str
    state_hash: str
    global_embedding: torch.Tensor
    segment_embeddings: torch.Tensor
    token_embeddings: Optional[torch.Tensor] = None
    content_token_embeddings: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class LogicalOption:
    option_id: str
    criterion_text: str
    aliases: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = ()
    counterexamples: tuple[str, ...] = ()
    value: float | None = None


@dataclass(frozen=True)
class CompiledSchema:
    schema_hash: str
    encoder_hash: str
    primitive: Primitive
    question_text: str
    options: tuple[LogicalOption, ...]
    question_embedding: torch.Tensor
    option_embeddings: torch.Tensor
    option_token_embeddings: torch.Tensor | None = None
    option_token_mask: torch.Tensor | None = None
    question_token_embeddings: torch.Tensor | None = None
    question_token_mask: torch.Tensor | None = None
    question_content_token_mask: torch.Tensor | None = None
    option_token_ids: torch.Tensor | None = None
    option_content_token_mask: torch.Tensor | None = None
