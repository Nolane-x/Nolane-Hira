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


@dataclass(frozen=True)
class LogicalOption:
    option_id: str
    criterion_text: str
    aliases: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = ()
    counterexamples: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledSchema:
    schema_hash: str
    encoder_hash: str
    primitive: Primitive
    question_text: str
    options: tuple[LogicalOption, ...]
    question_embedding: torch.Tensor
    option_embeddings: torch.Tensor
