from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PrivateCorrectionRepresentationFork(nn.Module):
    """S44 post-encoder private correction representation.

    Native signatures, native logits, and question-token inputs are detached
    internally. Only A/B/W belong to the correction branch.
    """

    def __init__(
        self,
        *,
        native_dimension: int = 256,
        hidden_dimension: int = 64,
        query_norm_epsilon: float = 1e-12,
        private_norm_epsilon: float = 1e-12,
        residual_scale: float = 1.0,
        adapter_seed: int = 65044,
        train_correction: bool = True,
    ):
        super().__init__()
        if native_dimension != 256:
            raise ValueError("S44 native dimension is frozen at 256")
        if hidden_dimension != 64:
            raise ValueError("S44 hidden dimension is frozen at 64")
        if query_norm_epsilon != 1e-12:
            raise ValueError("S44 query norm epsilon is frozen at 1e-12")
        if private_norm_epsilon != 1e-12:
            raise ValueError("S44 private norm epsilon is frozen at 1e-12")
        if residual_scale != 1.0:
            raise ValueError("S44 residual scale is frozen at 1.0")
        if adapter_seed != 65044:
            raise ValueError("S44 adapter seed is frozen at 65044")

        self.native_dimension = int(native_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.query_norm_epsilon = float(query_norm_epsilon)
        self.private_norm_epsilon = float(private_norm_epsilon)
        self.residual_scale = float(residual_scale)
        self.adapter_seed = int(adapter_seed)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.adapter_seed)
        std = math.sqrt(2.0 / (2 * self.native_dimension))
        a = torch.randn(
            self.hidden_dimension,
            2 * self.native_dimension,
            generator=generator,
            dtype=torch.float32,
        ) * std

        self.adapter_a = nn.Parameter(a, requires_grad=bool(train_correction))
        self.adapter_b = nn.Parameter(
            torch.zeros(
                self.native_dimension,
                self.hidden_dimension,
                dtype=torch.float32,
            ),
            requires_grad=bool(train_correction),
        )
        self.bilinear_weight = nn.Parameter(
            torch.zeros(
                self.native_dimension,
                self.native_dimension,
                dtype=torch.float32,
            ),
            requires_grad=bool(train_correction),
        )

    @property
    def adapter_a_parameter_count(self) -> int:
        return int(self.adapter_a.numel())

    @property
    def adapter_b_parameter_count(self) -> int:
        return int(self.adapter_b.numel())

    @property
    def private_adapter_parameter_count(self) -> int:
        return self.adapter_a_parameter_count + self.adapter_b_parameter_count

    @property
    def bilinear_parameter_count(self) -> int:
        return int(self.bilinear_weight.numel())

    @property
    def correction_parameter_count(self) -> int:
        return self.private_adapter_parameter_count + self.bilinear_parameter_count

    def correction_parameters(self) -> list[nn.Parameter]:
        return [self.adapter_a, self.adapter_b, self.bilinear_weight]

    def query_summary(
        self,
        *,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        if question_tokens.ndim != 3:
            raise ValueError("S44 question_tokens must be [B,Q,256]")
        if question_tokens.shape[-1] != self.native_dimension:
            raise ValueError("S44 question dimension changed")
        if question_mask.shape != question_tokens.shape[:2] or question_mask.dtype != torch.bool:
            raise ValueError("S44 question_mask mismatch")
        if bool((question_mask.sum(-1) < 1).any()):
            raise ValueError("S44 requires question content")

        tokens = question_tokens.detach()
        mask = question_mask.to(tokens.dtype)
        count = mask.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (tokens * mask[..., None]).sum(dim=1) / count
        query = F.normalize(mean, dim=-1, eps=self.query_norm_epsilon)
        if not bool(torch.isfinite(query).all()):
            raise ValueError("S44 query summary produced non-finite values")
        return query

    def private_residual(
        self,
        *,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if signatures.ndim != 3 or signatures.shape[-1] != self.native_dimension:
            raise ValueError("S44 signatures must be [B,K,256]")
        base = signatures.detach()
        query = self.query_summary(
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        if query.shape[0] != base.shape[0]:
            raise ValueError("S44 signature/query batch mismatch")

        expanded_query = query[:, None, :].expand(-1, base.shape[1], -1)
        x = torch.cat([base, expanded_query], dim=-1)

        a = self.adapter_a.to(dtype=base.dtype)
        b = self.adapter_b.to(dtype=base.dtype)
        hidden = F.gelu(F.linear(x, a, bias=None))
        residual = F.linear(hidden, b, bias=None)
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("S44 private residual produced non-finite values")
        return residual, query

    def private_signatures(
        self,
        *,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual, query = self.private_residual(
            signatures=signatures,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        private = F.normalize(
            signatures.detach() + residual,
            dim=-1,
            eps=self.private_norm_epsilon,
        )
        if not bool(torch.isfinite(private).all()):
            raise ValueError("S44 private signature produced non-finite values")
        return private, query

    def correction_residual(
        self,
        *,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        private, query = self.private_signatures(
            signatures=signatures,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        weight = self.bilinear_weight.to(dtype=private.dtype)
        residual = torch.einsum("bkd,de,be->bk", private, weight, query)
        residual = self.residual_scale * residual
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("S44 correction residual produced non-finite values")
        return residual

    def correction_logits(
        self,
        *,
        native_logits: Tensor,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        if native_logits.shape != signatures.shape[:2]:
            raise ValueError("S44 native logit/signature shape mismatch")
        residual = self.correction_residual(
            signatures=signatures,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        logits = native_logits.detach() + residual
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("S44 corrected logits produced non-finite values")
        return logits

    def correction_state_dict(self) -> dict[str, Tensor]:
        return {
            "adapter.a": self.adapter_a.detach().cpu().clone(),
            "adapter.b": self.adapter_b.detach().cpu().clone(),
            "bilinear.weight": self.bilinear_weight.detach().cpu().clone(),
        }

    def load_correction_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected = {"adapter.a", "adapter.b", "bilinear.weight"}
        if set(state_dict) != expected:
            raise ValueError("S44 correction checkpoint keys changed")
        shapes = {
            "adapter.a": (self.hidden_dimension, 2 * self.native_dimension),
            "adapter.b": (self.native_dimension, self.hidden_dimension),
            "bilinear.weight": (self.native_dimension, self.native_dimension),
        }
        targets = {
            "adapter.a": self.adapter_a,
            "adapter.b": self.adapter_b,
            "bilinear.weight": self.bilinear_weight,
        }
        with torch.no_grad():
            for key in sorted(expected):
                value = state_dict[key]
                if tuple(value.shape) != shapes[key]:
                    raise ValueError(f"S44 correction shape changed: {key}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"S44 correction tensor non-finite: {key}")
                targets[key].copy_(
                    value.to(device=targets[key].device, dtype=targets[key].dtype)
                )
        for p in self.correction_parameters():
            p.requires_grad_(not freeze)


__all__ = ["PrivateCorrectionRepresentationFork"]
