from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork


class QueryQuotientPrivateCorrectionFork(PrivateCorrectionRepresentationFork):
    """S48 private correction with an option-difference query quotient.

    Capacity and initialization remain exactly S44/S45. The only change is that
    the raw normalized query summary is projected through the span induced by
    centered option signatures before it reaches A/B/W.
    """

    def query_quotient(
        self,
        *,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if signatures.ndim != 3 or signatures.shape[-1] != self.native_dimension:
            raise ValueError("S48 signatures must be [B,K,256]")
        if signatures.shape[1] < 2:
            raise ValueError("S48 requires K >= 2")

        base = signatures.detach()
        raw_query = super().query_summary(
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        if raw_query.shape[0] != base.shape[0]:
            raise ValueError("S48 signature/query batch mismatch")

        centered = base - base.mean(dim=1, keepdim=True)
        coefficients = torch.einsum("bd,bkd->bk", raw_query, centered)
        projected = torch.einsum("bk,bkd->bd", coefficients, centered)
        quotient = F.normalize(
            projected,
            dim=-1,
            eps=self.query_norm_epsilon,
        )
        if not bool(torch.isfinite(quotient).all()):
            raise ValueError("S48 quotient query produced non-finite values")
        return quotient, raw_query

    def private_residual(
        self,
        *,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if signatures.ndim != 3 or signatures.shape[-1] != self.native_dimension:
            raise ValueError("S48 signatures must be [B,K,256]")
        base = signatures.detach()
        quotient, _raw_query = self.query_quotient(
            signatures=base,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )

        expanded_query = quotient[:, None, :].expand(-1, base.shape[1], -1)
        x = torch.cat([base, expanded_query], dim=-1)

        a = self.adapter_a.to(dtype=base.dtype)
        b = self.adapter_b.to(dtype=base.dtype)
        hidden = F.gelu(F.linear(x, a, bias=None))
        residual = F.linear(hidden, b, bias=None)
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("S48 private residual produced non-finite values")
        return residual, quotient

    def private_signatures(
        self,
        *,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual, quotient = self.private_residual(
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
            raise ValueError("S48 private signature produced non-finite values")
        return private, quotient

    def correction_residual(
        self,
        *,
        signatures: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        private, quotient = self.private_signatures(
            signatures=signatures,
            question_tokens=question_tokens,
            question_mask=question_mask,
        )
        weight = self.bilinear_weight.to(dtype=private.dtype)
        residual = torch.einsum("bkd,de,be->bk", private, weight, quotient)
        residual = self.residual_scale * residual
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("S48 correction residual produced non-finite values")
        return residual


__all__ = ["QueryQuotientPrivateCorrectionFork"]
