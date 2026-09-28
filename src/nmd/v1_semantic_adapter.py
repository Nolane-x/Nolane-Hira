from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .symmetric_semantic import SymmetricSemanticScorer
from .v1_triadic_semantic import _aggregate_triadic, _validate_inputs


class SharedResidualSemanticAdapter(nn.Module):
    """Shared S4 token adapter applied before frozen W28 projection.

    The same adapter parameters are used for state, question and option tokens.
    The up projection is zero-initialized, giving an exact identity function at
    construction while keeping the down projection available for learning once
    the residual path opens.
    """

    def __init__(self, d_model: int = 256, bottleneck: int = 32):
        super().__init__()
        self.d_model = int(d_model)
        self.bottleneck = int(bottleneck)
        if self.d_model < 1 or self.bottleneck < 1:
            raise ValueError("adapter dimensions must be positive")
        self.down = nn.Linear(self.d_model, self.bottleneck, bias=False)
        self.up = nn.Linear(self.bottleneck, self.d_model, bias=False)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.shape[-1] != self.d_model:
            raise ValueError("adapter input d_model mismatch")
        residual = self.up(F.gelu(self.down(tokens)))
        adapted = tokens + residual
        if not bool(torch.isfinite(adapted).all()):
            raise ValueError("semantic adapter produced non-finite values")
        return adapted


class AdaptedTriadicSemanticScorer(SymmetricSemanticScorer):
    """S4 learned semantic representation adaptation + parameter-free triad.

    Only the shared d_model adapter is learned. W28 remains frozen and there is
    no learned decision scorer after projection.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_rel: int = 128,
        bottleneck: int = 32,
    ):
        super().__init__(d_model=d_model, d_rel=d_rel)
        self.adapter = SharedResidualSemanticAdapter(
            d_model=d_model,
            bottleneck=bottleneck,
        )

    @property
    def adapter_parameter_count(self) -> int:
        return self.adapter.parameter_count

    @property
    def adapter_trainable_parameter_count(self) -> int:
        return self.adapter.trainable_parameter_count

    def freeze_adapter(self) -> None:
        self.adapter.freeze()

    def load_adapter_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        freeze: bool = True,
    ) -> None:
        expected = {
            "adapter.down.weight": (
                self.adapter.bottleneck,
                self.d_model,
            ),
            "adapter.up.weight": (
                self.d_model,
                self.adapter.bottleneck,
            ),
        }
        if set(state_dict) != set(expected):
            raise ValueError("Hira v1 S4 adapter checkpoint keys changed")
        own = self.state_dict()
        with torch.no_grad():
            for key, shape in expected.items():
                value = state_dict[key]
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(
                        f"Hira v1 S4 adapter shape mismatch: {key}"
                    )
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(
                        f"Hira v1 S4 adapter tensor non-finite: {key}"
                    )
                own[key].copy_(
                    value.to(
                        device=own[key].device,
                        dtype=own[key].dtype,
                    )
                )
        if freeze:
            self.freeze_adapter()

    def _adapt_project(self, tokens: Tensor) -> Tensor:
        adapted = self.adapter(tokens)
        return self._project(adapted)

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_tokens: Tensor,
        question_mask: Tensor,
        option_view_tokens: Tensor,
        option_view_token_mask: Tensor,
        option_view_mask: Tensor,
    ) -> Tensor:
        _validate_inputs(
            self,
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_tokens=question_tokens,
            question_mask=question_mask,
            option_view_tokens=option_view_tokens,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )

        state = self._adapt_project(state_tokens)
        question = self._adapt_project(question_tokens)
        options = self._adapt_project(option_view_tokens)

        triad = torch.einsum(
            "bsd,bqd,bkvtd->bkvtsq",
            state,
            question,
            options,
        ) / sqrt(float(self.d_rel))

        return _aggregate_triadic(
            triad,
            state_mask=state_mask,
            question_mask=question_mask,
            option_view_token_mask=option_view_token_mask,
            option_view_mask=option_view_mask,
        )


__all__ = [
    "AdaptedTriadicSemanticScorer",
    "SharedResidualSemanticAdapter",
]
