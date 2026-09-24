from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


CalibrationMode = Literal[
    "primitive-temperature",
    "primitive-temperature-noul-bias",
]
CALIBRATION_MODES: tuple[CalibrationMode, ...] = (
    "primitive-temperature",
    "primitive-temperature-noul-bias",
)

_MIN_LOG_TEMPERATURE = math.log(0.1)
_MAX_LOG_TEMPERATURE = math.log(10.0)
_MAX_ABS_NOUL_BIAS = 10.0


class TypedReliabilityCalibrator(nn.Module):
    """Tiny post-HIRA typed calibration layer.

    Temperature parameters are indexed by the frozen primitive IDs:
    0=choice, 1=score, 2=noul. The optional noul bias is applied only
    to the semantic true option supplied through noul_true_mask.
    """

    def __init__(self, mode: CalibrationMode):
        super().__init__()
        if mode not in CALIBRATION_MODES:
            raise ValueError(f"unsupported calibration mode: {mode}")
        self.mode = mode
        self.log_temperature = nn.Parameter(torch.zeros(3))
        if mode == "primitive-temperature-noul-bias":
            self.noul_true_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("noul_true_bias", None)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def temperatures(self) -> Tensor:
        return self.log_temperature.clamp(
            _MIN_LOG_TEMPERATURE,
            _MAX_LOG_TEMPERATURE,
        ).exp()

    def forward(
        self,
        logits: Tensor,
        qtype: Tensor,
        *,
        noul_true_mask: Tensor | None = None,
    ) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("calibrator logits must be [B,K]")
        if qtype.ndim != 1 or qtype.shape[0] != logits.shape[0]:
            raise ValueError("calibrator qtype must be [B]")
        if qtype.numel() and (
            int(qtype.min().item()) < 0 or int(qtype.max().item()) > 2
        ):
            raise ValueError("calibrator qtype must use primitive IDs 0..2")
        if not torch.isfinite(logits).all():
            raise ValueError("calibrator logits must be finite")

        temperature = self.temperatures().to(
            device=logits.device,
            dtype=logits.dtype,
        )[qtype].unsqueeze(-1)
        calibrated = logits / temperature

        if self.noul_true_bias is not None:
            noul_rows = qtype == 2
            if bool(noul_rows.any()):
                if (
                    noul_true_mask is None
                    or noul_true_mask.shape != logits.shape
                    or noul_true_mask.dtype != torch.bool
                ):
                    raise ValueError(
                        "noul-bias calibration requires bool true mask [B,K]"
                    )
                counts = noul_true_mask[noul_rows].sum(dim=-1)
                if not bool((counts == 1).all()):
                    raise ValueError(
                        "each noul row must identify exactly one semantic true option"
                    )
                active = noul_true_mask & noul_rows[:, None]
                bias = self.noul_true_bias.clamp(
                    -_MAX_ABS_NOUL_BIAS,
                    _MAX_ABS_NOUL_BIAS,
                ).to(device=logits.device, dtype=logits.dtype)
                calibrated = calibrated + active.to(logits.dtype) * bias

        if not torch.isfinite(calibrated).all():
            raise ValueError("calibrator produced non-finite logits")
        return calibrated


def count_calibrator_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
