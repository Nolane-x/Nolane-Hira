from types import SimpleNamespace

import torch
from torch import nn

from nmd.semantic import HFAutoSemanticEncoder
from nmd.v1_s29_semantic_core import (
    HIRA_V1_S29_LORA_PARAMETER_COUNT,
    HIRA_V1_S29_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S29_TOTAL_PARAMETER_COUNT,
)


def test_s29_declared_capacity():
    assert HIRA_V1_S29_LORA_PARAMETER_COUNT == 36864
    assert HIRA_V1_S29_PROJECTION_PARAMETER_COUNT == 32768
    assert HIRA_V1_S29_TOTAL_PARAMETER_COUNT == 69632
