from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .local_runtime import A13_REVISION
from .semantic import HFAutoSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_a13_lora import load_a13_lora_state_dict
from .v1_s17_semantic_core import (
    HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
)

CONTROL_SCHEMA="hira-v1-s34-control-relation-checkpoint-v1"
CONTROL_KIND="matched-control-s13-relation-s17-shell-a13-w28"
TRANSPORT_SCHEMA="hira-v1-s34-transport-relation-checkpoint-v1"
TRANSPORT_KIND="matched-entropic-transport-relation-s17-shell-a13-w28"

_LORA_KEYS={
    *(f"lora.{i}.a" for i in range(4)),
    *(f"lora.{i}.b" for i in range(4)),
}
_LORA_SHAPES={
    **{f"lora.{i}.a":(8,256) for i in range(4)},
    **{f"lora.{i}.b":(256,8) for i in range(4)},
}


def _spec(arm:str):
    if arm=="control":
        return CONTROL_SCHEMA,CONTROL_KIND
    if arm=="transport":
        return TRANSPORT_SCHEMA,TRANSPORT_KIND
    raise ValueError("S34 checkpoint arm must be control or transport")


def load_hira_v1_s34_checkpoint(
    checkpoint_path:str|Path,
    *,
    arm:str,
    expected_sha256:str|None=None,
    initialization_t0_sha256:str=W28_T0_CHECKPOINT_SHA256,
    semantic_revision:str=A13_REVISION,
):
    schema,kind=_spec(arm)
    path=Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual=file_sha256(path)
    if expected_sha256 is not None and actual!=expected_sha256:
        raise RuntimeError(f"Hira v1 S34 checkpoint SHA mismatch: {actual} != {expected_sha256}")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("schema_version")!=schema:
        raise RuntimeError("unexpected Hira v1 S34 checkpoint schema")
    if payload.get("kind")!=kind:
        raise RuntimeError("unexpected Hira v1 S34 checkpoint kind")
    if payload.get("arm")!=arm:
        raise RuntimeError("Hira v1 S34 checkpoint arm changed")
    if int(payload.get("lora_parameter_count",-1))!=16_384:
        raise RuntimeError("S34 LoRA count changed")
    if int(payload.get("projection_parameter_count",-1))!=32_768:
        raise RuntimeError("S34 projection count changed")
    if int(payload.get("total_parameter_count",-1))!=HIRA_V1_S17_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("S34 total count changed")
    if int(payload.get("lora_rank",-1))!=8:
        raise RuntimeError("S34 LoRA rank changed")
    if payload.get("initialization_t0_sha256")!=initialization_t0_sha256:
        raise RuntimeError("S34 T0 identity changed")
    if payload.get("semantic_revision")!=semantic_revision:
        raise RuntimeError("S34 semantic revision changed")

    lora=payload.get("lora_state_dict")
    projection=payload.get("projection_state_dict")
    if not isinstance(lora,Mapping) or not isinstance(projection,Mapping):
        raise RuntimeError("S34 checkpoint state missing")
    lora={str(k):v for k,v in lora.items()}
    projection={str(k):v for k,v in projection.items()}
    if set(lora)!=_LORA_KEYS:
        raise RuntimeError("S34 LoRA keys changed")
    if set(projection)!={"projection.weight"}:
        raise RuntimeError("S34 projection keys changed")

    clean={}
    for key,shape in _LORA_SHAPES.items():
        value=lora[key]
        if not isinstance(value,Tensor) or tuple(value.shape)!=shape:
            raise RuntimeError(f"S34 LoRA shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"S34 LoRA non-finite: {key}")
        clean[key]=value.detach().float().clone()

    weight=projection["projection.weight"]
    if not isinstance(weight,Tensor) or tuple(weight.shape)!=(128,256):
        raise RuntimeError("S34 projection shape changed")
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError("S34 projection non-finite")

    epoch=int(payload.get("selected_dev_epoch",-1))
    if not 1<=epoch<=24:
        raise RuntimeError("S34 selected DEV epoch invalid")

    return clean,{"projection.weight":weight.detach().float().clone()},{
        "sha256":actual,
        "arm":arm,
        "selected_dev_epoch":epoch,
        "total_parameter_count":HIRA_V1_S17_TOTAL_PARAMETER_COUNT,
        "semantic_revision":semantic_revision,
    }


def build_frozen_hira_v1_s34_candidate(
    encoder:HFAutoSemanticEncoder,
    t0_checkpoint_path:str|Path,
    candidate_checkpoint_path:str|Path,
    *,
    arm:str,
    expected_t0_sha256:str=W28_T0_CHECKPOINT_SHA256,
    expected_candidate_sha256:str|None=None,
    semantic_revision:str=A13_REVISION,
):
    if encoder.revision!=semantic_revision:
        raise RuntimeError("S34 encoder revision changed")
    runtime=build_hira_v1_s17_norm_balanced_core(
        encoder,t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=True,train_projection=True,
    )
    lora,projection,meta=load_hira_v1_s34_checkpoint(
        candidate_checkpoint_path,arm=arm,
        expected_sha256=expected_candidate_sha256,
        initialization_t0_sha256=expected_t0_sha256,
        semantic_revision=semantic_revision,
    )
    load_a13_lora_state_dict(runtime.encoder,lora,freeze=True)
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S34 replay scorer missing")
    scorer.load_projection_state_dict(projection,freeze=True)
    if sum(p.numel() for p in runtime.parameters() if p.requires_grad)!=0:
        raise RuntimeError("S34 frozen replay has trainable parameters")
    runtime.eval()
    return runtime,meta


__all__=[
    "CONTROL_SCHEMA","CONTROL_KIND","TRANSPORT_SCHEMA","TRANSPORT_KIND",
    "load_hira_v1_s34_checkpoint","build_frozen_hira_v1_s34_candidate",
]
