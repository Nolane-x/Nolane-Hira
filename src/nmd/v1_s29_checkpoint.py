from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .local_runtime import A13_REVISION
from .semantic import HFAutoSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_a13_fullblock_lora import load_a13_full_block_lora_state_dict
from .v1_s29_semantic_core import (
    HIRA_V1_S29_LORA_PARAMETER_COUNT,
    HIRA_V1_S29_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S29_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s29_full_block_lora_core,
)

S29_CHECKPOINT_SCHEMA="hira-v1-s29-full-block-lora-checkpoint-v1"
S29_CHECKPOINT_KIND="full-block-lora-s17-shell-a13-w28"
S29_LORA_KEYS={
    *(f"lora.{i}.a" for i in range(6)),
    *(f"lora.{i}.b" for i in range(6)),
}
S29_PROJECTION_KEYS={"projection.weight"}
_SHAPES={
    "lora.0.a":(8,256),"lora.0.b":(256,8),
    "lora.1.a":(8,256),"lora.1.b":(256,8),
    "lora.2.a":(8,256),"lora.2.b":(256,8),
    "lora.3.a":(8,256),"lora.3.b":(256,8),
    "lora.4.a":(8,256),"lora.4.b":(1024,8),
    "lora.5.a":(8,1024),"lora.5.b":(256,8),
}


def load_hira_v1_s29_checkpoint(
    checkpoint_path: str|Path,
    *,
    expected_sha256: str|None=None,
    initialization_t0_sha256: str=W28_T0_CHECKPOINT_SHA256,
    semantic_revision: str=A13_REVISION,
):
    path=Path(checkpoint_path)
    if not path.is_file(): raise FileNotFoundError(path)
    actual=file_sha256(path)
    if expected_sha256 is not None and actual!=expected_sha256:
        raise RuntimeError(f"Hira v1 S29 checkpoint SHA mismatch: {actual} != {expected_sha256}")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("schema_version")!=S29_CHECKPOINT_SCHEMA: raise RuntimeError("unexpected Hira v1 S29 checkpoint schema")
    if payload.get("kind")!=S29_CHECKPOINT_KIND: raise RuntimeError("unexpected Hira v1 S29 checkpoint kind")
    if int(payload.get("lora_parameter_count",-1))!=HIRA_V1_S29_LORA_PARAMETER_COUNT: raise RuntimeError("S29 LoRA count changed")
    if int(payload.get("projection_parameter_count",-1))!=HIRA_V1_S29_PROJECTION_PARAMETER_COUNT: raise RuntimeError("S29 projection count changed")
    if int(payload.get("total_parameter_count",-1))!=HIRA_V1_S29_TOTAL_PARAMETER_COUNT: raise RuntimeError("S29 total count changed")
    if int(payload.get("lora_rank",-1))!=8: raise RuntimeError("S29 rank changed")
    if payload.get("initialization_t0_sha256")!=initialization_t0_sha256: raise RuntimeError("S29 T0 identity changed")
    if payload.get("semantic_revision")!=semantic_revision: raise RuntimeError("S29 semantic revision changed")
    lora=payload.get("lora_state_dict")
    projection=payload.get("projection_state_dict")
    if not isinstance(lora,Mapping) or not isinstance(projection,Mapping): raise RuntimeError("S29 checkpoint state missing")
    lora={str(k):v for k,v in lora.items()}
    projection={str(k):v for k,v in projection.items()}
    if set(lora)!=S29_LORA_KEYS: raise RuntimeError("S29 LoRA keys changed")
    if set(projection)!=S29_PROJECTION_KEYS: raise RuntimeError("S29 projection keys changed")
    clean={}
    for key,shape in _SHAPES.items():
        value=lora[key]
        if not isinstance(value,Tensor) or tuple(value.shape)!=shape: raise RuntimeError(f"S29 LoRA shape changed: {key}")
        if not bool(torch.isfinite(value).all()): raise RuntimeError(f"S29 LoRA non-finite: {key}")
        clean[key]=value.detach().float().clone()
    weight=projection["projection.weight"]
    if not isinstance(weight,Tensor) or tuple(weight.shape)!=(128,256): raise RuntimeError("S29 projection shape changed")
    if not bool(torch.isfinite(weight).all()): raise RuntimeError("S29 projection non-finite")
    epoch=int(payload.get("selected_dev_epoch",-1))
    if not 1<=epoch<=24: raise RuntimeError("S29 selected epoch invalid")
    return clean,{"projection.weight":weight.detach().float().clone()},{
        "sha256":actual,
        "selected_dev_epoch":epoch,
        "total_parameter_count":HIRA_V1_S29_TOTAL_PARAMETER_COUNT,
        "semantic_revision":semantic_revision,
    }


def build_frozen_hira_v1_s29_candidate(
    encoder:HFAutoSemanticEncoder,
    t0_checkpoint_path:str|Path,
    candidate_checkpoint_path:str|Path,
    *,
    expected_t0_sha256:str=W28_T0_CHECKPOINT_SHA256,
    expected_candidate_sha256:str|None=None,
    semantic_revision:str=A13_REVISION,
):
    if encoder.revision!=semantic_revision: raise RuntimeError("S29 encoder revision changed")
    runtime=build_hira_v1_s29_full_block_lora_core(
        encoder,t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=True,train_projection=True,
    )
    lora,projection,meta=load_hira_v1_s29_checkpoint(
        candidate_checkpoint_path,
        expected_sha256=expected_candidate_sha256,
        initialization_t0_sha256=expected_t0_sha256,
        semantic_revision=semantic_revision,
    )
    load_a13_full_block_lora_state_dict(runtime.encoder,lora,freeze=True)
    scorer=runtime.projection_triadic_scorer
    if scorer is None: raise RuntimeError("S29 replay scorer missing")
    scorer.load_projection_state_dict(projection,freeze=True)
    trainable=sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable!=0: raise RuntimeError(f"S29 frozen replay has trainable parameters: {trainable}")
    runtime.eval()
    return runtime,meta


__all__=[
    "S29_CHECKPOINT_SCHEMA","S29_CHECKPOINT_KIND","S29_LORA_KEYS","S29_PROJECTION_KEYS",
    "load_hira_v1_s29_checkpoint","build_frozen_hira_v1_s29_candidate",
]
