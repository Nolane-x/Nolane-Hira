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
    HIRA_V1_S17_PROJECTION_PARAMETER_COUNT,
    build_hira_v1_s17_norm_balanced_core,
)

S32_LORA_PARAMETER_COUNT=16_384
S32_TOTAL_PARAMETER_COUNT=HIRA_V1_S17_TOTAL_PARAMETER_COUNT

_ARM_SPEC={
    "gold":{
        "schema":"hira-v1-s32-gold-global-relation-checkpoint-v1",
        "kind":"matched-gold-global-relation-s17-shell-a13-w28",
    },
    "matrix":{
        "schema":"hira-v1-s32-matrix-global-relation-checkpoint-v1",
        "kind":"matched-matrix-global-relation-s17-shell-a13-w28",
    },
}

_LORA_KEYS={
    *(f"lora.{i}.a" for i in range(4)),
    *(f"lora.{i}.b" for i in range(4)),
}


def load_hira_v1_s32_checkpoint(
    checkpoint_path:str|Path,
    *,
    arm:str,
    expected_sha256:str|None=None,
    initialization_t0_sha256:str=W28_T0_CHECKPOINT_SHA256,
    semantic_revision:str=A13_REVISION,
):
    if arm not in _ARM_SPEC:
        raise ValueError("S32 checkpoint arm must be gold or matrix")
    spec=_ARM_SPEC[arm]

    path=Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual=file_sha256(path)
    if expected_sha256 is not None and actual!=expected_sha256:
        raise RuntimeError(
            f"Hira v1 S32 checkpoint SHA mismatch: {actual} != {expected_sha256}"
        )

    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("schema_version")!=spec["schema"]:
        raise RuntimeError("unexpected Hira v1 S32 checkpoint schema")
    if payload.get("kind")!=spec["kind"]:
        raise RuntimeError("unexpected Hira v1 S32 checkpoint kind")
    if payload.get("arm")!=arm:
        raise RuntimeError("Hira v1 S32 checkpoint arm changed")
    if int(payload.get("lora_parameter_count",-1))!=S32_LORA_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S32 LoRA count changed")
    if int(payload.get("projection_parameter_count",-1))!=HIRA_V1_S17_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S32 projection count changed")
    if int(payload.get("total_parameter_count",-1))!=S32_TOTAL_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S32 total count changed")
    if int(payload.get("lora_rank",-1))!=8:
        raise RuntimeError("Hira v1 S32 rank changed")
    if payload.get("initialization_t0_sha256")!=initialization_t0_sha256:
        raise RuntimeError("Hira v1 S32 T0 identity changed")
    if payload.get("semantic_revision")!=semantic_revision:
        raise RuntimeError("Hira v1 S32 semantic revision changed")

    lora=payload.get("lora_state_dict")
    projection=payload.get("projection_state_dict")
    if not isinstance(lora,Mapping) or not isinstance(projection,Mapping):
        raise RuntimeError("Hira v1 S32 checkpoint state missing")
    lora={str(k):v for k,v in lora.items()}
    projection={str(k):v for k,v in projection.items()}
    if set(lora)!=_LORA_KEYS:
        raise RuntimeError("Hira v1 S32 LoRA keys changed")
    if set(projection)!={"projection.weight"}:
        raise RuntimeError("Hira v1 S32 projection keys changed")

    clean={}
    for i in range(4):
        a=lora[f"lora.{i}.a"]
        b=lora[f"lora.{i}.b"]
        if not isinstance(a,Tensor) or tuple(a.shape)!=(8,256):
            raise RuntimeError(f"Hira v1 S32 LoRA A shape changed: {i}")
        if not isinstance(b,Tensor) or tuple(b.shape)!=(256,8):
            raise RuntimeError(f"Hira v1 S32 LoRA B shape changed: {i}")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise RuntimeError(f"Hira v1 S32 LoRA tensor non-finite: {i}")
        clean[f"lora.{i}.a"]=a.detach().float().clone()
        clean[f"lora.{i}.b"]=b.detach().float().clone()

    weight=projection["projection.weight"]
    if not isinstance(weight,Tensor) or tuple(weight.shape)!=(128,256):
        raise RuntimeError("Hira v1 S32 projection shape changed")
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError("Hira v1 S32 projection non-finite")

    epoch=int(payload.get("selected_dev_epoch",-1))
    if not 1<=epoch<=24:
        raise RuntimeError("Hira v1 S32 selected epoch invalid")

    return clean,{"projection.weight":weight.detach().float().clone()},{
        "sha256":actual,
        "arm":arm,
        "selected_dev_epoch":epoch,
        "total_parameter_count":S32_TOTAL_PARAMETER_COUNT,
        "semantic_revision":semantic_revision,
    }


def build_frozen_hira_v1_s32_candidate(
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
        raise RuntimeError("S32 encoder revision changed")

    runtime=build_hira_v1_s17_norm_balanced_core(
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=True,
        train_projection=True,
    )
    lora,projection,meta=load_hira_v1_s32_checkpoint(
        candidate_checkpoint_path,
        arm=arm,
        expected_sha256=expected_candidate_sha256,
        initialization_t0_sha256=expected_t0_sha256,
        semantic_revision=semantic_revision,
    )
    load_a13_lora_state_dict(runtime.encoder,lora,freeze=True)
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S32 replay scorer missing")
    scorer.load_projection_state_dict(projection,freeze=True)

    trainable=sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable!=0:
        raise RuntimeError(f"S32 frozen replay has trainable parameters: {trainable}")
    runtime.eval()
    return runtime,meta


__all__=[
    "S32_LORA_PARAMETER_COUNT",
    "S32_TOTAL_PARAMETER_COUNT",
    "load_hira_v1_s32_checkpoint",
    "build_frozen_hira_v1_s32_candidate",
]
