from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .local_runtime import A13_REVISION
from .semantic import HFAutoSemanticEncoder
from .semantic_core import W28_T0_CHECKPOINT_SHA256
from .typed_competitive_cache import file_sha256
from .v1_a13_ffn_lora import load_a13_ffn_only_lora_state_dict
from .v1_a13_lora import load_a13_lora_state_dict
from .v1_s30_semantic_core import (
    S30_ATTENTION_LORA_PARAMETER_COUNT,
    S30_ATTENTION_TOTAL_PARAMETER_COUNT,
    S30_FFN_LORA_PARAMETER_COUNT,
    S30_FFN_TOTAL_PARAMETER_COUNT,
    S30_PROJECTION_PARAMETER_COUNT,
    build_s30_attention_arm,
    build_s30_ffn_arm,
)

ATTENTION_SCHEMA="hira-v1-s30-attention-only-checkpoint-v1"
ATTENTION_KIND="matched-attention-only-s17-shell-a13-w28"
FFN_SCHEMA="hira-v1-s30-ffn-only-checkpoint-v1"
FFN_KIND="matched-ffn-only-s17-shell-a13-w28"


def _spec(arm: str):
    if arm=="attention":
        return {
            "schema":ATTENTION_SCHEMA,
            "kind":ATTENTION_KIND,
            "lora_count":S30_ATTENTION_LORA_PARAMETER_COUNT,
            "total_count":S30_ATTENTION_TOTAL_PARAMETER_COUNT,
            "keys":{
                *(f"lora.{i}.a" for i in range(4)),
                *(f"lora.{i}.b" for i in range(4)),
            },
            "shapes":{
                **{f"lora.{i}.a":(8,256) for i in range(4)},
                **{f"lora.{i}.b":(256,8) for i in range(4)},
            },
            "builder":build_s30_attention_arm,
            "loader":load_a13_lora_state_dict,
        }
    if arm=="ffn":
        return {
            "schema":FFN_SCHEMA,
            "kind":FFN_KIND,
            "lora_count":S30_FFN_LORA_PARAMETER_COUNT,
            "total_count":S30_FFN_TOTAL_PARAMETER_COUNT,
            "keys":{"lora.0.a","lora.0.b","lora.1.a","lora.1.b"},
            "shapes":{
                "lora.0.a":(8,256),
                "lora.0.b":(1024,8),
                "lora.1.a":(8,1024),
                "lora.1.b":(256,8),
            },
            "builder":build_s30_ffn_arm,
            "loader":load_a13_ffn_only_lora_state_dict,
        }
    raise ValueError("S30 checkpoint arm must be attention or ffn")


def load_hira_v1_s30_checkpoint(
    checkpoint_path: str|Path,
    *,
    arm: str,
    expected_sha256: str|None=None,
    initialization_t0_sha256: str=W28_T0_CHECKPOINT_SHA256,
    semantic_revision: str=A13_REVISION,
):
    spec=_spec(arm)
    path=Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual=file_sha256(path)
    if expected_sha256 is not None and actual!=expected_sha256:
        raise RuntimeError(
            f"Hira v1 S30 checkpoint SHA mismatch: {actual} != {expected_sha256}"
        )
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("schema_version")!=spec["schema"]:
        raise RuntimeError("unexpected Hira v1 S30 checkpoint schema")
    if payload.get("kind")!=spec["kind"]:
        raise RuntimeError("unexpected Hira v1 S30 checkpoint kind")
    if payload.get("arm")!=arm:
        raise RuntimeError("Hira v1 S30 checkpoint arm changed")
    if int(payload.get("lora_parameter_count",-1))!=spec["lora_count"]:
        raise RuntimeError("Hira v1 S30 LoRA count changed")
    if int(payload.get("projection_parameter_count",-1))!=S30_PROJECTION_PARAMETER_COUNT:
        raise RuntimeError("Hira v1 S30 projection count changed")
    if int(payload.get("total_parameter_count",-1))!=spec["total_count"]:
        raise RuntimeError("Hira v1 S30 total count changed")
    if int(payload.get("lora_rank",-1))!=8:
        raise RuntimeError("Hira v1 S30 LoRA rank changed")
    if payload.get("initialization_t0_sha256")!=initialization_t0_sha256:
        raise RuntimeError("Hira v1 S30 T0 identity changed")
    if payload.get("semantic_revision")!=semantic_revision:
        raise RuntimeError("Hira v1 S30 semantic revision changed")

    lora=payload.get("lora_state_dict")
    projection=payload.get("projection_state_dict")
    if not isinstance(lora,Mapping) or not isinstance(projection,Mapping):
        raise RuntimeError("Hira v1 S30 checkpoint state missing")
    lora={str(k):v for k,v in lora.items()}
    projection={str(k):v for k,v in projection.items()}
    if set(lora)!=spec["keys"]:
        raise RuntimeError("Hira v1 S30 LoRA keys changed")
    if set(projection)!={"projection.weight"}:
        raise RuntimeError("Hira v1 S30 projection keys changed")

    clean={}
    for key,shape in spec["shapes"].items():
        value=lora[key]
        if not isinstance(value,Tensor) or tuple(value.shape)!=shape:
            raise RuntimeError(f"Hira v1 S30 LoRA shape changed: {key}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Hira v1 S30 LoRA tensor non-finite: {key}")
        clean[key]=value.detach().float().clone()

    weight=projection["projection.weight"]
    if not isinstance(weight,Tensor) or tuple(weight.shape)!=(128,256):
        raise RuntimeError("Hira v1 S30 projection shape changed")
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError("Hira v1 S30 projection tensor non-finite")

    epoch=int(payload.get("selected_dev_epoch",-1))
    if not 1<=epoch<=24:
        raise RuntimeError("Hira v1 S30 selected DEV epoch invalid")

    return clean,{"projection.weight":weight.detach().float().clone()},{
        "sha256":actual,
        "arm":arm,
        "selected_dev_epoch":epoch,
        "total_parameter_count":spec["total_count"],
        "semantic_revision":semantic_revision,
    }


def build_frozen_hira_v1_s30_candidate(
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
        raise RuntimeError("S30 encoder revision changed")
    spec=_spec(arm)
    runtime=spec["builder"](
        encoder,
        t0_checkpoint_path,
        expected_t0_sha256=expected_t0_sha256,
        train_lora=True,
        train_projection=True,
    )
    lora,projection,meta=load_hira_v1_s30_checkpoint(
        candidate_checkpoint_path,
        arm=arm,
        expected_sha256=expected_candidate_sha256,
        initialization_t0_sha256=expected_t0_sha256,
        semantic_revision=semantic_revision,
    )
    spec["loader"](runtime.encoder,lora,freeze=True)
    scorer=runtime.projection_triadic_scorer
    if scorer is None:
        raise RuntimeError("S30 replay scorer missing")
    scorer.load_projection_state_dict(projection,freeze=True)
    trainable=sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    if trainable!=0:
        raise RuntimeError(f"S30 frozen replay has trainable parameters: {trainable}")
    runtime.eval()
    return runtime,meta


__all__=[
    "ATTENTION_SCHEMA","ATTENTION_KIND","FFN_SCHEMA","FFN_KIND",
    "load_hira_v1_s30_checkpoint","build_frozen_hira_v1_s30_candidate",
]
