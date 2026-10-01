import tempfile
from pathlib import Path

import torch

from nmd.v1_s30_checkpoint import (
    ATTENTION_KIND,
    ATTENTION_SCHEMA,
    FFN_KIND,
    FFN_SCHEMA,
    load_hira_v1_s30_checkpoint,
)


def _base(schema,kind,arm,lora_count,total,lora):
    return {
        "schema_version":schema,
        "kind":kind,
        "arm":arm,
        "lora_parameter_count":lora_count,
        "projection_parameter_count":32768,
        "total_parameter_count":total,
        "lora_rank":8,
        "selected_dev_epoch":1,
        "semantic_revision":"4226d9e4d2c08703e5cb0491b479bfc6a1607181",
        "initialization_t0_sha256":"1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f",
        "lora_state_dict":lora,
        "projection_state_dict":{"projection.weight":torch.zeros(128,256)},
    }


def test_s30_attention_checkpoint_shapes():
    lora={}
    for i in range(4):
        lora[f"lora.{i}.a"]=torch.zeros(8,256)
        lora[f"lora.{i}.b"]=torch.zeros(256,8)
    payload=_base(ATTENTION_SCHEMA,ATTENTION_KIND,"attention",16384,49152,lora)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"a.pt"
        torch.save(payload,p)
        got,proj,meta=load_hira_v1_s30_checkpoint(p,arm="attention")
    assert set(got)==set(lora)
    assert tuple(proj["projection.weight"].shape)==(128,256)
    assert meta["total_parameter_count"]==49152


def test_s30_ffn_checkpoint_shapes():
    lora={
        "lora.0.a":torch.zeros(8,256),
        "lora.0.b":torch.zeros(1024,8),
        "lora.1.a":torch.zeros(8,1024),
        "lora.1.b":torch.zeros(256,8),
    }
    payload=_base(FFN_SCHEMA,FFN_KIND,"ffn",20480,53248,lora)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"f.pt"
        torch.save(payload,p)
        got,proj,meta=load_hira_v1_s30_checkpoint(p,arm="ffn")
    assert set(got)==set(lora)
    assert tuple(got["lora.1.a"].shape)==(8,1024)
    assert tuple(proj["projection.weight"].shape)==(128,256)
    assert meta["total_parameter_count"]==53248
