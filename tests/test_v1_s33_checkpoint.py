import tempfile
from pathlib import Path

import torch

from nmd.v1_s33_checkpoint import (
    CONTROL_KIND,
    CONTROL_SCHEMA,
    QUERY_KIND,
    QUERY_SCHEMA,
    load_hira_v1_s33_checkpoint,
)


def _payload(schema,kind,arm):
    lora={}
    for i in range(4):
        lora[f"lora.{i}.a"]=torch.zeros(8,256)
        lora[f"lora.{i}.b"]=torch.zeros(256,8)
    return {
        "schema_version":schema,
        "kind":kind,
        "arm":arm,
        "lora_parameter_count":16384,
        "projection_parameter_count":32768,
        "total_parameter_count":49152,
        "lora_rank":8,
        "selected_dev_epoch":1,
        "semantic_revision":"4226d9e4d2c08703e5cb0491b479bfc6a1607181",
        "initialization_t0_sha256":"1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f",
        "lora_state_dict":lora,
        "projection_state_dict":{"projection.weight":torch.zeros(128,256)},
    }


def test_s33_control_and_query_checkpoint_schemas():
    for schema,kind,arm in (
        (CONTROL_SCHEMA,CONTROL_KIND,"control"),
        (QUERY_SCHEMA,QUERY_KIND,"query"),
    ):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/f"{arm}.pt"
            torch.save(_payload(schema,kind,arm),p)
            lora,projection,meta=load_hira_v1_s33_checkpoint(p,arm=arm)
        assert len(lora)==8
        assert tuple(projection["projection.weight"].shape)==(128,256)
        assert meta["arm"]==arm
        assert meta["total_parameter_count"]==49152
