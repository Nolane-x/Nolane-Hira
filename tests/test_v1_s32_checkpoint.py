import tempfile
from pathlib import Path

import torch

from nmd.v1_s32_checkpoint import load_hira_v1_s32_checkpoint


def _payload(arm):
    lora={}
    for i in range(4):
        lora[f"lora.{i}.a"]=torch.zeros(8,256)
        lora[f"lora.{i}.b"]=torch.zeros(256,8)
    return {
        "schema_version":f"hira-v1-s32-{arm}-global-relation-checkpoint-v1",
        "kind":f"matched-{arm}-global-relation-s17-shell-a13-w28",
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


def test_s32_gold_and_matrix_checkpoint_schemas():
    for arm in ("gold","matrix"):
        payload=_payload(arm)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/f"{arm}.pt"
            torch.save(payload,p)
            lora,proj,meta=load_hira_v1_s32_checkpoint(p,arm=arm)
        assert len(lora)==8
        assert tuple(proj["projection.weight"].shape)==(128,256)
        assert meta["arm"]==arm
        assert meta["total_parameter_count"]==49152
