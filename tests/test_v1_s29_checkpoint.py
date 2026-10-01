import tempfile
from pathlib import Path
import torch
from nmd.v1_s29_checkpoint import S29_CHECKPOINT_KIND,S29_CHECKPOINT_SCHEMA,load_hira_v1_s29_checkpoint

def test_s29_checkpoint_accepts_exact_six_module_shapes():
    lora={}
    shapes=[
        ((8,256),(256,8)),((8,256),(256,8)),((8,256),(256,8)),((8,256),(256,8)),
        ((8,256),(1024,8)),((8,1024),(256,8)),
    ]
    for i,(a,b) in enumerate(shapes):
        lora[f"lora.{i}.a"]=torch.zeros(a)
        lora[f"lora.{i}.b"]=torch.zeros(b)
    payload={
        "schema_version":S29_CHECKPOINT_SCHEMA,
        "kind":S29_CHECKPOINT_KIND,
        "lora_parameter_count":36864,
        "projection_parameter_count":32768,
        "total_parameter_count":69632,
        "lora_rank":8,
        "selected_dev_epoch":1,
        "semantic_revision":"4226d9e4d2c08703e5cb0491b479bfc6a1607181",
        "initialization_t0_sha256":"1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f",
        "lora_state_dict":lora,
        "projection_state_dict":{"projection.weight":torch.zeros(128,256)},
    }
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"c.pt"; torch.save(payload,p)
        got_lora,got_proj,meta=load_hira_v1_s29_checkpoint(p)
    assert set(got_lora)==set(lora)
    assert tuple(got_lora["lora.4.b"].shape)==(1024,8)
    assert tuple(got_lora["lora.5.a"].shape)==(8,1024)
    assert tuple(got_proj["projection.weight"].shape)==(128,256)
    assert meta["total_parameter_count"]==69632
