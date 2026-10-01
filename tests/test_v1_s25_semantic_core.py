import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256
from nmd.v1_s25_semantic_core import (
    HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT,
    HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT,
    HIRA_V1_S25_TOTAL_PARAMETER_COUNT,
    build_hira_v1_s25_decoupled_projection_core,
    get_s25_relation_projection,
)


class _FakeConfig:
    model_type = "bert"
    hidden_size = 256
    num_hidden_layers = 6
    num_attention_heads = 8
    _name_or_path = "fake-a13"


class _Self(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(256, 256)
        self.key = torch.nn.Linear(256, 256)
        self.value = torch.nn.Linear(256, 256)


class _Output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(256, 256)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self = _Self()
        self.output = _Output()


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()


class _EncoderBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.ModuleList([_Layer() for _ in range(6)])


class _FakeBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.encoder = _EncoderBody()


class _Tokenizer:
    name_or_path = "fake-tokenizer"


def _encoder():
    return HFAutoSemanticEncoder(
        _FakeBert(),
        tokenizer=_Tokenizer(),
        revision="fake-revision",
        max_length=64,
    )


def _write_t0(tmp_path):
    path = tmp_path / "t0.pt"
    torch.manual_seed(25001)
    weight = torch.randn(128, 256)
    torch.save(
        {
            "schema_version": "r8-w28-candidate-checkpoint-v1",
            "candidate": "T0",
            "kind": "projection",
            "projection_weight": weight,
        },
        path,
    )
    return path, file_sha256(path), weight


def test_s25_exact_surface_and_private_initialization(tmp_path):
    t0, sha, expected = _write_t0(tmp_path)
    runtime = build_hira_v1_s25_decoupled_projection_core(
        _encoder(),
        t0,
        expected_t0_sha256=sha,
        train_lora=True,
        train_primary_projection=True,
        train_relation_projection=True,
    )
    primary = runtime.projection_triadic_scorer.projection
    relation = get_s25_relation_projection(runtime)

    assert HIRA_V1_S25_SHARED_LORA_PARAMETER_COUNT == 16384
    assert HIRA_V1_S25_PRIMARY_PROJECTION_PARAMETER_COUNT == 32768
    assert HIRA_V1_S25_RELATION_PROJECTION_PARAMETER_COUNT == 32768
    assert HIRA_V1_S25_TOTAL_PARAMETER_COUNT == 81920
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 81920

    assert primary.weight.data_ptr() != relation.weight.data_ptr()
    assert torch.equal(primary.weight.detach().cpu(), expected)
    assert torch.equal(relation.weight.detach().cpu(), expected)
    assert torch.equal(primary.weight.detach(), relation.weight.detach())


def test_s25_can_freeze_each_surface_independently(tmp_path):
    t0, sha, _ = _write_t0(tmp_path)
    runtime = build_hira_v1_s25_decoupled_projection_core(
        _encoder(),
        t0,
        expected_t0_sha256=sha,
        train_lora=False,
        train_primary_projection=True,
        train_relation_projection=False,
    )
    primary = runtime.projection_triadic_scorer.projection
    relation = get_s25_relation_projection(runtime)
    assert primary.weight.requires_grad
    assert not relation.weight.requires_grad
    assert sum(p.numel() for p in runtime.parameters() if p.requires_grad) == 32768


def test_s25_original_a13_and_hira_remain_frozen(tmp_path):
    t0, sha, _ = _write_t0(tmp_path)
    runtime = build_hira_v1_s25_decoupled_projection_core(
        _encoder(),
        t0,
        expected_t0_sha256=sha,
    )
    original = sum(
        p.numel()
        for name, p in runtime.encoder.model.named_parameters()
        if p.requires_grad and ".lora_" not in name
    )
    assert original == 0
    assert not any(p.requires_grad for p in runtime.hira.parameters())
