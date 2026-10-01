import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.typed_competitive_cache import file_sha256
from nmd.v1_a13_lora import a13_lora_state_dict, iter_a13_lora_modules
from nmd.v1_s27_checkpoint import (
    S27_CHECKPOINT_KIND,
    S27_CHECKPOINT_SCHEMA,
    build_frozen_hira_v1_s27_candidate,
    load_hira_v1_s27_checkpoint,
)
from nmd.v1_s25_semantic_core import get_s25_relation_projection
from nmd.v1_s27_semantic_core import build_hira_v1_s27_blockwise_canonicalization_core


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
    torch.manual_seed(25021)
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
    return path, file_sha256(path)


def _write_checkpoint(tmp_path, t0, t0_sha):
    runtime = build_hira_v1_s27_blockwise_canonicalization_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
    )
    torch.manual_seed(25022)
    for module in iter_a13_lora_modules(runtime.encoder):
        with torch.no_grad():
            module.lora_a.normal_(0.0, 0.03)
            module.lora_b.normal_(0.0, 0.02)
    primary = runtime.projection_triadic_scorer.projection
    relation = get_s25_relation_projection(runtime)
    with torch.no_grad():
        primary.weight.normal_(0.0, 0.04)
        relation.weight.normal_(0.0, 0.05)

    path = tmp_path / "s27.pt"
    torch.save(
        {
            "schema_version": S27_CHECKPOINT_SCHEMA,
            "kind": S27_CHECKPOINT_KIND,
            "lora_parameter_count": 16384,
            "primary_projection_parameter_count": 32768,
            "relation_projection_parameter_count": 32768,
            "total_parameter_count": 81920,
            "relation_operator_added_parameter_count": 0,
            "canonicalization_added_parameter_count": 0,
            "factorized_signature_dimension": 256,
            "role_block_dimension": 128,
            "value_block_dimension": 128,
            "lora_rank": 8,
            "selected_dev_epoch": 11,
            "semantic_revision": "fake-revision",
            "initialization_t0_sha256": t0_sha,
            "lora_state_dict": a13_lora_state_dict(runtime.encoder),
            "primary_projection_state_dict": {
                "projection.weight": primary.weight.detach().cpu().clone(),
            },
            "relation_projection_state_dict": {
                "projection.weight": relation.weight.detach().cpu().clone(),
            },
        },
        path,
    )
    return path, file_sha256(path)


def test_s27_checkpoint_roundtrip_and_frozen_replay(tmp_path):
    t0, t0_sha = _write_t0(tmp_path)
    checkpoint, checkpoint_sha = _write_checkpoint(tmp_path, t0, t0_sha)

    lora, primary, relation, meta = load_hira_v1_s27_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha,
        initialization_t0_sha256=t0_sha,
        semantic_revision="fake-revision",
    )
    assert meta["selected_dev_epoch"] == 11
    assert set(lora) == {
        *(f"lora.{i}.a" for i in range(4)),
        *(f"lora.{i}.b" for i in range(4)),
    }
    assert set(primary) == {"projection.weight"}
    assert set(relation) == {"projection.weight"}
    assert not torch.equal(primary["projection.weight"], relation["projection.weight"])

    replay, operator, replay_meta = build_frozen_hira_v1_s27_candidate(
        _encoder(),
        t0,
        checkpoint,
        expected_t0_sha256=t0_sha,
        expected_candidate_sha256=checkpoint_sha,
        semantic_revision="fake-revision",
    )
    assert replay_meta["sha256"] == checkpoint_sha
    assert operator.factorization_added_parameter_count == 0
    assert sum(p.numel() for p in replay.parameters() if p.requires_grad) == 0
    p = replay.projection_triadic_scorer.projection.weight
    r = get_s25_relation_projection(replay).weight
    assert p.data_ptr() != r.data_ptr()
    assert torch.equal(p.detach().cpu(), primary["projection.weight"])
    assert torch.equal(r.detach().cpu(), relation["projection.weight"])


def test_s27_checkpoint_rejects_shared_projection_alias_schema(tmp_path):
    t0, t0_sha = _write_t0(tmp_path)
    checkpoint, _ = _write_checkpoint(tmp_path, t0, t0_sha)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["relation_projection_state_dict"] = {}
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    try:
        load_hira_v1_s27_checkpoint(
            bad,
            initialization_t0_sha256=t0_sha,
            semantic_revision="fake-revision",
        )
    except RuntimeError as exc:
        assert "relation projection keys" in str(exc)
    else:
        raise AssertionError("S27 malformed relation projection must fail")
