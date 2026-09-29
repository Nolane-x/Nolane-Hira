import torch

from nmd.semantic import HFAutoSemanticEncoder
from nmd.v1_a13_lora import a13_lora_state_dict, iter_a13_lora_modules
from nmd.v1_s16_checkpoint import (
    build_frozen_hira_v1_s16_candidate,
    load_hira_v1_s16_checkpoint,
)
from nmd.v1_s16_semantic_core import build_hira_v1_s16_shared_gradient_surgery_core


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
    from nmd.typed_competitive_cache import file_sha256

    path = tmp_path / "t0.pt"
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


def test_s16_checkpoint_roundtrip_and_frozen_replay(tmp_path):
    from nmd.typed_competitive_cache import file_sha256

    torch.manual_seed(18101)
    t0, t0_sha = _write_t0(tmp_path)
    runtime = build_hira_v1_s16_shared_gradient_surgery_core(
        _encoder(),
        t0,
        expected_t0_sha256=t0_sha,
        train_lora=True,
        train_projection=True,
    )
    for module in iter_a13_lora_modules(runtime.encoder):
        with torch.no_grad():
            module.lora_a.normal_(mean=0.0, std=0.03)
            module.lora_b.normal_(mean=0.0, std=0.02)

    scorer = runtime.projection_triadic_scorer
    assert scorer is not None
    with torch.no_grad():
        scorer.projection.weight.normal_(mean=0.0, std=0.04)

    lora = a13_lora_state_dict(runtime.encoder)
    projection = {"projection.weight": scorer.projection.weight.detach().cpu().clone()}

    checkpoint = tmp_path / "s16.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s16-shared-gradient-surgery-checkpoint-v1",
            "kind": "shared-gradient-surgery-a13-w28",
            "lora_parameter_count": 16384,
            "projection_parameter_count": 32768,
            "total_parameter_count": 49152,
            "lora_rank": 8,
            "selected_dev_epoch": 9,
            "semantic_revision": "fake-revision",
            "initialization_t0_sha256": t0_sha,
            "lora_state_dict": lora,
            "projection_state_dict": projection,
        },
        checkpoint,
    )
    checkpoint_sha = file_sha256(checkpoint)

    loaded_lora, loaded_projection, metadata = load_hira_v1_s16_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha,
        initialization_t0_sha256=t0_sha,
        semantic_revision="fake-revision",
    )
    assert metadata["selected_dev_epoch"] == 9
    for key in lora:
        assert torch.equal(loaded_lora[key], lora[key])
    assert torch.equal(
        loaded_projection["projection.weight"],
        projection["projection.weight"],
    )

    replay, replay_meta = build_frozen_hira_v1_s16_candidate(
        _encoder(),
        t0,
        checkpoint,
        expected_t0_sha256=t0_sha,
        expected_shared_gradient_sha256=checkpoint_sha,
        semantic_revision="fake-revision",
    )
    assert replay_meta["sha256"] == checkpoint_sha
    assert sum(p.numel() for p in replay.parameters() if p.requires_grad) == 0


def test_s16_checkpoint_rejects_wrong_kind(tmp_path):
    t0, t0_sha = _write_t0(tmp_path)
    bad = tmp_path / "bad.pt"
    torch.save(
        {
            "schema_version": "hira-v1-s16-shared-gradient-surgery-checkpoint-v1",
            "kind": "wrong",
            "lora_parameter_count": 16384,
            "projection_parameter_count": 32768,
            "total_parameter_count": 49152,
            "lora_rank": 8,
            "selected_dev_epoch": 1,
            "semantic_revision": "fake-revision",
            "initialization_t0_sha256": t0_sha,
            "lora_state_dict": {},
            "projection_state_dict": {},
        },
        bad,
    )
    try:
        load_hira_v1_s16_checkpoint(
            bad,
            initialization_t0_sha256=t0_sha,
            semantic_revision="fake-revision",
        )
    except RuntimeError as exc:
        assert "kind" in str(exc)
    else:
        raise AssertionError("wrong S16 checkpoint kind must fail")
