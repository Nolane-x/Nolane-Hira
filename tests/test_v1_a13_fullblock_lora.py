from types import SimpleNamespace

import torch
from torch import nn

from nmd.semantic import HFAutoSemanticEncoder
from nmd.v1_a13_fullblock_lora import (
    ATTENTION_LORA_PARAMETER_COUNT,
    FFN_INTERMEDIATE_LORA_PARAMETER_COUNT,
    FFN_LORA_PARAMETER_COUNT,
    FFN_OUTPUT_LORA_PARAMETER_COUNT,
    FULL_BLOCK_LORA_PARAMETER_COUNT,
    a13_full_block_lora_state_dict,
    inject_a13_last_full_block_lora,
    iter_a13_full_block_lora_modules,
    load_a13_full_block_lora_state_dict,
)
from nmd.v1_a13_lora import LoRALinear


class _SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(256, 256)
        self.key = nn.Linear(256, 256)
        self.value = nn.Linear(256, 256)


class _AttentionOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(256, 256)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = _SelfAttention()
        self.output = _AttentionOutput()


class _Intermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(256, 1024)


class _Output(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1024, 256)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()
        self.intermediate = _Intermediate()
        self.output = _Output()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleNamespace(
            layer=nn.ModuleList([_Layer() for _ in range(6)])
        )
        self.config = SimpleNamespace(
            model_type="bert",
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
        )


def _encoder() -> HFAutoSemanticEncoder:
    encoder = object.__new__(HFAutoSemanticEncoder)
    nn.Module.__init__(encoder)
    encoder.model = _Model()
    encoder.d_model = 256
    return encoder


def test_s29_full_block_lora_exact_surface_and_modules():
    encoder = _encoder()
    receipt = inject_a13_last_full_block_lora(encoder)
    modules = iter_a13_full_block_lora_modules(encoder)

    assert receipt.attention_trainable_parameters == ATTENTION_LORA_PARAMETER_COUNT
    assert receipt.ffn_trainable_parameters == FFN_LORA_PARAMETER_COUNT
    assert receipt.trainable_parameters == FULL_BLOCK_LORA_PARAMETER_COUNT
    assert receipt.original_trainable_parameters == 0
    assert receipt.wrapped_modules == (
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    )
    assert len(modules) == 6
    assert all(isinstance(module, LoRALinear) for module in modules)
    assert modules[4].lora_parameter_count == FFN_INTERMEDIATE_LORA_PARAMETER_COUNT
    assert modules[5].lora_parameter_count == FFN_OUTPUT_LORA_PARAMETER_COUNT
    assert sum(p.numel() for p in encoder.model.parameters() if p.requires_grad) == 36864


def test_s29_zero_init_full_block_is_exact_linear_identity():
    encoder = _encoder()
    final_layer = encoder.model.encoder.layer[-1]
    x_attention = torch.randn(3, 7, 256)
    x_ffn = torch.randn(3, 7, 256)
    x_output = torch.randn(3, 7, 1024)

    before_q = final_layer.attention.self.query(x_attention).detach()
    before_intermediate = final_layer.intermediate.dense(x_ffn).detach()
    before_output = final_layer.output.dense(x_output).detach()

    inject_a13_last_full_block_lora(encoder)
    final_layer = encoder.model.encoder.layer[-1]

    assert torch.equal(
        final_layer.attention.self.query(x_attention).detach(),
        before_q,
    )
    assert torch.equal(
        final_layer.intermediate.dense(x_ffn).detach(),
        before_intermediate,
    )
    assert torch.equal(
        final_layer.output.dense(x_output).detach(),
        before_output,
    )


def test_s29_full_block_checkpoint_roundtrip():
    source = _encoder()
    inject_a13_last_full_block_lora(source)
    source_modules = iter_a13_full_block_lora_modules(source)
    with torch.no_grad():
        for index, module in enumerate(source_modules):
            module.lora_a.fill_(0.01 * (index + 1))
            module.lora_b.fill_(0.02 * (index + 1))
    state = a13_full_block_lora_state_dict(source)

    target = _encoder()
    inject_a13_last_full_block_lora(target)
    load_a13_full_block_lora_state_dict(target, state, freeze=True)
    replay = a13_full_block_lora_state_dict(target)

    assert set(replay) == set(state)
    for key in state:
        assert torch.equal(replay[key], state[key])
    assert all(
        not p.requires_grad
        for module in iter_a13_full_block_lora_modules(target)
        for p in (module.lora_a, module.lora_b)
    )


def test_s29_ffn_lora_b_receives_gradient():
    encoder = _encoder()
    inject_a13_last_full_block_lora(encoder)
    modules = iter_a13_full_block_lora_modules(encoder)

    x = torch.randn(2, 5, 256)
    hidden = modules[4](x)
    y = modules[5](hidden)
    loss = y.square().mean()
    loss.backward()

    assert modules[4].lora_b.grad is not None
    assert float(modules[4].lora_b.grad.abs().sum()) > 0.0
    assert modules[5].lora_b.grad is not None
    assert float(modules[5].lora_b.grad.abs().sum()) > 0.0
