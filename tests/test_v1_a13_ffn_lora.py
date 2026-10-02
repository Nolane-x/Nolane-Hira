from types import SimpleNamespace

import torch
from torch import nn

from nmd.semantic import HFAutoSemanticEncoder
from nmd.v1_a13_ffn_lora import (
    FFN_INTERMEDIATE_LORA_PARAMETER_COUNT,
    FFN_ONLY_LORA_PARAMETER_COUNT,
    FFN_OUTPUT_LORA_PARAMETER_COUNT,
    a13_ffn_only_lora_state_dict,
    inject_a13_last_ffn_only_lora,
    iter_a13_ffn_only_lora_modules,
    load_a13_ffn_only_lora_state_dict,
)


class _SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Linear(256,256)
        self.key=nn.Linear(256,256)
        self.value=nn.Linear(256,256)


class _AttentionOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense=nn.Linear(256,256)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self=_SelfAttention()
        self.output=_AttentionOutput()


class _Intermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense=nn.Linear(256,1024)


class _Output(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense=nn.Linear(1024,256)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=_Attention()
        self.intermediate=_Intermediate()
        self.output=_Output()


class _EncoderStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.ModuleList([_Layer() for _ in range(6)])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=_EncoderStack()
        self.config=SimpleNamespace(
            model_type="bert",
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
        )


def _encoder():
    encoder=object.__new__(HFAutoSemanticEncoder)
    nn.Module.__init__(encoder)
    encoder.model=_Model()
    encoder.d_model=256
    return encoder


def test_s30_ffn_only_exact_surface():
    encoder=_encoder()
    receipt=inject_a13_last_ffn_only_lora(encoder)
    modules=iter_a13_ffn_only_lora_modules(encoder)
    assert receipt.wrapped_modules==("intermediate.dense","output.dense")
    assert receipt.trainable_parameters==FFN_ONLY_LORA_PARAMETER_COUNT==20480
    assert modules[0].lora_parameter_count==FFN_INTERMEDIATE_LORA_PARAMETER_COUNT==10240
    assert modules[1].lora_parameter_count==FFN_OUTPUT_LORA_PARAMETER_COUNT==10240
    assert sum(p.numel() for p in encoder.model.parameters() if p.requires_grad)==20480


def test_s30_ffn_only_zero_init_is_exact_identity():
    encoder=_encoder()
    final=encoder.model.encoder.layer[-1]
    x1=torch.randn(3,5,256)
    x2=torch.randn(3,5,1024)
    before1=final.intermediate.dense(x1).detach()
    before2=final.output.dense(x2).detach()
    inject_a13_last_ffn_only_lora(encoder)
    final=encoder.model.encoder.layer[-1]
    assert torch.equal(final.intermediate.dense(x1).detach(),before1)
    assert torch.equal(final.output.dense(x2).detach(),before2)


def test_s30_ffn_only_checkpoint_roundtrip():
    source=_encoder()
    inject_a13_last_ffn_only_lora(source)
    modules=iter_a13_ffn_only_lora_modules(source)
    with torch.no_grad():
        modules[0].lora_a.fill_(0.01)
        modules[0].lora_b.fill_(0.02)
        modules[1].lora_a.fill_(0.03)
        modules[1].lora_b.fill_(0.04)
    state=a13_ffn_only_lora_state_dict(source)
    target=_encoder()
    inject_a13_last_ffn_only_lora(target)
    load_a13_ffn_only_lora_state_dict(target,state,freeze=True)
    replay=a13_ffn_only_lora_state_dict(target)
    assert set(replay)==set(state)
    assert all(torch.equal(replay[k],state[k]) for k in state)
    assert all(
        not p.requires_grad
        for module in iter_a13_ffn_only_lora_modules(target)
        for p in (module.lora_a,module.lora_b)
    )


def test_s30_ffn_b_gradients_are_live():
    encoder=_encoder()
    inject_a13_last_ffn_only_lora(encoder)
    modules=iter_a13_ffn_only_lora_modules(encoder)
    x=torch.randn(2,5,256)
    y=modules[1](modules[0](x))
    y.square().mean().backward()
    assert modules[0].lora_b.grad is not None
    assert float(modules[0].lora_b.grad.abs().sum())>0.0
    assert modules[1].lora_b.grad is not None
    assert float(modules[1].lora_b.grad.abs().sum())>0.0
