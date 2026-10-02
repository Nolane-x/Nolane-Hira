import torch
import torch.nn.functional as F

from nmd.v1_native_relation_geometry import NativeA13RelationCanonicalizer
from nmd.v1_query_gated_signature_readout import NativeQueryGatedSignatureCorrectnessReadout


def _inputs(*, batch=3, k=4, views=2, s=5, q=3, t=4, d=256):
    torch.manual_seed(37001 + k)
    state = torch.randn(batch, s, d)
    question = torch.randn(batch, q, d)
    options = torch.randn(batch, k, views, t, d)
    sm = torch.ones(batch, s, dtype=torch.bool)
    qm = torch.ones(batch, q, dtype=torch.bool)
    otm = torch.ones(batch, k, views, t, dtype=torch.bool)
    ovm = torch.ones(batch, k, views, dtype=torch.bool)
    return dict(
        state_tokens=state,
        state_mask=sm,
        question_tokens=question,
        question_mask=qm,
        option_view_tokens=options,
        option_view_token_mask=otm,
        option_view_mask=ovm,
    )


def _base():
    return NativeA13RelationCanonicalizer(
        role_temperature=0.10,
        pair_temperature=0.10,
        contrastive_temperature=0.10,
        native_dimension=256,
    )


def _readout(*, train=True):
    return NativeQueryGatedSignatureCorrectnessReadout(
        role_temperature=0.10,
        pair_temperature=0.10,
        contrastive_temperature=0.10,
        native_dimension=256,
        query_norm_epsilon=1e-12,
        residual_scale=1.0,
        train_readout=train,
    )


def test_s37_exact_parameter_surface_and_zero_init():
    op = _readout()
    assert op.added_parameter_count == 256
    assert op.parameter_count == 256
    assert tuple(op.readout_weight.shape) == (256,)
    assert torch.count_nonzero(op.readout_weight).item() == 0
    assert {name for name, _ in op.named_parameters()} == {"readout_weight"}


def test_s37_zero_init_exact_identity_to_native_base():
    x = _inputs()
    base = _base()
    op = _readout()
    b_logits, b_sig, _ = base(**x)
    t_logits, t_sig, _ = op(**x)
    assert torch.equal(t_logits, b_logits)
    assert torch.equal(t_sig, b_sig)


def test_s37_query_summary_is_permutation_and_padding_invariant():
    x = _inputs(batch=2, q=4)
    op = _readout()
    q = x["question_tokens"]
    qm = x["question_mask"]
    base = op.query_summary(question_tokens=q, question_mask=qm)

    perm = torch.tensor([2, 0, 3, 1])
    permuted = op.query_summary(
        question_tokens=q[:, perm],
        question_mask=qm[:, perm],
    )
    assert torch.allclose(permuted, base, rtol=0.0, atol=2e-6)

    padded = torch.cat([q, torch.randn(2, 3, 256)], dim=1)
    padded_mask = torch.cat([qm, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
    padded_summary = op.query_summary(
        question_tokens=padded,
        question_mask=padded_mask,
    )
    assert torch.allclose(padded_summary, base, rtol=0.0, atol=2e-6)


def test_s37_nonzero_readout_responds_to_query_content():
    op = _readout()
    with torch.no_grad():
        op.readout_weight.copy_(torch.linspace(-0.2, 0.2, 256))

    torch.manual_seed(37111)
    signatures = F.normalize(torch.randn(2, 4, 256), dim=-1)
    q1 = torch.randn(2, 3, 256)
    q2 = q1.clone()
    q2[:, :, :128] *= -1.0
    qm = torch.ones(2, 3, dtype=torch.bool)

    r1 = op.readout_residual(
        signatures=signatures,
        question_tokens=q1,
        question_mask=qm,
    )
    r2 = op.readout_residual(
        signatures=signatures,
        question_tokens=q2,
        question_mask=qm,
    )
    assert float((r1 - r2).abs().max()) > 1e-6


def test_s37_arbitrary_k_three_and_seven():
    op = _readout()
    for k in (3, 7):
        logits, sig, _ = op(**_inputs(k=k))
        assert tuple(logits.shape) == (3, k)
        assert tuple(sig.shape) == (3, k, 256)
        assert bool(torch.isfinite(logits).all())
        assert bool(torch.isfinite(sig).all())


def test_s37_logical_option_permutation_equivariance():
    x = _inputs(k=7)
    op = _readout()
    with torch.no_grad():
        op.readout_weight.copy_(torch.linspace(-0.1, 0.1, 256))
    logits, sig, _ = op(**x)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    px = dict(x)
    px["option_view_tokens"] = x["option_view_tokens"][:, perm]
    px["option_view_token_mask"] = x["option_view_token_mask"][:, perm]
    px["option_view_mask"] = x["option_view_mask"][:, perm]
    plog, psig, _ = op(**px)
    assert torch.allclose(plog, logits[:, perm], rtol=0.0, atol=2e-6)
    assert torch.allclose(psig, sig[:, perm], rtol=0.0, atol=2e-6)


def test_s37_question_token_permutation_and_masked_padding_forward_invariance():
    x = _inputs(batch=2, q=4)
    op = _readout()
    with torch.no_grad():
        op.readout_weight.copy_(torch.linspace(-0.05, 0.05, 256))
    logits, sig, _ = op(**x)

    perm = torch.tensor([2, 0, 3, 1])
    px = dict(x)
    px["question_tokens"] = x["question_tokens"][:, perm]
    px["question_mask"] = x["question_mask"][:, perm]
    plog, psig, _ = op(**px)
    assert torch.allclose(plog, logits, rtol=0.0, atol=2e-6)
    assert torch.allclose(psig, sig, rtol=0.0, atol=2e-6)

    pad = dict(x)
    pad["question_tokens"] = torch.cat(
        [x["question_tokens"], torch.randn(2, 2, 256)],
        dim=1,
    )
    pad["question_mask"] = torch.cat(
        [x["question_mask"], torch.zeros(2, 2, dtype=torch.bool)],
        dim=1,
    )
    qlog, qsig, _ = op(**pad)
    assert torch.allclose(qlog, logits, rtol=0.0, atol=2e-6)
    assert torch.allclose(qsig, sig, rtol=0.0, atol=2e-6)


def test_s37_readout_gradient_is_live():
    x = _inputs(batch=4, k=4)
    op = _readout()
    logits, _, _ = op(**x)
    gold = torch.tensor([0, 1, 2, 3])
    loss = F.cross_entropy(logits, gold)
    loss.backward()
    assert op.readout_weight.grad is not None
    assert float(op.readout_weight.grad.abs().sum()) > 0.0


def test_s37_checkpoint_roundtrip():
    source = _readout()
    with torch.no_grad():
        source.readout_weight.copy_(torch.linspace(-0.25, 0.25, 256))
    state = source.readout_state_dict()
    target = _readout()
    target.load_readout_state_dict(state, freeze=True)
    replay = target.readout_state_dict()
    assert torch.equal(replay["readout.weight"], state["readout.weight"])
    assert target.readout_weight.requires_grad is False
