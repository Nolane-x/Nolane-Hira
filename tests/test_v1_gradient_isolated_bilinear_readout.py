import torch
import torch.nn.functional as F

from nmd.v1_native_relation_geometry import NativeA13RelationCanonicalizer
from nmd.v1_gradient_isolated_bilinear_readout import (
    NativeGradientIsolatedBilinearCorrectnessReadout,
)


def _inputs(*, batch=3, k=4, views=2, s=5, q=4, t=4, d=256):
    torch.manual_seed(39001 + k)
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
    return NativeGradientIsolatedBilinearCorrectnessReadout(
        role_temperature=0.10,
        pair_temperature=0.10,
        contrastive_temperature=0.10,
        native_dimension=256,
        query_norm_epsilon=1e-12,
        residual_scale=1.0,
        train_readout=train,
    )


def _off_diagonal_weight():
    w = torch.zeros(256, 256)
    idx = torch.arange(256)
    w[idx, (idx + 19) % 256] = torch.linspace(-0.2, 0.2, 256)
    return w


def test_s39_exact_parameter_surface_and_zero_init():
    op = _readout()
    assert op.added_parameter_count == 65536
    assert op.parameter_count == 65536
    assert tuple(op.bilinear_weight.shape) == (256, 256)
    assert torch.count_nonzero(op.bilinear_weight).item() == 0
    assert {name for name, _ in op.named_parameters()} == {"bilinear_weight"}


def test_s39_zero_init_exact_evaluation_identity_to_native_base():
    x = _inputs()
    base = _base()
    op = _readout()
    b_logits, b_sig, _ = base(**x)
    t_logits, t_sig, _ = op(**x)
    assert torch.equal(t_logits, b_logits)
    assert torch.equal(t_sig, b_sig)


def test_s39_correction_path_is_hard_detached_upstream():
    torch.manual_seed(39111)
    op = _readout()
    native_logits = torch.randn(4, 4, requires_grad=True)
    signatures = F.normalize(torch.randn(4, 4, 256), dim=-1).detach().requires_grad_(True)
    question = torch.randn(4, 3, 256, requires_grad=True)
    qm = torch.ones(4, 3, dtype=torch.bool)
    gold = torch.tensor([0, 1, 2, 3])

    logits = op.correction_logits(
        native_logits=native_logits,
        signatures=signatures,
        question_tokens=question,
        question_mask=qm,
    )
    loss = F.cross_entropy(logits, gold)
    loss.backward()

    assert op.bilinear_weight.grad is not None
    assert float(op.bilinear_weight.grad.abs().sum()) > 0.0
    diag = torch.diagonal(op.bilinear_weight.grad)
    assert float(op.bilinear_weight.grad.abs().sum() - diag.abs().sum()) > 0.0
    assert native_logits.grad is None or float(native_logits.grad.abs().sum()) == 0.0
    assert signatures.grad is None or float(signatures.grad.abs().sum()) == 0.0
    assert question.grad is None or float(question.grad.abs().sum()) == 0.0


def test_s39_query_summary_is_permutation_and_padding_invariant():
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


def test_s39_nonzero_map_responds_to_query_and_signature():
    op = _readout()
    with torch.no_grad():
        op.bilinear_weight.copy_(_off_diagonal_weight())

    torch.manual_seed(39222)
    signatures = F.normalize(torch.randn(2, 4, 256), dim=-1)
    question = torch.randn(2, 3, 256)
    qm = torch.ones(2, 3, dtype=torch.bool)

    base = op.detached_readout_residual(
        signatures=signatures,
        question_tokens=question,
        question_mask=qm,
    )

    q2 = question.clone()
    q2[:, :, :128] *= -1.0
    changed_q = op.detached_readout_residual(
        signatures=signatures,
        question_tokens=q2,
        question_mask=qm,
    )
    assert float((base - changed_q).abs().max()) > 1e-6

    s2 = signatures.clone()
    s2[:, :, 128:] *= -1.0
    changed_s = op.detached_readout_residual(
        signatures=s2,
        question_tokens=question,
        question_mask=qm,
    )
    assert float((base - changed_s).abs().max()) > 1e-6


def test_s39_arbitrary_k_three_and_seven():
    op = _readout()
    for k in (3, 7):
        logits, sig, _ = op(**_inputs(k=k))
        assert tuple(logits.shape) == (3, k)
        assert tuple(sig.shape) == (3, k, 256)
        assert bool(torch.isfinite(logits).all())
        assert bool(torch.isfinite(sig).all())


def test_s39_logical_option_permutation_equivariance():
    x = _inputs(k=7)
    op = _readout()
    with torch.no_grad():
        op.bilinear_weight.copy_(_off_diagonal_weight())
    logits, sig, _ = op(**x)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    px = dict(x)
    px["option_view_tokens"] = x["option_view_tokens"][:, perm]
    px["option_view_token_mask"] = x["option_view_token_mask"][:, perm]
    px["option_view_mask"] = x["option_view_mask"][:, perm]
    plog, psig, _ = op(**px)
    assert torch.allclose(plog, logits[:, perm], rtol=0.0, atol=2e-6)
    assert torch.allclose(psig, sig[:, perm], rtol=0.0, atol=2e-6)


def test_s39_question_token_permutation_and_masked_padding_forward_invariance():
    x = _inputs(batch=2, q=4)
    op = _readout()
    with torch.no_grad():
        op.bilinear_weight.copy_(_off_diagonal_weight())
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


def test_s39_checkpoint_roundtrip():
    source = _readout()
    with torch.no_grad():
        source.bilinear_weight.copy_(_off_diagonal_weight())
    state = source.readout_state_dict()
    target = _readout()
    target.load_readout_state_dict(state, freeze=True)
    replay = target.readout_state_dict()
    assert torch.equal(replay["bilinear.weight"], state["bilinear.weight"])
    assert target.bilinear_weight.requires_grad is False
