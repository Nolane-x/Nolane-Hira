import torch
import torch.nn.functional as F

from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork


def _inputs(*, batch=3, k=4, q=5, d=256):
    g = torch.Generator().manual_seed(44001 + k)
    signatures = F.normalize(torch.randn(batch, k, d, generator=g), dim=-1)
    question = torch.randn(batch, q, d, generator=g)
    mask = torch.ones(batch, q, dtype=torch.bool)
    native_logits = torch.randn(batch, k, generator=g)
    return native_logits, signatures, question, mask


def _op(*, train=True):
    return PrivateCorrectionRepresentationFork(train_correction=train)


def test_s44_exact_parameter_surface_and_initialization():
    op = _op()
    assert op.adapter_a_parameter_count == 32768
    assert op.adapter_b_parameter_count == 16384
    assert op.private_adapter_parameter_count == 49152
    assert op.bilinear_parameter_count == 65536
    assert op.correction_parameter_count == 114688
    assert tuple(op.adapter_a.shape) == (64, 512)
    assert tuple(op.adapter_b.shape) == (256, 64)
    assert tuple(op.bilinear_weight.shape) == (256, 256)
    assert torch.count_nonzero(op.adapter_b).item() == 0
    assert torch.count_nonzero(op.bilinear_weight).item() == 0
    assert bool(torch.isfinite(op.adapter_a).all())


def test_s44_adapter_a_is_deterministic_without_global_rng_dependency():
    torch.manual_seed(1)
    a = _op().adapter_a.detach().clone()
    torch.manual_seed(999999)
    b = _op().adapter_a.detach().clone()
    assert torch.equal(a, b)


def test_s44_zero_init_corrected_logits_are_exact_native_identity():
    native, sig, question, mask = _inputs()
    op = _op()
    corrected = op.correction_logits(
        native_logits=native,
        signatures=sig,
        question_tokens=question,
        question_mask=mask,
    )
    assert torch.equal(corrected, native)


def test_s44_zero_b_private_residual_is_exact_zero_and_private_signature_is_close():
    _native, sig, question, mask = _inputs()
    op = _op()
    residual, _q = op.private_residual(
        signatures=sig,
        question_tokens=question,
        question_mask=mask,
    )
    private, _q2 = op.private_signatures(
        signatures=sig,
        question_tokens=question,
        question_mask=mask,
    )
    assert torch.count_nonzero(residual).item() == 0
    assert torch.allclose(private, sig, rtol=0.0, atol=2e-6)


def test_s44_correction_inputs_are_detached_from_native_graph():
    native, sig, question, mask = _inputs(batch=2)
    native = native.requires_grad_()
    sig = sig.requires_grad_()
    question = question.requires_grad_()
    op = _op()

    corrected = op.correction_logits(
        native_logits=native,
        signatures=sig,
        question_tokens=question,
        question_mask=mask,
    )
    gold = torch.tensor([0, 1])
    loss = F.cross_entropy(corrected, gold)
    loss.backward()

    assert native.grad is None
    assert sig.grad is None
    assert question.grad is None
    assert op.bilinear_weight.grad is not None
    assert float(op.bilinear_weight.grad.abs().sum()) > 0.0


def test_s44_private_branch_a_b_w_all_live_under_nonzero_probe():
    native, sig, question, mask = _inputs(batch=4)
    op = _op()
    g = torch.Generator().manual_seed(44111)
    with torch.no_grad():
        op.adapter_b.copy_(torch.randn(256, 64, generator=g) * 0.01)
        op.bilinear_weight.copy_(torch.randn(256, 256, generator=g) * 0.01)

    corrected = op.correction_logits(
        native_logits=native,
        signatures=sig,
        question_tokens=question,
        question_mask=mask,
    )
    gold = torch.tensor([0, 1, 2, 3])
    F.cross_entropy(corrected, gold).backward()

    for p in (op.adapter_a, op.adapter_b, op.bilinear_weight):
        assert p.grad is not None
        assert float(p.grad.abs().sum()) > 0.0


def test_s44_private_branch_is_not_equivalent_to_w_only_under_probe():
    _native, sig, question, mask = _inputs(batch=2)
    op = _op()
    g = torch.Generator().manual_seed(44222)
    with torch.no_grad():
        op.adapter_b.copy_(torch.randn(256, 64, generator=g) * 0.02)
        op.bilinear_weight.copy_(torch.randn(256, 256, generator=g) * 0.01)

    private_residual = op.correction_residual(
        signatures=sig,
        question_tokens=question,
        question_mask=mask,
    )
    q = op.query_summary(question_tokens=question, question_mask=mask)
    w_only = torch.einsum(
        "bkd,de,be->bk",
        sig,
        op.bilinear_weight.to(sig.dtype),
        q,
    )
    assert float((private_residual - w_only).abs().max()) > 1e-6


def test_s44_logical_option_permutation_equivariance():
    native, sig, question, mask = _inputs(k=7)
    op = _op()
    g = torch.Generator().manual_seed(44333)
    with torch.no_grad():
        op.adapter_b.copy_(torch.randn(256, 64, generator=g) * 0.01)
        op.bilinear_weight.copy_(torch.randn(256, 256, generator=g) * 0.01)

    base = op.correction_logits(
        native_logits=native,
        signatures=sig,
        question_tokens=question,
        question_mask=mask,
    )
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    moved = op.correction_logits(
        native_logits=native[:, perm],
        signatures=sig[:, perm],
        question_tokens=question,
        question_mask=mask,
    )
    assert torch.allclose(moved, base[:, perm], rtol=0.0, atol=2e-6)


def test_s44_question_permutation_and_padding_invariance():
    native, sig, question, mask = _inputs(batch=2, q=4)
    op = _op()
    g = torch.Generator().manual_seed(44444)
    with torch.no_grad():
        op.adapter_b.copy_(torch.randn(256, 64, generator=g) * 0.01)
        op.bilinear_weight.copy_(torch.randn(256, 256, generator=g) * 0.01)

    base = op.correction_logits(
        native_logits=native,
        signatures=sig,
        question_tokens=question,
        question_mask=mask,
    )
    perm = torch.tensor([2, 0, 3, 1])
    moved = op.correction_logits(
        native_logits=native,
        signatures=sig,
        question_tokens=question[:, perm],
        question_mask=mask[:, perm],
    )
    assert torch.allclose(moved, base, rtol=0.0, atol=2e-6)

    padded_q = torch.cat([question, torch.randn(2, 3, 256)], dim=1)
    padded_m = torch.cat([mask, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
    padded = op.correction_logits(
        native_logits=native,
        signatures=sig,
        question_tokens=padded_q,
        question_mask=padded_m,
    )
    assert torch.allclose(padded, base, rtol=0.0, atol=2e-6)


def test_s44_arbitrary_k_three_and_seven():
    op = _op()
    for k in (3, 7):
        native, sig, question, mask = _inputs(k=k)
        logits = op.correction_logits(
            native_logits=native,
            signatures=sig,
            question_tokens=question,
            question_mask=mask,
        )
        private, q = op.private_signatures(
            signatures=sig,
            question_tokens=question,
            question_mask=mask,
        )
        assert tuple(logits.shape) == (3, k)
        assert tuple(private.shape) == (3, k, 256)
        assert tuple(q.shape) == (3, 256)


def test_s44_checkpoint_roundtrip():
    source = _op()
    g = torch.Generator().manual_seed(44555)
    with torch.no_grad():
        source.adapter_b.copy_(torch.randn(256, 64, generator=g) * 0.01)
        source.bilinear_weight.copy_(torch.randn(256, 256, generator=g) * 0.01)

    state = source.correction_state_dict()
    target = _op()
    target.load_correction_state_dict(state, freeze=True)
    replay = target.correction_state_dict()

    assert set(replay) == {"adapter.a", "adapter.b", "bilinear.weight"}
    for key in replay:
        assert torch.equal(replay[key], state[key])
    assert all(not p.requires_grad for p in target.correction_parameters())
