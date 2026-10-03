import torch
import torch.nn.functional as F

from nmd.v1_private_correction_fork import PrivateCorrectionRepresentationFork

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from hira_v1_s45_a0_cross_view_consistent_private_correction import _js_divergence


def test_s45_js_is_symmetric_zero_on_identity_and_positive_on_shift():
    a=torch.tensor([[2.0,0.5,-1.0,0.0],[0.0,1.0,2.0,-0.5]])
    b=a.roll(shifts=1,dims=-1)
    zero=float(_js_divergence(a,a))
    ab=float(_js_divergence(a,b))
    ba=float(_js_divergence(b,a))
    assert zero <= 1e-12
    assert ab > 0.0
    assert abs(ab-ba) <= 1e-7


def test_s45_js_only_updates_private_a_b_w_and_detaches_native_inputs():
    g=torch.Generator().manual_seed(450045)
    batch,k,q,d=4,4,5,256

    native_c=torch.randn(batch,k,generator=g,requires_grad=True)
    native_p=torch.randn(batch,k,generator=g,requires_grad=True)
    sig_c=F.normalize(torch.randn(batch,k,d,generator=g),dim=-1).requires_grad_()
    sig_p=F.normalize(torch.randn(batch,k,d,generator=g),dim=-1).requires_grad_()
    question_c=torch.randn(batch,q,d,generator=g,requires_grad=True)
    question_p=torch.randn(batch,q,d,generator=g,requires_grad=True)
    mask=torch.ones(batch,q,dtype=torch.bool)

    op=PrivateCorrectionRepresentationFork(train_correction=True)
    with torch.no_grad():
        op.adapter_b.copy_(torch.randn(256,64,generator=g)*0.02)
        op.bilinear_weight.copy_(torch.randn(256,256,generator=g)*0.02)

    cc=op.correction_logits(
        native_logits=native_c,
        signatures=sig_c,
        question_tokens=question_c,
        question_mask=mask,
    )
    cp=op.correction_logits(
        native_logits=native_p,
        signatures=sig_p,
        question_tokens=question_p,
        question_mask=mask,
    )
    js=_js_divergence(cc,cp)
    assert bool(torch.isfinite(js))
    assert float(js)>0.0

    grads=torch.autograd.grad(js,op.correction_parameters(),retain_graph=True)
    assert all(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum())>0.0 for g in grads)

    native_grads=torch.autograd.grad(
        js,
        [native_c,native_p,sig_c,sig_p,question_c,question_p],
        allow_unused=True,
    )
    assert all(g is None for g in native_grads)
