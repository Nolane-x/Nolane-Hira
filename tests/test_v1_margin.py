import torch

from nmd.v1_margin import (
    gold_vs_all_negative_margin_loss,
    gold_vs_max_wrong_margin,
    margin_satisfaction_rate,
    top1_top2_margin,
)


def test_s9_all_negative_margin_zero_when_every_wrong_is_far_enough():
    logits = torch.tensor([
        [1.0, 0.5, 0.0, -0.3],
        [0.2, 1.2, 0.4, 0.1],
    ])
    gold = torch.tensor([0, 1], dtype=torch.long)
    loss = gold_vs_all_negative_margin_loss(logits, gold, margin=0.20)
    assert torch.equal(loss, torch.tensor(0.0))


def test_s9_all_negative_margin_penalizes_every_violating_wrong_option():
    logits = torch.tensor([[0.30, 0.25, 0.20, -1.0]], requires_grad=True)
    gold = torch.tensor([0], dtype=torch.long)
    # deficits at m=.20: 0.15, 0.10, 0.00 -> mean = 0.083333...
    loss = gold_vs_all_negative_margin_loss(logits, gold, margin=0.20)
    assert torch.allclose(loss, torch.tensor(1.0 / 12.0), atol=1e-7)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_s9_signed_gold_margin_detects_correct_and_wrong_separation():
    logits = torch.tensor([
        [1.0, 0.4, 0.2, 0.1],
        [0.3, 0.8, 1.1, 0.2],
    ])
    gold = torch.tensor([0, 1], dtype=torch.long)
    margins = gold_vs_max_wrong_margin(logits, gold)
    assert torch.allclose(margins, torch.tensor([0.6, -0.3]), atol=1e-7)


def test_s9_margin_satisfaction_rate_uses_strongest_wrong():
    logits = torch.tensor([
        [1.0, 0.7, 0.1, 0.0],
        [0.5, 0.4, 0.3, 0.2],
        [0.0, 0.1, 0.5, 0.4],
        [0.3, 0.1, 0.0, -0.4],
    ])
    gold = torch.tensor([0, 0, 2, 0], dtype=torch.long)
    # signed margins: .3, .1, .1, .2; exact floating point at .2 is
    # intentionally not relied on here, so threshold .19 gives rows 1 and 4.
    rate = margin_satisfaction_rate(logits, gold, margin=0.19)
    assert rate.item() == 0.5


def test_s9_top1_top2_margin_is_order_agnostic():
    logits = torch.tensor([
        [1.2, 0.7, -1.0, 0.1],
        [0.1, 0.9, 0.4, 1.3],
    ])
    margins = top1_top2_margin(logits)
    assert torch.allclose(margins, torch.tensor([0.5, 0.4]), atol=1e-7)


def test_s9_margin_rejects_bad_gold_shape_and_nonfinite_logits():
    logits = torch.randn(3, 4)
    try:
        gold_vs_all_negative_margin_loss(
            logits,
            torch.tensor([[0, 1, 2]], dtype=torch.long),
        )
    except ValueError as exc:
        assert "gold shape" in str(exc)
    else:
        raise AssertionError("bad gold shape must fail")

    bad = logits.clone()
    bad[0, 0] = float("nan")
    try:
        top1_top2_margin(bad)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite logits must fail")
