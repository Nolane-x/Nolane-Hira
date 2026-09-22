import torch

from nmd.functional_trust import functional_trust_metrics, trust_safe


def test_identical_teacher_student_is_fully_trusted():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]])
    teacher = torch.softmax(logits, dim=-1)

    metrics = functional_trust_metrics(logits, teacher)

    assert metrics.finite
    assert metrics.n == 2
    assert metrics.teacher_argmax_agreement_count == 2
    assert metrics.teacher_argmax_agreement_fraction == 1.0
    assert abs(metrics.mean_teacher_kl) < 1e-6
    assert trust_safe(metrics, mean_kl_ceiling=1e-4)


def test_argmax_change_fails_full_agreement_even_with_finite_logits():
    teacher_logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    teacher = torch.softmax(teacher_logits, dim=-1)
    student = torch.tensor([[1.0, 2.0, 0.0], [0.0, 2.0, 1.0]])

    metrics = functional_trust_metrics(student, teacher)

    assert metrics.finite
    assert metrics.teacher_argmax_agreement_count == 1
    assert metrics.teacher_argmax_agreement_fraction == 0.5
    assert not trust_safe(metrics, mean_kl_ceiling=10.0)


def test_nonfinite_student_fails_closed():
    teacher = torch.tensor([[0.7, 0.2, 0.1]])
    student = torch.tensor([[float("nan"), 0.0, 0.0]])

    metrics = functional_trust_metrics(student, teacher)

    assert not metrics.finite
    assert not trust_safe(metrics, mean_kl_ceiling=1e-4)


def test_kl_ceiling_is_enforced():
    teacher = torch.tensor([[0.99, 0.005, 0.005]])
    student = torch.tensor([[1.0, 0.9, 0.8]])

    metrics = functional_trust_metrics(student, teacher)

    assert metrics.teacher_argmax_agreement_count == 1
    assert metrics.mean_teacher_kl > 1e-4
    assert not trust_safe(metrics, mean_kl_ceiling=1e-4)
