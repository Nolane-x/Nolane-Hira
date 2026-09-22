import torch

from nmd.functional_damage import (
    finite_sample_relative_accuracy_pass,
    teacher_kl_loss,
    weighted_batch_gradient_row,
)


def test_teacher_kl_is_zero_for_identical_distribution():
    logits = torch.tensor(
        [[1.0, 2.0, -0.5], [0.2, -0.1, 0.7]],
        dtype=torch.float64,
    )
    teacher = torch.softmax(logits, dim=-1)
    loss = teacher_kl_loss(logits.clone(), teacher)
    assert abs(float(loss)) < 1e-12


def test_teacher_kl_is_positive_for_drifted_student():
    teacher_logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64)
    teacher = torch.softmax(teacher_logits, dim=-1)
    student = torch.tensor([[-1.0, 0.0, 2.0]], dtype=torch.float64)
    loss = teacher_kl_loss(student, teacher)
    assert float(loss) > 0.1


def test_teacher_kl_never_backpropagates_into_teacher():
    teacher_logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    teacher = torch.softmax(teacher_logits, dim=-1)
    student = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss = teacher_kl_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert teacher_logits.grad is None


def test_weighted_gradient_row_uses_sqrt_sample_fraction():
    grad = torch.tensor([2.0, 4.0], dtype=torch.float64)
    row = weighted_batch_gradient_row(grad, batch_n=25, total_n=100)
    assert torch.allclose(row, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_relative_accuracy_boundary_uses_exact_counts_not_float32_roundoff():
    baseline = float(torch.tensor(799 / 1500, dtype=torch.float32))
    candidate = float(torch.tensor(796 / 1500, dtype=torch.float32))
    assert candidate < baseline - 0.002
    assert finite_sample_relative_accuracy_pass(
        candidate,
        baseline,
        n=1500,
        tolerance=0.002,
    )


def test_relative_accuracy_rejects_one_extra_error_beyond_count_budget():
    baseline = float(torch.tensor(799 / 1500, dtype=torch.float32))
    candidate = float(torch.tensor(795 / 1500, dtype=torch.float32))
    assert not finite_sample_relative_accuracy_pass(
        candidate,
        baseline,
        n=1500,
        tolerance=0.002,
    )
