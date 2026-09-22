from nmd.sequential_trust import (
    SafeStepCandidate,
    choose_largest_safe_step,
    exact_zero_drop,
)


def row(
    eta,
    *,
    kl=1e-6,
    agree=6000,
    n=6000,
    finite=True,
    bit=True,
    benefit=1.0,
    l2=0.01,
):
    return SafeStepCandidate(
        eta=eta,
        mean_teacher_kl=kl,
        teacher_argmax_agreement_count=agree,
        trust_n=n,
        finite=finite,
        non_relation_bit_identical=bit,
        first_order_structural_predicted_benefit=benefit,
        applied_update_l2=l2,
    )


def test_choose_largest_safe_step_prefers_largest_eta():
    chosen = choose_largest_safe_step(
        [row(1/4096), row(1/512), row(1/1024)],
        mean_kl_ceiling=1e-4,
    )
    assert chosen is not None
    assert chosen.eta == 1/512


def test_choose_largest_safe_step_rejects_single_argmax_flip():
    chosen = choose_largest_safe_step(
        [row(1/64, agree=5999), row(1/128, agree=6000)],
        mean_kl_ceiling=1e-4,
    )
    assert chosen is not None
    assert chosen.eta == 1/128


def test_choose_largest_safe_step_rejects_kl_over_budget():
    chosen = choose_largest_safe_step(
        [row(1/64, kl=2e-4), row(1/128, kl=9e-5)],
        mean_kl_ceiling=1e-4,
    )
    assert chosen is not None
    assert chosen.eta == 1/128


def test_choose_largest_safe_step_returns_none_when_all_unsafe():
    assert choose_largest_safe_step(
        [
            row(1/64, agree=5999),
            row(1/128, benefit=0.0),
            row(1/256, finite=False),
        ],
        mean_kl_ceiling=1e-4,
    ) is None


def test_exact_zero_drop_is_count_space():
    assert exact_zero_drop(candidate_correct=800, baseline_correct=800)
    assert exact_zero_drop(candidate_correct=801, baseline_correct=800)
    assert not exact_zero_drop(candidate_correct=799, baseline_correct=800)
