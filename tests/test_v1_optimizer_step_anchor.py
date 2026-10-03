import torch

from nmd.v1_optimizer_step_anchor import (
    adamw_candidate_deltas,
    apply_parameter_deltas,
    initialize_adamw_state,
    project_adamw_runtime_delta_against_anchor,
)


def _std_optimizer(params):
    return torch.optim.AdamW(
        params,
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )


def _assert_state_matches(custom, optimizer, params):
    for state, param in zip(custom, params):
        ref = optimizer.state[param]
        assert state.step == int(ref["step"].item())
        assert torch.equal(state.exp_avg, ref["exp_avg"])
        assert torch.equal(state.exp_avg_sq, ref["exp_avg_sq"])


def test_s41_adamw_candidate_matches_pytorch_multistep():
    torch.manual_seed(41001)
    custom_params = [
        torch.randn(5, 7, dtype=torch.float32),
        torch.randn(11, dtype=torch.float32),
    ]
    std_params = [torch.nn.Parameter(x.clone()) for x in custom_params]
    custom_params = [torch.nn.Parameter(x.clone()) for x in custom_params]

    states = initialize_adamw_state(custom_params)
    opt = _std_optimizer(std_params)

    for step in range(1, 6):
        g = torch.Generator().manual_seed(41100 + step)
        grads = [
            torch.randn(p.shape, generator=g, dtype=p.dtype)
            for p in custom_params
        ]

        deltas, next_states = adamw_candidate_deltas(
            custom_params,
            grads,
            states,
            lr=2e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        before = [p.detach().clone() for p in custom_params]
        apply_parameter_deltas(custom_params, deltas)

        opt.zero_grad(set_to_none=True)
        for p, grad in zip(std_params, grads):
            p.grad = grad.clone()
        opt.step()

        for p, q, old, delta in zip(custom_params, std_params, before, deltas):
            assert torch.equal(p.detach(), q.detach())
            assert torch.equal(p.detach() - old, delta)

        states = next_states
        _assert_state_matches(states, opt, std_params)


def test_s41_adamw_candidate_includes_decoupled_weight_decay():
    custom = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float32))
    standard = torch.nn.Parameter(custom.detach().clone())
    grad = torch.zeros_like(custom)

    states = initialize_adamw_state([custom])
    deltas, next_states = adamw_candidate_deltas(
        [custom],
        [grad],
        states,
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    before = custom.detach().clone()
    apply_parameter_deltas([custom], deltas)

    opt = _std_optimizer([standard])
    standard.grad = grad.clone()
    opt.step()

    assert torch.equal(custom.detach(), standard.detach())
    assert bool((deltas[0] != 0).any())
    assert torch.all(deltas[0] * before <= 0)
    _assert_state_matches(next_states, opt, [standard])


def test_s41_actual_step_projection_removes_anchor_increasing_component():
    delta = [torch.tensor([2.0, 1.0]), torch.tensor([0.5])]
    anchor = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    out, diag = project_adamw_runtime_delta_against_anchor(delta, anchor)
    assert diag.projected is True
    assert diag.pre_dot > 0.0
    assert abs(diag.post_dot) < 1e-6
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0]), rtol=0.0, atol=1e-6)
    assert torch.equal(out[1], delta[1])


def test_s41_actual_step_projection_is_identity_when_anchor_safe():
    delta = [torch.tensor([-2.0, 1.0]), torch.tensor([0.5])]
    anchor = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    out, diag = project_adamw_runtime_delta_against_anchor(delta, anchor)
    assert diag.projected is False
    assert diag.pre_dot <= 0.0
    assert diag.post_dot == diag.pre_dot
    assert all(torch.equal(x, y) for x, y in zip(out, delta))


def test_s41_actual_step_projection_zero_anchor_is_identity():
    delta = [torch.tensor([2.0, 1.0])]
    anchor = [torch.zeros(2)]
    out, diag = project_adamw_runtime_delta_against_anchor(delta, anchor)
    assert diag.projected is False
    assert diag.anchor_gradient_norm == 0.0
    assert diag.pre_dot == 0.0
    assert diag.post_dot == 0.0
    assert torch.equal(out[0], delta[0])


def test_s41_w_delta_can_be_kept_bitwise_outside_runtime_projection():
    runtime_delta = [torch.tensor([3.0, 1.0])]
    anchor = [torch.tensor([1.0, 0.0])]
    w_delta = torch.randn(4, 4)
    before = w_delta.clone()
    out, diag = project_adamw_runtime_delta_against_anchor(runtime_delta, anchor)
    assert diag.projected is True
    assert torch.equal(w_delta, before)
    assert not torch.equal(out[0], runtime_delta[0])


def test_s41_adamw_candidate_delta_parity_is_direct_movement_parity():
    # Mixed scales expose why old + (candidate-old) is not a valid bitwise
    # parity test in float32. S41's contract is on the optimizer movement.
    custom = torch.nn.Parameter(torch.tensor(
        [1.0e8, -1.0e8, 1.0, -1.0, 1.0e-7, -1.0e-7],
        dtype=torch.float32,
    ))
    standard = torch.nn.Parameter(custom.detach().clone())
    grad = torch.tensor([0.3, -0.2, 0.7, -0.9, 0.4, -0.5], dtype=torch.float32)

    states = initialize_adamw_state([custom])
    deltas, next_states = adamw_candidate_deltas(
        [custom], [grad], states,
        lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )

    opt = _std_optimizer([standard])
    standard.grad = grad.clone()
    opt.step()

    standard_delta = standard.detach() - custom.detach()
    assert torch.equal(deltas[0], standard_delta)
    _assert_state_matches(next_states, opt, [standard])
