import torch

from nmd.v1_s25_gradient_ownership import apply_s25_decoupled_gradient_update


def test_s25_gradient_router_keeps_private_surfaces_isolated():
    shared = torch.nn.Parameter(torch.tensor([0.7, -0.4]))
    primary = torch.nn.Parameter(torch.tensor([0.2, 0.5]))
    relation = torch.nn.Parameter(torch.tensor([-0.3, 0.9]))

    primary_block = ((2.0 * shared + 3.0 * primary) ** 2).sum()
    relation_block = ((-1.5 * shared + 4.0 * relation) ** 2).sum()

    d = apply_s25_decoupled_gradient_update(
        primary_block=primary_block,
        relation_block=relation_block,
        shared=[shared],
        primary_private=[primary],
        relation_private=[relation],
    )

    assert d.primary_to_relation_private_max_abs == 0.0
    assert d.relation_to_primary_private_max_abs == 0.0
    assert d.primary_private_l1 > 0.0
    assert d.relation_private_l1 > 0.0
    assert d.primary_shared_l1 > 0.0
    assert d.relation_shared_l1 > 0.0
    assert shared.grad is not None and torch.isfinite(shared.grad).all()
    assert primary.grad is not None and torch.isfinite(primary.grad).all()
    assert relation.grad is not None and torch.isfinite(relation.grad).all()


def test_s25_gradient_router_rejects_overlapping_groups():
    shared = torch.nn.Parameter(torch.tensor([1.0]))
    relation = torch.nn.Parameter(torch.tensor([2.0]))
    primary_block = (shared.square()).sum()
    relation_block = (relation.square()).sum()

    try:
        apply_s25_decoupled_gradient_update(
            primary_block=primary_block,
            relation_block=relation_block,
            shared=[shared],
            primary_private=[shared],
            relation_private=[relation],
        )
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("S25 overlapping parameter groups must fail")
