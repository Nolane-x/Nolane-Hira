import torch
import torch.nn.functional as F

from nmd.v1_blockwise_factor_canonicalization import (
    FACTORIZED_BLOCK_DIMENSION,
    blockwise_cross_view_factor_canonicalization_loss,
    split_factorized_role_value_signature,
)


def _orthogonal_factorized(batch: int = 1, k: int = 4) -> torch.Tensor:
    role = torch.zeros(batch, k, FACTORIZED_BLOCK_DIMENSION)
    value = torch.zeros(batch, k, FACTORIZED_BLOCK_DIMENSION)
    for i in range(k):
        role[:, i, i] = 1.0
        value[:, i, 16 + i] = 1.0
    return F.normalize(torch.cat((role, value), dim=-1), dim=-1)


def test_s27_split_is_exact_128_plus_128():
    x = _orthogonal_factorized()
    role, value = split_factorized_role_value_signature(x)
    assert role.shape[-1] == 128
    assert value.shape[-1] == 128
    assert torch.isfinite(role).all()
    assert torch.isfinite(value).all()


def test_s27_identical_well_separated_views_have_near_zero_loss():
    x = _orthogonal_factorized()
    loss = blockwise_cross_view_factor_canonicalization_loss(x, x)
    assert float(loss.total) < 1e-7
    assert float(loss.role_total) < 1e-7
    assert float(loss.value_total) < 1e-7


def test_s27_role_only_mismatch_localizes_to_role_block():
    canonical = _orthogonal_factorized()
    paraphrase = canonical.clone()
    paraphrase[:, 0, :128] = canonical[:, 1, :128]
    loss = blockwise_cross_view_factor_canonicalization_loss(canonical, paraphrase)
    assert float(loss.role_total) > 0.1
    assert float(loss.value_total) < 1e-7


def test_s27_value_only_mismatch_localizes_to_value_block():
    canonical = _orthogonal_factorized()
    paraphrase = canonical.clone()
    paraphrase[:, 0, 128:] = canonical[:, 1, 128:]
    loss = blockwise_cross_view_factor_canonicalization_loss(canonical, paraphrase)
    assert float(loss.value_total) > 0.1
    assert float(loss.role_total) < 1e-7


def test_s27_both_factor_mismatch_activates_both_blocks():
    canonical = _orthogonal_factorized()
    paraphrase = canonical.clone()
    paraphrase[:, 0, :128] = canonical[:, 1, :128]
    paraphrase[:, 0, 128:] = canonical[:, 2, 128:]
    loss = blockwise_cross_view_factor_canonicalization_loss(canonical, paraphrase)
    assert float(loss.role_total) > 0.1
    assert float(loss.value_total) > 0.1


def test_s27_block_swap_swaps_component_losses():
    canonical = _orthogonal_factorized()
    paraphrase = canonical.clone()
    paraphrase[:, 0, :128] = canonical[:, 1, :128]
    first = blockwise_cross_view_factor_canonicalization_loss(canonical, paraphrase)

    c_role, c_value = split_factorized_role_value_signature(canonical)
    p_role, p_value = split_factorized_role_value_signature(paraphrase)
    canonical_swapped = F.normalize(torch.cat((c_value, c_role), dim=-1), dim=-1)
    paraphrase_swapped = F.normalize(torch.cat((p_value, p_role), dim=-1), dim=-1)
    second = blockwise_cross_view_factor_canonicalization_loss(
        canonical_swapped,
        paraphrase_swapped,
    )
    assert torch.allclose(first.role_total, second.value_total, atol=1e-7, rtol=0)
    assert torch.allclose(first.value_total, second.role_total, atol=1e-7, rtol=0)
    assert torch.allclose(first.total, second.total, atol=1e-7, rtol=0)


def test_s27_option_permutation_preserves_blockwise_loss():
    canonical = _orthogonal_factorized()
    paraphrase = canonical.clone()
    paraphrase[:, 0, :128] = canonical[:, 1, :128]
    base = blockwise_cross_view_factor_canonicalization_loss(canonical, paraphrase)
    perm = torch.tensor([2, 0, 3, 1])
    moved = blockwise_cross_view_factor_canonicalization_loss(
        canonical[:, perm],
        paraphrase[:, perm],
    )
    assert torch.allclose(base.total, moved.total, atol=1e-7, rtol=0)
    assert torch.allclose(base.role_total, moved.role_total, atol=1e-7, rtol=0)
    assert torch.allclose(base.value_total, moved.value_total, atol=1e-7, rtol=0)


def test_s27_invalid_width_rejected():
    x = torch.randn(1, 4, 255)
    try:
        split_factorized_role_value_signature(x)
    except ValueError:
        pass
    else:
        raise AssertionError("S27 accepted non-256D signature")
