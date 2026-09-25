from __future__ import annotations

from pathlib import Path

from nmd.anchor_preserving_residual_authority import all_w15_text_atoms
from nmd.anchor_preserving_residual_training import (
    CANDIDATES,
    EPOCHS,
    SEEDS,
    TRAINABLE_CANDIDATES,
)


def test_w15_candidate_set_budget_and_seed_identity_are_frozen():
    assert CANDIDATES == (
        "production-frozen-control",
        "multiview-anchor-control",
        "unbounded-residual-control",
        "bounded-residual-primary",
        "bounded-residual-replica",
    )
    assert TRAINABLE_CANDIDATES == CANDIDATES[2:]
    assert EPOCHS == 8
    assert SEEDS == {
        "production-frozen-control": 0,
        "multiview-anchor-control": 0,
        "unbounded-residual-control": 2303,
        "bounded-residual-primary": 2311,
        "bounded-residual-replica": 2323,
    }
    assert (
        SEEDS["bounded-residual-primary"]
        != SEEDS["bounded-residual-replica"]
    )


def test_w15_confirm_capability_literal_is_confined_to_confirm_script():
    root = Path(__file__).resolve().parents[1]
    sentinel = "allow_" + "confirm=True"
    allowed = root / "scripts" / "r8_w15_confirm.py"

    found = []
    for path in [
        root / "src" / "nmd" / "anchor_preserving_residual_authority.py",
        root / "scripts" / "r8_w15_build_cache.py",
        root / "scripts" / "r8_w15_train_candidate.py",
        root / "scripts" / "r8_w15_freeze_candidates.py",
        allowed,
        root / "tests" / "test_anchor_preserving_residual_authority.py",
        root / "tests" / "test_anchor_preserving_residual_execution.py",
    ]:
        if not path.exists():
            continue
        count = path.read_text(encoding="utf-8").count(sentinel)
        if count:
            found.append((path, count))

    assert found == [(allowed, 2)]


def test_w15_confirm_script_requires_frozen_candidate_boundary():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "r8_w15_confirm.py"
    ).read_text(encoding="utf-8")
    assert 'all_candidates_frozen_before_confirm") is not True' in source
    assert 'confirm_ce_exposed") is not False' in source
    assert 'confirm_cf_exposed") is not False' in source
    assert "primary_replica_seed_independence" in source
    assert "w15_verdict" in source


def test_w15_exact_text_atoms_are_fresh_against_all_prior_authorities():
    from scripts.r8_w15_build_cache import _prior_text_atoms

    overlap = sorted(all_w15_text_atoms() & _prior_text_atoms())
    assert overlap == []


def test_w15_train_dev_builder_never_imports_confirm_capability():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "r8_w15_build_cache.py"
    ).read_text(encoding="utf-8")
    assert "generate_w15_confirm" not in source
    assert "generate_w15_train" in source
    assert "generate_w15_dev" in source
    assert "prior_exact_text_overlap" in source
