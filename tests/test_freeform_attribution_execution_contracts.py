from pathlib import Path


def test_only_confirm_script_can_materialize_w7b_confirm():
    root = Path(__file__).resolve().parents[1]
    sentinel = "allow_" + "confirm=True"
    allowed = root / "scripts" / "r8_w7b_confirm.py"
    assert allowed.read_text(encoding="utf-8").count(sentinel) == 2

    protected = [
        root / "src" / "nmd" / "freeform_attribution_authority.py",
        root / "src" / "nmd" / "freeform_attribution_training.py",
        root / "scripts" / "r8_w7b_build_cache.py",
        root / "scripts" / "r8_w7b_train_candidate.py",
        root / "scripts" / "r8_w7b_freeze_candidates.py",
        Path(__file__),
    ]
    for path in protected:
        assert sentinel not in path.read_text(encoding="utf-8"), path
