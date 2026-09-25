from pathlib import Path


def test_only_confirm_script_can_materialize_w7_confirm():
    root = Path(__file__).resolve().parents[1]
    sentinel = "allow_" + "confirm=True"
    allowed = root / "scripts" / "r8_w7_confirm.py"
    assert allowed.read_text(encoding="utf-8").count(sentinel) == 2

    protected = [
        root / "src" / "nmd" / "conjunctive_authority.py",
        root / "src" / "nmd" / "conjunctive_cache.py",
        root / "src" / "nmd" / "conjunctive_training.py",
        root / "scripts" / "r8_w7_build_cache.py",
        root / "scripts" / "r8_w7_train_candidate.py",
        root / "scripts" / "r8_w7_freeze_candidates.py",
        Path(__file__),
    ]
    for path in protected:
        assert sentinel not in path.read_text(encoding="utf-8"), path
