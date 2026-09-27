from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import zipfile

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def include_repo_file(path: Path, root: Path, out: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if out == path or out in path.parents:
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--t0-dir", type=Path, required=True)
    parser.add_argument("--w34-dir", type=Path, required=True)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--r2-dir", type=Path, required=True)
    parser.add_argument("--confirm-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    handoff_source = root / "research" / "HIRA-V0-MAINLINE-M1-HANDOFF.md"
    closure_source = root / "research" / "HIRA-V0-MAINLINE-M1-CLOSURE.md"
    if not handoff_source.is_file() or not closure_source.is_file():
        raise RuntimeError("M1 handoff/closure is missing")

    handoff_out = out / "HIRA-V0-MAINLINE-M1-HANDOFF.md"
    closure_out = out / "HIRA-V0-MAINLINE-M1-CLOSURE.md"
    shutil.copy2(handoff_source, handoff_out)
    shutil.copy2(closure_source, closure_out)

    staged = (
        (args.t0_dir.resolve(), out / "base" / "W28-T0"),
        (args.w34_dir.resolve(), out / "base" / "W34-provisional-transfer"),
        (args.r1_dir.resolve(), out / "evidence" / "r1-train-dev"),
        (args.r2_dir.resolve(), out / "evidence" / "r2-train-dev"),
        (args.confirm_dir.resolve(), out / "evidence" / "r2-sealed-confirm"),
    )
    for source, destination in staged:
        if not source.is_dir():
            raise RuntimeError(f"M1 bundle input missing: {source}")
        copy_tree(source, destination)

    archive = out / "NOLANE-HIRA-V0-MAINLINE-M1-FULL.zip"
    manifest = out / "HIRA-V0-MAINLINE-M1-INTEGRITY.sha256"

    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for path in sorted(root.rglob("*")):
            if include_repo_file(path, root, out):
                zf.write(path, Path("repo") / path.relative_to(root))

        for path in sorted(out.rglob("*")):
            if not path.is_file() or path in {archive, manifest}:
                continue
            zf.write(path, Path("bundle") / path.relative_to(out))

    lines = [
        f"{file_sha256(archive)}  {archive.name}",
        f"{file_sha256(handoff_out)}  {handoff_out.name}",
        f"{file_sha256(closure_out)}  {closure_out.name}",
    ]
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path in {archive, manifest, handoff_out, closure_out}:
            continue
        lines.append(f"{file_sha256(path)}  {path.relative_to(out).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"M1_BUNDLE={archive}")
    print(f"M1_HANDOFF={handoff_out}")
    print(f"M1_CLOSURE={closure_out}")
    print(f"M1_MANIFEST={manifest}")
    print(f"M1_BUNDLE_SHA256={file_sha256(archive)}")


if __name__ == "__main__":
    main()
