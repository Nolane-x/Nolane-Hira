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
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    handoff_source = root / "research" / "R8-W33-HANDOFF.md"
    closure_source = root / "research" / "R8-W33-CLOSURE.md"
    if not handoff_source.is_file() or not closure_source.is_file():
        raise RuntimeError("W33 handoff/closure is missing")

    handoff_out = out / "R8-W33-HANDOFF.md"
    closure_out = out / "R8-W33-CLOSURE.md"
    shutil.copy2(handoff_source, handoff_out)
    shutil.copy2(closure_source, closure_out)

    staged = (
        (args.t0_dir.resolve(), out / "base" / "W28-T0"),
        (args.qualification_dir.resolve(), out / "evidence" / "qualification"),
    )
    for source, destination in staged:
        if not source.is_dir():
            raise RuntimeError(f"W33 bundle input missing: {source}")
        copy_tree(source, destination)

    archive = out / "NOLANE-HIRA-R8-W33-FULL.zip"
    manifest = out / "R8-W33-INTEGRITY.sha256"

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

    print(f"W33_BUNDLE={archive}")
    print(f"W33_HANDOFF={handoff_out}")
    print(f"W33_CLOSURE={closure_out}")
    print(f"W33_MANIFEST={manifest}")
    print(f"W33_BUNDLE_SHA256={file_sha256(archive)}")


if __name__ == "__main__":
    main()
