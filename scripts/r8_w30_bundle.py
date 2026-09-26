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


def include_file(path: Path, root: Path, out_dir: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if out_dir == path or out_dir in path.parents:
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--confirm-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    handoff_source = root / "research" / "R8-W30-HANDOFF.md"
    handoff_out = out / "R8-W30-HANDOFF.md"
    shutil.copy2(handoff_source, handoff_out)

    archive = out / "NOLANE-HIRA-R8-W30-FULL.zip"
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for path in sorted(root.rglob("*")):
            if include_file(path, root, out):
                zf.write(path, Path("repo") / path.relative_to(root))

        for source_dir, prefix in (
            (args.train_dir.resolve(), Path("authority") / "train"),
            (args.confirm_dir.resolve(), Path("authority") / "confirm"),
        ):
            for path in sorted(source_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, prefix / path.relative_to(source_dir))

        zf.write(handoff_source, Path("handoff") / handoff_source.name)

    manifest = out / "R8-W30-INTEGRITY.sha256"
    lines = [
        f"{file_sha256(archive)}  {archive.name}",
        f"{file_sha256(handoff_out)}  {handoff_out.name}",
    ]
    for directory, label in (
        (args.train_dir.resolve(), "train"),
        (args.confirm_dir.resolve(), "confirm"),
    ):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                lines.append(
                    f"{file_sha256(path)}  authority/{label}/{path.relative_to(directory).as_posix()}"
                )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"W30_BUNDLE={archive}")
    print(f"W30_HANDOFF={handoff_out}")
    print(f"W30_MANIFEST={manifest}")
    print(f"W30_BUNDLE_SHA256={file_sha256(archive)}")


if __name__ == "__main__":
    main()
