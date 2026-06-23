"""Extract a ZIP or TAR archive with a small resumable state file."""

from __future__ import annotations

import argparse
import json
import tarfile
import time
import zipfile
from pathlib import Path


def safe_state_name(path: Path) -> str:
    return path.name.replace("\\", "_").replace("/", "_").replace(":", "_")


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_safe_tar_members(archive: tarfile.TarFile, output_root: Path) -> None:
    output_root_resolved = output_root.resolve()
    for member in archive.getmembers():
        target = (output_root / member.name).resolve()
        if output_root_resolved not in target.parents and target != output_root_resolved:
            raise ValueError(f"Unsafe tar member path: {member.name}")


def extract_zip(archive_path: Path, output_root: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")
        members = archive.infolist()
        archive.extractall(output_root)
    return {"format": "zip", "member_count": len(members)}


def extract_tar(archive_path: Path, output_root: Path) -> dict[str, object]:
    with tarfile.open(archive_path) as archive:
        ensure_safe_tar_members(archive, output_root)
        members = archive.getmembers()
        archive.extractall(output_root)
    return {"format": "tar", "member_count": len(members)}


def extract_archive(archive_path: Path, output_root: Path) -> dict[str, object]:
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        return extract_zip(archive_path, output_root)
    if name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz"):
        return extract_tar(archive_path, output_root)
    raise ValueError(f"Unsupported archive type: {archive_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    state_root = args.output_root / "_extract_state"
    state_path = state_root / f"{safe_state_name(args.archive)}.json"
    if state_path.exists() and not args.overwrite:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "extracted":
            print(json.dumps({"status": "already_extracted", "state_path": str(state_path)}, indent=2))
            return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    result = extract_archive(args.archive, args.output_root)
    state = {
        "archive": str(args.archive),
        "output_root": str(args.output_root),
        "status": "extracted",
        "archive_size": args.archive.stat().st_size,
        "completed_at_unix": int(time.time()),
        **result,
    }
    write_json(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
