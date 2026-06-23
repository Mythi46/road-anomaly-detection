"""Extract nested ZIP members from a larger ZIP archive."""

from __future__ import annotations

import argparse
import fnmatch
import json
import tempfile
import time
import zipfile
from pathlib import Path


def copy_stream(source, destination: Path) -> int:
    total = 0
    with destination.open("wb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            total += len(chunk)
    return total


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def safe_state_name(member_name: str) -> str:
    return (
        member_name.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".zip", "")
    )


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_member(
    outer: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    output_root: Path,
    state_root: Path,
    overwrite: bool,
) -> dict[str, object]:
    state_path = state_root / f"{safe_state_name(info.filename)}.json"
    if state_path.exists() and not overwrite:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "extracted":
            return {"member": info.filename, "status": "already_extracted", "state_path": str(state_path)}

    with tempfile.TemporaryDirectory(prefix="nested_zip_") as temp_name:
        temp_zip = Path(temp_name) / Path(info.filename).name
        with outer.open(info) as source:
            bytes_written = copy_stream(source, temp_zip)

        with zipfile.ZipFile(temp_zip) as nested:
            bad_member = nested.testzip()
            if bad_member:
                raise ValueError(f"Nested ZIP {info.filename} is corrupt at {bad_member}")
            members = nested.infolist()
            nested.extractall(output_root)

    state = {
        "member": info.filename,
        "status": "extracted",
        "compressed_size": info.compress_size,
        "nested_zip_size": info.file_size,
        "bytes_written": bytes_written,
        "member_count": len(members),
        "output_root": str(output_root),
        "completed_at_unix": int(time.time()),
    }
    write_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--member-pattern",
        action="append",
        default=[],
        help="fnmatch pattern for nested ZIP members. Repeatable. Default: *.zip",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    patterns = args.member_pattern or ["*.zip"]
    state_root = args.output_root / "_extract_state"
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    with zipfile.ZipFile(args.zip) as outer:
        bad_member = outer.testzip()
        if bad_member:
            raise ValueError(f"Outer ZIP is corrupt at {bad_member}")
        nested_infos = [
            info
            for info in outer.infolist()
            if info.filename.lower().endswith(".zip") and matches_any(info.filename, patterns)
        ]
        for info in nested_infos:
            print(f"Extracting nested ZIP: {info.filename}")
            results.append(
                extract_member(
                    outer=outer,
                    info=info,
                    output_root=args.output_root,
                    state_root=state_root,
                    overwrite=args.overwrite,
                )
            )

    summary = {
        "zip": str(args.zip),
        "output_root": str(args.output_root),
        "patterns": patterns,
        "results": results,
    }
    write_json(state_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
