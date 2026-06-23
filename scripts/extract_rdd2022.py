"""Extract RDD2022's nested country ZIP files.

The Figshare package contains one large ZIP, which contains one ZIP per
country/platform. This script extracts those nested archives in a resumable,
country-by-country way and writes small state files under the output folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = REPO_ROOT / "data" / "public" / "raw" / "rdd2022" / "RDD2022_released_through_CRDDC2022.zip"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "public" / "extracted" / "rdd2022"


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


def read_state(state_path: Path) -> dict[str, object] | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def write_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_nested_zip(
    outer_zip: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    output_root: Path,
    state_root: Path,
    overwrite: bool,
    keep_nested_zip: bool,
) -> dict[str, object]:
    country = Path(info.filename).stem
    state_path = state_root / f"{country}.json"
    existing_state = read_state(state_path)
    if existing_state and existing_state.get("status") == "extracted" and not overwrite:
        return {"country": country, "status": "already_extracted", "state_path": str(state_path)}

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"rdd2022_{country}_") as temp_name:
        temp_zip = Path(temp_name) / f"{country}.zip"
        with outer_zip.open(info) as source:
            bytes_written = copy_stream(source, temp_zip)

        with zipfile.ZipFile(temp_zip) as nested:
            bad_member = nested.testzip()
            if bad_member:
                raise ValueError(f"Nested ZIP for {country} is corrupt at {bad_member}")
            members = nested.infolist()
            nested.extractall(output_root)

        if keep_nested_zip:
            saved_zip = output_root / "_nested_zips" / temp_zip.name
            saved_zip.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_zip, saved_zip)
        else:
            saved_zip = None

    state = {
        "country": country,
        "status": "extracted",
        "outer_member": info.filename,
        "compressed_size": info.compress_size,
        "nested_zip_size": info.file_size,
        "bytes_written": bytes_written,
        "member_count": len(members),
        "output_root": str(output_root),
        "saved_nested_zip": str(saved_zip) if saved_zip else None,
        "completed_at_unix": int(time.time()),
    }
    write_state(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--country",
        action="append",
        help="Country/platform name to extract, e.g. Japan. Repeatable. Defaults to all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-nested-zips", action="store_true")
    args = parser.parse_args()

    requested = {country.lower() for country in args.country or []}
    state_root = args.output_root / "_extract_state"

    if not args.zip.exists():
        raise FileNotFoundError(args.zip)
    if not zipfile.is_zipfile(args.zip):
        raise ValueError(f"Not a valid ZIP file: {args.zip}")

    results = []
    with zipfile.ZipFile(args.zip) as outer:
        bad_member = outer.testzip()
        if bad_member:
            raise ValueError(f"Outer ZIP is corrupt at {bad_member}")

        nested_infos = [info for info in outer.infolist() if info.filename.lower().endswith(".zip")]
        for info in nested_infos:
            country = Path(info.filename).stem
            if requested and country.lower() not in requested:
                continue
            print(f"Extracting {country} ...")
            results.append(
                extract_nested_zip(
                    outer_zip=outer,
                    info=info,
                    output_root=args.output_root,
                    state_root=state_root,
                    overwrite=args.overwrite,
                    keep_nested_zip=args.keep_nested_zips,
                )
            )

    summary = {
        "zip": str(args.zip),
        "output_root": str(args.output_root),
        "requested": sorted(requested),
        "results": results,
    }
    summary_path = state_root / "summary.json"
    write_state(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
