"""Prepare public dataset folders and fetch scriptable metadata/assets.

This script intentionally keeps large downloads opt-in. It can fetch the
small RDD2022 Figshare metadata files immediately, while recording large
files and manual-download sources for later handling.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "PyYAML is required. Install with: pip install -r RoadAnomalyDetection/requirements.txt"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "datasets" / "catalog.yaml"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "public"
USER_AGENT = "road-anomaly-detection-dataset-prep/0.1"


def fetch_json(url: str, accept: str | None = None) -> Any:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path, expected_size: int | None = None) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_destination = destination.with_name(f"{destination.name}.part")
    resume_from = partial_destination.stat().st_size if partial_destination.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if resume_from and status != 206:
            resume_from = 0
        mode = "ab" if resume_from else "wb"
        total = resume_from
        with partial_destination.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                total += len(chunk)

    if expected_size is not None and total != expected_size:
        raise IOError(f"Incomplete download for {destination}: expected {expected_size}, got {total}")
    partial_destination.replace(destination)
    return total


def probe_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, dict) or "datasets" not in catalog:
        raise ValueError(f"Invalid catalog: {path}")
    return catalog


def select_datasets(catalog: dict[str, Any], ids: list[str] | None) -> list[dict[str, Any]]:
    datasets = catalog["datasets"]
    by_id = {item["id"]: item for item in datasets}
    if ids:
        missing = [dataset_id for dataset_id in ids if dataset_id not in by_id]
        if missing:
            raise ValueError(f"Unknown dataset id(s): {', '.join(missing)}")
        return [by_id[dataset_id] for dataset_id in ids]

    order = catalog.get("download_order") or [item["id"] for item in datasets]
    return [by_id[dataset_id] for dataset_id in order if dataset_id in by_id]


def figshare_article_id(url: str) -> str | None:
    match = re.search(r"/articles/(?:dataset|journal_contribution|figure)/[^/]+/(\d+)", url)
    return match.group(1) if match else None


def mendeley_dataset_parts(url: str) -> tuple[str, str | None] | None:
    match = re.search(r"data\.mendeley\.com/datasets/([a-z0-9]+)(?:/(\d+))?", url)
    if not match:
        return None
    return match.group(1), match.group(2)


def prepare_dirs(data_root: Path, dataset_id: str) -> dict[str, Path]:
    paths = {
        "raw": data_root / "raw" / dataset_id,
        "converted": data_root / "converted" / dataset_id,
        "manifest": data_root / "manifests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def handle_figshare(
    dataset: dict[str, Any],
    paths: dict[str, Path],
    max_file_size_bytes: int,
    include_large: bool,
    dry_run: bool,
) -> dict[str, Any]:
    article_id = figshare_article_id(dataset["source_url"])
    if not article_id:
        raise ValueError(f"Could not parse Figshare article id: {dataset['source_url']}")

    article_url = f"https://api.figshare.com/v2/articles/{article_id}"
    article = fetch_json(article_url)
    files = article.get("files", [])

    article_path = paths["manifest"] / f"{dataset['id']}_figshare_article.json"
    files_path = paths["manifest"] / f"{dataset['id']}_files.json"
    if not dry_run:
        article_path.write_text(json.dumps(article, indent=2, ensure_ascii=False), encoding="utf-8")
        files_path.write_text(json.dumps(files, indent=2, ensure_ascii=False), encoding="utf-8")

    file_results = []
    for file_info in files:
        name = file_info["name"]
        size = int(file_info.get("size") or 0)
        destination = paths["raw"] / name
        should_download = include_large or size <= max_file_size_bytes
        result = {
            "name": name,
            "size": size,
            "download_url": file_info.get("download_url"),
            "destination": str(destination),
            "status": "planned" if should_download else "skipped_large",
        }
        if should_download and not dry_run:
            if destination.exists() and destination.stat().st_size == size:
                result["status"] = "already_exists"
            else:
                bytes_written = download_file(file_info["download_url"], destination, size)
                result["status"] = "downloaded"
                result["bytes_written"] = bytes_written
        file_results.append(result)

    return {
        "id": dataset["id"],
        "source_type": "figshare",
        "article_api": article_url,
        "title": article.get("title"),
        "files": file_results,
    }


def handle_mendeley(
    dataset: dict[str, Any],
    paths: dict[str, Path],
    max_file_size_bytes: int,
    include_large: bool,
    dry_run: bool,
) -> dict[str, Any]:
    parts = mendeley_dataset_parts(dataset["source_url"])
    api_status: dict[str, Any] = {"status": "not_attempted"}
    zip_status: dict[str, Any] = {"status": "not_attempted"}
    if parts:
        dataset_id, version = parts
        version_for_zip = version or "latest"
        api_url = f"https://api.mendeley.com/datasets/{dataset_id}"
        if version:
            api_url = f"{api_url}?version={version}"
        try:
            dataset_info = fetch_json(api_url, accept="application/vnd.mendeley-dataset.1+json")
            api_status = {"status": "ok", "api_url": api_url, "dataset": dataset_info}
            if not dry_run:
                (paths["manifest"] / f"{dataset['id']}_mendeley_dataset.json").write_text(
                    json.dumps(dataset_info, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except urllib.error.HTTPError as exc:
            api_status = {"status": "blocked", "api_url": api_url, "http_status": exc.code}
        except Exception as exc:  # pragma: no cover - network dependent
            api_status = {"status": "error", "api_url": api_url, "error": repr(exc)}

        if version:
            zip_url = f"https://data.mendeley.com/public-api/zip/{dataset_id}/download/{version_for_zip}"
            destination = paths["raw"] / f"{dataset['id']}.zip"
            size = probe_content_length(zip_url)
            should_download = include_large or size is None or size <= max_file_size_bytes
            zip_status = {
                "status": "planned" if should_download else "skipped_large",
                "zip_url": zip_url,
                "content_length": size,
                "destination": str(destination),
            }
            if should_download and not dry_run:
                if size is not None and destination.exists() and destination.stat().st_size == size:
                    zip_status["status"] = "already_exists"
                else:
                    bytes_written = download_file(zip_url, destination, size)
                    zip_status["status"] = "downloaded"
                    zip_status["bytes_written"] = bytes_written

    return {
        "id": dataset["id"],
        "source_type": "mendeley",
        "source_url": dataset["source_url"],
        "api_probe": api_status,
        "zip_download": zip_status,
        "next_action": "If zip_download is skipped_large, rerun with --include-large after confirming license.",
    }


def zenodo_record_id(url: str) -> str | None:
    match = re.search(r"zenodo\.org/records/(\d+)", url)
    return match.group(1) if match else None


def handle_zenodo(
    dataset: dict[str, Any],
    paths: dict[str, Path],
    max_file_size_bytes: int,
    include_large: bool,
    dry_run: bool,
) -> dict[str, Any]:
    url = dataset.get("zenodo_url") or dataset.get("data_url") or dataset.get("source_url")
    record_id = zenodo_record_id(url or "")
    if not record_id:
        raise ValueError(f"Could not parse Zenodo record id from {url}")

    api_url = f"https://zenodo.org/api/records/{record_id}"
    record = fetch_json(api_url)
    if not dry_run:
        (paths["manifest"] / f"{dataset['id']}_zenodo_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    file_results = []
    for file_info in record.get("files", []):
        name = file_info["key"]
        size = int(file_info.get("size") or 0)
        download_url = file_info.get("links", {}).get("self")
        destination = paths["raw"] / name
        should_download = include_large or size <= max_file_size_bytes
        result = {
            "name": name,
            "size": size,
            "download_url": download_url,
            "destination": str(destination),
            "status": "planned" if should_download else "skipped_large",
        }
        if should_download and not dry_run:
            if destination.exists() and destination.stat().st_size == size:
                result["status"] = "already_exists"
            else:
                bytes_written = download_file(download_url, destination, size)
                result["status"] = "downloaded"
                result["bytes_written"] = bytes_written
        file_results.append(result)

    return {
        "id": dataset["id"],
        "source_type": "zenodo",
        "record_api": api_url,
        "title": record.get("title") or record.get("metadata", {}).get("title"),
        "doi": record.get("doi"),
        "files": file_results,
    }


def handle_source_page(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": dataset["id"],
        "source_type": "source_page",
        "source_url": dataset.get("source_url"),
        "data_url": dataset.get("data_url"),
        "next_action": "Open source/data URL, confirm license and available files before download.",
    }


def prepare_dataset(
    dataset: dict[str, Any],
    data_root: Path,
    max_file_size_bytes: int,
    include_large: bool,
    dry_run: bool,
) -> dict[str, Any]:
    paths = prepare_dirs(data_root, dataset["id"])
    url = dataset.get("source_url", "")

    zenodo_candidate = dataset.get("zenodo_url") or dataset.get("data_url") or url
    if "zenodo.org" in zenodo_candidate:
        return handle_zenodo(dataset, paths, max_file_size_bytes, include_large, dry_run)
    if "figshare.com" in url:
        return handle_figshare(dataset, paths, max_file_size_bytes, include_large, dry_run)
    if "data.mendeley.com" in url:
        return handle_mendeley(dataset, paths, max_file_size_bytes, include_large, dry_run)
    return handle_source_page(dataset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset", action="append", help="Dataset id to prepare. Repeatable.")
    parser.add_argument("--max-file-size-mb", type=float, default=50.0)
    parser.add_argument("--include-large", action="store_true", help="Allow downloads above max-file-size-mb.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    catalog = load_catalog(args.catalog)
    selected = select_datasets(catalog, args.dataset)
    max_file_size_bytes = int(args.max_file_size_mb * 1024 * 1024)

    args.data_root.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in selected:
        print(f"Preparing {dataset['id']} ...")
        result = prepare_dataset(
            dataset=dataset,
            data_root=args.data_root,
            max_file_size_bytes=max_file_size_bytes,
            include_large=args.include_large,
            dry_run=args.dry_run,
        )
        results.append(result)

    status = {
        "generated_at_unix": int(time.time()),
        "catalog": str(args.catalog),
        "data_root": str(args.data_root),
        "dry_run": args.dry_run,
        "include_large": args.include_large,
        "max_file_size_mb": args.max_file_size_mb,
        "datasets": results,
    }
    status_path = args.data_root / "download_status.json"
    if not args.dry_run:
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
