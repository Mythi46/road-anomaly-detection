# coding: utf-8
"""Walk the PoC folders and yield ``(image_path, label)`` pairs.

Any directory whose name is exactly ``before`` contributes images with the
``abnormal`` label; any ``after`` directory contributes ``normal`` images.
``repair`` and ``no_label`` directories are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

from . import config


LabelledImage = Tuple[Path, str]


def _iter_images(folder: Path) -> Iterator[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTS:
            yield path


def iter_labelled_images(roots: List[Path] | None = None) -> Iterator[LabelledImage]:
    """Yield ``(image_path, label)`` pairs from all configured PoC roots."""
    roots = roots if roots is not None else config.POC_ROOTS
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for sub in root.rglob("*"):
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if name == "before":
                label = config.LABEL_ABNORMAL
            elif name == "after":
                label = config.LABEL_NORMAL
            else:
                continue
            for img in _iter_images(sub):
                resolved = img.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield img, label


def collect_labelled_images(roots: List[Path] | None = None) -> List[LabelledImage]:
    return list(iter_labelled_images(roots))
