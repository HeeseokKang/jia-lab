from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable
import re


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
CHANNEL_TOKENS = ("BF", "561", "647")
INDEX_PATTERN = re.compile(r"^(R\d+_\d+_\d+_[^_]+)_FUCCI_", re.IGNORECASE)


def _detect_channel(filename: str) -> str | None:
    upper = filename.upper()
    for channel in CHANNEL_TOKENS:
        if channel in upper:
            return channel
    return None


def _extract_index(filename: str) -> str | None:
    stem = Path(filename).stem
    matched = INDEX_PATTERN.match(stem)
    if matched:
        return matched.group(1)

    normalized = re.sub(r"_(BF|561|647)(?:_|$)", "_", stem, flags=re.IGNORECASE)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized if normalized else None


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _choose_representative(paths: Iterable[Path], min_file_size_bytes: int) -> Path:
    candidates = [p for p in paths if p.stat().st_size >= min_file_size_bytes]
    if not candidates:
        candidates = list(paths)

    # Prefer the latest artifact because debugging/re-export steps often produce corrected files last.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def get_clean_file_list(data_root: Path | str, min_file_size_bytes: int = 1024) -> dict[str, list[Path]]:
    """
    Build a de-duplicated, analysis-ready file list grouped by channel.

    For each (channel, index) group, this function selects one representative file:
    1) Prefer files with size >= `min_file_size_bytes`
    2) Among valid candidates, choose the most recently modified file
    3) If no candidate passes the size threshold, choose the newest file anyway
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    grouped: dict[str, dict[str, list[Path]]] = {
        channel: defaultdict(list) for channel in CHANNEL_TOKENS
    }

    for path in root.rglob("*"):
        if not path.is_file() or not _is_image(path):
            continue

        channel = _detect_channel(path.name)
        if channel is None:
            continue

        index = _extract_index(path.name)
        if index is None:
            continue

        grouped[channel][index].append(path)

    clean: dict[str, list[Path]] = {channel: [] for channel in CHANNEL_TOKENS}
    for channel in CHANNEL_TOKENS:
        for index in sorted(grouped[channel]):
            selected = _choose_representative(grouped[channel][index], min_file_size_bytes)
            clean[channel].append(selected)

    return clean
