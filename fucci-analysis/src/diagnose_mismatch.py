from collections import defaultdict
from pathlib import Path
import re
import sys


# Add repository root to import path so `configs.paths` resolves consistently.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import DATASET_1_RAW  # noqa: E402


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
CHANNELS = ("BF", "561", "647")
INDEX_PATTERN = re.compile(r"^(R\d+_\d+_\d+_[^_]+)_FUCCI_", re.IGNORECASE)


def detect_channel(filename: str) -> str | None:
    upper = filename.upper()
    for channel in CHANNELS:
        if channel in upper:
            return channel
    return None


def extract_index(filename: str) -> str | None:
    stem = Path(filename).stem
    match = INDEX_PATTERN.match(stem)
    if match:
        return match.group(1)

    # Fallback: strip the channel token and derive a normalized index surrogate.
    normalized = re.sub(r"_(BF|561|647)(?:_|$)", "_", stem, flags=re.IGNORECASE)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized if normalized else None


def main() -> None:
    data_root = Path(DATASET_1_RAW)
    if not data_root.exists():
        print(f"[ERROR] Raw data path does not exist: {data_root}")
        return

    channel_index_map: dict[str, set[str]] = {ch: set() for ch in CHANNELS}
    channel_files_by_index: dict[str, dict[str, list[str]]] = {
        ch: defaultdict(list) for ch in CHANNELS
    }

    for path in data_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        channel = detect_channel(path.name)
        if channel is None:
            continue

        index_key = extract_index(path.name)
        if index_key is None:
            continue

        channel_index_map[channel].add(index_key)
        rel_path = str(path.relative_to(data_root))
        channel_files_by_index[channel][index_key].append(rel_path)

    print(f"Raw data root: {data_root}")
    for channel in CHANNELS:
        print(f"{channel} unique index count: {len(channel_index_map[channel])}")

    only_647 = sorted(channel_index_map["647"] - (channel_index_map["BF"] | channel_index_map["561"]))
    print("\n[Indexes present only in 647]")
    if only_647:
        for idx in only_647:
            print(f"- {idx}")
            for fname in sorted(channel_files_by_index["647"][idx]):
                print(f"  - {fname}")
    else:
        print("- None detected")

    print("\n[Duplicated indexes by channel]")
    has_duplicate = False
    for channel in CHANNELS:
        duplicates = {
            idx: files
            for idx, files in channel_files_by_index[channel].items()
            if len(files) > 1
        }
        if not duplicates:
            continue

        has_duplicate = True
        print(f"\nChannel {channel}:")
        for idx in sorted(duplicates):
            files_sorted = sorted(duplicates[idx])
            print(f"- {idx} ({len(files_sorted)} files)")
            for fname in files_sorted[:10]:
                print(f"  - {fname}")
            if len(files_sorted) > 10:
                print(f"  - ... ({len(files_sorted) - 10} more)")

    if not has_duplicate:
        print("- No duplicated indexes detected.")


if __name__ == "__main__":
    main()
