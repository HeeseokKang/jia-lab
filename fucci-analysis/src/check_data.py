from pathlib import Path
import sys


# Add repository root to import path so `configs.paths` resolves consistently.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import DATASET_1_RAW  # noqa: E402


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
CHANNEL_KEYWORDS = ("BF", "561", "647")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def main() -> None:
    data_root = Path(DATASET_1_RAW)
    if not data_root.exists():
        print(f"[ERROR] Raw data path does not exist: {data_root}")
        return

    image_files = [p for p in data_root.rglob("*") if p.is_file() and is_image_file(p)]
    print(f"Raw data root: {data_root}")
    print(f"Total image files discovered: {len(image_files)}")

    channel_counts = {key: 0 for key in CHANNEL_KEYWORDS}
    for file_path in image_files:
        filename_upper = file_path.name.upper()
        for key in CHANNEL_KEYWORDS:
            if key in filename_upper:
                channel_counts[key] += 1

    for key in CHANNEL_KEYWORDS:
        print(f"Channel {key} file count: {channel_counts[key]}")

    counts = [channel_counts[key] for key in CHANNEL_KEYWORDS]
    if len(set(counts)) != 1:
        print("[WARNING] Channel file counts are imbalanced across BF/561/647.")
    else:
        print("[OK] Channel file counts are balanced across BF/561/647.")


if __name__ == "__main__":
    main()
