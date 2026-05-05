from pathlib import Path
import sys


# Add repository root to import path so `configs.paths` resolves consistently.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import DATASET_1_RAW  # noqa: E402
from shared.utils import get_clean_file_list  # noqa: E402


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

    raw_counts = {key: 0 for key in CHANNEL_KEYWORDS}
    for file_path in image_files:
        filename_upper = file_path.name.upper()
        for key in CHANNEL_KEYWORDS:
            if key in filename_upper:
                raw_counts[key] += 1

    cleaned_by_channel = get_clean_file_list(data_root)
    cleaned_counts = {key: len(cleaned_by_channel[key]) for key in CHANNEL_KEYWORDS}

    print("\nRaw vs Cleaned file counts")
    print(f"{'Channel':<10}{'Raw':>10}{'Cleaned':>12}")
    print("-" * 32)
    for key in CHANNEL_KEYWORDS:
        print(f"{key:<10}{raw_counts[key]:>10}{cleaned_counts[key]:>12}")

    raw_values = [raw_counts[key] for key in CHANNEL_KEYWORDS]
    if len(set(raw_values)) != 1:
        print("\n[WARNING] Raw channel counts are imbalanced across BF/561/647.")
    else:
        print("\n[OK] Raw channel counts are balanced across BF/561/647.")

    cleaned_values = [cleaned_counts[key] for key in CHANNEL_KEYWORDS]
    if len(set(cleaned_values)) == 1 and cleaned_values[0] == 2146:
        print("[SUCCESS] Data alignment successful.")
    else:
        print("[WARNING] Cleaned channel counts are not aligned to 2146 per channel.")


if __name__ == "__main__":
    main()
