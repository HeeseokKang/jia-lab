from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re
import sys

import pandas as pd
import tifffile


# Add repository root to import path so configs resolve consistently.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import DATASET_1_RAW, RESULT_ROOT  # noqa: E402


FILENAME_PATTERN = re.compile(
    r"^R(?P<row>\d+)_(?P<col>\d+)_(?P<z>\d+)_(?P<cellline>[^_]+)_FUCCI_(?P<channel>[A-Za-z0-9]+)\.tiff$",
    re.IGNORECASE,
)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Quantify background trend by computing quadrant means for each image "
            "across all timepoints/wells."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(DATASET_1_RAW),
        help="Raw dataset root containing timepoint folders (default: DATASET_1_RAW).",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="647",
        help="Target channel token in filename (default: 647).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULT_ROOT / "background",
        help="Directory where background quantification CSV files are written.",
    )
    return parser.parse_args()


def list_timepoint_dirs(data_root: Path) -> list[Path]:
    return sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def quadrant_means(image) -> dict[str, float]:
    h, w = image.shape
    h2, w2 = h // 2, w // 2
    return {
        "top_left": float(image[:h2, :w2].mean()),
        "top_right": float(image[:h2, w2:].mean()),
        "bottom_left": float(image[h2:, :w2].mean()),
        "bottom_right": float(image[h2:, w2:].mean()),
    }


def build_background_table(data_root: Path, channel: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_channel = channel.upper()
    timepoint_dirs = list_timepoint_dirs(data_root)

    if not timepoint_dirs:
        raise FileNotFoundError(f"No numeric timepoint folders found under: {data_root}")

    for tp_dir in timepoint_dirs:
        timepoint = int(tp_dir.name)
        for file_path in sorted(tp_dir.glob("*.tif*")):
            match = FILENAME_PATTERN.match(file_path.name)
            if not match:
                continue

            if match.group("channel").upper() != target_channel:
                continue

            image = tifffile.imread(file_path)
            if image.ndim != 2:
                raise ValueError(f"Expected 2D image, got shape {image.shape}: {file_path}")

            quad = quadrant_means(image)
            image_mean = float(image.mean())
            rows.append(
                {
                    "timepoint": timepoint,
                    "well_id": f"R{match.group('row')}_{match.group('col')}",
                    "row": int(match.group("row")),
                    "col": int(match.group("col")),
                    "z": int(match.group("z")),
                    "cellline": match.group("cellline"),
                    "channel": target_channel,
                    "image_mean": image_mean,
                    **quad,
                    "source_path": str(file_path),
                }
            )

    if not rows:
        raise RuntimeError(
            f"No matching TIFF files found for channel {target_channel} under {data_root}"
        )

    df = pd.DataFrame(rows).sort_values(["timepoint", "row", "col"]).reset_index(drop=True)
    quadrant_cols = ["top_left", "top_right", "bottom_left", "bottom_right"]

    # Median subtraction surrogate background (per timepoint across wells).
    for col in ["image_mean", *quadrant_cols]:
        median_col = f"{col}_tp_median"
        corrected_col = f"{col}_median_subtracted"
        df[median_col] = df.groupby("timepoint")[col].transform("median")
        df[corrected_col] = df[col] - df[median_col]

    return df


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    df = build_background_table(data_root=data_root, channel=args.channel)
    details_csv = output_dir / f"quadrant_background_{args.channel}.csv"
    summary_csv = output_dir / f"timepoint_background_summary_{args.channel}.csv"

    df.to_csv(details_csv, index=False)

    summary = (
        df.groupby("timepoint", as_index=False)
        .agg(
            wells=("well_id", "nunique"),
            image_mean_global_median=("image_mean", "median"),
            image_mean_global_std=("image_mean", "std"),
            top_left_median=("top_left", "median"),
            top_right_median=("top_right", "median"),
            bottom_left_median=("bottom_left", "median"),
            bottom_right_median=("bottom_right", "median"),
        )
        .sort_values("timepoint")
    )
    summary.to_csv(summary_csv, index=False)

    print(f"[DONE] Channel: {str(args.channel).upper()}")
    print(f"[DONE] Data root: {data_root}")
    print(f"[DONE] Timepoints analyzed: {df['timepoint'].nunique()}")
    print(f"[DONE] Wells analyzed: {df['well_id'].nunique()}")
    print(f"[DONE] Detail table: {details_csv}")
    print(f"[DONE] Timepoint summary: {summary_csv}")


if __name__ == "__main__":
    main()
