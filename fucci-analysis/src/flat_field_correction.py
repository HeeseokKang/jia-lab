from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re
import sys

import numpy as np
import tifffile


# Add repository root to import path so configs resolve consistently.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import DATASET_1_RAW, RESULT_ROOT  # noqa: E402


TARGET_CHANNELS = ("561", "647")
FILENAME_PATTERN = re.compile(
    r"^R(?P<row>\d+)_(?P<col>\d+)_(?P<z>\d+)_(?P<cellline>[^_]+)_FUCCI_(?P<channel>[A-Za-z0-9]+)\.tiff$",
    re.IGNORECASE,
)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Apply pseudo-flat-field correction using t=0 well/channel references "
            "for channels 561 and 647."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(DATASET_1_RAW),
        help="Raw timelapse root containing numeric timepoint folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULT_ROOT / "corrected",
        help="Output folder for corrected TIFF images.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Small denominator floor to avoid divide-by-zero.",
    )
    parser.add_argument(
        "--strategy-561",
        type=str,
        choices=("raw", "t0", "min"),
        default="raw",
        help=(
            "Correction strategy for channel 561: "
            "'raw' keeps raw image (recommended), "
            "'t0' uses t=0 reference, "
            "'min' uses per-pixel minimum across all timepoints."
        ),
    )
    parser.add_argument(
        "--strategy-647",
        type=str,
        choices=("t0", "min"),
        default="t0",
        help="Correction strategy for channel 647.",
    )
    return parser.parse_args()


def list_timepoint_dirs(data_root: Path) -> list[Path]:
    return sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def parse_filename(path: Path):
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None
    return match.groupdict()


def load_t0_references(data_root: Path, eps: float):
    t0_dir = data_root / "0"
    if not t0_dir.exists():
        raise FileNotFoundError(f"Timepoint folder not found: {t0_dir}")

    references: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    # key: (well_id, channel) -> (reference_image, reference_median)
    for path in sorted(t0_dir.glob("*.tif*")):
        meta = parse_filename(path)
        if meta is None:
            continue

        channel = meta["channel"].upper()
        if channel not in TARGET_CHANNELS:
            continue

        well_id = f"R{meta['row']}_{meta['col']}"
        ref = tifffile.imread(path).astype(np.float32)
        if ref.ndim != 2:
            raise ValueError(f"Expected 2D reference image, got {ref.shape}: {path}")

        ref_median = float(np.median(ref))
        references[(well_id, channel)] = (np.maximum(ref, eps), ref_median)

    if not references:
        raise RuntimeError("No t=0 reference images found for channels 561/647.")
    return references


def load_min_references(data_root: Path, eps: float):
    # key: (well_id, channel) -> reference_min_image
    references: dict[tuple[str, str], np.ndarray] = {}
    timepoint_dirs = list_timepoint_dirs(data_root)
    if not timepoint_dirs:
        raise RuntimeError(f"No numeric timepoint directories found under: {data_root}")

    for tp_dir in timepoint_dirs:
        for path in sorted(tp_dir.glob("*.tif*")):
            meta = parse_filename(path)
            if meta is None:
                continue
            channel = meta["channel"].upper()
            if channel not in TARGET_CHANNELS:
                continue

            well_id = f"R{meta['row']}_{meta['col']}"
            key = (well_id, channel)
            img = tifffile.imread(path).astype(np.float32)
            if img.ndim != 2:
                raise ValueError(f"Expected 2D image, got {img.shape}: {path}")

            if key not in references:
                references[key] = img
            else:
                references[key] = np.minimum(references[key], img)

    if not references:
        raise RuntimeError("No images found to compute per-pixel minimum reference.")

    # Denominator floor for numeric stability.
    for key in list(references.keys()):
        references[key] = np.maximum(references[key], eps)
    return references


def correct_image(raw: np.ndarray, ref: np.ndarray, ref_median: float, eps: float) -> np.ndarray:
    raw_f = raw.astype(np.float32)
    corrected = raw_f / np.maximum(ref, eps) * ref_median
    corrected = np.clip(corrected, 0, np.iinfo(np.uint16).max)
    return corrected.astype(np.uint16)


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    output_root: Path = args.output_root
    eps: float = args.eps
    strategy_561: str = args.strategy_561
    strategy_647: str = args.strategy_647

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    strategies = {"561": strategy_561, "647": strategy_647}
    needs_t0 = any(v == "t0" for v in strategies.values())
    needs_min = any(v == "min" for v in strategies.values())

    t0_refs = load_t0_references(data_root, eps=eps) if needs_t0 else {}
    min_refs = load_min_references(data_root, eps=eps) if needs_min else {}

    timepoint_dirs = list_timepoint_dirs(data_root)
    if not timepoint_dirs:
        raise RuntimeError(f"No numeric timepoint directories found under: {data_root}")

    total_written = 0
    skipped_non_target = 0
    skipped_no_reference = 0
    copied_raw_561 = 0

    for tp_dir in timepoint_dirs:
        out_tp_dir = output_root / tp_dir.name
        out_tp_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(tp_dir.glob("*.tif*")):
            meta = parse_filename(path)
            if meta is None:
                continue

            channel = meta["channel"].upper()
            if channel not in TARGET_CHANNELS:
                skipped_non_target += 1
                continue

            well_id = f"R{meta['row']}_{meta['col']}"

            raw = tifffile.imread(path)
            if raw.ndim != 2:
                raise ValueError(f"Expected 2D raw image, got {raw.shape}: {path}")

            out_path = out_tp_dir / path.name
            strategy = strategies[channel]
            if strategy == "raw":
                tifffile.imwrite(out_path, raw)
                copied_raw_561 += 1
                total_written += 1
                continue

            key = (well_id, channel)
            if strategy == "t0":
                if key not in t0_refs:
                    skipped_no_reference += 1
                    continue
                ref, ref_median = t0_refs[key]
                corrected = correct_image(raw, ref=ref, ref_median=ref_median, eps=eps)
            elif strategy == "min":
                if key not in min_refs:
                    skipped_no_reference += 1
                    continue
                ref = min_refs[key]
                ref_median = float(np.median(ref))
                corrected = correct_image(raw, ref=ref, ref_median=ref_median, eps=eps)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

            tifffile.imwrite(out_path, corrected)
            total_written += 1

    print(f"[DONE] Data root: {data_root}")
    print(f"[DONE] Output root: {output_root}")
    print(
        "[DONE] Strategies: "
        f"561={strategy_561}, "
        f"647={strategy_647}"
    )
    print(f"[DONE] Corrected images written: {total_written}")
    print(f"[INFO] 561 raw passthrough images written: {copied_raw_561}")
    print(f"[INFO] Skipped non-target channel images (e.g., BF): {skipped_non_target}")
    print(f"[INFO] Skipped due to missing t=0 reference: {skipped_no_reference}")


if __name__ == "__main__":
    main()
