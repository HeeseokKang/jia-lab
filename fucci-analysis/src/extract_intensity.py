"""Per-cell FUCCI intensity extraction for one well at one timepoint.

Inputs:
- BF mask  (int label image from segment_test.py)
- 647     (flat-field corrected, from analysis/.../corrected/)
- 561     (raw — no correction needed for this channel)

Output:
- CSV with one row per cell: cell_id, centroid_x, centroid_y, area,
  mean_intensity_561, mean_intensity_647, ratio_647_561
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import RESULT_ROOT  # noqa: E402


DATA_ROOT = Path(
    "/data/Project_Data/Voltage_CellCycle/Data/20260413_FUCCI_Timelapse/"
    "timelapse_plus_bf_2026-04-16_12-11-45.768458"
)
CORRECTED_ROOT = RESULT_ROOT / "corrected"
SEG_DIR = RESULT_ROOT / "segmentation_test"

WELL_ID = "R1_1"
TIMEPOINT = 30
CELLLINE = "3T3"
Z = 0


def main() -> None:
    stem = f"{WELL_ID}_t{TIMEPOINT:03d}"

    mask_path = SEG_DIR / f"{stem}_BF_mask.npy"
    img_561_path = DATA_ROOT / str(TIMEPOINT) / f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI_561.tiff"
    img_647_path = CORRECTED_ROOT / str(TIMEPOINT) / f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI_647.tiff"

    for p in (mask_path, img_561_path, img_647_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    print(f"[LOAD] mask -> {mask_path}")
    mask = np.load(mask_path)
    print(f"[LOAD] 561  -> {img_561_path}")
    img_561 = tifffile.imread(str(img_561_path))
    print(f"[LOAD] 647  -> {img_647_path}")
    img_647 = tifffile.imread(str(img_647_path))

    if mask.shape != img_561.shape or mask.shape != img_647.shape:
        raise ValueError(
            f"Shape mismatch: mask={mask.shape}, 561={img_561.shape}, 647={img_647.shape}"
        )

    # Stack into (H, W, 2) so regionprops emits per-channel intensity_mean columns.
    intensity = np.stack(
        [img_561.astype(np.float32), img_647.astype(np.float32)], axis=-1
    )

    props = regionprops_table(
        mask.astype(np.int32),
        intensity_image=intensity,
        properties=("label", "area", "centroid", "intensity_mean"),
    )
    df = pd.DataFrame(props).rename(
        columns={
            "label": "cell_id",
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
            "intensity_mean-0": "mean_intensity_561",
            "intensity_mean-1": "mean_intensity_647",
        }
    )

    # NaN ratio when 561 mean is 0 to avoid divide-by-zero artifacts.
    denom = df["mean_intensity_561"].replace(0, np.nan)
    df["ratio_647_561"] = df["mean_intensity_647"] / denom

    df = df[
        [
            "cell_id",
            "centroid_x",
            "centroid_y",
            "area",
            "mean_intensity_561",
            "mean_intensity_647",
            "ratio_647_561",
        ]
    ].sort_values("cell_id").reset_index(drop=True)

    out_csv = SEG_DIR / f"{stem}_intensity.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    n_cells = len(df)
    if n_cells == 0:
        print("[SUMMARY] No cells found in mask.")
    else:
        print(
            f"[SUMMARY] cells={n_cells} | "
            f"median 561={df['mean_intensity_561'].median():.1f} | "
            f"median 647={df['mean_intensity_647'].median():.1f} | "
            f"median ratio 647/561={df['ratio_647_561'].median():.3f}"
        )


if __name__ == "__main__":
    main()
