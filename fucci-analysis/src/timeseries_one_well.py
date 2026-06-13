"""Single-well FUCCI ratio timeseries (no cell tracking).

For one well (default R1_1) iterate over every timepoint folder, run
Cellpose-SAM on BF, then extract per-cell mean intensity for 561 (raw) and
647 (flat-field corrected). Aggregate and plot the per-timepoint median
ratio with a 10th-90th percentile band, after a small area filter to
suppress obvious bubble/debris false positives.

Cell ids are NOT consistent across timepoints; tracking is a separate step.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
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
MASKS_DIR = SEG_DIR / "masks"

WELL_ID = "R1_1"
CELLLINE = "3T3"
Z = 0
TIMEPOINT_RANGE = range(0, 67)
AREA_MIN_PX = 400  # drops debris/bubble false positives in the population summary


def file_paths(timepoint: int) -> tuple[Path, Path, Path]:
    base = f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI"
    bf = DATA_ROOT / str(timepoint) / f"{base}_BF.tiff"
    c561 = DATA_ROOT / str(timepoint) / f"{base}_561.tiff"
    c647 = CORRECTED_ROOT / str(timepoint) / f"{base}_647.tiff"
    return bf, c561, c647


def extract_per_cell(
    mask: np.ndarray, img_561: np.ndarray, img_647: np.ndarray
) -> pd.DataFrame:
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
    denom = df["mean_intensity_561"].replace(0, np.nan)
    df["ratio_647_561"] = df["mean_intensity_647"] / denom
    return df


def main() -> None:
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    print("[CELLPOSE] Loading Cellpose-SAM (cpsam) once before the timepoint loop")
    model = models.CellposeModel(gpu=True, model_type="cpsam")

    per_t_frames: list[pd.DataFrame] = []
    t_start = time.time()

    for t in TIMEPOINT_RANGE:
        bf_path, c561_path, c647_path = file_paths(t)
        for p in (bf_path, c561_path, c647_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing input at t={t}: {p}")

        bf_img = tifffile.imread(str(bf_path))
        masks, _flows, _styles = model.eval(bf_img, diameter=None)
        n_cells = int(masks.max())

        mask_path = MASKS_DIR / f"{WELL_ID}_t{t:03d}_BF_mask.npy"
        np.save(mask_path, masks.astype(np.int32))

        img_561 = tifffile.imread(str(c561_path))
        img_647 = tifffile.imread(str(c647_path))
        df_t = extract_per_cell(masks, img_561, img_647)
        df_t.insert(0, "timepoint", t)
        per_t_frames.append(df_t)

        elapsed = time.time() - t_start
        med_ratio = df_t["ratio_647_561"].median()
        med_str = f"{med_ratio:.3f}" if pd.notna(med_ratio) else "NaN"
        print(
            f"[t={t:02d}] cells={n_cells:3d} "
            f"median_ratio={med_str} "
            f"elapsed={elapsed:6.1f}s -> {mask_path.name}"
        )

    all_df = pd.concat(per_t_frames, ignore_index=True)[
        [
            "timepoint",
            "cell_id",
            "area",
            "centroid_x",
            "centroid_y",
            "mean_intensity_561",
            "mean_intensity_647",
            "ratio_647_561",
        ]
    ]
    out_csv = SEG_DIR / f"{WELL_ID}_all_timepoints.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}  rows={len(all_df)}")

    filtered = all_df[all_df["area"] > AREA_MIN_PX]
    summary = (
        filtered.groupby("timepoint")["ratio_647_561"]
        .agg(
            median="median",
            p10=lambda s: s.quantile(0.10),
            p90=lambda s: s.quantile(0.90),
            n="count",
        )
        .reset_index()
        .sort_values("timepoint")
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax.fill_between(
        summary["timepoint"],
        summary["p10"],
        summary["p90"],
        color="tab:blue",
        alpha=0.2,
        label="10th–90th percentile",
    )
    ax.plot(
        summary["timepoint"],
        summary["median"],
        color="tab:blue",
        lw=2,
        label="median",
    )
    ax.set_xlabel("timepoint")
    ax.set_ylabel("ratio 647 / 561 (per-cell mean)")
    ax.set_title(
        f"{WELL_ID} | per-cell FUCCI ratio across timepoints "
        f"(area > {AREA_MIN_PX} px)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plot_path = SEG_DIR / f"{WELL_ID}_ratio_timeseries.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {plot_path}")


if __name__ == "__main__":
    main()
