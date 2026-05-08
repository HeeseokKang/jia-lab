"""Compare corrected vs raw 647 nuclear intensity for one tracked well.

This diagnostic keeps the tracking, area filter, and BF-mask erosion identical
to nuclear_intensity_one_well.py. The only variable is the 647 source:

- corrected 647 from analysis/.../corrected/
- raw 647 from DATA_ROOT

If raw 647 has much larger across-track variation than corrected 647, the
flat-field correction likely suppressed biological FUCCI signal.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_erosion


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
N_FRAMES = 67
EROSION_ITERS = 3
TRACK_MIN_LENGTH = 5
AREA_MIN_PX = 400


def image_paths(t: int) -> tuple[Path, Path, Path, Path]:
    base = f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI"
    mask = MASKS_DIR / f"{WELL_ID}_t{t:03d}_BF_mask.npy"
    img_561 = DATA_ROOT / str(t) / f"{base}_561.tiff"
    img_647_corrected = CORRECTED_ROOT / str(t) / f"{base}_647.tiff"
    img_647_raw = DATA_ROOT / str(t) / f"{base}_647.tiff"
    return mask, img_561, img_647_corrected, img_647_raw


def erode_cell(cell_mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(cell_mask, iterations=EROSION_ITERS)
    return eroded if eroded.any() else cell_mask


def filter_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    lengths = tracks.groupby("track_id")["timepoint"].nunique()
    keep_tracks = set(lengths[lengths >= TRACK_MIN_LENGTH].index)
    return tracks[
        tracks["track_id"].isin(keep_tracks) & (tracks["area"] > AREA_MIN_PX)
    ].copy()


def extract_comparison(filtered: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for t, group in filtered.groupby("timepoint"):
        t = int(t)
        mask_path, img_561_path, c647_path, raw647_path = image_paths(t)
        for path in (mask_path, img_561_path, c647_path, raw647_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing input at t={t}: {path}")

        mask = np.load(mask_path)
        img_561 = tifffile.imread(str(img_561_path))
        img_647_corrected = tifffile.imread(str(c647_path))
        img_647_raw = tifffile.imread(str(raw647_path))

        for _, row in group.iterrows():
            cell_id = int(row["cell_id"])
            cell_mask = mask == cell_id
            if not cell_mask.any():
                continue
            nuclear_mask = erode_cell(cell_mask)
            rows.append(
                {
                    "track_id": int(row["track_id"]),
                    "timepoint": t,
                    "cell_id": cell_id,
                    "nuclear_area": int(nuclear_mask.sum()),
                    "mean_561": float(img_561[nuclear_mask].mean()),
                    "mean_647_corrected": float(img_647_corrected[nuclear_mask].mean()),
                    "mean_647_raw": float(img_647_raw[nuclear_mask].mean()),
                }
            )

        print(f"[t={t:02d}] processed cells={len(group)}")

    return pd.DataFrame(rows).sort_values(["track_id", "timepoint"]).reset_index(drop=True)


def plot_trace_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    full_tracks: set[int],
) -> None:
    for track_id, sub in df.groupby("track_id"):
        is_full = int(track_id) in full_tracks
        ax.plot(
            sub["timepoint"],
            sub[value_col],
            color="tab:red" if is_full else "0.35",
            lw=1.4 if is_full else 0.5,
            alpha=0.9 if is_full else 0.35,
        )
    ax.set_title(title)
    ax.set_xlabel("timepoint")
    ax.set_ylabel("mean 647")
    ax.grid(True, alpha=0.3)


def plot_hist_axis(ax: plt.Axes, values: pd.Series, title: str, color: str) -> None:
    ax.hist(values.dropna(), bins=20, color=color, edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("per-track mean 647")
    ax.set_ylabel("N tracks")
    ax.grid(True, axis="y", alpha=0.3)


def make_diagnostic_plot(df: pd.DataFrame, out_path: Path) -> tuple[float, float, float]:
    lengths = df.groupby("track_id")["timepoint"].nunique()
    full_tracks = set(int(t) for t in lengths[lengths == N_FRAMES].index)

    full = df[df["track_id"].isin(full_tracks)]
    per_track_full = full.groupby("track_id").agg(
        mean_647_corrected=("mean_647_corrected", "mean"),
        mean_647_raw=("mean_647_raw", "mean"),
    )

    std_corrected = float(per_track_full["mean_647_corrected"].std(ddof=1))
    std_raw = float(per_track_full["mean_647_raw"].std(ddof=1))
    std_ratio = float(std_raw / std_corrected) if std_corrected != 0 else np.inf

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot_trace_axis(
        axes[0, 0],
        df,
        "mean_647_corrected",
        f"Corrected 647 traces (all filtered tracks; red=full, n={len(full_tracks)})",
        full_tracks,
    )
    plot_trace_axis(
        axes[0, 1],
        df,
        "mean_647_raw",
        f"Raw 647 traces (all filtered tracks; red=full, n={len(full_tracks)})",
        full_tracks,
    )
    plot_hist_axis(
        axes[1, 0],
        per_track_full["mean_647_corrected"],
        "Full-duration tracks: corrected 647",
        "tab:blue",
    )
    plot_hist_axis(
        axes[1, 1],
        per_track_full["mean_647_raw"],
        "Full-duration tracks: raw 647",
        "tab:orange",
    )
    fig.suptitle(
        f"{WELL_ID} | corrected vs raw 647 diagnostic | erosion={EROSION_ITERS}px",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return std_corrected, std_raw, std_ratio


def main() -> None:
    tracks_csv = SEG_DIR / f"{WELL_ID}_tracks.csv"
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Tracks CSV missing: {tracks_csv}")

    print(f"[LOAD] {tracks_csv}")
    tracks = pd.read_csv(tracks_csv)
    filtered = filter_tracks(tracks)
    print(
        f"[INFO] filtered tracks: rows={len(filtered)}, "
        f"tracks={filtered['track_id'].nunique()} "
        f"(length>={TRACK_MIN_LENGTH}, area>{AREA_MIN_PX})"
    )

    comparison = extract_comparison(filtered)
    out_csv = SEG_DIR / f"{WELL_ID}_nuclear_647_comparison.csv"
    comparison.to_csv(
        out_csv,
        index=False,
        columns=[
            "track_id",
            "timepoint",
            "cell_id",
            "nuclear_area",
            "mean_561",
            "mean_647_corrected",
            "mean_647_raw",
        ],
    )
    print(f"[SAVED] {out_csv} rows={len(comparison)}")

    plot_path = SEG_DIR / f"{WELL_ID}_647_correction_diagnosis.png"
    std_corrected, std_raw, std_ratio = make_diagnostic_plot(comparison, plot_path)
    print(f"[SAVED] {plot_path}")

    print("[DIAGNOSTIC] std across full-duration track means")
    print(f"[DIAGNOSTIC] std mean_647_corrected = {std_corrected:.3f}")
    print(f"[DIAGNOSTIC] std mean_647_raw       = {std_raw:.3f}")
    print(f"[DIAGNOSTIC] std_raw / std_corrected = {std_ratio:.3f}")


if __name__ == "__main__":
    main()
