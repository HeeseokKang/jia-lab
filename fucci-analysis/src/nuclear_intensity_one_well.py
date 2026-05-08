"""Per-track FUCCI intensity from eroded nuclear regions (well R1_1).

Reuses BF masks from segmentation_test/masks/ and the linked tracks from
R1_1_tracks.csv. For each filtered (track length >= 5 frames AND area > 400 px)
detection, erode the BF cell mask by `EROSION_ITERS` pixels to approximate the
nuclear region, then extract mean intensity of 561 (raw) and 647 (raw)
within that region. Tracks of length >= TRACE_MIN_LENGTH are plotted as
per-track time traces, with full-duration tracks highlighted in red.

Also writes qc_population_t30.png with filtered cells colored by raw mean_647.
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
from cellpose import utils as cp_utils
from scipy.ndimage import binary_erosion


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import RESULT_ROOT  # noqa: E402


DATA_ROOT = Path(
    "/data/Project_Data/Voltage_CellCycle/Data/20260413_FUCCI_Timelapse/"
    "timelapse_plus_bf_2026-04-16_12-11-45.768458"
)
SEG_DIR = RESULT_ROOT / "segmentation_test"
MASKS_DIR = SEG_DIR / "masks"

WELL_ID = "R1_1"
CELLLINE = "3T3"
Z = 0
EROSION_ITERS = 3
TRACK_MIN_LENGTH = 5      # frames, filter applied to tracks_df
AREA_MIN_PX = 400
TRACE_MIN_LENGTH = 20     # frames, filter for trace plots
N_FRAMES = 67             # total timepoints (0..66)

QC_TIMEPOINTS = (0, 22, 44, 66)
POPULATION_T = 30
CROP_HALF = 50            # 100x100 crop


def file_paths(t: int) -> tuple[Path, Path, Path]:
    base = f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI"
    bf_mask = MASKS_DIR / f"{WELL_ID}_t{t:03d}_BF_mask.npy"
    img_561 = DATA_ROOT / str(t) / f"{base}_561.tiff"
    # 647 raw used -- t0-reference correction found to kill signal
    # (std_raw/std_corrected = 52.6x, see nuclear_intensity_raw_comparison.py).
    img_647 = DATA_ROOT / str(t) / f"{base}_647.tiff"
    return bf_mask, img_561, img_647


def auto_contrast(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.7) -> np.ndarray:
    arr = image.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, (lo_pct, hi_pct))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def erode_cell(cell_mask: np.ndarray, iterations: int = EROSION_ITERS) -> np.ndarray:
    eroded = binary_erosion(cell_mask, iterations=iterations)
    if not eroded.any():
        return cell_mask
    return eroded


def filter_tracks(tracks_df: pd.DataFrame) -> pd.DataFrame:
    track_lengths = tracks_df.groupby("track_id")["timepoint"].nunique()
    long_tracks = set(track_lengths[track_lengths >= TRACK_MIN_LENGTH].index)
    return tracks_df[
        tracks_df["track_id"].isin(long_tracks)
        & (tracks_df["area"] > AREA_MIN_PX)
    ].copy()


def compute_nuclear_intensity(filtered: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict] = []
    for t, group in filtered.groupby("timepoint"):
        bf_mask_path, img_561_path, img_647_path = file_paths(int(t))
        bf_mask = np.load(bf_mask_path)
        img_561 = tifffile.imread(str(img_561_path))
        img_647 = tifffile.imread(str(img_647_path))

        for _, row in group.iterrows():
            cell_id = int(row["cell_id"])
            cell_mask = bf_mask == cell_id
            if not cell_mask.any():
                continue
            eroded = erode_cell(cell_mask)
            nuclear_area = int(eroded.sum())
            out_rows.append(
                {
                    "track_id": int(row["track_id"]),
                    "timepoint": int(t),
                    "cell_id": cell_id,
                    "area": float(row["area"]),
                    "nuclear_area": nuclear_area,
                    "mean_561": float(img_561[eroded].mean()),
                    "mean_647": float(img_647[eroded].mean()),
                }
            )
        print(f"[t={int(t):02d}] processed cells={len(group)}")

    df = pd.DataFrame(out_rows).sort_values(["track_id", "timepoint"]).reset_index(drop=True)
    df["ratio_647_561"] = df["mean_647"] / df["mean_561"].replace(0, np.nan)
    return df


def plot_traces(df: pd.DataFrame, value_col: str, ylabel: str, out_path: Path) -> None:
    lengths = df.groupby("track_id")["timepoint"].nunique()
    keep = lengths[lengths >= TRACE_MIN_LENGTH].index
    full_duration = set(lengths[lengths == N_FRAMES].index)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    n_full = 0
    for tid in keep:
        sub = df[df["track_id"] == tid].sort_values("timepoint")
        is_full = tid in full_duration
        if is_full:
            n_full += 1
        ax.plot(
            sub["timepoint"],
            sub[value_col],
            color="tab:red" if is_full else "0.4",
            lw=1.4 if is_full else 0.6,
            alpha=0.9 if is_full else 0.45,
        )

    ax.set_xlabel("timepoint")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{WELL_ID} | per-track {ylabel} | length >= {TRACE_MIN_LENGTH} "
        f"({len(keep)} tracks, red = full duration {N_FRAMES} frames, n={n_full})",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_qc_segmentation_grid(filtered: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        len(QC_TIMEPOINTS), 2,
        figsize=(11, 5 * len(QC_TIMEPOINTS)),
        constrained_layout=True,
    )
    for r, t in enumerate(QC_TIMEPOINTS):
        bf_mask_path, _, _ = file_paths(t)
        bf_path = DATA_ROOT / str(t) / f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI_BF.tiff"
        bf_img = tifffile.imread(str(bf_path))
        bf_mask = np.load(bf_mask_path)
        outlines = cp_utils.masks_to_outlines(bf_mask)

        keep_ids = {
            int(c) for c in filtered.loc[filtered["timepoint"] == t, "cell_id"]
        }
        nuclear_union = np.zeros_like(bf_mask, dtype=bool)
        for cid in keep_ids:
            cell_mask = bf_mask == cid
            if not cell_mask.any():
                continue
            nuclear_union |= erode_cell(cell_mask)

        bf_disp = auto_contrast(bf_img)

        ax_a = axes[r, 0]
        ax_a.imshow(bf_disp, cmap="gray", vmin=0, vmax=1)
        ov_a = np.zeros((*outlines.shape, 4), dtype=np.float32)
        ov_a[..., 0] = 1.0
        ov_a[..., 3] = outlines.astype(np.float32)
        ax_a.imshow(ov_a)
        ax_a.set_title(f"BF + all cell mask outlines | t={t}", fontsize=10)
        ax_a.set_xticks([])
        ax_a.set_yticks([])

        ax_b = axes[r, 1]
        ax_b.imshow(bf_disp, cmap="gray", vmin=0, vmax=1)
        ov_b = np.zeros((*nuclear_union.shape, 4), dtype=np.float32)
        ov_b[..., 0] = 1.0
        ov_b[..., 1] = 1.0
        ov_b[..., 3] = nuclear_union.astype(np.float32) * 0.45
        ax_b.imshow(ov_b)
        ax_b.set_title(
            f"BF + filtered nuclear masks (eroded {EROSION_ITERS}px) | t={t}",
            fontsize=10,
        )
        ax_b.set_xticks([])
        ax_b.set_yticks([])

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_single_cell_spotlight(
    intensity_df: pd.DataFrame, tracks_df: pd.DataFrame, out_path: Path
) -> None:
    lengths = intensity_df.groupby("track_id")["timepoint"].nunique()
    full_tracks = lengths[lengths == N_FRAMES].index
    if len(full_tracks) == 0:
        candidates = lengths.sort_values(ascending=False).index[:1]
        print("[QC][warn] No full-duration tracks survived filter, using longest")
    else:
        candidates = full_tracks

    means_647 = (
        intensity_df[intensity_df["track_id"].isin(candidates)]
        .groupby("track_id")["mean_647"]
        .mean()
    )
    chosen = int(means_647.idxmax())
    print(f"[QC] spotlight track_id={chosen}, mean_647={means_647.loc[chosen]:.1f}")

    fig, axes = plt.subplots(
        2, len(QC_TIMEPOINTS),
        figsize=(3.5 * len(QC_TIMEPOINTS), 7),
        constrained_layout=True,
    )

    for c, t in enumerate(QC_TIMEPOINTS):
        sub = intensity_df[
            (intensity_df["track_id"] == chosen) & (intensity_df["timepoint"] == t)
        ]
        track_row = tracks_df[
            (tracks_df["track_id"] == chosen) & (tracks_df["timepoint"] == t)
        ]
        if len(sub) == 0 or len(track_row) == 0:
            for r in range(2):
                axes[r, c].axis("off")
                axes[r, c].set_title(f"t={t}\n(missing)", fontsize=10)
            continue

        cx = int(round(track_row.iloc[0]["centroid_x"]))
        cy = int(round(track_row.iloc[0]["centroid_y"]))

        bf_path = DATA_ROOT / str(t) / f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI_BF.tiff"
        c647_path = DATA_ROOT / str(t) / f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI_647.tiff"
        bf_img = tifffile.imread(str(bf_path))
        c647_img = tifffile.imread(str(c647_path))

        h, w = bf_img.shape
        y0 = max(0, cy - CROP_HALF); y1 = min(h, cy + CROP_HALF)
        x0 = max(0, cx - CROP_HALF); x1 = min(w, cx + CROP_HALF)

        bf_crop = auto_contrast(bf_img[y0:y1, x0:x1])
        c647_crop = auto_contrast(c647_img[y0:y1, x0:x1])

        axes[0, c].imshow(bf_crop, cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"BF | t={t}", fontsize=10)
        axes[0, c].set_xticks([]); axes[0, c].set_yticks([])

        m647 = sub.iloc[0]["mean_647"]
        axes[1, c].imshow(c647_crop, cmap="magma", vmin=0, vmax=1)
        axes[1, c].set_title(f"647 | t={t} | mean_647={m647:.0f}", fontsize=10)
        axes[1, c].set_xticks([]); axes[1, c].set_yticks([])

    fig.suptitle(
        f"{WELL_ID} | track_id={chosen} (full-duration, highest mean_647)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_population_snapshot(intensity_df: pd.DataFrame, out_path: Path) -> None:
    t = POPULATION_T
    sub = intensity_df[intensity_df["timepoint"] == t]
    if len(sub) == 0:
        print(f"[WARN] no filtered cells at t={t}; skipping population snapshot")
        return

    bf_mask_path, _, c647_path = file_paths(t)
    bf_mask = np.load(bf_mask_path)
    c647_img = tifffile.imread(str(c647_path))
    bg = auto_contrast(c647_img)

    paint = np.full(bf_mask.shape, np.nan, dtype=np.float32)
    for _, row in sub.iterrows():
        cell_mask = bf_mask == int(row["cell_id"])
        if not cell_mask.any():
            continue
        eroded = erode_cell(cell_mask)
        paint[eroded] = float(row["mean_647"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 9), constrained_layout=True)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)

    finite_paint = paint[np.isfinite(paint)]
    vmin = float(np.percentile(finite_paint, 5)) if finite_paint.size else 0.0
    vmax = float(np.percentile(finite_paint, 95)) if finite_paint.size else 1.0

    ma = np.ma.masked_invalid(paint)
    im = ax.imshow(ma, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.85)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("mean_647 (raw)")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"{WELL_ID} | t={t} | filtered nuclear regions colored by mean_647 "
        f"(n={len(sub)})",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    SEG_DIR.mkdir(parents=True, exist_ok=True)

    tracks_csv = SEG_DIR / f"{WELL_ID}_tracks.csv"
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Tracks CSV missing: {tracks_csv}")

    print(f"[LOAD] {tracks_csv}")
    tracks_df = pd.read_csv(tracks_csv)
    print(
        f"[INFO] tracks raw: rows={len(tracks_df)}, "
        f"tracks={tracks_df['track_id'].nunique()}"
    )

    filtered = filter_tracks(tracks_df)
    print(
        f"[INFO] after filter (length>={TRACK_MIN_LENGTH}, area>{AREA_MIN_PX}): "
        f"rows={len(filtered)}, tracks={filtered['track_id'].nunique()}"
    )

    intensity_df = compute_nuclear_intensity(filtered)
    out_csv = SEG_DIR / f"{WELL_ID}_nuclear_intensity.csv"
    intensity_df.to_csv(
        out_csv,
        index=False,
        columns=[
            "track_id",
            "timepoint",
            "cell_id",
            "area",
            "nuclear_area",
            "mean_561",
            "mean_647",
        ],
    )
    print(f"[SAVED] {out_csv}  rows={len(intensity_df)}")

    full_lengths = intensity_df.groupby("track_id")["timepoint"].nunique()
    full_track_ids = full_lengths[full_lengths == N_FRAMES].index
    full = intensity_df[intensity_df["track_id"].isin(full_track_ids)].copy()
    if len(full) > 0:
        per_track = full.groupby("track_id").agg(
            mean_561=("mean_561", "mean"),
            mean_647=("mean_647", "mean"),
            ratio_647_561=("ratio_647_561", "mean"),
        )
        print("[STATS] full-duration tracks only")
        print(f"[STATS] std mean_647 (raw) = {per_track['mean_647'].std(ddof=1):.3f}")
        print(f"[STATS] std mean_561 (raw) = {per_track['mean_561'].std(ddof=1):.3f}")
        print(
            "[STATS] ratio_647_561 range = "
            f"{per_track['ratio_647_561'].min():.3f} - "
            f"{per_track['ratio_647_561'].max():.3f}"
        )

    for col, label, fname in [
        ("mean_561", "mean_561 (raw)", f"{WELL_ID}_trace_561.png"),
        ("mean_647", "mean_647 (raw)", f"{WELL_ID}_trace_647.png"),
        ("ratio_647_561", "ratio 647 / 561", f"{WELL_ID}_trace_ratio.png"),
    ]:
        out = SEG_DIR / fname
        plot_traces(intensity_df, col, label, out)
        print(f"[SAVED] {out}")

    plot_population_snapshot(intensity_df, SEG_DIR / "qc_population_t30.png")
    print(f"[SAVED] {SEG_DIR / 'qc_population_t30.png'}")


if __name__ == "__main__":
    main()
