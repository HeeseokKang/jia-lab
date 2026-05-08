"""Frame-to-frame cell tracking for one well using trackpy.

Reads the per-timepoint Cellpose masks produced by timeseries_one_well.py,
extracts centroid + area per cell with regionprops, then links detections
across timepoints by centroid proximity. Track ids are saved alongside the
per-cell rows.

QC characterizes how many tracks are real (full-duration, or entering/exiting
through the FOV edge) vs. suspicious mid-FOV ghost tracks. Division/lineage
is not handled here.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from skimage.measure import regionprops_table


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import RESULT_ROOT  # noqa: E402


SEG_DIR = RESULT_ROOT / "segmentation_test"
MASKS_DIR = SEG_DIR / "masks"

WELL_ID = "R1_1"
TIMEPOINT_RANGE = range(0, 67)
FIRST_T = TIMEPOINT_RANGE.start
LAST_T = TIMEPOINT_RANGE.stop - 1

SEARCH_RANGE_PX = 50  # max centroid jump allowed between consecutive frames
MEMORY_FRAMES = 1     # tolerate one missed detection so brief Cellpose misses are forgiven
EDGE_PX = 50          # distance from FOV border that counts as "edge"


def build_centroid_table() -> tuple[pd.DataFrame, tuple[int, int]]:
    rows: list[pd.DataFrame] = []
    img_shape: tuple[int, int] | None = None

    for t in TIMEPOINT_RANGE:
        mask_path = MASKS_DIR / f"{WELL_ID}_t{t:03d}_BF_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        mask = np.load(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {mask.shape}: {mask_path}")
        if img_shape is None:
            img_shape = (int(mask.shape[0]), int(mask.shape[1]))
        elif (int(mask.shape[0]), int(mask.shape[1])) != img_shape:
            raise ValueError(f"Shape mismatch at t={t}: {mask.shape} vs {img_shape}")

        if mask.max() == 0:
            continue

        props = regionprops_table(
            mask.astype(np.int32),
            properties=("label", "area", "centroid"),
        )
        df = pd.DataFrame(props).rename(
            columns={
                "label": "cell_id",
                "centroid-0": "centroid_y",
                "centroid-1": "centroid_x",
            }
        )
        df.insert(0, "timepoint", t)
        rows.append(df)

    centroids = pd.concat(rows, ignore_index=True)
    assert img_shape is not None
    return centroids, img_shape


def link_with_trackpy(centroids: pd.DataFrame) -> pd.DataFrame:
    # trackpy expects columns named x, y, frame.
    feats = centroids.rename(
        columns={"centroid_x": "x", "centroid_y": "y", "timepoint": "frame"}
    )
    linked = tp.link(feats, search_range=SEARCH_RANGE_PX, memory=MEMORY_FRAMES)
    linked = linked.rename(
        columns={
            "x": "centroid_x",
            "y": "centroid_y",
            "frame": "timepoint",
            "particle": "track_id",
        }
    )
    linked["track_id"] = linked["track_id"].astype(int)
    return linked


def compute_qc(linked: pd.DataFrame, shape: tuple[int, int]) -> dict[str, int]:
    h, w = shape
    first_idx = linked.groupby("track_id")["timepoint"].idxmin()
    last_idx = linked.groupby("track_id")["timepoint"].idxmax()
    first = linked.loc[first_idx].set_index("track_id")
    last = linked.loc[last_idx].set_index("track_id")

    starts_t0 = first["timepoint"] == FIRST_T
    ends_tlast = last["timepoint"] == LAST_T

    def at_edge(x: pd.Series, y: pd.Series) -> pd.Series:
        return (
            (x <= EDGE_PX)
            | (x >= w - EDGE_PX)
            | (y <= EDGE_PX)
            | (y >= h - EDGE_PX)
        )

    first_at_edge = at_edge(first["centroid_x"], first["centroid_y"])
    last_at_edge = at_edge(last["centroid_x"], last["centroid_y"])

    return {
        "n_tracks_total": int(first.shape[0]),
        "n_tracks_full_duration": int((starts_t0 & ends_tlast).sum()),
        "n_tracks_edge_endpoint": int((first_at_edge | last_at_edge).sum()),
        "n_tracks_suspicious_mid_fov": int(
            (~starts_t0 & ~ends_tlast & ~first_at_edge & ~last_at_edge).sum()
        ),
    }


def plot_track_length_distribution(
    linked: pd.DataFrame, qc: dict[str, int], out_path: Path
) -> None:
    lengths = linked.groupby("track_id")["timepoint"].nunique()
    n_frames = LAST_T - FIRST_T + 1

    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
    ax.hist(
        lengths,
        bins=np.arange(1, n_frames + 2) - 0.5,
        edgecolor="black",
        color="tab:blue",
        alpha=0.85,
    )
    ax.set_xlabel("Track length (number of frames present)")
    ax.set_ylabel("Number of tracks")
    ax.set_title(
        f"{WELL_ID} | track length distribution | "
        f"total={qc['n_tracks_total']}, full={qc['n_tracks_full_duration']}, "
        f"edge_endpoint={qc['n_tracks_edge_endpoint']}, "
        f"suspicious_mid_fov={qc['n_tracks_suspicious_mid_fov']}",
        fontsize=10,
    )
    ax.axvline(n_frames, color="tab:green", lw=1.2, linestyle="--", label=f"full duration ({n_frames})")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    SEG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] reading masks from {MASKS_DIR}")
    centroids, shape = build_centroid_table()
    print(
        f"[INFO] image shape={shape}, "
        f"total per-frame detections={len(centroids)}, "
        f"frames={centroids['timepoint'].nunique()}"
    )

    print(f"[LINK] trackpy.link(search_range={SEARCH_RANGE_PX}, memory={MEMORY_FRAMES})")
    linked = link_with_trackpy(centroids)

    linked = linked[
        [
            "timepoint",
            "cell_id",
            "track_id",
            "centroid_x",
            "centroid_y",
            "area",
        ]
    ].sort_values(["track_id", "timepoint"]).reset_index(drop=True)

    out_csv = SEG_DIR / f"{WELL_ID}_tracks.csv"
    linked.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}  rows={len(linked)}")

    qc = compute_qc(linked, shape)
    for k, v in qc.items():
        print(f"[QC] {k} = {v}")

    plot_path = SEG_DIR / f"{WELL_ID}_track_qc.png"
    plot_track_length_distribution(linked, qc, plot_path)
    print(f"[SAVED] {plot_path}")


if __name__ == "__main__":
    main()
