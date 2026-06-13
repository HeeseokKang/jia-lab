"""Prepare a TINY manual GT-lineage annotation set (~24 cells) for Stage 03.

Selects well-separated, long-lived candidate nuclei from the centroid pilot
trajectories (R0/fov0 t0-100), then emits the minimal products an annotator
needs to build a small ground-truth lineage for measuring division recall and
ID-swap rate (the metrics neither tracker can self-report):

  gt_candidates.csv        seed cells (id, t0 centroid, area, nn_dist, track_len)
  gt_lineage_template.csv  empty annotation template (one row per cell-frame)
  gt_candidates_overlay.png  t0/t50/t100 montage with numbered candidate markers

Planning/prototyping product only — does NOT itself create ground truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.io import get_fov_timeseries, parse_dataset  # noqa: E402

REGION, FOV = "R0", 0


def _norm(im, lo=1.0, hi=99.5):
    a, b = np.percentile(im, [lo, hi])
    return np.clip((im.astype(np.float32) - a) / max(b - a, 1e-6), 0, 1)


def farthest_point_sample(xy: np.ndarray, k: int) -> list[int]:
    """Greedy farthest-point sampling for spatial spread."""
    if len(xy) <= k:
        return list(range(len(xy)))
    chosen = [int(np.argmax(xy[:, 0] + xy[:, 1]))]  # deterministic seed
    d = cdist(xy, xy[chosen])
    while len(chosen) < k:
        mind = d.min(axis=1)
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        d = np.minimum(d, cdist(xy, xy[[nxt]]))
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--acquisition", default="timelapse_2026-05-05_18-12-11.466141")
    ap.add_argument("--n-cells", type=int, default=24)
    ap.add_argument("--nn-min-px", type=float, default=35.0)
    ap.add_argument("--min-track-len", type=int, default=70)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    pilot_dir = data_root / "analysis" / "tracking" / f"pilot_{REGION}_fov{FOV}_t0-100"
    traj = pd.read_csv(pilot_dir / "trajectories.csv")
    out_dir = data_root / "analysis" / "tracking" / "gt_lineage_prep"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = int(traj["timepoint"].min())
    length = traj.groupby("track_id")["timepoint"].nunique()
    at0 = traj[traj["timepoint"] == t0].copy()
    xy0 = at0[["centroid_x", "centroid_y"]].to_numpy()
    D = cdist(xy0, xy0)
    np.fill_diagonal(D, np.inf)
    at0["nn_dist"] = D.min(axis=1)
    at0["track_len"] = at0["track_id"].map(length).astype(int)

    pool = at0[(at0["nn_dist"] >= args.nn_min_px) & (at0["track_len"] >= args.min_track_len)]
    if len(pool) < args.n_cells:  # relax if too strict
        pool = at0[at0["track_len"] >= args.min_track_len].sort_values("nn_dist", ascending=False)
    pool = pool.reset_index(drop=True)
    idx = farthest_point_sample(pool[["centroid_x", "centroid_y"]].to_numpy(), args.n_cells)
    cand = pool.iloc[idx].reset_index(drop=True)
    cand.insert(0, "gt_cell_id", range(1, len(cand) + 1))

    cand_out = cand[["gt_cell_id", "track_id", "timepoint", "centroid_x", "centroid_y", "area", "nn_dist", "track_len"]]
    cand_out.to_csv(out_dir / "gt_candidates.csv", index=False)

    # empty annotation template: annotator fills mask_label/centroid per frame + events
    template = pd.DataFrame(columns=[
        "gt_cell_id", "timepoint", "mask_label", "centroid_x", "centroid_y",
        "event",  # none | division | death | leaves_fov | enters
        "parent_gt_cell_id",  # set on the two daughters at a division
        "notes",
    ])
    template.to_csv(out_dir / "gt_lineage_template.csv", index=False)

    # QC overlay montage at t0/t50/t100 with numbered candidate markers
    df = parse_dataset(data_root / args.acquisition)
    h2b_paths = get_fov_timeseries(df, region=REGION, fov=FOV)["mTagBFP2"]
    show_tps = [t0, t0 + 50, t0 + 100]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for ax, tp in zip(axes, show_tps):
        ax.imshow(_norm(tifffile.imread(h2b_paths[tp])), cmap="gray", interpolation="nearest")
        sub = traj[(traj["timepoint"] == tp) & (traj["track_id"].isin(cand["track_id"]))]
        id_map = dict(zip(cand["track_id"], cand["gt_cell_id"]))
        for r in sub.itertuples():
            ax.plot(r.centroid_x, r.centroid_y, "o", mfc="none", mec="#ffcc00", ms=14, mew=1.5)
            ax.text(r.centroid_x + 6, r.centroid_y - 6, str(id_map[r.track_id]), color="#ff44aa", fontsize=9, weight="bold")
        ax.set_title(f"t={tp:04d}  ({len(sub)} of {len(cand)} candidates present)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"GT-lineage candidate seeds — {REGION}/fov{FOV} — {len(cand)} cells (numbered)", fontsize=13)
    fig.savefig(out_dir / "gt_candidates_overlay.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[gt-prep] selected {len(cand)} candidate cells -> {out_dir}")
    print(f"[gt-prep]   nn_dist range [{cand['nn_dist'].min():.0f},{cand['nn_dist'].max():.0f}] px; track_len min {cand['track_len'].min()}")
    print(f"[gt-prep]   files: gt_candidates.csv, gt_lineage_template.csv, gt_candidates_overlay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
