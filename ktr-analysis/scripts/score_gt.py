"""Score trackers against the tiny GT lineage — ID-swap rate + division recall/precision.

SKELETON (Stage 03). Runs once the annotator has produced
`analysis/tracking/gt_lineage_prep/gt_lineage_filled.csv` (see that folder's
README for the format). Until then it exits cleanly with instructions.

Both trackers segment the SAME `stardist_h2b` masks, so a tracker's
`cell_in_frame_id` == the StarDist label == the GT `mask_label` at a given frame
— that shared key is how GT cells are mapped onto each tracker's `track_id`.

GT format (gt_lineage_filled.csv), per the annotation template:
  gt_cell_id, timepoint, mask_label, centroid_x, centroid_y,
  event{none|division|death|leaves_fov|enters}, parent_gt_cell_id, notes

Usage:
  python scripts/score_gt.py --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse \
      [--match-frames 2 --match-px 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REGION, FOV = "R0", 0
TRACKERS = {
    "centroid": "pilot_R0_fov0_t0-100/trajectories.csv",
    "ultrack": "pilot_ultrack_R0_fov0_t0-100/trajectories.csv",
}


# ----------------------------------------------------------------------------
# ID-swap rate
# ----------------------------------------------------------------------------
def id_swap_rate(gt: pd.DataFrame, traj: pd.DataFrame) -> dict:
    """For each GT cell, map its per-frame (timepoint, mask_label) to the tracker's
    track_id (join on timepoint + cell_in_frame_id==mask_label). A 'swap' is a
    consecutive-frame transition where that track_id changes *without* a GT
    division on the mother (divisions are legitimate identity changes)."""
    key = traj.rename(columns={"cell_in_frame_id": "mask_label"})[["timepoint", "mask_label", "track_id"]]
    g = gt.merge(key, on=["timepoint", "mask_label"], how="left").sort_values(["gt_cell_id", "timepoint"])
    swaps = links = unmapped = 0
    for _gid, sub in g.groupby("gt_cell_id"):
        tids = sub["track_id"].to_numpy()
        evs = sub["event"].fillna("none").to_numpy()
        for i in range(1, len(tids)):
            if pd.isna(tids[i]) or pd.isna(tids[i - 1]):
                unmapped += 1
                continue
            links += 1
            if tids[i] != tids[i - 1] and evs[i - 1] != "division":
                swaps += 1
    return {
        "links_evaluated": int(links),
        "id_swaps": int(swaps),
        "id_swap_rate": round(swaps / links, 4) if links else None,
        "unmapped_links": int(unmapped),
    }


# ----------------------------------------------------------------------------
# Division recall / precision
# ----------------------------------------------------------------------------
def gt_divisions(gt: pd.DataFrame) -> np.ndarray:
    d = gt[gt["event"].fillna("none") == "division"]
    return d[["timepoint", "centroid_x", "centroid_y"]].to_numpy(dtype=float)


def tracker_divisions(traj: pd.DataFrame) -> np.ndarray:
    par = pd.to_numeric(traj.get("parent_track_id"), errors="coerce")
    dau = traj.assign(_p=par).dropna(subset=["_p"])
    if dau.empty:
        return np.zeros((0, 3))
    births = dau.sort_values("timepoint").groupby("track_id").head(1)
    rows = []
    for p, grp in births.groupby("_p"):
        if grp["track_id"].nunique() >= 2:  # a real split: >=2 daughters
            rows.append([grp["timepoint"].min(), grp["centroid_x"].mean(), grp["centroid_y"].mean()])
    return np.array(rows, dtype=float) if rows else np.zeros((0, 3))


def match_divisions(gt_div: np.ndarray, trk_div: np.ndarray, kf: int, px: float) -> dict:
    matched_gt = 0
    used = set()
    for g in gt_div:
        for j, t in enumerate(trk_div):
            if j in used:
                continue
            if abs(g[0] - t[0]) <= kf and np.hypot(g[1] - t[1], g[2] - t[2]) <= px:
                matched_gt += 1
                used.add(j)
                break
    n_gt, n_trk = len(gt_div), len(trk_div)
    return {
        "gt_divisions": int(n_gt),
        "tracker_divisions": int(n_trk),
        "matched": int(matched_gt),
        "recall": round(matched_gt / n_gt, 3) if n_gt else None,
        "precision": round(len(used) / n_trk, 3) if n_trk else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--match-frames", type=int, default=2)
    ap.add_argument("--match-px", type=float, default=30.0)
    args = ap.parse_args()

    base = args.data_root.resolve() / "analysis" / "tracking"
    gt_path = base / "gt_lineage_prep" / "gt_lineage_filled.csv"
    if not gt_path.exists():
        print(f"[score_gt] GT not found: {gt_path}")
        print("[score_gt] Skeleton ready — fill the annotation template per "
              "gt_lineage_prep/README.md, then re-run. Nothing scored.")
        return 0

    gt = pd.read_csv(gt_path)
    gt_div = gt_divisions(gt)
    results = {}
    for name, rel in TRACKERS.items():
        p = base / rel
        if not p.exists():
            results[name] = {"error": f"missing {p}"}
            continue
        traj = pd.read_csv(p)
        results[name] = {
            "id_swap": id_swap_rate(gt, traj),
            "division": match_divisions(gt_div, tracker_divisions(traj), args.match_frames, args.match_px),
        }

    out = base / "gt_scores.json"
    out.write_text(json.dumps({"params": vars(args) | {"data_root": str(args.data_root)}, "results": results}, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"[score_gt] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
