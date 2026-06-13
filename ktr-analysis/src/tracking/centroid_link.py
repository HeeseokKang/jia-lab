"""Minimal reusable tracking interface — centroid-linking backend (v1 pilot).

This is the *tracker-agnostic* interface for Stage 03: extract per-frame
detections from label masks, link them into tracks, and emit the locked
longitudinal schema. The linking backend here is dependency-light
centroid-linking (scipy Hungarian + a small memory for gaps) — chosen for the
pilot because trackpy is not installed and Ultrack lives in its own env.

Swap path (schema unchanged; bump TRACKING_POLICY_VERSION):
  v1 centroid-linking (here)  →  trackpy / btrack  →  Ultrack (primary GT, ultrack_env)

The functions below ARE the minimal interface a production tracker must satisfy:
  extract_detections(masks) -> detections df
  link(detections, params)  -> detections df + track_id   [swappable backend]
  to_trajectories(df, ...)  -> longitudinal-schema df
No mitosis logic in v1: parent_track_id / lineage / generation are left at root
defaults; division *candidates* are surfaced separately for inspection only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage import measure

# Locked longitudinal schema (CLAUDE.md). Trackers fill track_id; v1 leaves the
# lineage columns at root defaults until a mitosis-aware backend (btrack/Ultrack).
SCHEMA_COLUMNS = [
    "region", "fov", "timepoint", "cell_in_frame_id", "track_id", "lineage_id",
    "parent_track_id", "generation", "mitosis_event_frame",
    "centroid_x", "centroid_y", "area", "nuclear_mask_path",
]

_BIG = 1e6


def extract_detections(masks: Sequence[np.ndarray], mask_paths: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-frame detections (centroid, area) from a sequence of label masks."""
    rows = []
    for t, m in enumerate(masks):
        for p in measure.regionprops(m):
            cy, cx = p.centroid
            rows.append({
                "timepoint": int(t),
                "cell_in_frame_id": int(p.label),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "area": float(p.area),
                "nuclear_mask_path": (mask_paths[t] if mask_paths else ""),
            })
    return pd.DataFrame(rows)


def link_centroids(det: pd.DataFrame, search_range_px: float = 25.0, memory: int = 2) -> pd.DataFrame:
    """Greedy-optimal (Hungarian) frame-to-frame centroid linking with gap memory.

    A track lost at frame L may be re-acquired at frames L+1 … L+memory within
    ``search_range_px``. Returns ``det`` with a ``track_id`` column.
    """
    det = det.sort_values(["timepoint", "cell_in_frame_id"]).reset_index(drop=True)
    frames = sorted(det["timepoint"].unique())
    next_tid = 0
    active: dict[int, dict] = {}            # track_id -> {last_frame, cx, cy}
    track_of: dict[tuple[int, int], int] = {}

    for t in frames:
        cur = det[det["timepoint"] == t]
        cand = [(tid, info) for tid, info in active.items() if t - info["last_frame"] <= memory + 1]
        assigned = set()
        if cand and len(cur):
            cur_xy = cur[["centroid_x", "centroid_y"]].to_numpy()
            cost = np.full((len(cand), len(cur)), _BIG)
            for i, (_, info) in enumerate(cand):
                d = np.hypot(cur_xy[:, 0] - info["cx"], cur_xy[:, 1] - info["cy"])
                cost[i] = np.where(d <= search_range_px, d, _BIG)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] < _BIG:
                    tid = cand[r][0]
                    row = cur.iloc[c]
                    track_of[(t, int(row.cell_in_frame_id))] = tid
                    active[tid] = {"last_frame": t, "cx": row.centroid_x, "cy": row.centroid_y}
                    assigned.add(c)
        for c in range(len(cur)):
            if c not in assigned:
                row = cur.iloc[c]
                tid = next_tid
                next_tid += 1
                track_of[(t, int(row.cell_in_frame_id))] = tid
                active[tid] = {"last_frame": t, "cx": row.centroid_x, "cy": row.centroid_y}
        active = {tid: info for tid, info in active.items() if t - info["last_frame"] <= memory + 1}

    det["track_id"] = [track_of[(int(r.timepoint), int(r.cell_in_frame_id))] for r in det.itertuples()]
    return det


def to_trajectories(det: pd.DataFrame, region: str, fov: int) -> pd.DataFrame:
    """Project a linked detection table onto the locked longitudinal schema.

    v1 (centroid-linking): every track is its own lineage root —
    lineage_id = track_id, parent_track_id / mitosis_event_frame = NA, generation = 0.
    """
    df = det.copy()
    df["region"] = region
    df["fov"] = int(fov)
    df["lineage_id"] = df["track_id"]
    df["parent_track_id"] = pd.NA
    df["generation"] = 0
    df["mitosis_event_frame"] = pd.NA
    return df[SCHEMA_COLUMNS]


def track_quality_metrics(traj: pd.DataFrame, n_frames: int) -> dict:
    """Continuity / fragmentation / ID-persistence summary (no GT lineage needed)."""
    g = traj.groupby("track_id")["timepoint"]
    lengths = g.nunique()
    spans = g.agg(lambda s: int(s.max() - s.min() + 1))
    n_tracks = int(len(lengths))
    n_det = int(len(traj))
    mean_cells = n_det / max(n_frames, 1)
    full = int((lengths == n_frames).sum())
    gaps = (spans - lengths)
    return {
        "n_tracks": n_tracks,
        "n_detections": n_det,
        "mean_cells_per_frame": round(mean_cells, 1),
        "fragmentation_ratio": round(n_tracks / mean_cells, 2),   # tracks per avg cell; ~1 ideal, >1 = fragmented
        "median_track_len": int(lengths.median()),
        "mean_track_len": round(float(lengths.mean()), 1),
        "full_window_tracks": full,
        "id_persistence": round(full / n_tracks, 3),              # frac of tracks spanning the whole window
        "tracks_with_gaps": int((gaps > 0).sum()),
        "singleton_tracks": int((lengths == 1).sum()),            # 1-frame tracks (likely spurious / over-seg)
    }


def division_candidates(traj: pd.DataFrame, window_end: int, search_range_px: float = 30.0, k: int = 2) -> dict:
    """Heuristic split detection: a track ends mid-window and >=2 new tracks
    appear within ``search_range_px`` and ``k`` frames. Inspection only — true
    division recall needs a tracked DAPI GT lineage (deeper Stage 03)."""
    s = traj.sort_values("timepoint")
    last = s.groupby("track_id").tail(1)
    first = s.groupby("track_id").head(1)
    ends = last[last["timepoint"] < window_end]
    n_cand = 0
    for e in ends.itertuples():
        starts = first[(first["timepoint"] > e.timepoint) & (first["timepoint"] <= e.timepoint + k)]
        if len(starts):
            d = np.hypot(starts["centroid_x"] - e.centroid_x, starts["centroid_y"] - e.centroid_y)
            if int((d <= search_range_px).sum()) >= 2:
                n_cand += 1
    return {"division_candidates": int(n_cand), "track_terminations_midwindow": int(len(ends))}
