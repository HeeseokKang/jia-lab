"""Stage 05 — longitudinal substrate-benchmark harness (skeleton).

Substrate-AGNOSTIC: give it any tracking ``trajectories.parquet`` that follows
the lab longitudinal schema (region, fov, timepoint, cell_in_frame_id, track_id,
lineage_id, parent_track_id, generation, mitosis_event_frame, centroid_x/y, area)
and a ``--label`` (e.g. ``h2b_R0fov0``, ``bf_R0fov0``, ``phaseTian_R0fov0``).

It computes the *intrinsic* (anchor-free) half of the benchmark — track-duration /
survival statistics, window-length dependence, density-over-time, and a review
queue of suspicious divisions + ID-switch candidates linked back to frame ranges.
The *anchor-relative* half (score vs a nuclear lineage GT: id-swap, division
recall, per-cytokinesis split/merge/miss) stays in score_gt.py and is merged in a
later step; this harness deliberately works WITHOUT a GT so it runs on every FOV
and on tomorrow's QPM substrates the same way.

Outputs (under --out, default <data-root>/analysis/qc/stage05/<label>/):
  per_track_summary.parquet     one row per track_id (duration, gaps, drift, ...)
  survival_overall.csv          totals + fraction surviving >=1/2/4/8/24h
  survival_by_window.csv        same stats inside tiled 4/8/24/48h windows
  density_over_time.csv         cells per frame vs hour
  review_queue/division_events.csv     all divisions, scored by suspicion
  review_queue/idswitch_candidates.csv large single-frame centroid jumps (gap-links)
  review_queue/napari_worklist.csv     top-N events: frame range + crop bbox + ids
  figures/*.png                 duration hist, survival curve, density, window bars
  meta.json                     params + headline numbers

Run (ultrack_env has pandas+pyarrow+matplotlib):
  /opt/miniconda/envs/ultrack_env/bin/python scripts/stage05_benchmark.py \
      --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse \
      --traj   <...>/analysis/tracking/R0_fov0/trajectories.parquet \
      --label  h2b_R0fov0 --interval-min 5 --px-um 1.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_HOURS = [1, 2, 4, 8, 24]
WINDOW_HOURS = [4, 8, 24, 48]


def _f2h(frames: np.ndarray | float, interval_min: float) -> np.ndarray | float:
    return np.asarray(frames) * interval_min / 60.0


def per_track_summary(df: pd.DataFrame, interval_min: float) -> pd.DataFrame:
    """One row per track_id over the whole movie."""
    rows = []
    for tid, g in df.sort_values("timepoint").groupby("track_id"):
        tp = g["timepoint"].to_numpy()
        cx = g["centroid_x"].to_numpy()
        cy = g["centroid_y"].to_numpy()
        span = int(tp.max() - tp.min())
        n_obs = len(tp)
        jumps = np.hypot(np.diff(cx), np.diff(cy)) if n_obs > 1 else np.array([0.0])
        rows.append(
            {
                "track_id": int(tid),
                "lineage_id": int(g["lineage_id"].iloc[0]),
                "generation": int(g["generation"].iloc[0]),
                "has_parent": bool(pd.notna(g["parent_track_id"].iloc[0])),
                "first_frame": int(tp.min()),
                "last_frame": int(tp.max()),
                "n_obs": n_obs,
                "span_frames": span,
                "duration_h": round(_f2h(span + 1, interval_min), 4),  # inclusive
                "n_gaps": int(span + 1 - n_obs),
                "mean_area": float(g["area"].mean()),
                "max_jump_px": float(jumps.max()),
                "mean_jump_px": float(jumps.mean()),
            }
        )
    return pd.DataFrame(rows)


def survival_stats(per_track: pd.DataFrame, interval_min: float, n_frames: int) -> dict:
    dur = per_track["duration_h"].to_numpy()
    movie_h = _f2h(n_frames, interval_min)
    out = {
        "n_tracks": int(len(per_track)),
        "n_lineages": int(per_track["lineage_id"].nunique()),
        "fragmentation_index": round(len(per_track) / max(per_track["lineage_id"].nunique(), 1), 3),
        "movie_duration_h": round(float(movie_h), 3),
        "duration_h_median": round(float(np.median(dur)), 3),
        "duration_h_p25": round(float(np.percentile(dur, 25)), 3),
        "duration_h_p75": round(float(np.percentile(dur, 75)), 3),
        "duration_h_max": round(float(dur.max()), 3),
    }
    for h in SURVIVAL_HOURS:
        out[f"frac_surviving_{h}h"] = round(float((dur >= h).mean()), 4)
    out["frac_full_movie"] = round(float((dur >= movie_h - 1e-6).mean()), 4)
    return out


def survival_by_window(df: pd.DataFrame, interval_min: float, n_frames: int) -> pd.DataFrame:
    """Tile the movie into non-overlapping windows of each length; recompute
    track stats CLIPPED to the window. Shows duration/density dependence on
    window position (early/low-density vs late/high-density)."""
    rows = []
    for wh in WINDOW_HOURS:
        wlen = int(round(wh * 60 / interval_min))  # frames per window
        if wlen >= n_frames:  # whole movie is one window
            starts = [0]
            wlen = n_frames
        else:
            starts = list(range(0, n_frames - wlen + 1, wlen))
        for wi, t0 in enumerate(starts):
            t1 = min(t0 + wlen - 1, n_frames - 1)
            w = df[(df["timepoint"] >= t0) & (df["timepoint"] <= t1)]
            if w.empty:
                continue
            pt = per_track_summary(w, interval_min)
            dur = pt["duration_h"].to_numpy()
            win_h = _f2h(t1 - t0 + 1, interval_min)
            divs = w[w["parent_track_id"].notna()].groupby("track_id").head(1)
            n_div = int(divs.groupby(["parent_track_id", "timepoint"]).ngroups) if not divs.empty else 0
            rows.append(
                {
                    "window_h": wh,
                    "window_idx": wi,
                    "t0": t0,
                    "t1": t1,
                    "n_tracks": int(len(pt)),
                    "n_lineages": int(pt["lineage_id"].nunique()),
                    "fragmentation_index": round(len(pt) / max(pt["lineage_id"].nunique(), 1), 3),
                    "mean_density": round(float(w.groupby("timepoint").size().mean()), 1),
                    "duration_h_median": round(float(np.median(dur)), 3),
                    "frac_full_window": round(float((dur >= win_h - 1e-6).mean()), 4),
                    "n_divisions": n_div,
                }
            )
    return pd.DataFrame(rows)


def division_events(df: pd.DataFrame, per_track: pd.DataFrame, interval_min: float) -> pd.DataFrame:
    """One row per division (mother -> daughters). Suspicion score so the worst
    rows float to the top of the review queue."""
    dur_by_tid = dict(zip(per_track["track_id"], per_track["duration_h"]))
    daughters = df[df["parent_track_id"].notna()].sort_values("timepoint")
    births = daughters.groupby("track_id").head(1)
    rows = []
    for (parent, bframe), g in births.groupby(["parent_track_id", "timepoint"]):
        dids = g["track_id"].astype(int).tolist()
        d_durs = [dur_by_tid.get(d, 0.0) for d in dids]
        cx = float(g["centroid_x"].mean())
        cy = float(g["centroid_y"].mean())
        min_d = min(d_durs) if d_durs else 0.0
        # suspicion: lone daughter (no sibling), or a very short-lived daughter
        suspicion = 0.0
        if len(dids) < 2:
            suspicion += 2.0
        suspicion += max(0.0, 1.0 - min_d / 1.0)  # daughter shorter than 1h -> up to +1
        rows.append(
            {
                "parent_track_id": int(parent),
                "birth_frame": int(bframe),
                "birth_hour": round(_f2h(bframe, interval_min), 3),
                "n_daughters": len(dids),
                "daughter_track_ids": ";".join(map(str, dids)),
                "min_daughter_dur_h": round(min_d, 3),
                "centroid_x": round(cx, 1),
                "centroid_y": round(cy, 1),
                "suspicion": round(suspicion, 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["suspicion", "birth_frame"], ascending=[False, True])


def idswitch_candidates(per_track: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    """Within-track single-frame jumps larger than the tracker's max_distance can
    only arise via gap-closing links -> candidate questionable identity. Proxy
    for ID switches when no GT is available."""
    c = per_track[per_track["max_jump_px"] > max_distance].copy()
    return c.sort_values("max_jump_px", ascending=False)[
        ["track_id", "lineage_id", "first_frame", "last_frame", "duration_h", "max_jump_px"]
    ]


def napari_worklist(div: pd.DataFrame, top_n: int, n_frames: int, pad: int = 5, crop: int = 120) -> pd.DataFrame:
    rows = []
    for _, r in div.head(top_n).iterrows():
        bf = int(r["birth_frame"])
        cx, cy = int(r["centroid_x"]), int(r["centroid_y"])
        rows.append(
            {
                "event_kind": "division",
                "event_id": f"div_p{int(r['parent_track_id'])}_t{bf}",
                "parent_track_id": int(r["parent_track_id"]),
                "daughter_track_ids": r["daughter_track_ids"],
                "frame_start": max(0, bf - pad),
                "frame_end": min(n_frames - 1, bf + pad),
                "center_x": cx,
                "center_y": cy,
                "crop_x0": max(0, cx - crop),
                "crop_y0": max(0, cy - crop),
                "crop_x1": cx + crop,
                "crop_y1": cy + crop,
                "suspicion": r["suspicion"],
                "verdict": "",  # human fills: real_division / merge / miss / artifact
            }
        )
    return pd.DataFrame(rows)


def make_figures(per_track, surv, by_window, density, fig_dir, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[stage05] matplotlib unavailable, skipping figures: {e}", flush=True)
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    dur = per_track["duration_h"].to_numpy()

    # 1. duration histogram
    plt.figure(figsize=(6, 4))
    plt.hist(dur, bins=40, color="#4477aa")
    plt.xlabel("track duration (h)"); plt.ylabel("# tracks"); plt.title(f"Track duration — {label}")
    plt.tight_layout(); plt.savefig(fig_dir / "duration_hist.png", dpi=120); plt.close()

    # 2. survival curve (fraction surviving >= h)
    hs = np.linspace(0, dur.max(), 100)
    frac = [(dur >= h).mean() for h in hs]
    plt.figure(figsize=(6, 4))
    plt.plot(hs, frac, color="#aa3377")
    for h in SURVIVAL_HOURS:
        if h <= dur.max():
            plt.axvline(h, ls=":", c="grey", lw=0.8)
    plt.xlabel("hours"); plt.ylabel("fraction of tracks surviving >= h"); plt.ylim(0, 1)
    plt.title(f"Track survival — {label}"); plt.tight_layout()
    plt.savefig(fig_dir / "survival_curve.png", dpi=120); plt.close()

    # 3. density over time
    plt.figure(figsize=(6, 4))
    plt.plot(density["hour"], density["n_cells"], color="#228833")
    plt.xlabel("hour"); plt.ylabel("cells per frame"); plt.title(f"Detection density — {label}")
    plt.tight_layout(); plt.savefig(fig_dir / "density_over_time.png", dpi=120); plt.close()

    # 4. duration-vs-window (median + frac_full) per window length
    if not by_window.empty:
        plt.figure(figsize=(6, 4))
        for wh, sub in by_window.groupby("window_h"):
            plt.scatter(sub["mean_density"], sub["frac_full_window"], label=f"{wh}h", s=30)
        plt.xlabel("mean density (cells/frame)"); plt.ylabel("frac tracks spanning full window")
        plt.legend(title="window"); plt.title(f"Survival vs density/window — {label}")
        plt.tight_layout(); plt.savefig(fig_dir / "survival_vs_window.png", dpi=120); plt.close()
    print(f"[stage05] figures -> {fig_dir}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--traj", required=True, type=Path)
    ap.add_argument("--label", required=True, help="substrate label e.g. h2b_R0fov0")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--interval-min", type=float, default=5.0)
    ap.add_argument("--px-um", type=float, default=1.3)
    ap.add_argument("--max-distance", type=float, default=25.0, help="tracker max_distance (px) for jump proxy")
    ap.add_argument("--top-n-review", type=int, default=60)
    args = ap.parse_args()

    out = args.out or (args.data_root / "analysis" / "qc" / "stage05" / args.label)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_queue").mkdir(exist_ok=True)

    df = pd.read_parquet(args.traj)
    n_frames = int(df["timepoint"].max()) + 1
    print(f"[stage05] {args.label}: {len(df)} rows, {df['track_id'].nunique()} tracks, "
          f"{n_frames} frames", flush=True)

    per_track = per_track_summary(df, args.interval_min)
    per_track.to_parquet(out / "per_track_summary.parquet", index=False)

    surv = survival_stats(per_track, args.interval_min, n_frames)
    pd.DataFrame([surv]).to_csv(out / "survival_overall.csv", index=False)

    by_window = survival_by_window(df, args.interval_min, n_frames)
    by_window.to_csv(out / "survival_by_window.csv", index=False)

    density = (
        df.groupby("timepoint").size().rename("n_cells").reset_index()
        .assign(hour=lambda d: _f2h(d["timepoint"].to_numpy(), args.interval_min))
    )
    density.to_csv(out / "density_over_time.csv", index=False)

    div = division_events(df, per_track, args.interval_min)
    div.to_csv(out / "review_queue" / "division_events.csv", index=False)

    idsw = idswitch_candidates(per_track, args.max_distance)
    idsw.to_csv(out / "review_queue" / "idswitch_candidates.csv", index=False)

    wl = napari_worklist(div, args.top_n_review, n_frames)
    wl.to_csv(out / "review_queue" / "napari_worklist.csv", index=False)

    make_figures(per_track, surv, by_window, density, out / "figures", args.label)

    meta = {
        "STAGE05_VERSION": "v0-skeleton-2026-06-08",
        "label": args.label,
        "traj": str(args.traj),
        "params": {"interval_min": args.interval_min, "px_um": args.px_um,
                   "max_distance": args.max_distance, "survival_hours": SURVIVAL_HOURS,
                   "window_hours": WINDOW_HOURS},
        "headline": surv,
        "n_divisions": int(len(div)),
        "n_suspicious_divisions": int((div["suspicion"] > 1.0).sum()),
        "n_idswitch_candidates": int(len(idsw)),
        "outputs": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"headline": surv, "n_divisions": len(div),
                      "n_suspicious": int((div['suspicion'] > 1.0).sum()),
                      "n_idswitch_candidates": len(idsw)}, indent=2), flush=True)
    print(f"[stage05] outputs -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
