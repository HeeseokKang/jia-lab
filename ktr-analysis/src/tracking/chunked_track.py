"""Stage 03 #2 — temporal chunking + seam stitch for Ultrack scaling.

Full-FOV Ultrack is infeasible (R0/fov0 = 3.5h / 81 GB). Cost is set by per-solve
node count (= cells x frames); the only lever that cuts it is fewer frames per
solve. This module solves overlapping temporal windows independently (each
pilot-scale, ~13 GB) and stitches them into one global-id trajectory set.

KEY ENABLER: every window consumes the IDENTICAL StarDist mask stack, so a cell
at a frame is exactly `(timepoint, cell_in_frame_id)` (cell_in_frame_id == the
StarDist mask_label). Cross-window identity is therefore matched by EXACT shared
`(timepoint, cell_in_frame_id)` in the overlap (mutual-best by shared count) — not
fuzzy spatial matching. Motion is tiny (p99 = 7.1 px/frame), so this is unambiguous.

Fixed params (locked 2026-05-22): B_det~=45k/chunk, overlap O=15, max_distance=12.
Window sizing is adaptive in principle (W = B_det / cells(t)); the CLI also accepts
explicit `--windows` (used by the seam-validation test).

CLI (seam-validation example, runs per-window Ultrack then stitches):
    /opt/miniconda/envs/ultrack_env/bin/python -m src.tracking.chunked_track \
        --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse --region R0 --fov 0 \
        --windows 0-60,45-100 --max-distance 12 --masks-stage validation \
        --out-root <DATA>/analysis/tracking/_bench_chunk/seamtest \
        --monolith <DATA>/analysis/tracking/_bench_maxdist/md12_t0-100/R0_fov0/trajectories.parquet

NO full chunked production run is launched by this module.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TRACK_SCENE = REPO / "scripts" / "track_ultrack_scene.py"
SCHEMA = [
    "region", "fov", "timepoint", "cell_in_frame_id", "track_id", "lineage_id",
    "parent_track_id", "generation", "mitosis_event_frame",
    "centroid_x", "centroid_y", "area", "nuclear_mask_path",
]


# ---------------------------------------------------------------- planning
def plan_windows(cells_per_frame: dict[int, float], b_det: int = 45000,
                 overlap: int = 15, w_cap: int = 240) -> list[tuple[int, int]]:
    """Adaptive temporal windows by detection budget. cells_per_frame maps t->count.
    Each window grows until cumulative detections ~= b_det (cap W at w_cap); the
    next window starts `overlap` frames before the previous end."""
    tps = sorted(cells_per_frame)
    t0_all, t1_all = tps[0], tps[-1]
    windows, start = [], t0_all
    while start <= t1_all:
        cum, end = 0.0, start
        while end <= t1_all and (end - start) < w_cap:
            cum += cells_per_frame.get(end, 0.0)
            if cum >= b_det:
                break
            end += 1
        end = min(end, t1_all)
        windows.append((start, end))
        if end >= t1_all:
            break
        start = max(start + 1, end - overlap + 1)
    return windows


# ---------------------------------------------------------------- stitching
def _overlap_match(df_prev: pd.DataFrame, df_cur: pd.DataFrame, lo: int, hi: int) -> dict[int, int]:
    """Mutual-best match cur.track_id -> prev.track_id over shared (timepoint,
    cell_in_frame_id) in [lo,hi]. Robust to a single in-overlap swap."""
    def win(df):
        d = df[(df.timepoint >= lo) & (df.timepoint <= hi) & (df.cell_in_frame_id > 0)]
        return d[["timepoint", "cell_in_frame_id", "track_id"]]
    m = win(df_prev).merge(win(df_cur), on=["timepoint", "cell_in_frame_id"], suffixes=("_p", "_c"))
    if m.empty:
        return {}
    cnt = m.groupby(["track_id_p", "track_id_c"]).size().reset_index(name="n")
    best_c_for_p = cnt.loc[cnt.groupby("track_id_p")["n"].idxmax()]      # p -> c
    best_p_for_c = cnt.loc[cnt.groupby("track_id_c")["n"].idxmax()]      # c -> p
    p_to_c = dict(zip(best_c_for_p.track_id_p, best_c_for_p.track_id_c))
    c_to_p = dict(zip(best_p_for_c.track_id_c, best_p_for_c.track_id_p))
    return {int(c): int(p) for c, p in c_to_p.items() if p_to_c.get(p) == c}  # mutual only


def _root_gen(tid: int, parent_of: dict[int, int]) -> tuple[int, int]:
    g, seen, cur = 0, set(), tid
    while cur in parent_of and cur not in seen:
        seen.add(cur); cur = parent_of[cur]; g += 1
    return cur, g


def stitch(dfs: list[pd.DataFrame], parents: list[dict[int, int]],
           windows: list[tuple[int, int]]) -> tuple[pd.DataFrame, dict[int, int], dict]:
    """Fold N overlapping windows into one global-id trajectory set + lineage.
    Returns (merged_df, global_parent_of, diagnostics)."""
    n = len(dfs)
    gid: dict[tuple[int, int], int] = {}
    nextg = 1
    for tid in dfs[0].track_id.unique():
        gid[(0, int(tid))] = nextg; nextg += 1
    seam_match_stats = []
    for i in range(1, n):
        lo = max(windows[i][0], windows[i - 1][0])
        hi = min(windows[i][1], windows[i - 1][1])
        mutual = _overlap_match(dfs[i - 1], dfs[i], lo, hi)
        matched = 0
        for tid in dfs[i].track_id.unique():
            tid = int(tid)
            if tid in mutual:
                gid[(i, tid)] = gid[(i - 1, int(mutual[tid]))]; matched += 1
            else:
                gid[(i, tid)] = nextg; nextg += 1
        n_cur = dfs[i].track_id.nunique()
        seam_match_stats.append({"seam": [windows[i - 1][1], windows[i][0]], "overlap": [lo, hi],
                                 "cur_tracks": int(n_cur), "matched": int(matched)})

    # authoritative split = midpoint of each consecutive overlap
    splits = [(max(windows[i + 1][0], windows[i][0]) + min(windows[i + 1][1], windows[i][1])) // 2
              for i in range(n - 1)]
    owned, rows = [], []
    for i in range(n):
        lo_own = -10**9 if i == 0 else splits[i - 1] + 1
        hi_own = 10**9 if i == n - 1 else splits[i]
        sub = dfs[i][(dfs[i].timepoint >= lo_own) & (dfs[i].timepoint <= hi_own)].copy()
        sub["track_id"] = sub["track_id"].map(lambda t: gid[(i, int(t))])
        rows.append(sub)
        owned.append((lo_own, hi_own))
    merged = pd.concat(rows, ignore_index=True).sort_values(["track_id", "timepoint"])

    # global parent_of, preferring the window that owns the daughter's birth frame
    born = merged.groupby("track_id").timepoint.min().to_dict()
    cand: dict[int, list[tuple[int, int]]] = {}
    for i in range(n):
        for c_loc, p_loc in parents[i].items():
            gc, gp = gid.get((i, int(c_loc))), gid.get((i, int(p_loc)))
            if gc is not None and gp is not None:
                cand.setdefault(gc, []).append((i, gp))
    gparent: dict[int, int] = {}
    for gc, opts in cand.items():
        bf = born.get(gc, None)
        owner = None
        if bf is not None:
            for i, (lo_own, hi_own) in enumerate(owned):
                if lo_own <= bf <= hi_own:
                    owner = i; break
        pick = next((gp for (i, gp) in opts if i == owner), opts[0][1])
        gparent[int(gc)] = int(pick)

    # recompute lineage globally
    merged["parent_track_id"] = merged.track_id.map(lambda t: gparent.get(int(t), pd.NA))
    merged["lineage_id"] = merged.track_id.map(lambda t: _root_gen(int(t), gparent)[0])
    merged["generation"] = merged.track_id.map(lambda t: _root_gen(int(t), gparent)[1])
    merged["mitosis_event_frame"] = merged.track_id.map(
        lambda t: born[int(t)] if int(t) in gparent else pd.NA)
    diag = {"splits": splits, "owned_ranges": owned, "seam_match": seam_match_stats,
            "n_global_tracks": int(merged.track_id.nunique()),
            "n_divisions": int(sum(1 for _, c in pd.Series(list(gparent.values())).value_counts().items() if c >= 2))}
    return merged[SCHEMA], gparent, diag


def seam_continuity(stitched: pd.DataFrame, monolith: pd.DataFrame, split: int) -> dict:
    """Fraction of cells the MONOLITH tracks continuously across split->split+1 that
    the STITCH also keeps as one global id (matched by exact mask_label per frame)."""
    mlo = monolith[monolith.timepoint == split][["cell_in_frame_id", "track_id"]]
    mhi = monolith[monolith.timepoint == split + 1][["cell_in_frame_id", "track_id"]]
    cont = mlo.merge(mhi, on="track_id", suffixes=("_lo", "_hi"))
    slo = stitched[stitched.timepoint == split].set_index("cell_in_frame_id").track_id.to_dict()
    shi = stitched[stitched.timepoint == split + 1].set_index("cell_in_frame_id").track_id.to_dict()
    same = tot = 0
    for _, r in cont.iterrows():
        g1, g2 = slo.get(r.cell_in_frame_id_lo), shi.get(r.cell_in_frame_id_hi)
        if g1 is not None and g2 is not None:
            tot += 1; same += int(g1 == g2)
    return {"split": split, "monolith_cross_seam_tracks": int(tot),
            "kept_continuous": int(same),
            "seam_continuity_rate": round(same / tot, 4) if tot else None}


# ---------------------------------------------------------------- CLI
def _run_window(data_root, region, fov, t0, t1, md, masks_stage, n_workers, out_root):
    out = out_root / f"win_{t0}-{t1}"
    cmd = [sys.executable, str(TRACK_SCENE), "--data-root", str(data_root),
           "--region", region, "--fov", str(fov), "--masks-stage", masks_stage,
           "--t0", str(t0), "--t1", str(t1), "--max-distance", str(md),
           "--n-workers", str(n_workers), "--out-root", str(out)]
    print(f"[chunked] start window t{t0}-{t1}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"window t{t0}-{t1} failed rc={r.returncode}\n{r.stderr[-2000:]}")
    base = out / f"{region}_fov{fov}"
    df = pd.read_parquet(base / "trajectories.parquet")
    pj = json.loads((base / "lineage.graph.json").read_text())["parent_of"]
    meta = json.loads((base / "meta.json").read_text())
    print(f"[chunked] done window t{t0}-{t1}: {meta['runtime_sec']:.0f}s "
          f"peakRSS={meta['peak_rss_mb']:.0f}MB det={meta['n_detections']}", flush=True)
    return df, {int(k): int(v) for k, v in pj.items()}, meta


def _cells_per_frame_from_parquet(path: Path) -> dict[int, float]:
    d = pd.read_parquet(path, columns=["timepoint", "cell_in_frame_id"])
    return d.groupby("timepoint").size().astype(float).to_dict()


def main() -> int:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--region", default="R0")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--windows", required=True, help="'auto' (adaptive) or comma list e.g. 0-60,45-100")
    ap.add_argument("--b-det", type=int, default=45000, help="adaptive detection budget per chunk")
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--w-cap", type=int, default=240)
    ap.add_argument("--density-parquet", type=Path, default=None,
                    help="parquet to derive cells/frame for adaptive planning (e.g. S1 monolith)")
    ap.add_argument("--max-distance", type=float, default=12.0)
    ap.add_argument("--masks-stage", choices=["auto", "full", "validation"], default="validation")
    ap.add_argument("--n-workers", type=int, default=1, help="ultrack-internal workers PER chunk")
    ap.add_argument("--workers", type=int, default=6, help="concurrent chunk processes")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--monolith", type=Path, default=None, help="parquet for seam-continuity + consistency")
    args = ap.parse_args()

    if args.windows.strip() == "auto":
        dens_src = args.density_parquet or args.monolith
        if not (dens_src and Path(dens_src).exists()):
            raise SystemExit("--windows auto needs --density-parquet (or --monolith) to derive cells/frame")
        cells = _cells_per_frame_from_parquet(dens_src)
        windows = plan_windows(cells, b_det=args.b_det, overlap=args.overlap, w_cap=args.w_cap)
    else:
        windows = [tuple(int(x) for x in w.split("-")) for w in args.windows.split(",")]
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"[chunked] {len(windows)} windows (b_det={args.b_det} O={args.overlap} "
          f"workers={args.workers} md={args.max_distance}): {windows}", flush=True)

    # parallel per-chunk solve (each chunk = 1 subprocess; pool size = --workers)
    t_wall = time.time()
    results: dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_window, args.data_root, args.region, args.fov, t0, t1,
                          args.max_distance, args.masks_stage, args.n_workers, args.out_root): i
                for i, (t0, t1) in enumerate(windows)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()   # raises (and aborts) if any chunk failed
    wall = time.time() - t_wall
    dfs = [results[i][0] for i in range(len(windows))]
    parents = [results[i][1] for i in range(len(windows))]
    metas = [results[i][2] for i in range(len(windows))]

    merged, gparent, diag = stitch(dfs, parents, windows)
    sdir = args.out_root / "stitched"
    sdir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(sdir / "trajectories.parquet", engine="pyarrow", index=False)

    # per-chunk + scaling telemetry
    pw_rt = [m["runtime_sec"] for m in metas]
    pw_rss = [m["peak_rss_mb"] for m in metas]
    tlen = merged.groupby("track_id").size()
    diag["scaling"] = {
        "n_chunks": len(windows), "concurrent_workers": args.workers,
        "wall_sec": round(wall, 1), "wall_min": round(wall / 60, 1),
        "per_chunk_runtime_sec": {"max": round(max(pw_rt), 1), "sum": round(sum(pw_rt), 1),
                                  "mean": round(sum(pw_rt) / len(pw_rt), 1)},
        "per_chunk_peak_rss_mb": {"max": round(max(pw_rss), 1), "mean": round(sum(pw_rss) / len(pw_rss), 1)},
        "n_detections_stitched": int(len(merged)),
        "short_tracks_le3": int((tlen <= 3).sum()),
    }
    if args.monolith and Path(args.monolith).exists():
        mono = pd.read_parquet(args.monolith)
        diag["seam_continuity"] = [seam_continuity(merged, mono, s) for s in diag["splits"]]
        mono_div = int(sum(1 for _, c in pd.to_numeric(mono.get("parent_track_id"), errors="coerce")
                           .dropna().value_counts().items() if c >= 2))
        mtlen = mono.groupby("track_id").size()
        diag["vs_monolith"] = {
            "monolith_parquet": str(args.monolith),
            "n_tracks": {"stitched": diag["n_global_tracks"], "monolith": int(mono.track_id.nunique())},
            "n_detections": {"stitched": int(len(merged)), "monolith": int(len(mono))},
            "divisions": {"stitched": diag["n_divisions"], "monolith": mono_div},
            "short_tracks_le3": {"stitched": int((tlen <= 3).sum()), "monolith": int((mtlen <= 3).sum())},
            "seam_continuity_min": min((s["seam_continuity_rate"] for s in diag["seam_continuity"]
                                        if s["seam_continuity_rate"] is not None), default=None),
        }
    (sdir / "stitch_meta.json").write_text(json.dumps(
        {"windows": windows, "max_distance": args.max_distance, **diag}, indent=2, default=str))
    print(json.dumps({k: diag[k] for k in ("scaling", "vs_monolith", "seam_continuity") if k in diag},
                     indent=2, default=str), flush=True)
    print(f"[chunked] stitched -> {sdir/'trajectories.parquet'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
