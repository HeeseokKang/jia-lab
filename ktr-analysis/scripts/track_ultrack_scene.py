"""Stage 03 production per-scene Ultrack runner (generalized from the Stage-03 Ultrack pilot).

Runs the validated production tracker (Ultrack 0.7.1, mitosis-aware, CBC solver)
on the fresh-StarDist H2B masks of ONE scene (region, fov), emits the locked
13-col longitudinal schema as parquet, the lineage graph, and a meta.json with
runtime / peak-RSS / density telemetry.

Mask-source policy (resolved by data 2026-05-21): tracking division recall is
mask-source dependent (fresh StarDist recall 0.80 vs Bill nuc_labels 0.30), so
this runner consumes ONLY fresh StarDist masks:
    full:       <analysis>/segmentation/full/{region}_fov{fov}/stardist_h2b/t####.npy
    validation: <analysis>/segmentation/validation/{region}_fov{fov}/stardist_h2b/  (R0/fov0 only)

MUST run in the ultrack env:
    /opt/miniconda/envs/ultrack_env/bin/python scripts/track_ultrack_scene.py \
        --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse --region R0 --fov 0

Restartable: skips a scene whose output dir already has a _SUCCESS marker
(unless --force). Failure-isolated: raises on this scene only; the driver decides
whether to continue the sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

GT_CELL = "stardist_h2b"
TRACKING_POLICY_VERSION = "v-ultrack-2026-05-21-prod"

SCHEMA_COLUMNS = [
    "region", "fov", "timepoint", "cell_in_frame_id", "track_id", "lineage_id",
    "parent_track_id", "generation", "mitosis_event_frame",
    "centroid_x", "centroid_y", "area", "nuclear_mask_path",
]


def _to_parent(p):
    """Ultrack/napari graph values may be scalar or a [parent] list."""
    if isinstance(p, (list, tuple, np.ndarray)):
        return int(p[0]) if len(p) else None
    return int(p)


def _resolve_masks_dir(analysis: Path, region: str, fov: int, stage: str, cell: str = GT_CELL) -> Path:
    full = analysis / "segmentation" / "full" / f"{region}_fov{fov}" / cell
    val = analysis / "segmentation" / "validation" / f"{region}_fov{fov}" / cell
    if stage == "full":
        return full
    if stage == "validation":
        return val
    # auto: prefer full (production), fall back to validation (R0/fov0)
    if full.is_dir() and any(full.glob("t*.npy")):
        return full
    return val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--region", default="R0")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=-1, help="-1 = last available frame")
    ap.add_argument("--masks-stage", choices=["auto", "full", "validation"], default="auto")
    ap.add_argument("--cell", default=GT_CELL,
                    help="mask subdir under the seg tree: stardist_h2b (default) or cellpose_bf, etc.")
    ap.add_argument("--max-distance", type=float, default=25.0)
    ap.add_argument("--min-area", type=int, default=10)
    ap.add_argument("--max-area", type=int, default=100000)
    ap.add_argument("--n-workers", type=int, default=0,
                    help="Ultrack internal workers; 0=leave default (match pilot). "
                         "Set to 1 when running N scenes concurrently to avoid core oversubscription")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default <analysis>/tracking")
    ap.add_argument("--working-dir", type=Path, default=None,
                    help="SQLite candidate-graph scratch; default /tmp/ultrack_work/<scene>_<pid> "
                         "(root fs nvme1n1, separate device from /data masks — see cost-reduction #3)")
    ap.add_argument("--keep-working-dir", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if _SUCCESS exists")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    analysis = data_root / "analysis"
    region, fov = args.region, args.fov
    masks_dir = _resolve_masks_dir(analysis, region, fov, args.masks_stage, args.cell)
    if not (masks_dir.is_dir() and any(masks_dir.glob("t*.npy"))):
        raise FileNotFoundError(f"no {args.cell} masks for {region}/fov{fov} at {masks_dir}")

    avail = sorted(int(p.stem[1:]) for p in masks_dir.glob("t*.npy"))
    t1 = avail[-1] if args.t1 < 0 else args.t1
    tps = [t for t in range(args.t0, t1 + 1) if t in set(avail)]
    n_frames = len(tps)

    out_root = (args.out_root or (analysis / "tracking")).resolve()
    out_dir = out_root / f"{region}_fov{fov}"
    out_dir.mkdir(parents=True, exist_ok=True)
    success_marker = out_dir / "_SUCCESS"
    if success_marker.exists() and not args.force:
        print(f"[ultrack-scene] SKIP {region}/fov{fov}: _SUCCESS present ({success_marker})")
        return 0

    print(f"[ultrack-scene] {region}/fov{fov} frames {args.t0}-{t1} ({n_frames}) masks={masks_dir}")
    paths = [str(masks_dir / f"t{tp:04d}.npy") for tp in tps]
    masks = [np.load(p) for p in paths]
    stack = np.stack(masks).astype(np.int32)
    areas_per_frame = [np.bincount(m.ravel()) for m in masks]
    cells_per_frame = [int(np.count_nonzero(np.unique(m))) for m in masks]
    mean_cells = float(np.mean(cells_per_frame))

    from ultrack import MainConfig, Tracker

    cfg = MainConfig()
    if args.working_dir is not None:
        wd = Path(args.working_dir)
    else:
        # Cost-reduction #3 (locked 2026-05-21): default the SQLite candidate-graph
        # scratch to /tmp (root fs = nvme1n1), a DIFFERENT physical device than /data
        # (nvme0n1) where masks are read. This stops DB writes from contending with
        # mask reads on one device's I/O queue (S1 had both on /data → 11.9M voluntary
        # ctx-switches, ~68% CPU util). No accuracy impact. Pair with off-hours/idle
        # runs to also remove the ~2.77x CPU-contention factor measured vs an idle box.
        wd = Path(f"/tmp/ultrack_work/{region}_fov{fov}_{os.getpid()}")
    wd.mkdir(parents=True, exist_ok=True)
    wd = str(wd)
    for attr in ("working_dir", "working_directory"):
        if hasattr(cfg.data_config, attr):
            setattr(cfg.data_config, attr, wd)
    if args.n_workers > 0 and hasattr(cfg.data_config, "n_workers"):
        cfg.data_config.n_workers = args.n_workers
    cfg.segmentation_config.min_area = args.min_area
    cfg.segmentation_config.max_area = args.max_area
    cfg.linking_config.max_distance = args.max_distance

    t_start = time.time()
    tracker = Tracker(cfg)
    tracker.track(labels=stack, overwrite="all")
    tracks_df, graph = tracker.to_tracks_layer()
    runtime = time.time() - t_start

    tdf = tracks_df.reset_index(drop=True)
    parent_of: dict[int, int] = {}
    if isinstance(graph, dict):
        for c, p in graph.items():
            pp = _to_parent(p)
            if pp is not None:
                parent_of[int(c)] = pp

    def root_and_gen(tid: int):
        g, seen, cur = 0, set(), tid
        while cur in parent_of and cur not in seen:
            seen.add(cur)
            cur = parent_of[cur]
            g += 1
        return cur, g

    born = tdf.groupby("track_id")["t"].min().to_dict()

    rows = []
    for r in tdf.itertuples():
        tid, t = int(r.track_id), int(r.t)
        y, x = float(r.y), float(r.x)
        lab = int(masks[t][min(int(round(y)), masks[t].shape[0] - 1),
                            min(int(round(x)), masks[t].shape[1] - 1)])
        area = float(areas_per_frame[t][lab]) if 0 < lab < len(areas_per_frame[t]) else float("nan")
        lin, gen = root_and_gen(tid)
        is_daughter = tid in parent_of
        rows.append({
            "region": region, "fov": fov, "timepoint": tps[t],
            "cell_in_frame_id": lab, "track_id": tid, "lineage_id": lin,
            "parent_track_id": parent_of.get(tid, pd.NA),
            "generation": gen,
            "mitosis_event_frame": (tps[int(born[tid])] if is_daughter else pd.NA),
            "centroid_x": x, "centroid_y": y, "area": area,
            "nuclear_mask_path": paths[t],
        })
    traj = pd.DataFrame(rows)[SCHEMA_COLUMNS]
    traj_parquet = out_dir / "trajectories.parquet"
    traj.to_parquet(traj_parquet, engine="pyarrow", index=False)

    n_div = sum(1 for _, c in Counter(parent_of.values()).items() if c >= 2)
    lineage_roots = sorted({root_and_gen(int(t))[0] for t in traj["track_id"].unique()})
    graph_json = {
        "parent_of": {str(c): int(p) for c, p in parent_of.items()},
        "lineage_roots": [int(x) for x in lineage_roots],
        "n_divisions": int(n_div),
        "n_daughter_tracks": int(len(parent_of)),
    }
    (out_dir / "lineage.graph.json").write_text(json.dumps(graph_json, indent=2))

    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    meta = {
        "TRACKING_POLICY_VERSION": TRACKING_POLICY_VERSION,
        "IMPL": "ultrack 0.7.1 (CBC/Coin-OR ILP, mitosis-aware)",
        "stage": "03-production-per-scene",
        "input": {"cell": args.cell, "masks_dir": str(masks_dir), "masks_stage": args.masks_stage,
                  "region": region, "fov": fov, "timepoints": [args.t0, t1], "n_frames": n_frames},
        "params": {"max_distance": args.max_distance, "min_area": args.min_area,
                   "max_area": args.max_area, "n_workers": args.n_workers},
        "density": {"mean_cells_per_frame": round(mean_cells, 1),
                    "min_cells_per_frame": int(np.min(cells_per_frame)),
                    "max_cells_per_frame": int(np.max(cells_per_frame))},
        "n_tracks": int(traj["track_id"].nunique()),
        "n_lineages": int(len(lineage_roots)),
        "n_detections": int(len(traj)),
        "division_events_detected": int(n_div),
        "daughter_tracks": int(len(parent_of)),
        "runtime_sec": round(runtime, 2),
        "sec_per_frame": round(runtime / max(n_frames, 1), 3),
        "peak_rss_mb": peak_rss_mb,
        "working_dir": wd,
        "pid": os.getpid(),
        "outputs": {"trajectories_parquet": str(traj_parquet),
                    "lineage_graph_json": str(out_dir / "lineage.graph.json")},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if not args.keep_working_dir:
        shutil.rmtree(wd, ignore_errors=True)
    success_marker.write_text(json.dumps({
        "region": region, "fov": fov, "n_frames": n_frames,
        "runtime_sec": meta["runtime_sec"], "divisions": n_div,
        "peak_rss_mb": peak_rss_mb, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2))

    print(f"[ultrack-scene] {region}/fov{fov} done: {n_frames}f in {runtime:.1f}s "
          f"({meta['sec_per_frame']}s/f) | tracks={meta['n_tracks']} div={n_div} "
          f"| peakRSS={peak_rss_mb}MB | mean_cells/frame={mean_cells:.0f}")
    print(f"[ultrack-scene] out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
