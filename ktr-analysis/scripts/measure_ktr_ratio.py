"""Stage 04 — per-track ERK-KTR cytoplasm/nucleus (C/N) ratio.

Consumes the Stage 03 trajectories (locked 13-col schema) + the fresh-StarDist
H2B nuclear masks they reference + the raw mKate2 (KTR sensor) channel, and emits
a per-(track, timepoint) measurement table.

Key fact (verified 2026-05-24): in trajectories.parquet, `cell_in_frame_id` IS
the label value in the per-row `nuclear_mask_path` mask, so the nucleus for a
(track, frame) row is exactly `mask == cell_in_frame_id` — no centroid matching.

Cytoplasmic ring = `skimage.segmentation.expand_labels(mask, distance=expansion_px)`
restricted to background pixels (`mask == 0`). expand_labels grows each nucleus
into nearby background only, splitting contested background between neighbours by
distance, so the ring is automatically neighbour-excluded.

DESIGN (knob-deferral): the background-subtraction policy and the exact ratio
definition are NOT baked into this heavy pass. We store raw components
(nuc/ring mean, median, area, sum) + per-frame background percentiles, so the
ratio can be recomputed cheaply afterwards under any policy. A clearly-labelled
exploratory ratio is included for convenience only.

Run (ultrack_env has numpy/scipy/skimage/pandas/pyarrow):
    /opt/miniconda/envs/ultrack_env/bin/python scripts/measure_ktr_ratio.py \
        --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse --region R0 --fov 0

--smoke processes only {0,100,566} into a *_smoke output dir for validation.
Restartable: skips if _SUCCESS present (unless --force). Frame-failure isolated:
a missing/unreadable raw frame records NaN for that frame's rows and continues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

try:
    from skimage.segmentation import expand_labels
except Exception as e:  # pragma: no cover
    print(f"[stage04] FATAL: need scikit-image (expand_labels): {e}", file=sys.stderr)
    raise

# raw reader: prefer tifffile, fall back to skimage.io
try:
    import tifffile
    def _imread(p): return tifffile.imread(str(p))
except Exception:  # pragma: no cover
    from skimage.io import imread as _sk_imread
    def _imread(p): return _sk_imread(str(p))

ACQ_DEFAULT = "timelapse_2026-05-05_18-12-11.466141"
SMOKE_TPS = (0, 100, 566)

# carried straight through from the Stage 03 schema
SCHEMA_PASSTHROUGH = [
    "region", "fov", "timepoint", "cell_in_frame_id", "track_id", "lineage_id",
    "parent_track_id", "generation", "mitosis_event_frame",
    "centroid_x", "centroid_y", "area", "nuclear_mask_path",
]


def raw_channel_path(data_root: Path, acq: str, region: str, fov: int, tp: int,
                     channel: str) -> Path | None:
    regdigit = region[1:] if region.upper().startswith("R") else region
    folder = data_root / acq / str(tp)
    cands = sorted(folder.glob(f"R{regdigit}_{fov}_*_{channel}_KTR.tif*"))
    return cands[0] if cands else None


def measure_frame(mask: np.ndarray, img: np.ndarray, expansion_px: int):
    """Return dict label -> component measurements + frame background percentiles."""
    labels = np.unique(mask)
    labels = labels[labels != 0]
    img = img.astype(np.float64, copy=False)

    bg_pixels = img[mask == 0]
    bg = {
        "bg_p05": float(np.percentile(bg_pixels, 5)) if bg_pixels.size else np.nan,
        "bg_p50": float(np.percentile(bg_pixels, 50)) if bg_pixels.size else np.nan,
        "bg_p97": float(np.percentile(bg_pixels, 97)) if bg_pixels.size else np.nan,
    }
    if labels.size == 0:
        return {}, bg

    ring_img = np.where(mask == 0, expand_labels(mask, distance=expansion_px), 0)

    nuc_mean = ndi.mean(img, mask, labels)
    nuc_med = ndi.median(img, mask, labels)
    nuc_sum = ndi.sum(img, mask, labels)
    nuc_area = ndi.sum(np.ones_like(img), mask, labels)
    ring_mean = ndi.mean(img, ring_img, labels)
    ring_med = ndi.median(img, ring_img, labels)
    ring_sum = ndi.sum(img, ring_img, labels)
    ring_area = ndi.sum(np.ones_like(img), ring_img, labels)

    out = {}
    for i, lab in enumerate(labels):
        ra = float(ring_area[i])
        out[int(lab)] = {
            "nuc_mean": float(nuc_mean[i]), "nuc_median": float(nuc_med[i]),
            "nuc_sum": float(nuc_sum[i]), "nuc_area_px": float(nuc_area[i]),
            "ring_mean": float(ring_mean[i]) if ra else np.nan,
            "ring_median": float(ring_med[i]) if ra else np.nan,
            "ring_sum": float(ring_sum[i]), "ring_area_px": ra,
        }
    return out, bg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--region", default="R0")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--acq", default=ACQ_DEFAULT, help="acquisition subdir under data-root")
    ap.add_argument("--channel", default="mKate2", help="KTR sensor channel token in filenames")
    ap.add_argument("--traj", type=Path, default=None,
                    help="trajectories.parquet; default <analysis>/tracking/{region}_fov{fov}/")
    ap.add_argument("--expansion-px", type=int, default=12,
                    help="cyto-ring dilation (mirrors the prior cyto-ring run's max_expansion_px=12)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default <analysis>/ratio/{region}_fov{fov}")
    ap.add_argument("--smoke", action="store_true", help=f"only timepoints {SMOKE_TPS}, *_smoke out dir")
    ap.add_argument("--force", action="store_true", help="re-run even if _SUCCESS present")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    analysis = data_root / "analysis"
    scene = f"{args.region}_fov{args.fov}"
    traj_path = args.traj or (analysis / "tracking" / scene / "trajectories.parquet")
    out_dir = args.out_dir or (analysis / "ratio" / (scene + ("_smoke" if args.smoke else "")))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    success = out_dir / "_SUCCESS"
    if success.exists() and not args.force:
        print(f"[stage04] SKIP {scene}: _SUCCESS present ({success})")
        return 0

    if not traj_path.exists():
        raise FileNotFoundError(f"trajectories not found: {traj_path}")
    tdf = pd.read_parquet(traj_path)
    tdf = tdf[(tdf["region"] == args.region) & (tdf["fov"] == args.fov)].copy()
    tps = sorted(tdf["timepoint"].unique().tolist())
    if args.smoke:
        tps = [t for t in tps if t in SMOKE_TPS]
        tdf = tdf[tdf["timepoint"].isin(tps)]
    print(f"[stage04] {scene}: {len(tdf)} rows over {len(tps)} timepoints "
          f"(t{tps[0]}..t{tps[-1]}), expansion_px={args.expansion_px}", flush=True)

    t_start = time.time()
    measures, bg_rows = {}, []
    n_missing = 0
    for k, tp in enumerate(tps):
        rows_tp = tdf[tdf["timepoint"] == tp]
        mask_path = rows_tp["nuclear_mask_path"].iloc[0]
        mask = np.load(mask_path)
        raw = raw_channel_path(data_root, args.acq, args.region, args.fov, tp, args.channel)
        if raw is None:
            n_missing += 1
            print(f"[stage04]   t{tp:04d}: NO raw {args.channel} TIFF -> NaN", flush=True)
            bg_rows.append({"timepoint": tp, "bg_p05": np.nan, "bg_p50": np.nan, "bg_p97": np.nan})
            continue
        img = _imread(raw)
        if img.shape != mask.shape:
            raise ValueError(f"t{tp}: raw {img.shape} != mask {mask.shape} ({raw})")
        per_label, bg = measure_frame(mask, img, args.expansion_px)
        measures[tp] = per_label
        bg_rows.append({"timepoint": tp, **bg})
        if k % 50 == 0 or k == len(tps) - 1:
            print(f"[stage04]   {k+1}/{len(tps)} t{tp:04d} cells={len(per_label)} "
                  f"({(time.time()-t_start):.0f}s)", flush=True)

    comp_cols = ["nuc_mean", "nuc_median", "nuc_sum", "nuc_area_px",
                 "ring_mean", "ring_median", "ring_sum", "ring_area_px"]
    bg_by_tp = {r["timepoint"]: r for r in bg_rows}
    out_rows = []
    for r in tdf.itertuples(index=False):
        d = {c: getattr(r, c) for c in SCHEMA_PASSTHROUGH}
        comp = measures.get(r.timepoint, {}).get(int(r.cell_in_frame_id))
        for c in comp_cols:
            d[c] = comp[c] if comp else np.nan
        bg = bg_by_tp.get(r.timepoint, {})
        d["bg_p05"] = bg.get("bg_p05", np.nan)
        d["bg_p50"] = bg.get("bg_p50", np.nan)
        d["bg_p97"] = bg.get("bg_p97", np.nan)
        out_rows.append(d)
    df = pd.DataFrame(out_rows)

    # EXPLORATORY ratios only — policy not yet locked; recompute downstream freely.
    eps = 1e-9
    df["ktr_cn_raw_exploratory"] = df["ring_mean"] / (df["nuc_mean"] + eps)
    nuc_bg = (df["nuc_mean"] - df["bg_p05"]).clip(lower=eps)
    df["ktr_cn_bgsub_exploratory"] = (df["ring_mean"] - df["bg_p05"]) / nuc_bg
    df["ktr_cn_median_exploratory"] = df["ring_median"] / (df["nuc_median"] + eps)

    out_parquet = out_dir / "per_track_ratio.parquet"
    df.to_parquet(out_parquet, engine="pyarrow", index=False)

    valid = df["ktr_cn_raw_exploratory"].replace([np.inf, -np.inf], np.nan).dropna()
    meta = {
        "STAGE04_VERSION": "v1-2026-05-24",
        "operation": "per-track-ktr-cn-ratio",
        "scene": scene,
        "inputs": {"trajectories": str(traj_path),
                   "raw_channel": args.channel, "acquisition": args.acq,
                   "nuclear_masks": "per-row nuclear_mask_path (stardist_h2b)"},
        "params": {"expansion_px": args.expansion_px,
                   "ring_def": "expand_labels(distance=expansion_px) restricted to mask==0 (neighbour-excluded)",
                   "background": "per-frame percentiles of mask==0 pixels (p05/p50/p97); policy NOT locked",
                   "ratio_status": "EXPLORATORY — 3 variants stored; final C/N policy deferred"},
        "n_timepoints": len(tps), "n_rows": int(len(df)),
        "n_tracks": int(df["track_id"].nunique()),
        "n_frames_missing_raw": n_missing,
        "ktr_cn_raw_exploratory": {
            "median": float(valid.median()) if len(valid) else None,
            "mean": float(valid.mean()) if len(valid) else None,
            "p10": float(valid.quantile(.10)) if len(valid) else None,
            "p90": float(valid.quantile(.90)) if len(valid) else None,
            "note": "sanity anchor: prior frame-wise cyto-ring run reported median KTR_CN ~1.035",
        },
        "runtime_sec": round(time.time() - t_start, 1),
        "outputs": {"per_track_ratio_parquet": str(out_parquet)},
        "pid": os.getpid(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    success.write_text(json.dumps({
        "scene": scene, "n_rows": int(len(df)), "n_tracks": int(df["track_id"].nunique()),
        "median_ktr_cn_raw": meta["ktr_cn_raw_exploratory"]["median"],
        "runtime_sec": meta["runtime_sec"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2))

    print(f"[stage04] done: {len(df)} rows, {df['track_id'].nunique()} tracks, "
          f"median KTR C/N (raw, exploratory)={meta['ktr_cn_raw_exploratory']['median']} "
          f"in {meta['runtime_sec']}s", flush=True)
    print(f"[stage04] out: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
