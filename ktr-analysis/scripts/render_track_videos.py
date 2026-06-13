"""Render a simple track-overlay video for one tracking output (Stage 03 review).

Goal: a Bill-style *simple* movie the human can eyeball for identity breaks and
divisions — NOT a quantitative figure. Design choices:

  - Background: chosen raw channel (default BF), percentile-normalised, grayscale.
  - Mask outlines are coloured by ``track_id`` with a *deterministic* colour
    (same track -> same colour for the whole movie). So an ID swap shows up as a
    cell that suddenly changes colour; a division shows up as two daughters
    taking new colours from the mother.
  - Division births (rows with a non-null ``parent_track_id`` at their first
    frame) get a hollow ring marker for a few frames so the eye catches them.

Reads per-frame label masks ``t%04d.npy`` (uint32, label == ``cell_in_frame_id``)
and the tracker ``trajectories.parquet``. Writes PNG frames to a temp dir then
encodes an mp4 with the system ffmpeg.

Run in ultrack_env (has pandas+pyarrow+tifffile+skimage+imageio):
    /opt/miniconda/envs/ultrack_env/bin/python scripts/render_track_videos.py \
        --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse \
        --traj   .../bf_ultrack/R0_fov0/trajectories.parquet \
        --masks  .../segmentation/validation/R0_fov0/cellpose_bf \
        --channel BF --t0 0 --t1 100 --out .../_review/bf_ultrack_t0-100.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.io import get_fov_timeseries, parse_dataset  # noqa: E402


def _normalize(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros(img.shape, np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def _track_color_lut(max_track_id: int, seed: int = 0) -> np.ndarray:
    """Deterministic vivid RGB per track_id (index 0 reserved -> dim grey)."""
    rng = np.random.default_rng(seed)
    hues = rng.permutation(max_track_id + 1) / max(max_track_id, 1)
    import colorsys

    lut = np.zeros((max_track_id + 1, 3), np.uint8)
    for tid in range(1, max_track_id + 1):
        r, g, b = colorsys.hsv_to_rgb(float(hues[tid]), 0.85, 1.0)
        lut[tid] = (np.array([r, g, b]) * 255).astype(np.uint8)
    lut[0] = (90, 90, 90)  # unmapped detection
    return lut


def _draw_ring(rgb: np.ndarray, cy: int, cx: int, radius: int = 14) -> None:
    from skimage.draw import circle_perimeter

    h, w = rgb.shape[:2]
    for rr in (radius, radius + 1):
        yy, xx = circle_perimeter(cy, cx, rr, shape=(h, w))
        rgb[yy, xx] = (255, 255, 255)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--traj", required=True, type=Path, help="trajectories.parquet")
    ap.add_argument("--masks", required=True, type=Path, help="dir with t%04d.npy")
    ap.add_argument("--out", required=True, type=Path, help="output .mp4")
    ap.add_argument("--channel", default="BF", choices=["BF", "mTagBFP2", "mKate2"])
    ap.add_argument("--region", default="R0")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--acquisition", default="timelapse_2026-05-05_18-12-11.466141")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=100)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--div-marker-frames", type=int, default=4,
                    help="frames to keep a division ring visible after birth")
    args = ap.parse_args()

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.traj)
    df = df[(df["region"] == args.region) & (df["fov"] == args.fov)].copy()
    max_tid = int(df["track_id"].max())
    lut = _track_color_lut(max_tid)

    # division births: first frame of each track that has a parent
    births: dict[int, list[tuple[int, int]]] = {}
    daughters = df[df["parent_track_id"].notna()]
    for tid, g in daughters.groupby("track_id"):
        first = g.loc[g["timepoint"].idxmin()]
        births.setdefault(int(first["timepoint"]), []).append(
            (int(round(first["centroid_y"])), int(round(first["centroid_x"])))
        )

    # raw frame paths via the repo's io loader
    fdf = parse_dataset(args.data_root / args.acquisition)
    series = get_fov_timeseries(fdf, args.region, args.fov)
    raw_paths = series[args.channel]

    tmp = Path(tempfile.mkdtemp(prefix="trkvid_"))
    selem = disk(1)
    n = 0
    try:
        for i, t in enumerate(range(args.t0, args.t1 + 1)):
            mpath = args.masks / f"t{t:04d}.npy"
            if t >= len(raw_paths) or not mpath.exists():
                print(f"  [skip] t={t} (missing raw or mask)", flush=True)
                continue
            raw = tifffile.imread(raw_paths[t])
            mask = np.load(mpath, allow_pickle=True).astype(np.int64)

            bg = (_normalize(raw) * 255).astype(np.uint8)
            rgb = np.repeat(bg[:, :, None], 3, axis=2)

            # label -> track_id for this frame
            sub = df[df["timepoint"] == t]
            lab2trk = np.zeros(int(mask.max()) + 1, np.int64)
            labs = sub["cell_in_frame_id"].to_numpy()
            trks = sub["track_id"].to_numpy()
            inb = labs <= mask.max()
            lab2trk[labs[inb].astype(int)] = trks[inb].astype(int)

            trk_img = lab2trk[mask]  # H,W of track ids (0 where unmapped/bg)
            bnd = find_boundaries(mask, mode="inner")
            bnd = dilation(bnd, selem)
            color = lut[np.clip(trk_img, 0, max_tid)]
            rgb[bnd] = color[bnd]

            # division rings (this frame + a short tail)
            for bt in range(t - args.div_marker_frames + 1, t + 1):
                for (cy, cx) in births.get(bt, []):
                    _draw_ring(rgb, cy, cx)

            iio.imwrite(tmp / f"f{i:05d}.png", rgb)
            n += 1
            if i % 25 == 0:
                print(f"  t={t} ({i+1} frames)", flush=True)

        if n == 0:
            print("ERROR: no frames rendered", file=sys.stderr)
            return 2

        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp / "f%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            return 3
        print(f"OK  {out}  ({n} frames, {args.fps} fps)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
